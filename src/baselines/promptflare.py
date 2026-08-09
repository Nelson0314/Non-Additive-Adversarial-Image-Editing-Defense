"""PromptFlare —— `NAHOHYUN-SKKU/PromptFlare`，HEAD `5e9ad004b0e06cfe69608a65e38d2aa24f870142`。

逐行佐證見 `docs/_audit_promptflare_photoguard.md` §PromptFlare 1–5，
裁決見 `docs/reference/SOURCE_AUDIT.md` §2 與 §2.5。

必須換掉 attention processor，不可改用 pre-hook
──────────────────────────────────────────────────────────────────────

`promptflare.py:66-68` 對 UNet 中所有名稱以 `attn2` 結尾的模組
`set_processor(MyAttnProcessor2_0(...))`。本專案的 `src/models/attention.py`
刻意不換 processor（理由見該檔 docstring：換掉會改變 UNet 自己的注意力
計算路徑，「有沒有開這個目標」不再是單一變因），但**那是為了擷取分佈供
我們自己的損失使用**，是另一件事。

PromptFlare 記錄的是 cross-attention 模組**經 `to_out` 之後的輸出**
（`A·V·W_out`，不是論文 Eq. 10 的 `A·V`），其值取決於注意力如何計算。
改用 pre-hook 重算會得到另一個量，那就變成我們設計的變體。故此處照原樣
換 processor，並與 `src/models/attention.py` 完全隔離、不共用擷取層。

三項落差
──────────────────────────────────────────────────────────────────────

1. **eps 實際只有論文宣稱的一半。** `protect.py:23` 的 `eps /= 255` 施加在
   `utils.py:10` 產生的 `[-1,1]` 張量上，故論文的 12/255 在像素域只有
   **6/255**（以 repo 內 sample 實測原圖與 protected 圖的最大絕對差為 7，
   多數 ≤ 6，與推論一致）。照論文數字實作等於給它兩倍預算。
2. `cal_loss` 是**跨層加總**，不是論文 Eq. 12 的 `E_l`（對層取期望）。
   sign 更新下不影響方向，但損失曲線的尺度隨層數而變。
3. `loss_depth = [1024, 256, 64]` 是 **token 數白名單**（依 latent 解析度
   篩層），不是層編號。`h = 4096`（64×64）那一層不在白名單內。
   **此值與影像解析度綁定**：原作寫死的清單只在 512² 成立。本移植把它改寫
   成它所依據的規則——「觀察到的層級去掉最外（token 數最大）那一級」——
   並由 `resolve_loss_depth` 在每次 `prepare` 時實測決定。該規則在
   512²／SD1.x 上還原原值（測試釘住），在 SDXL／1024² 上給出 `{1024}`
   （70 層中的 60 層）。推導見 `resolve_loss_depth` 的 docstring。

另：全 repo 無任何 seed，`torch.randn` 未固定，原作的保護結果不可逐位元
重現。本實作以顯式 generator 固定，屬我方加強（不改變方法）。

移植到 img2img（`modified_from_paper=True`）
──────────────────────────────────────────────────────────────────────

原作是 inpainting：`x_adv` 只經由 9 通道輸入的 `masked_image_latents`
進入計算圖（`promptflare.py:82-94`：`masked_adv = adv * (1 - cur_mask)`，
梯度亦乘 `(1 - cur_mask)`），而 `latents` 是與 `x_adv` 無關的隨機噪聲。
img2img 沒有那個通道，若照搬「latents 為隨機噪聲」，損失對 `x_adv` 的
梯度**恆為零**。故本移植把 `x_adv` 改由 `E(x_adv)` 加噪到 `strength`
對應的 t 進入 latent，並把 mask 設為全圖（損失涵蓋整張、擾動涵蓋整張）。
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.pgd import BaselineSpec, ValueRange
from src.models.sd import cat_cond

PROMPTFLARE_RANGE = ValueRange(
    -1.0,
    1.0,
    "utils.py:10 `torch.from_numpy(image)/127.5 - 1.0`；promptflare.py:106 clamp(-1, 1)",
)

# `promptflare.py:73`
QUALITY_TAG_PROMPT = (
    "professional photography, best quality, ultra high res, photo, art, "
    "high quality, realistic, anime, masterpiece, best quality, artistic, "
    "detail, 4k, 8k"
)

# `promptflare.py:8-11`，硬編碼、無 CLI 開關。
PF_NUM_INFERENCE_STEPS = 4
PF_K = 1
PF_LOSS_MASK = True
PF_LOSS_DEPTH = (1024, 256, 64)


class AttnController:
    """`attention_control.py:5-54` 逐字移植。

    `pred` 是批次第 0 項（未遮蔽的完整 prompt），`target` 是批次第 1 項
    （只保留 BOS）。被 `detach()` 的是 **BOS-masked 那一側**，即最佳化把
    「正常 cross-attention 輸出」拉向「只 attend BOS 的輸出」；論文 Eq. 11
    未指明哪一側 detach。
    """

    def __init__(self, mask: torch.Tensor):
        self.attn2_preds: List[torch.Tensor] = []
        self.attn2_targets: List[torch.Tensor] = []
        self.masks = {512: mask}

    def __call__(self, hidden_states, module_name):
        hidden_states = hidden_states.clone()
        pred, target = hidden_states.chunk(2)
        _, h, _ = pred.shape
        self.make_mask(h)
        if module_name.endswith("attn2"):
            self.attn2_preds.append(pred)
            self.attn2_targets.append(target)

    def reset(self):
        self.attn2_preds = []
        self.attn2_targets = []

    def make_mask(self, h):
        if h not in self.masks:
            size = int(np.sqrt(h))
            new_mask = F.interpolate(self.masks[512], size=(size, size))
            new_mask = new_mask.flatten().unsqueeze(0).unsqueeze(2)
            self.masks[h] = new_mask

    def set_mask(self, mask):
        self.masks = {512: mask}

    def cal_loss(self, loss_mask, loss_depth):
        text_losses = 0
        for pred, target in zip(self.attn2_preds, self.attn2_targets):
            _, h, _ = pred.shape
            text_loss = 0
            if h in loss_depth:
                if loss_mask:
                    pred = pred * self.masks[h]
                    target = target * self.masks[h]
                text_loss = (pred - target.detach()).norm(p=2)
            text_losses += text_loss
        self.reset()
        return text_losses


class MyAttnProcessor2_0:
    """`attention_control.py:56-143` 逐字移植，僅一處為版本相容的改寫。

    改寫：原作對 `to_q/to_k/to_v/to_out[0]` 傳 `scale=scale`
    （diffusers 0.21.2 的 `LoRACompatibleLinear` 簽名）。PromptFlare 從未
    傳入非預設的 `scale`，其值恆為 1.0，而 1.0 對非 LoRA 線性層是恆等；
    新版 diffusers 的 `nn.Linear` 不接受該引數。故此處不傳，數值等價。

    `attention_mask` 分支保留：BOS 遮罩就是經由 UNet 的
    `encoder_attention_mask` 轉成 `-10000.0` 偏置後走這一路
    （diffusers `unet_2d_condition.py` 的
    `(1 - mask) * -10000.0`，即論文 Eq. 9 的常數 c）。
    """

    def __init__(self, attn_controller, module_name):
        self.attn_controller = attn_controller
        self.module_name = module_name
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MyAttnProcessor2_0 需要 PyTorch 2.0 以上")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if args or kwargs:
            # 新版 diffusers 若多傳了引數，靜默忽略會讓注意力算出別的東西
            # 而完全沒有症狀。此處直接中止。
            raise RuntimeError(
                f"{self.module_name}：processor 收到未預期的引數 "
                f"args={args} kwargs={list(kwargs)}；"
                "PromptFlare 的 processor 是逐字移植的，多出來的引數必須先查證其語意"
            )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.module_name.endswith("attn2"):
            self.attn_controller(hidden_states, self.module_name)
        return hidden_states


class PromptFlareContext:
    """安裝／還原 processor 的責任在這裡。`run_pgd` 結束時呼叫 `close()`。"""

    def __init__(self, sd, spec, emb2, encoder_attention_mask, t0,
                 controller, loss_depth, generator, mask=None):
        self.spec = spec
        # 給定時回到原作：latents 為隨機噪聲，x_adv 只經後 4 個通道進入。
        self.mask = mask
        self.emb2 = emb2
        self.encoder_attention_mask = encoder_attention_mask
        self.t0 = t0
        self.controller = controller
        self.loss_depth = loss_depth
        self.generator = generator
        self._sd = sd
        self._saved = []
        for n, m in sd.unet.named_modules():
            if n.endswith("attn2"):
                self._saved.append((m, m.get_processor()))
                m.set_processor(MyAttnProcessor2_0(controller, n))
        if not self._saved:
            raise RuntimeError("UNet 中找不到任何 attn2 層")

    def close(self):
        """還原原本的 processor。不還原會污染共用同一個 SDWrapper 的後續實驗。"""
        for module, proc in self._saved:
            module.set_processor(proc)
        self._saved = []


def observed_token_counts(sd, ctx: "PromptFlareContext", h: int, w: int) -> List[int]:
    """跑一次空白前向，收集各 attn2 層實際看到的 token 數。

    不由 UNet config 推導，是因為推導要同時假設「哪些 block 有 attention」
    與「各 block 的下採樣倍率」；那兩個假設在 SD1.x 與 SDXL 之間不同，
    寫錯不會有症狀（白名單少涵蓋幾層，損失照樣有值）。實際前向一次即得，
    成本相對 400 步 PGD 可忽略。
    """
    z = torch.zeros(
        sd.latent_shape(h, w), device=ctx.emb2.device, dtype=sd.unet.dtype
    )
    ctx.controller.reset()
    with torch.no_grad():
        sd.unet_forward(
            torch.cat([z] * 2),
            ctx.t0,
            ctx.emb2,
            encoder_attention_mask=ctx.encoder_attention_mask,
        )
    counts = sorted({int(p.shape[1]) for p in ctx.controller.attn2_preds})
    ctx.controller.reset()
    return counts


def resolve_loss_depth(sd, ctx: "PromptFlareContext", h: int, w: int) -> List[int]:
    """決定 token 數白名單：**觀察到的層級去掉最外（token 數最大）那一級**。

    原作把 `loss_depth` 寫死為 `[1024, 256, 64]`（`promptflare.py:11`），
    對應 512² 影像、latent 64² 下的 attn2 層級 `{4096, 1024, 256, 64}`
    去掉 4096。原值與解析度綁定，SDXL／1024² 沿用會涵蓋錯的層且無症狀
    （見 `SOURCE_AUDIT §2.5`）。

    此處實作的規則在 512²／SD1.x 上**還原原值**（由
    `tests/test_baselines.py` 釘住），故不是另立一套判準，而是把原作寫死的
    那個清單改寫成它所依據的規則。兩種讀法在 SDXL 上給出同一個答案：

    - 逐字讀「token 數白名單 `{1024, 256, 64}`」：SDXL／1024² 的 attn2 只有
      `{4096, 1024}` 兩級，交集為 `{1024}`。
    - 讀成「排除最外層」：去掉 4096，同樣得 `{1024}`。

    兩者一致，故此項不是自由選擇。SDXL 的 70 層 attn2 中，10 層在 4096
    （down_blocks[1] 4 層、up_blocks[1] 6 層），其餘 60 層在 1024。
    """
    counts = observed_token_counts(sd, ctx, h, w)
    if len(counts) < 2:
        raise NotImplementedError(
            f"只觀察到 {len(counts)} 個 attn2 層級（token 數 {counts}）。"
            "「排除最外層」之後白名單會是空的，損失恆為 0 且無症狀。"
            "此形狀下必須另行決定白名單"
        )
    depth = counts[:-1]
    if (h, w) == (512, 512) and counts[-1] == 4096:
        # 原作的情形。規則若在此處與寫死的清單不符，就是規則寫錯了。
        expected = sorted(PF_LOSS_DEPTH)
        if depth != expected:
            raise RuntimeError(
                f"512² 下推導出的白名單 {depth} 與原作寫死的 "
                f"{expected} 不符；規則有誤，不得沿用"
            )
    return depth


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    strength: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    mask01: Optional[torch.Tensor] = None,
    prompt: str = QUALITY_TAG_PROMPT,
    seed: int = 0,
    **_,
) -> PromptFlareContext:
    if strength is None and mask is None:
        raise NotImplementedError(
            "PromptFlare 的 img2img 版缺 strength：原作是 inpainting，latent 由 "
            "torch.randn 起跑（`promptflare.py:29-31`），x_adv 只經 masked_image_latents "
            "進入計算圖，因此原始碼沒有『攻擊時的 strength』這個數。"
            "請由呼叫端依本專案的威脅模型指定"
        )

    h, w = x01.shape[-2], x01.shape[-1]
    device = x01.device
    if mask01 is None:
        # 本輪設為全圖：損失涵蓋整張影像。原作的 mask 為待重繪區，
        # 且噪聲只加在保留區；img2img 沒有這個切分，見模組 docstring。
        mask01 = torch.ones(1, 1, h, w, device=device, dtype=x01.dtype)

    emb = sd.encode_text(prompt).detach()
    # `promptflare.py:22`：兩列是同一個 prompt。`cat_cond` 而非 `.repeat`——
    # SDXL 的條件是 SDXLPrompt，沒有 `repeat` 方法，且 pooled 那一半也要跟著複製。
    emb2 = cat_cond([emb, emb])

    # `promptflare.py:40-41`：第 0 列全 1（不遮罩），第 1 列只留位置 0（BOS）。
    encoder_attention_mask = torch.ones(2, 77, device=device, dtype=emb.dtype)
    encoder_attention_mask[1][1:] = 0

    # 原作以 `k=1` 只在排程的第一個 timestep 上算一次損失，而該排程是
    # inpainting pipeline 的預設（strength=1）。img2img 的第一個 timestep
    # 由呼叫端的 strength 決定，那是本專案的移植。
    t0 = (sd.num_train_timesteps - 1 if mask is not None else
          min(int(sd.num_train_timesteps * strength),
              sd.num_train_timesteps - 1))
    controller = AttnController(mask01)
    generator = torch.Generator(device=device).manual_seed(seed)
    ctx = PromptFlareContext(
        sd, spec, emb2, encoder_attention_mask, t0, controller,
        None, generator, mask=mask,
    )
    ctx.loss_depth = resolve_loss_depth(sd, ctx, h, w)
    return ctx


def loss_fn(sd, x_adv: torch.Tensor, ctx: PromptFlareContext) -> torch.Tensor:
    """`k = 1`，只在第一個 timestep 上算一次損失（`promptflare.py:44-58`）。

    `torch.stack(text_losses).mean()` 在 `k=1` 下只是對單一元素取平均，
    `latents = pred_noise` 是死碼，兩者都不影響結果。
    """
    ctx.controller.reset()
    if ctx.mask is not None:
        # **原生形態**（`promptflare.py:82-94`）：`latents` 是與 `x_adv` 無關的
        # 隨機噪聲，`x_adv` 只經 `masked_adv = adv * (1 - cur_mask)` 之後的
        # `masked_image_latents` 進入計算圖。img2img 沒有那個通道，照搬會讓
        # 損失對 `x_adv` 的梯度**恆為零**——那正是本專案改由加噪 latent 進入
        # 的理由（見模組 docstring）。遮罩給定時切回原作。
        lat = sd.latent_shape(x_adv.shape[-2], x_adv.shape[-1])
        with torch.no_grad():
            z = torch.randn(lat, generator=ctx.generator, device=x_adv.device,
                            dtype=x_adv.dtype)
        m = F.interpolate(ctx.mask.to(z.device, z.dtype), size=z.shape[-2:],
                          mode="nearest")
        mpost = sd.vae.encode(
            (x_adv * (1.0 - ctx.mask.to(x_adv))).to(sd.vae.dtype)).latent_dist
        z_masked = mpost.sample(ctx.generator) * sd.scaling_factor
        z = torch.cat([z, m, z_masked.to(z.dtype)], dim=1)
    else:
        post = sd.vae.encode(x_adv.to(sd.vae.dtype)).latent_dist
        z0 = post.sample(ctx.generator) * sd.scaling_factor
        noise = torch.randn(
            z0.shape, generator=ctx.generator, device=z0.device, dtype=z0.dtype
        )
        abar = sd.alphas_cumprod(z0.device)[ctx.t0].to(z0.dtype)
        z = abar.sqrt() * z0 + (1 - abar).sqrt() * noise
    z2 = torch.cat([z] * 2)

    # `sd.unet_forward` 而非 `sd.unet`：SDXL 的條件是 SDXLPrompt，UNet 另需
    # added_cond_kwargs。`encoder_attention_mask` 原樣轉交——它作用在 token
    # 軸（77），SDXL 串接兩個 encoder 改變的是最後一維，token 軸不變，
    # 故 BOS decoy 的語意在兩種模型上一致。
    sd.unet_forward(
        z2,
        ctx.t0,
        ctx.emb2,
        encoder_attention_mask=ctx.encoder_attention_mask,
    )
    return ctx.controller.cal_loss(
        loss_mask=PF_LOSS_MASK, loss_depth=ctx.loss_depth
    )


SPEC = BaselineSpec(
    name="promptflare",
    source="NAHOHYUN-SKKU/PromptFlare@5e9ad004b0e06cfe69608a65e38d2aa24f870142",
    value_range=PROMPTFLARE_RANGE,
    eps=12.0 / 255.0,            # `protect.py:23` eps/=255，施加於 [-1,1]（沒有乘 2）
    eps_pixel01=6.0 / 255.0,     # 論文宣稱 12/255，實際只有一半
    norm="linf",
    steps=400,                   # `protect.py:54` --epochs 預設 400
    step_size=2.0 / 255.0,       # `protect.py:24` step/=255，同樣施加於 [-1,1]
    step_schedule="constant",    # 無步長衰減
    update_rule="sign",
    objective="minimize",        # `adv = adv - avg_grad.sign() * step_size`
    init_rule="none",            # `adv = src_image_orig.clone()`，無隨機初始化
    grad_reps=1,                 # `promptflare.py:75`
    needs_target_image=False,
    needs_mask=True,             # 本輪設為全圖
    # `promptflare.py:82-94`：`masked_adv = adv * (1 - cur_mask)`，
    # 且**梯度亦乘 `(1 - cur_mask)`**。與 PhotoGuard-c 同一項忠實處置。
    grad_outside_mask=True,
    modified_from_paper=True,
    modification_note=(
        "(1) mask 設為全圖：損失涵蓋整張影像，且不再有「噪聲只加在保留區」的切分；"
        "(2) 由 inpainting 移植到 img2img：原作 x_adv 只經 9 通道的 masked_image_latents "
        "進入計算圖、latents 為隨機噪聲；img2img 沒有該通道，若照搬則梯度恆為零，"
        "故改由 E(x_adv) 加噪到 strength 對應的 t 進入 latent；"
        "(3) timesteps[0] 改由 strength 決定，原作為 4 步排程的最大 t；"
        "(4) 原作全 repo 無 seed，本實作以顯式 generator 固定隨機性。"
    ),
    discrepancy_note=(
        "eps 論文宣稱 12/255，實際施加於 [-1,1] 故像素域只有 6/255（sample 實測最大差 7）；"
        "step size 同理為 1/255；cal_loss 是跨層加總而非論文 Eq. 12 的 E_l；"
        "記錄的是 A·V·W_out（to_out 之後）而非論文 Eq. 10 的 A·V；"
        "使用 SDPA 融合核，attention map 從未實體化，不記錄 Q/K/V 也不記錄 softmax 機率；"
        "全 repo 無 seed，原作結果不可逐位元重現。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "num_inference_steps": PF_NUM_INFERENCE_STEPS,
        "k": PF_K,
        "loss_mask": PF_LOSS_MASK,
        "loss_depth": list(PF_LOSS_DEPTH),
        "loss_depth_meaning": (
            "原作寫死的 token 數白名單（非層編號），只在 512² 成立；"
            "實際使用的白名單由 resolve_loss_depth 依「去掉最外一級」實測決定，"
            "512² 下還原此值，SDXL/1024² 下為 [1024]"
        ),
        "prompt": QUALITY_TAG_PROMPT,
        "attn_layers": "只有 attn2（attn1 在原始碼中被註解掉）",
        "model": "runwayml/stable-diffusion-inpainting（原作）",
    },
)
