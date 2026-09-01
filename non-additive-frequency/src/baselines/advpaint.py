"""AdvPaint —— Jeon et al.（`JoonsungJeon/AdvPaint`）。

出處：官方 `AdvPaint.py`（本次以 raw 檔逐行核對，行為與
`docs/reference/SOURCE_AUDIT.md` §1 一致，另補一項該文件未記的落差，見下）。
逐字佐證見 `docs/_audit_advpaint_dia_promptflare.md` §1；下列行號以該文件
記載的 raw 檔（`main` 分支，13891 bytes）為準。

損失只有四個成分
──────────────────────────────────────────────────────────────────────

    loss = (loss_query + loss_key + loss_value + loss_cross_q) / length

    loss_query   -= (GT_self_query[t][path]  - self_query[t][path]).norm(p=2)
    loss_key     -= (GT_self_key[t][path]    - self_key[t][path]).norm(p=2)
    loss_value   -= (GT_self_value[t][path]  - self_value[t][path]).norm(p=2)
    loss_cross_q -= (GT_cross_query[t][path] - cross_query[t][path]).norm(p=2)

即 **self-attention 的 Q／K／V 三者 + cross-attention 的 Q 一者**。
cross-attention 的 K 與 V **不在損失內**。論文摘要的「targets the self- and
cross-attention blocks」照字面實作會多攻擊兩個成分，那是另一個方法。

`length = len(self_query[timestep].keys())`，即單一 timestep 下記錄到的
path 數（self-attention 層數）。它是常數，sign 更新下不改變方向。

四項與 `SOURCE_AUDIT` §1.4 的補充／修正
──────────────────────────────────────────────────────────────────────

1. **有隨機初始化。** `AdvPaint.py:76`
   `X_adv = X.clone().detach() + ((torch.rand(*X.shape)*2*eps-eps).to("cuda"))`。
   SOURCE_AUDIT §1.4 未記此項；漏掉它會讓起點不同、最終解落在球面的
   另一側。該行位於 mask 迴圈**之外**，即多個 mask 共用同一個起點且
   `X_adv` 跨 mask 累積。
2. eps／step_size 的函式預設為 0.06／0.03（`AdvPaint.py:57` 的簽章），
   **CLI 預設為 0.1／0.05**（`AdvPaint.py:372-373`）。依裁決取論文正文
   那組（0.06／0.03），因為那是論文報告數字所依據的設定。
3. `--prompt` 是 **required 且無預設**的 CLI 參數（`AdvPaint.py:370-371`），
   攻擊時以 `guidance_scale=7.5`（`AdvPaint.py:100`）走 classifier-free
   guidance（`prompt_embeds` 為 `[uncond; cond]` 兩列）。故「攻擊時用什麼
   prompt」沒有原始碼可依，必須由呼叫端指定。
4. **原作的梯度只走 masked-image 那一路。** `AdvPaint.py:206-214` 把
   `vae.encode(X_adv)` → `add_noise` 整段包在 `torch.no_grad()` 內，
   真正可微的是 `masked_image = X_adv * (mask_512 < 0.5)`（`AdvPaint.py:219`）
   之後的 `masked_image_latents`（9 通道 inpainting 輸入的後 4 通道）。
5. **GT 與每次迭代的雜訊是不同的抽樣。** GT 段（`AdvPaint.py:124-129`）
   與迭代段（`AdvPaint.py:207-214`）都呼叫 `latent_dist.sample(model.generator)`
   與 `randn_tensor(..., generator=model.generator)`，產生器狀態在兩者之間
   前進，故 GT 的 `noise` 與第 1…100 次迭代的 `noise` 兩兩皆不同。
   本移植照樣重現（`_forward_and_record` 共用同一個 generator）。

移植到 img2img（`modified_from_paper=True`）
──────────────────────────────────────────────────────────────────────

| 原始（inpainting） | 本移植（img2img） |
|---|---|
| 9 通道 `cat([latents, mask, masked_image_latents])` | 4 通道 |
| 梯度只經 masked-image latent | img2img 沒有那一路，梯度改經加噪後的 latent |
| 逐 mask 檔迴圈，`X_ori_masked = X_ori * (mask_512 < 0.5)` | mask 設為全圖，只跑一次 |
| `timesteps[0]` 來自 inpainting pipeline 的預設排程（50 步、strength=1） | 由呼叫端的 `strength` 決定，取 `int(num_train × strength)`，與 `SDWrapper.sdedit` 同式 |

第二列是**實質**改動：原作攻擊的是「模型看到的已知區域」如何影響注意力，
img2img 沒有已知／未知之分，只能經由整張圖的 latent 施力。
"""

from typing import List, Optional

import torch

from src.baselines.pgd import BaselineSpec, ValueRange
from src.models.sd import cat_cond

ADVPAINT_RANGE = ValueRange(
    -1.0,
    1.0,
    "AdvPaint.py:346 `preprocess(init_image)` → [-1,1]；pgd 呼叫 clamp_min=-1, clamp_max=1",
)


class QKVRecorder:
    """以 forward pre-hook 記錄各 attention 層的 Q／K／V。

    為什麼可以用 pre-hook 而不必換 processor（PromptFlare 則不可）：
    這裡要的是 `to_q` / `to_k` / `to_v` 三個線性層對**該層輸入**的投影。
    無論由哪個 processor 執行注意力，這三個投影都是同一組權重作用在同一個
    輸入上，數值相同；而 PromptFlare 記錄的是注意力**輸出**，其值取決於
    注意力如何計算（SDPA 融合核 vs 手寫），故不可代換。

    `.norm(p=2)` 是整個張量的 Frobenius norm，對 reshape／permute 不變，
    因此「原作存的是 head 展開後的形狀」不影響損失值。
    """

    def __init__(self, unet):
        self.unet = unet
        self.self_q: List[torch.Tensor] = []
        self.self_k: List[torch.Tensor] = []
        self.self_v: List[torch.Tensor] = []
        self.cross_q: List[torch.Tensor] = []
        self._handles = []
        self._layers = [
            (n, m)
            for n, m in unet.named_modules()
            if n.endswith("attn1") or n.endswith("attn2")
        ]
        if not self._layers:
            raise RuntimeError("UNet 中找不到任何 attn1／attn2 層")

    def _make_hook(self, name):
        is_self = name.endswith("attn1")

        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            if len(args) > 1:
                encoder_hidden_states = args[1]
            else:
                encoder_hidden_states = kwargs.get("encoder_hidden_states")

            # 依 diffusers `AttnProcessor.__call__` 的順序處理 group_norm 與
            # norm_cross。SD v1.x 的 attn1／attn2 兩者皆為 None（實測），
            # 但換模型時不照做會算出不同的東西。
            hs = hidden_states
            if getattr(module, "group_norm", None) is not None:
                hs = module.group_norm(hs.transpose(1, 2)).transpose(1, 2)

            query = module.to_q(hs)
            if is_self:
                enc = hs
                if getattr(module, "norm_cross", None) is not None and \
                        encoder_hidden_states is not None:
                    enc = module.norm_encoder_hidden_states(encoder_hidden_states)
                self.self_q.append(query)
                self.self_k.append(module.to_k(enc))
                self.self_v.append(module.to_v(enc))
            else:
                self.cross_q.append(query)

        return hook

    def __enter__(self):
        for name, layer in self._layers:
            self._handles.append(
                layer.register_forward_pre_hook(
                    self._make_hook(name), with_kwargs=True
                )
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def clear(self):
        self.self_q, self.self_k, self.self_v, self.cross_q = [], [], [], []


class AdvPaintContext:
    def __init__(self, sd, spec, emb2, t0, latent_shape, generator, gt,
                 mask=None):
        self.spec = spec
        self.mask = mask
        self.emb2 = emb2
        self.t0 = t0
        self.latent_shape = latent_shape
        self.generator = generator
        self.gt = gt                       # (self_q, self_k, self_v, cross_q)
        self.length = len(gt[0])
        self.recorder = QKVRecorder(sd.unet)

    def close(self):
        pass


def _forward_and_record(sd, mask, x_paper, emb2, t0, generator, recorder,
                        differentiable: bool):
    """跑一次 UNet 前向並取回 Q／K／V。

    `latents = add_noise(scale_factor · sample(E(x)), noise, t0)`，
    再 `cat([latents]*2)` 供 CFG 使用（`AdvPaint.py:220-236`）。
    `noise` 每次呼叫都是新的一抽——原作對 GT 與每一次迭代都呼叫
    `randn_tensor(..., generator=model.generator)`，產生器狀態會前進，
    故 GT 與各迭代所用的雜訊本來就不同。此處照樣重現。
    """
    ctx = torch.enable_grad() if differentiable else torch.no_grad()
    with ctx:
        # **原作的梯度只走 masked-image 那一路。** `AdvPaint.py:206-214` 把
        # `vae.encode(X_adv)` → `add_noise` 整段包在 `torch.no_grad()` 內，
        # 真正可微的是 `masked_image = X_adv * (mask_512 < 0.5)`
        # （`AdvPaint.py:219`）之後的 `masked_image_latents`。
        #
        # img2img 沒有那一路，故本專案的移植改讓梯度經加噪後的 latent——
        # 那是移植表第二列標為「實質改動」的那一項。遮罩給定時此處回到
        # 原作：帶噪 latent 切斷梯度，改由後 4 個通道進入。
        z_ctx = torch.no_grad() if mask is not None else ctx
        with z_ctx:
            post = sd.vae.encode(x_paper.to(sd.vae.dtype)).latent_dist
            z0 = post.sample(generator) * sd.scaling_factor
            noise = torch.randn(
                z0.shape, generator=generator, device=z0.device, dtype=z0.dtype
            )
            abar = sd.alphas_cumprod(z0.device)[t0].to(z0.dtype)
            z = abar.sqrt() * z0 + (1 - abar).sqrt() * noise
        if mask is not None:
            # `mask_512 < 0.5` 即保留**未遮罩**的區域，等價於 x·(1−m)。
            #
            # **一項未由原始碼查證的細節**：此處的 `masked_image_latents` 取
            # 後驗抽樣（與上面的 z0 一致，也與 diffusers inpaint pipeline 的
            # `retrieve_latents(..., generator)` 一致），但原作取 sample 還是
            # mode 未在既有的逐行紀錄中載明。兩者差一個 std·ε 項，會改變損失
            # 的數值但不改變梯度所走的路徑。取回原始碼確認之前，此項列入
            # `discrepancy_note`。
            m = torch.nn.functional.interpolate(
                mask.to(z.device, z.dtype), size=z.shape[-2:], mode="nearest")
            mpost = sd.vae.encode(
                (x_paper * (1.0 - mask.to(x_paper))).to(sd.vae.dtype)
            ).latent_dist
            z_masked = mpost.sample(generator) * sd.scaling_factor
            z = torch.cat([z, m, z_masked.to(z.dtype)], dim=1)
        z2 = torch.cat([z] * 2)
        recorder.clear()
        with recorder:
            # `sd.unet_forward` 而非 `sd.unet`：SDXL 的條件是 SDXLPrompt，
            # UNet 另需 added_cond_kwargs（pooled 嵌入 + time_ids），少傳會
            # 直接報錯；同樣的程式碼在 SD v1.x 上卻能跑。
            sd.unet_forward(z2, t0, emb2)
        return (
            list(recorder.self_q),
            list(recorder.self_k),
            list(recorder.self_v),
            list(recorder.cross_q),
        )


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    prompt: Optional[str] = None,
    strength: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    seed: int = 9999,
    **_,
) -> AdvPaintContext:
    """建立 GT 注意力投影（取自**原圖**）。

    `seed` 預設 9999（`AdvPaint.py:317` 的全域 `torch.manual_seed(seed)`）。
    `guidance_scale` 預設 7.5（`AdvPaint.py:97`）。
    """
    if prompt is None:
        raise NotImplementedError(
            "AdvPaint 的攻擊 prompt 沒有原始碼可依：`--prompt` 是 required 且無預設的"
            "CLI 參數，論文亦未給定攻擊時使用哪個 prompt。請由呼叫端明確指定"
            "（本專案為 prompt-free，見 DESIGN §2.1），並在報表標註為我方設定"
        )
    if strength is None and mask is None:
        raise NotImplementedError(
            "AdvPaint 的 img2img 版缺 strength：原作在 inpainting pipeline 的預設排程上"
            "取 `timesteps[0]`（50 步、strength=1），img2img 的第一個 timestep 由 "
            "strength 決定，原始碼沒有這個數。請由呼叫端依本專案的威脅模型指定"
        )

    vr = spec.value_range
    device = x01.device
    x_paper = vr.from01(x01.detach())

    emb_cond = sd.encode_text(prompt).detach()
    # **無條件分支取 `uncond_prompt()` 而非 `encode_text("")`。**
    # 原作跑在 SD v1.x 上，兩者恆等；stock SDXL base 的
    # `force_zeros_for_empty_prompt=true`，其無條件分支是零張量。
    # 威脅模型要求攻擊方是 stock 模型，用 `encode_text("")` 模擬的就不是
    # stock 行為，而該差異沒有任何症狀——影像照樣生得出來。
    emb_uncond = sd.uncond_prompt().detach()
    if guidance_scale == 1.0:
        raise NotImplementedError(
            "AdvPaint 固定以 guidance_scale=7.5 走 CFG（prompt_embeds 為兩列）。"
            "guidance_scale=1 時 pipeline 的 do_classifier_free_guidance 為 False，"
            "latent 不再複製兩份，記錄到的 Q/K/V 形狀與原作不同；該路徑未查證"
        )
    emb2 = cat_cond([emb_uncond, emb_cond])

    # inpainting 下原作取 pipeline 預設排程的 `timesteps[0]`（50 步、
    # strength=1），即整條排程的第一個 timestep。img2img 沒有那個排程，
    # 故本專案的移植改由呼叫端的 strength 決定——那正是移植表第四列。
    t0 = (sd.num_train_timesteps - 1 if mask is not None else
          min(int(sd.num_train_timesteps * strength),
              sd.num_train_timesteps - 1))
    generator = torch.Generator(device=device).manual_seed(seed)
    recorder = QKVRecorder(sd.unet)
    # GT 與迭代必須走**同一條**路徑。先前這裡傳 None，遮罩因此讀不到，
    # GT 會走 img2img 而迭代走 inpainting——兩組 Q/K/V 來自不同的前向，
    # 相減得到的距離量的是路徑差異而不是擾動。
    gt = _forward_and_record(
        sd, mask, x_paper, emb2, t0, generator, recorder,
        differentiable=False
    )
    gt = tuple([t.detach() for t in group] for group in gt)

    latent_shape = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    ctx = AdvPaintContext(sd, spec, emb2, t0, latent_shape, generator, gt,
                          mask=mask)
    return ctx


def loss_fn(sd, x_adv: torch.Tensor, ctx: AdvPaintContext) -> torch.Tensor:
    """`(loss_query + loss_key + loss_value + loss_cross_q) / length`。

    四項都是**負的** norm 累加（原始碼的 `-=`），故整體是要**最小化**的量，
    等價於最大化 GT 與對抗樣本之間的 Q／K／V 距離。骨幹以
    `objective="minimize"` 走 `X_adv − sign(grad)·step`，與
    `AdvPaint.py:296` 逐字相同。符號不吸收進骨幹，見 `pgd.run_pgd`。

    累加**不得**從 `x_adv.sum() * 0.0` 這類錨點起算。該錨點會讓 `x_adv`
    恆被計算圖使用，`pgd.run_pgd` 的 `torch.autograd.grad(loss, probe)`
    （`allow_unused=False`）就永遠不會拋出「輸入未被使用」——那是本專案
    唯一會偵測「hook 沒裝上／前向被 no_grad 切斷」的機制。拆掉它之後，
    梯度一旦斷了會回傳全零，`sign(0)=0` 使 x_adv 停在初始值，100 步跑完、
    loss 曲線是平線但仍有數值，看不出異常。故此處由第一個實際的 norm 項
    起算，路徑斷掉時 autograd 會直接拋錯。
    """
    cur = _forward_and_record(
        sd, ctx.mask, x_adv, ctx.emb2, ctx.t0, ctx.generator, ctx.recorder,
        differentiable=True,
    )
    terms = []
    for gt_group, cur_group in zip(ctx.gt, cur):
        if len(gt_group) != len(cur_group):
            raise RuntimeError(
                f"GT 與當前前向記錄到的層數不同（{len(gt_group)} vs {len(cur_group)}），"
                "hook 沒有正確安裝或 UNet 前向路徑改變了"
            )
        for gt_t, cur_t in zip(gt_group, cur_group):
            terms.append((gt_t - cur_t).norm(p=2))
    if not terms:
        raise RuntimeError(
            "四組 Q／K／V 記錄皆為空：QKVRecorder 的 hook 沒有攔截到任何 "
            "attention 層。損失無從計算"
        )
    return -torch.stack(terms).sum() / ctx.length


SPEC = BaselineSpec(
    name="advpaint",
    source="JoonsungJeon/AdvPaint（官方 AdvPaint.py）",
    value_range=ADVPAINT_RANGE,
    eps=0.06,                    # 函式預設；CLI 預設為 0.1
    eps_pixel01=0.03,
    norm="linf",
    steps=100,                   # 原始碼預設 100；論文正文寫 250
    step_size=0.03,              # 函式預設；CLI 預設為 0.05
    step_schedule="linear_decay_1pct",
    update_rule="sign",
    objective="minimize",        # `X_adv = X_adv - grad.sign() * actual_step_size`
    init_rule="uniform_linf",    # `X + (rand*2*eps - eps)`，SOURCE_AUDIT §1.4 未記
    grad_reps=1,
    needs_target_image=False,
    needs_mask=True,             # 本輪設為全圖
    modified_from_paper=True,
    modification_note=(
        "(1) mask 設為全圖，原作逐 mask 檔迴圈並只在 mask 外加擾動；"
        "(2) 由 inpainting 移植到 img2img：去掉 9 通道拼接，"
        "原作可微的路徑是 masked-image latent，本移植改為經加噪後的整張 latent；"
        "(3) timesteps[0] 改由呼叫端的 strength 決定，原作取 inpainting pipeline 預設排程；"
        "(4) 攻擊 prompt 由呼叫端指定，原作為 required CLI 參數無預設。"
    ),
    discrepancy_note=(
        "迭代次數論文正文 250、原始碼預設 100（取 100）；"
        "eps/step_size 函式預設 0.06/0.03、CLI 預設 0.1/0.05（取前者，即論文報告數字所依據者）；"
        "損失只含 self-attn 的 Q/K/V 與 cross-attn 的 Q，cross 的 K、V 不在內，"
        "與摘要「targets the self- and cross-attention blocks」的字面讀法不同；"
        "原始碼有均勻隨機初始化，SOURCE_AUDIT §1.4 未記。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "guidance_scale": 7.5,
        "seed_original": 9999,
        "timestep": "只取 timesteps[0]，不掃全部",
        "optimizer": "無，手動 sign 更新",
        "eps_cli": 0.1,
        "step_size_cli": 0.05,
        "model": "runwayml/stable-diffusion-inpainting（原作）",
    },
)
