"""Mist v1 —— Liang & Wu, arXiv:2305.12683（技術報告；主論文 ICML 2023 Oral）。

出處：`mist-project/mist`，commit `99f5f3c455022bef77ef25ba440a4289ea472d25`，
主程式 `mist_v3.py`。逐行佐證見 `docs/_audit_mist_diffvax.md` §1–§8，
複驗與裁決見 `docs/SOURCE_AUDIT_2026-08-05.md` §4。

三項與論文正文的落差
──────────────────────────────────────────────────────────────────────

1. **textural loss 是平方 L2 的和，不是 L2 範數。** `mist_v3.py:102`
   `self.fn = nn.MSELoss(reduction="sum")`。論文 Eq (3) 寫 `‖E(y) − E(x+δ)‖₂`。
   這**直接改變 `w = 1e4` 的有效尺度**：semantic loss 是 mean-reduction
   （512² 影像下 latent 有 4×64×64 = 16,384 個元素），textural 是
   sum-reduction 的平方，兩者尺度差極大。照論文式子實作再配 w=1e4，
   融合模式會跑在完全不同的操作點上。
2. **mode 編號與 README 相反。** 程式碼 `mist_v3.py:143-149` 是
   0 = semantic、1 = textural、2 = fused；`Readme.md:45` 寫成
   0 = textural、1 = semantic。以可執行的程式碼為準。
3. **eps 名目為 16/255 而非論文的 17/255。** `mist_v3.py:197-205`
   `'epsilon': epsilon/255.0 * (1-(-1))`——Mist 這裡**做對了**值域換算
   （乘 2），故 `[0,1]` 尺度確實是 16/255。論文的 17/255 只出現在
   `mist_v3.py:156-159` 的 docstring（宣稱取整後上界為 (eps+1)/255），
   **程式碼中沒有任何一行實作那個 +1/255**。

VAE 取樣而非取均值
──────────────────────────────────────────────────────────────────────

`get_first_stage_encoding` 用 `encoder_posterior.sample()`
（`ldm/models/diffusion/ddpmAttack.py:541-548`），不是 mode/mean。
本專案的 `SDWrapper.encode_image` 固定取 `latent_dist.mean`（為了決定性），
兩者不同，故此處**不能**用 `encode_image`，改為直接取 `sd.vae` 的後驗抽樣。
這是 Mist 的方法本身帶的隨機性：兩項損失每次前傳都含 VAE 後驗抽樣與
擴散時間步／雜訊兩重隨機性，100 步 PGD 的結果對 seed 敏感
（原始碼 `seed_everything(23)`）。
"""

from typing import Optional

import torch
import torch.utils.checkpoint as ckpt
from diffusers.utils.torch_utils import randn_tensor

from src.baselines.pgd import BaselineSpec, ValueRange
from src.models.sd import expand_cond

# `mist_v3.py:230-233` `img / 127.5 - 1.0`；PGD 的 `clip_min=-1.0`。
MIST_RANGE = ValueRange(
    -1.0,
    1.0,
    "mist_v3.py:230 `np.array(img)/127.5 - 1.0`；LinfPGDAttack clip_min=-1.0",
)

# `mist_v3.py:177-193`：`object` 旗標無法由命令列設定且 __main__ 未傳，
# 故 prompt 恆為 'a painting'。論文正文完全未提及 prompt 的選擇。
MIST_PROMPT = "a painting"


class MistContext:
    def __init__(self, sd, spec, z_target, emb, mode, rate, generator,
                 use_ckpt: bool = False, vae_ckpt: bool = False):
        self.spec = spec
        self.z_target = z_target
        self.emb = emb
        self.mode = mode
        self.rate = rate
        self.generator = generator
        # fused mode（本專案採用的預設）在**同一張計算圖**上放三樣東西：
        # textural 的一次 VAE 編碼、`_semantic_loss` 內的另一次 VAE 編碼、
        # 以及一次完整的 UNet 前向。不做 checkpoint 時 peak 是三者之和。
        #
        # 2026-08-07 實測於 RTX 3090（23.56 GB）、SDXL、1024²、bf16：mist 在
        # 第一次計算損失時即 `torch.OutOfMemoryError`，訊息為「23.14 GiB is
        # allocated by PyTorch」——**活著的張量**，不是快取碎片；且在全新
        # 行程中單獨跑同樣失敗（`b3_bird_03/diag_mist.log`），故不是跨格累積。
        # 512²／SD v1.4 的同一格則正常跑完（v14 三張圖 `done=7 failed=0`），
        # 即這是 1024² 專屬的容量問題。
        #
        # 2026-08-07 實測：只包 UNet **不夠**，mist 仍在同一個位置 OOM
        # （`b3_bird_03/diag_fix.log`）。主導成本是那兩次 1024² 的 VAE 編碼，
        # 故 `vae_ckpt` 也必須開。它的重參數化寫法見 `_encode_sampled`。
        #
        # 兩個開關分開，與 `SDWrapper` 的 `use_ckpt`／`vae_ckpt` 一致。重算的
        # 是同一個函式且噪聲已固定，**數值中性**，故兩者都不進 `config_hash`
        # ——與 `1024633` 對 DIA 的處理同一原則。
        self.use_ckpt = bool(use_ckpt)
        self.vae_ckpt = bool(vae_ckpt)
        self.mse_sum = torch.nn.MSELoss(reduction="sum")


def _encode_sampled(sd, x_paper: torch.Tensor, generator,
                    use_ckpt: bool = False) -> torch.Tensor:
    """`z = scale_factor · sample(q(z|x))`，輸入已在 `[-1,1]`。

    對應 `get_first_stage_encoding(encode_first_stage(x))`。**必須取樣**，
    見模組 docstring。`scale_factor` 取自模型設定（SD v1.x 為 0.18215，
    與 `configs/stable-diffusion/v1-inference-attack.yaml` 一致）。

    `use_ckpt` 為真時**不能直接把 `.sample(generator)` 包進 checkpoint**：
    `use_reentrant=False` 保存的是預設 RNG 狀態，管不到呼叫端持有的顯式
    `Generator`，重算會再抽一個樣本，於是反向傳播算的梯度屬於另一個函式
    ——沒有任何症狀的數值汙染。

    故此處把噪聲抽在區塊**外**，區塊內只做重參數化。diffusers 0.39.0 的
    `DiagonalGaussianDistribution.sample` 逐字為

        sample = randn_tensor(self.mean.shape, generator=generator,
                              device=self.parameters.device,
                              dtype=self.parameters.dtype)
        x = self.mean + self.std * sample

    本函式以同一個 `randn_tensor`、同一組 (shape, generator, device, dtype)
    抽噪聲，故產生的樣本與抽樣順序都與原路徑**逐位元相同**，重算則因 eps
    已固定而成為確定性。形狀在區塊內明確核對：不符時會 broadcast 成另一個
    張量而不報錯，那正是要擋掉的失效。
    """
    x = x_paper.to(sd.vae.dtype)
    if not use_ckpt:
        post = sd.vae.encode(x).latent_dist
        return post.sample(generator) * sd.scaling_factor

    shape = sd.latent_shape(x.shape[-2], x.shape[-1])
    eps = randn_tensor(tuple(shape), generator=generator,
                       device=sd.vae.device, dtype=sd.vae.dtype)

    def _enc(a, e):
        post = sd.vae.encode(a).latent_dist
        if post.mean.shape != e.shape:
            raise RuntimeError(
                f"重參數化的噪聲形狀 {tuple(e.shape)} 與後驗均值 "
                f"{tuple(post.mean.shape)} 不符。`latent_shape` 推導的形狀與 "
                "VAE 實際輸出不一致，直接相乘會 broadcast 成另一個張量而不報錯"
            )
        return (post.mean + post.std * e) * sd.scaling_factor

    return ckpt.checkpoint(_enc, x, eps, use_reentrant=False)


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    target01: Optional[torch.Tensor] = None,
    mode: int = 2,
    rate: float = 1e4,
    block_num: int = 1,
    seed: int = 23,
    use_ckpt: bool = False,
    vae_ckpt: bool = False,
    **_,
) -> MistContext:
    """預設值取自 CLI：`--mode` 2（fused）、`--rate` 1 → `10**(1+3)` = 1e4
    （`mist_utils.py:71-86`、`mist_v3.py:280`）、`--block_num` 1。
    """
    if block_num != 1:
        raise NotImplementedError(
            "Mist 的 block_num != 1 未實作：原始碼 `mist_v3.py:316` 會把影像與"
            "target 切成 block 後逐塊對齊，`pre_process` 的 RandomCrop 也隨之"
            "變成真的隨機裁切。本專案只查證了 block_num=1（RandomCrop 退化為"
            "identity）的路徑，逐塊對齊的細節未查證"
        )
    if target01 is None:
        raise NotImplementedError(
            "Mist 需要 target 影像 MIST.png（repo 根目錄，1440×1440，"
            "黑底白字密集平鋪、高對比硬邊）。該檔無法由描述重建，必須取得原檔。"
            "PhotoGuard-c 的零張量 target **不可**代用（見 SOURCE_AUDIT §3.4）"
        )

    vr = spec.value_range
    device = x01.device
    generator = torch.Generator(device=device).manual_seed(seed)

    tgt_paper = vr.from01(target01.to(device).detach())
    with torch.no_grad():
        z_target = _encode_sampled(sd, tgt_paper, generator).detach()
    emb = sd.encode_text(MIST_PROMPT).detach()
    return MistContext(sd, spec, z_target, emb, mode, rate, generator,
                       use_ckpt=use_ckpt, vae_ckpt=vae_ckpt)


def _semantic_loss(sd, x_paper: torch.Tensor, ctx: MistContext) -> torch.Tensor:
    """`L_semantic = E_{t~U(0,1000), ε}[ mean_CHW((ε − ε_θ(z_t,t,c))²) ]`。

    化簡自 `p_losses`（`ddpmAttack.py:1011-1044`）：預設
    `logvar_init=0`、`l_simple_weight=1`、`original_elbo_weight=0`，
    故 `loss = loss_simple.mean()`，其中
    `loss_simple = get_loss(model_output, noise, mean=False).mean([1,2,3])`。
    單次前傳只抽一個 `t` 與一個 `ε`（1 樣本 Monte Carlo）。

    `pre_process` 是 `RandomCrop(target_size)`；`block_num=1` 時
    `target_size == input_size`，退化為 identity（`prepare` 已擋掉其他情形）。
    """
    z = _encode_sampled(sd, x_paper, ctx.generator,
                        use_ckpt=ctx.vae_ckpt)
    n_train = sd.num_train_timesteps
    t = torch.randint(
        0, n_train, (z.shape[0],), generator=ctx.generator, device=z.device
    ).long()
    noise = torch.randn(
        z.shape, generator=ctx.generator, device=z.device, dtype=z.dtype
    )
    abar = sd.alphas_cumprod(z.device)[t].view(-1, 1, 1, 1).to(z.dtype)
    z_noisy = abar.sqrt() * z + (1 - abar).sqrt() * noise
    # `expand_cond` 而非 `.expand(...)`：SDXL 的條件是 SDXLPrompt，
    # 沒有 `expand` 方法，且 pooled 是二維、序列嵌入是三維，要分別處理。
    eps_pred = sd._eps(z_noisy, t, expand_cond(ctx.emb, z.shape[0]),
                       use_ckpt=ctx.use_ckpt)
    return ((noise - eps_pred) ** 2).mean(dim=[1, 2, 3]).mean()


def loss_fn(sd, x_adv: torch.Tensor, ctx: MistContext) -> torch.Tensor:
    """`mist_v3.py:141-149` 的三個分支，逐字對應。

        mode 0 (semantic): −L_semantic
        mode 1 (textural):  L_textural
        mode 2 (fused):     L_textural − w · L_semantic

    骨幹以 `objective="minimize"` 走（`Masked_PGD.py:68-71` 的
    `minimize=targeted=True` 把損失取負後再做 `+= sign` 上升，等價於對原
    損失下降）。整體即

        δ* = argmax_δ [ w·L_semantic(x+δ) − L_textural(x+δ, y) ]

    與論文 Eq (4) 同號。
    """
    if ctx.mode == 0:
        return -_semantic_loss(sd, x_adv, ctx)
    if ctx.mode == 1:
        zx = _encode_sampled(sd, x_adv, ctx.generator,
                             use_ckpt=ctx.vae_ckpt)
        return ctx.mse_sum(zx, ctx.z_target)
    if ctx.mode == 2:
        zx = _encode_sampled(sd, x_adv, ctx.generator,
                             use_ckpt=ctx.vae_ckpt)
        textural = ctx.mse_sum(zx, ctx.z_target)
        return textural - _semantic_loss(sd, x_adv, ctx) * ctx.rate
    raise ValueError(
        f"Mist 的 mode 只有 0（semantic）／1（textural）／2（fused），收到 {ctx.mode}"
    )


SPEC = BaselineSpec(
    name="mist",
    source="arXiv:2305.12683；mist-project/mist@99f5f3c455022bef77ef25ba440a4289ea472d25",
    value_range=MIST_RANGE,
    # `mist_v3.py:197-198`：epsilon/255.0 * (1-(-1))，即 16/255 × 2。
    eps=16.0 / 255.0 * (1.0 - (-1.0)),
    eps_pixel01=16.0 / 255.0,
    norm="linf",
    steps=100,                 # `mist_utils.py:44-52` --steps 預設 100
    # `mist_v3.py:152` alpha=1（無 CLI 開關，__main__ 未傳），
    # 同樣經 `alpha/255.0 * (1-(-1))`。
    step_size=1.0 / 255.0 * (1.0 - (-1.0)),
    step_schedule="constant",
    update_rule="sign",
    objective="minimize",
    init_rule="uniform_linf",  # `Masked_PGD.py:141-192` rand_init 預設 True 且未關閉
    grad_reps=1,
    needs_target_image=True,   # MIST.png，不可與 PhotoGuard 的零張量混用
    needs_mask=False,
    modified_from_paper=False,
    modification_note="",
    discrepancy_note=(
        "論文 Eq (3) 寫 ‖·‖₂，原始碼為 nn.MSELoss(reduction='sum')（平方 L2 之和），"
        "直接改變 w=1e4 的有效尺度；README 的 mode 對照表與程式碼相反（以程式碼為準："
        "0=semantic, 1=textural, 2=fused）；論文的 17/255 只存在於 docstring 的宣稱，"
        "程式的投影上界為 16/255；論文未說明 PGD 使用隨機初始化、也未說明 VAE 取樣而非取均值。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "mode": 2,
        "mode_meaning": "0=semantic, 1=textural, 2=fused（依原始碼，非 README）",
        "rate_w": 1e4,
        "prompt": MIST_PROMPT,
        "block_num": 1,
        "seed_original": 23,
        "target_file": "MIST.png（repo 根目錄，1440×1440）",
        "model": "CompVis Stable Diffusion v1.4 原始 ckpt",
    },
)
