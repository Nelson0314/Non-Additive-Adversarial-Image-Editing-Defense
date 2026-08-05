"""DIA —— Hong, Son, Lee, Woo, ICCV 2025（arXiv:2510.00778）。

出處：`sohn1029/DIA`（共同第一作者帳號，內容與論文所附匿名 repo 相同），
`attack/DIA_PT.py`、`attack/DIA_R.py`、`attack_setting.json`。
逐項佐證見 `docs/_audit_dia_apa.md` §對象一，值域裁決見
`docs/SOURCE_AUDIT_2026-08-05.md` §6 與 §10。

兩個變體
──────────────────────────────────────────────────────────────────────

| | DIA-PT | DIA-R |
|---|---|---|
| 損失 | `‖x̂_{0:T}(x_adv) − E(x_adv)‖₂`（latent 空間） | `‖x̃_{T:0}(x̂_{0:T}(x_adv)) − x_adv‖₂`（像素空間） |
| 前向 | 10 步反演 | 10 步反演 + 10 步重建 |
| 初始化 | `norm: "L1"` → L1 球內隨機初始化 | `norm: null` → 由乾淨影像出發 |
| VAE 編碼 | `.latent_dist.sample()` | `.latent_dist.mode()` |

**論文 Eq. 9／Eq. 10 寫 `‖·‖₂²`，程式碼是 `.norm(p=2)`（未平方）。**
sign 更新下不影響方向，此處照原始碼。

值域陷阱
──────────────────────────────────────────────────────────────────────

`eps=0.05` 是在 `[-1,1]` 上量的（`attack_benchmark.py:16-27` 的
`2·ToTensor()−1`、`clamp_min/max = ∓1`），故 `[0,1]` 尺度只有 **0.025
（≈6.4/255）**。`lr = 1/255` 同樣以 `[-1,1]` 計，`[0,1]` 尺度是 0.5/255。
做失真匹配時不可把 0.05 當成 `[0,1]` 的 L∞ 預算，否則預算會高出一倍。

反傳方式：本實作用 autograd，原實作用 FlowGrad 逐步 VJP
──────────────────────────────────────────────────────────────────────

原始碼把整條軌跡放在 `torch.no_grad()` 內，再依論文 Eq. 11 逐步以
`torch.autograd.grad` 回推 `lam`，最後一步用
`torch.autograd.functional.vjp` 穿過 VAE encoder（`DIA_PT.py:365-404`）。
那是**鏈鎖律的手動展開**，與 autograd 在精確算術下給出同一個梯度；差別
只在記憶體（原作 O(1)，本實作 O(軌跡長度)）與浮點捨入（原作對每一步
重新前傳一次）。目標函數與更新規則不受影響，故此處用 autograd。

但有一處**不是**單純的記憶體差異，必須逐字重現：原作的
`vjp_encode_fn` 用 `latent_dist.mode()`，而 DIA-PT 的前向值用
`latent_dist.sample()`（`DIA_PT.py:355, 388`）。即「值取樣、梯度走均值」。
本實作以直通式 `mode + (sample − mode).detach()` 重現這個組合，
而不是統一成其中一個——統一會改變梯度（`sample = mean + std·ε`，
`std` 亦為 x 的函數）。
"""

from typing import Optional

import torch

from src.baselines.pgd import BaselineSpec, ValueRange

DIA_RANGE = ValueRange(
    -1.0,
    1.0,
    "attack_benchmark.py:16-27 `2.0*ToTensor()-1.0`；attack_setting.json clamp_min/max = ∓1",
)

# `attack_setting.json`：兩個變體共用。
DIA_NUM_INFERENCE_STEPS = 10
DIA_CFG = 1
DIA_PROMPTS = {"uncond": "", "cond": "", "neg": ""}


def dia_timesteps(sd, num_inference_steps: int = DIA_NUM_INFERENCE_STEPS) -> torch.Tensor:
    """`custom_set_timesteps`（`DIA_PT.py:165-192`）產生的升冪格點。

        timesteps = round(linspace(0, 1, n) * (num_train_timesteps - 2) + 1e-6)
        timesteps += steps_offset

    原始碼以 `inversion_flag=False` 呼叫（降冪），再於反演迴圈中
    `reversed(...)` 走訪，淨結果即升冪。此處直接回傳升冪，等價。
    `steps_offset` 取自 scheduler 設定，不寫死。
    """
    import numpy as np

    n_train = sd.num_train_timesteps
    ts = np.linspace(0, 1, num_inference_steps) * (n_train - 2)
    ts = ts + 1e-6
    ts = ts.round().astype(np.int64)
    offset = sd.scheduler.config.steps_offset
    return torch.from_numpy(ts) + offset


def resolve_final_abar(sd, abar: torch.Tensor) -> torch.Tensor:
    """`prev_timestep < 0` 時 DDIM 用的 ᾱ（`DIA_PT.py:315-319`）。

    **這條退路會實際觸發，不是防禦性程式碼。** DIA 的格點最低點是 t=1
    （`dia_timesteps`），而 `prev_t = t − 100`，故第一步就是 `−99 < 0`，
    每條反演軌跡都會走到它一次。

    diffusers 的 `DDIMScheduler` 定義為

        final_alpha_cumprod = 1.0 if set_alpha_to_one else alphas_cumprod[0]

    stock SD 的 `scheduler/scheduler_config.json` 記 `set_alpha_to_one: false`，
    故該值為 `alphas_cumprod[0]`。DIA 由 `from_pretrained` 載入 pipeline、
    未覆寫該欄位（`DIA_PT.py` 全檔無 `set_alpha_to_one`），故取 `abar[0]`。

    **為什麼不能只靠 `getattr(sd.scheduler, ...)`**：SDXL 的預設排程器是
    `EulerDiscreteScheduler`，根本沒有這個屬性。而 DIA 只從排程器讀
    `alphas_cumprod`、自己實作 DDIM 步（`_ddim`），排程器的種類本不影響其
    數值——SDXL 與 SD v1.x 的 beta 排程相同（0.00085–0.012、scaled_linear、
    1000 步），`alphas_cumprod` 逐元素相同。

    排程器若**有**該屬性則交叉核對，不一致即拋出：那表示該檢查點的
    `set_alpha_to_one` 與 stock SD 不同，沿用會靜默算出另一條軌跡。

    未確認項：`set_alpha_to_one=True`（diffusers 的建構子預設）會給出 1.0，
    與 `abar[0] ≈ 0.99915` 相差 0.085%。DIA 的 repo 未附 scheduler 設定檔，
    上述判定依據的是 stock SD 的設定。差異量級記於此，供日後查核。
    """
    derived = abar[0]
    exposed = getattr(sd.scheduler, "final_alpha_cumprod", None)
    if exposed is not None:
        exposed = torch.as_tensor(exposed, device=abar.device, dtype=abar.dtype)
        if not torch.allclose(exposed, derived, atol=1e-6):
            raise NotImplementedError(
                f"排程器的 final_alpha_cumprod 為 {float(exposed):.6f}，與 "
                f"stock SD 的 alphas_cumprod[0]={float(derived):.6f} 不符。"
                "這表示該檢查點的 set_alpha_to_one 與 DIA 所依據的設定不同；"
                "沿用任一個都會靜默算出另一條反演軌跡，必須先確認出處"
            )
        return exposed
    return derived


class DIAContext:
    def __init__(self, sd, spec, variant, emb, ts, num_inference_steps, generator):
        self.spec = spec
        self.variant = variant
        self.emb = emb
        self.ts = ts
        self.num_inference_steps = num_inference_steps
        self.generator = generator
        abar = sd.alphas_cumprod(emb.device)
        self.abar = abar
        self.final_abar = resolve_final_abar(sd, abar)


def _ddim(z, eps, alpha_t, alpha_tm1):
    """`backward_ddim`（`utils/utils_general_H.py:32-34`）逐字。

        x' = (α_{t-1}^½ α_t^{-½})·x + α_{t-1}^½ · [ (1/α_{t-1} − 1)^½ − (1/α_t − 1)^½ ] · ε
    """
    return (alpha_tm1 ** 0.5 * alpha_t ** -0.5) * z + alpha_tm1 ** 0.5 * (
        (1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5
    ) * eps


def _step(sd, ctx: DIAContext, t: torch.Tensor, z: torch.Tensor, inversion: bool):
    """`custom_diffusion` 的 `cfg == 1` 分支（`DIA_PT.py:300-322`）。

    cfg 為 1 時只做**一次**無條件前向（prompt 為空字串），不是兩分支的
    classifier-free guidance。`prev_timestep = t − num_train // n_steps`，
    反演時把 `alpha_prod_t` 與 `alpha_prod_t_prev` 對調。
    """
    eps = sd._eps(z, t, ctx.emb)
    prev_t = int(t) - sd.num_train_timesteps // ctx.num_inference_steps
    a_t = ctx.abar[int(t)]
    a_prev = ctx.abar[prev_t] if prev_t >= 0 else ctx.final_abar
    if inversion:
        a_t, a_prev = a_prev, a_t
    return _ddim(z, eps, a_t, a_prev)


def _encode_pt(sd, x_paper, generator):
    """DIA-PT：值取自 `.sample()`，梯度走 `.mode()`。見模組 docstring。"""
    post = sd.vae.encode(x_paper.to(sd.vae.dtype)).latent_dist
    mode = post.mode() * sd.scaling_factor
    with torch.no_grad():
        sample = post.sample(generator) * sd.scaling_factor
    return mode + (sample - mode).detach()


def _encode_r(sd, x_paper):
    """DIA-R：前向與反傳都是 `.mode()`（`DIA_R.py:353`、`vjp_encode_fn`）。"""
    post = sd.vae.encode(x_paper.to(sd.vae.dtype)).latent_dist
    return post.mode() * sd.scaling_factor


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    num_inference_steps: int = DIA_NUM_INFERENCE_STEPS,
    seed: int = 1234,
    **_,
) -> DIAContext:
    """`seed` 預設 1234（`attack_benchmark.py --seed`）。

    prompt 三者皆為空字串、`forward_cfg = backward_cfg = 1`，即攻擊端的
    反演與重建都在**空 prompt、無 CFG** 下進行。
    """
    variant = spec.extras["variant"]
    emb = sd.encode_text(DIA_PROMPTS["uncond"]).detach()
    ts = dia_timesteps(sd, num_inference_steps).to(x01.device)
    generator = torch.Generator(device=x01.device).manual_seed(seed)
    return DIAContext(sd, spec, variant, emb, ts, num_inference_steps, generator)


def loss_fn(sd, x_adv: torch.Tensor, ctx: DIAContext) -> torch.Tensor:
    """兩個變體的損失，皆為未平方的 L2，皆由骨幹**最大化**。

    DIA-PT（`DIA_PT.py:388`）：`(x_T − z_0.detach()).norm(p=2)`
    DIA-R（`DIA_R.py:378`）：`(decode(z_rec) − x_adv.detach()).norm(p=2)`
    """
    if ctx.variant == "PT":
        z0 = _encode_pt(sd, x_adv, ctx.generator)
        z = z0
        for t in ctx.ts:                       # 升冪：反演
            z = _step(sd, ctx, t, z, inversion=True)
        return (z - z0.detach()).norm(p=2)

    if ctx.variant == "R":
        z0 = _encode_r(sd, x_adv)
        z = z0
        for t in ctx.ts:                       # 升冪：反演
            z = _step(sd, ctx, t, z, inversion=True)
        for t in reversed(ctx.ts):             # 降冪：重建（backward_point = 0，全走）
            z = _step(sd, ctx, t, z, inversion=False)
        # `decode_image`（`utils_general_H.py:312-318`）為 `vae.decode(z/0.18215).sample`，
        # **不裁切**。本專案的 `decode_latent` 會 clamp(0,1) 再換到 [0,1]，
        # 被裁切處梯度為零，故此處直接用 vae。
        x_rec = sd.vae.decode(z / sd.scaling_factor).sample
        return (x_rec.float() - x_adv.detach().float()).norm(p=2)

    raise ValueError(f"DIA 只有 PT 與 R 兩個變體，收到 {ctx.variant!r}")


def _spec(variant: str, init_rule: str) -> BaselineSpec:
    return BaselineSpec(
        name=f"dia_{variant.lower()}",
        source="arXiv:2510.00778（ICCV 2025）；sohn1029/DIA",
        value_range=DIA_RANGE,
        eps=0.05,                 # attack_setting.json，[-1,1] 值域
        eps_pixel01=0.025,        # ≈ 6.375/255
        norm="linf",
        steps=20,                 # attack_setting.json `iters`；論文 §4.1 亦為 20
        step_size=1.0 / 255.0,    # attack_setting.json `lr = 0.003921568627451`
        step_schedule="constant",
        update_rule="sign",
        objective="maximize",     # `grad_normalize`：X_adv + step_size * grad.sign()
        init_rule=init_rule,
        grad_reps=1,
        needs_target_image=False,
        needs_mask=False,
        modified_from_paper=False,
        modification_note="",
        discrepancy_note=(
            "論文 Eq. 9/10 寫 ‖·‖₂²，程式碼為 .norm(p=2) 未平方（sign 更新下不影響方向）；"
            "論文正文未給 step size，本值取自 attack_setting.json；"
            "論文未述值域，由程式碼確認為 [-1,1]，故 eps=0.05 在 [0,1] 只有 0.025。"
            "本實作以 autograd 取代原作的 FlowGrad 逐步 VJP（同一個梯度，記憶體不同）；"
            "DIA-PT 的『值取樣、梯度走均值』以直通式重現。"
        ),
        prepare=prepare,
        loss_fn=loss_fn,
        extras={
            "variant": variant,
            "num_inference_steps": DIA_NUM_INFERENCE_STEPS,
            "cfg": DIA_CFG,
            "prompts": dict(DIA_PROMPTS),
            "seed_original": 1234,
            "model": "CompVis/stable-diffusion-v1-4",
            "precision_original": "float16",
        },
    )


# `attack_setting.json`：DIA_PT 的 `norm` 為 `"L1"`（L1 球隨機初始化），
# DIA_R 的 `norm` 為 `null`（由乾淨影像出發，無隨機初始化）。
SPEC_PT = _spec("PT", "l1_ball")
SPEC_R = _spec("R", "none")
