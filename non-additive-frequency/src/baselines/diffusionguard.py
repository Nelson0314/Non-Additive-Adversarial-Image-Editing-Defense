"""DiffusionGuard（Choi et al., ICLR 2025）— 唯一明確為抗淨化設計的編輯防護。

為什麼納入
────────────────────────────────────────────────────────────────────
DEC-025 之後本專案的主軸是抗淨化，但**現有的強 baseline 沒有一個是為抗淨化
設計的**：`photoguard_c`／`mist`／`dia_r` 三篇都只求「讓編輯壞掉」。
DiffusionGuard 明確以抗 JPEG、抗 crop-and-resize、抗專用 adversarial cleaning
為賣點，是這個缺口的直接補位（survey §2.7）。

它的新意在**損失**，不在參數化
────────────────────────────────────────────────────────────────────
擾動本身就是普通的像素加性（ℓ∞ 球、sign 更新），與 `AdditiveParam` 無異。
真正不同的是損失：

    在**最高噪聲的那一個時間步**上，最大化 UNet 噪聲預測的 L2 範數。

原始碼（`attacks/attack_diffusionguard.py`）：

    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = timesteps_all[0].long()          # 只取第一個，即噪聲最大的
    latents = randn(...) * init_noise_sigma       # 每次迭代重抽
    latent_model_input = cat([latents, mask, masked_image_latents], dim=1)
    noise_pred = pipe.unet(latent_model_input, timesteps, text_embeddings)[0]
    loss = -(noise_pred.norm(p=2) / batch_size)

因此本檔對外提供的是 **`make_early_step_loss`**，可以配上任何參數化——包括
相位臂。「換一個損失」與「換一個參數化」在本專案是兩件分開的事
（`param_pgd` 模組 docstring）。

移植到 img2img：一處結構性的改寫，必須標註
────────────────────────────────────────────────────────────────────
原文的威脅模型是 **inpainting**，UNet 吃 9 通道：`[帶噪 latent(4), 遮罩(1),
遮罩後影像的 latent(4)]`。**影像與噪聲是分開的兩組通道**，所以它的
`latents` 可以是與影像無關的純噪聲。

本專案的威脅模型只有 img2img（2026-08-15 起），stock SD 的 UNet 是 4 通道，
沒有放影像的地方。影像只能經由 SDEdit 的起始 latent 進入：

    t₀ = min(int(1000 · strength), 999)                 # 與 `sd.sdedit` 同一式
    z   = √ᾱ_{t₀} · E(x_def) + √(1−ᾱ_{t₀}) · ε          # ε 每次迭代重抽
    loss = −‖ε_θ(z, t₀, emb)‖₂ / batch

**這不是換個名字而已**：原文的影像與噪聲各佔一組通道、互不混合；此處兩者被
加權混進同一個張量，梯度因此要穿過 `√ᾱ_{t₀}` 這個係數。strength 0.7 時
`√ᾱ_{700} = 0.287`，也就是影像那一份只剩三成的權重。`modified_from_paper`
必須為真。

另外**遮罩相關的一切都拿掉了**：原文的 mask-augmentation（每次迭代重抽一個
遮罩，使防護對測試時未知的遮罩也有效）是它的主要新意之一，在 img2img 下沒有
對應物。報表不可把本檔寫成「DiffusionGuard 的重現」，只能寫成「它的早期時間步
損失移植到 img2img」。

成本
────────────────────────────────────────────────────────────────────
**每一步都要一次 UNet 前向加反向**，而不是像 encoder 損失那樣只跑 VAE。
原生 800 步在 512² 上估計 2–4 分鐘/圖（VAE 損失是 20 秒級）。原文的
`num_inference_steps: 4` 只用來決定時間格點，不代表跑四步去噪。

定案超參數（repo 的 `config/diffusionguard.yaml`）
────────────────────────────────────────────────────────────────────
`iters = 800`、`eps = 0.06274509803921569`（**即 16/255**）、
`step_size = 0.00392156862745098`（**即 1/255**）、`batch_size = 1`、
`grad_reps = 1`、`num_inference_steps = 4`、`size = 512`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

PAPER_ITERS = 800
PAPER_EPS = 16.0 / 255.0            # config 寫成 0.06274509803921569
PAPER_STEP_SIZE = 1.0 / 255.0       # config 寫成 0.00392156862745098
PAPER_BATCH = 1
PAPER_GRAD_REPS = 1


def make_early_step_loss(
    sd,
    prompt: str,
    strength: float,
    *,
    batch: int = PAPER_BATCH,
    seed: Optional[int] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """`−‖ε_θ(z_{t₀}, t₀, emb)‖₂ / batch`，在 SDEdit 的**起始**時間步上。

    要**最小化**（與本專案所有 `loss_fn` 一致），故取負號——最小化負範數
    等於最大化範數。

    噪聲 ε **每次呼叫重抽**，與原始碼每個 iteration 重抽 `latents` 一致。
    `seed` 給定時抽樣可重現；`None` 時走全域亂數（原始碼即如此）。

    `batch > 1` 時同一張影像配 `batch` 組獨立噪聲，等於一個小規模的
    expectation-over-transformation。原文的預設是 1。

    在 δ = 0 處梯度非零（範數不是極值），故不受 FND-053 的零梯度陷阱影響。
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength 必須落在 (0,1]，收到 {strength}")
    emb = sd.encode_text(prompt).detach()
    t0 = min(int(sd.num_train_timesteps * strength), sd.num_train_timesteps - 1)
    gen = (torch.Generator(device="cpu").manual_seed(int(seed))
           if seed is not None else None)

    def loss(x01: torch.Tensor) -> torch.Tensor:
        abar = sd.alphas_cumprod(x01.device)
        z_src = sd.encode_image(x01)
        if batch > 1:
            z_src = z_src.expand(batch, -1, -1, -1)
        eps = torch.randn(z_src.shape, generator=gen,
                          dtype=z_src.dtype).to(z_src.device)
        z = abar[t0].sqrt() * z_src + (1 - abar[t0]).sqrt() * eps
        e = emb if batch == 1 else torch.cat([emb] * batch, dim=0)
        return -(sd.unet_forward(z, torch.tensor(t0, device=z.device), e
                                 ).norm(p=2) / batch)

    loss.t0 = t0          # 供報表記錄實際用到的時間步
    return loss


@dataclass(frozen=True)
class DiffusionGuardSpec:
    """移植到 img2img 的設定。`modified_from_paper` 恆為真——見模組 docstring。"""

    name: str = "diffusionguard"
    eps_pixel01: float = PAPER_EPS
    step_size: float = PAPER_STEP_SIZE
    iters: int = PAPER_ITERS
    batch: int = PAPER_BATCH
    modified_from_paper: bool = True
    modification_note: str = (
        "原文為 inpainting（9 通道 UNet，影像與噪聲分屬不同通道）；此處移植到 "
        "img2img，影像經 SDEdit 的起始 latent 以 √ᾱ_{t₀} 加權混入同一個張量，"
        "且 mask-augmentation 完全拿掉")
    source: str = "arXiv:2410.05694；超參數取自 repo 的 config/diffusionguard.yaml"

    def __post_init__(self):
        if not self.modified_from_paper:
            raise ValueError(
                "本檔是 img2img 的移植，不是原文的重現，"
                "modified_from_paper 不得為 False")
        if self.step_size > self.eps_pixel01:
            raise ValueError(
                f"step_size={self.step_size} 大於 eps={self.eps_pixel01}，"
                "第一步就會撞出球外")


SPEC_PAPER = DiffusionGuardSpec()


@dataclass
class DiffusionGuardResult:
    x_def: torch.Tensor
    spec: DiffusionGuardSpec
    t0: int
    history: List[Dict] = field(default_factory=list)


def run_diffusionguard(
    sd,
    x01: torch.Tensor,
    prompt: str,
    strength: float,
    spec: DiffusionGuardSpec = SPEC_PAPER,
    *,
    seed: Optional[int] = 0,
    log_every: int = 0,
) -> DiffusionGuardResult:
    """`attacks/common.py` 的 `generate_perturbation`，去掉遮罩那一路。

        adv ← adv − sign(∇_adv L) · step_size
        adv ← clamp(adv, x − ε, x + ε) 再夾到 [0,1]

    與原始碼的差別只有兩處，都在模組 docstring 說明：損失的輸入路徑（img2img）
    與遮罩（拿掉）。更新規則、步長、ε、迭代數全部照原文。

    **`prompt` 是攻擊方的編輯 prompt。** 原文的威脅模型假設防禦方不知道它，
    其 config 用空字串；本專案的 `apa_baseline` 慣例是把真實 prompt 餵給防禦
    （白盒），故此處由呼叫端明給並記錄在報表上。
    """
    loss_fn = make_early_step_loss(sd, prompt, strength,
                                   batch=spec.batch, seed=seed)
    x0 = x01.detach()
    adv = x0.clone()
    history: List[Dict] = []

    for i in range(spec.iters):
        adv = adv.detach().requires_grad_(True)
        loss = loss_fn(adv)
        (grad,) = torch.autograd.grad(loss, [adv])
        with torch.no_grad():
            adv = adv - grad.sign() * spec.step_size
            adv = torch.min(torch.max(adv, x0 - spec.eps_pixel01),
                            x0 + spec.eps_pixel01).clamp(0.0, 1.0)
        if log_every and (i % log_every == 0 or i == spec.iters - 1):
            history.append({"step": i, "loss": float(loss.detach())})
            print(f"    [diffusionguard] step {i:4d} "
                  f"loss {float(loss.detach()):.2f}", flush=True)

    return DiffusionGuardResult(adv.detach(), spec, loss_fn.t0, history)
