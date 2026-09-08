"""直接以**防禦效果**為目標：讓模型從防禦圖建出來的東西，離它從原圖建出來的
東西越遠越好。

與既有三個損失的差別
────────────────────────────────────────────────────────────────────
`encoder_target`、`latent_norm`、`image_guidance` 都是**同一種形狀**的目標：
把影像條件推向某個特定的點（灰圖的 latent、零張量、或 UNet 的無條件反應）。
它們是「讓模型看不到這張圖」。

本項不指定要推到哪裡，只要求**離原圖的結果遠**：

    L = − E_{t,ε} ‖ x̂₀(z_t ; c_I(x_def)) − x̂₀(z_t ; c_I(x)) ‖²

`x̂₀` 是由 `z_t` 與 UNet 的噪聲預測反解出來的「模型認為的乾淨影像」：

    x̂₀ = ( z_t − sqrt(1−ᾱ_t)·ε̂ ) / sqrt(ᾱ_t)

這是取樣器每一步實際在往哪裡走的量。兩個條件下的 `x̂₀` 差得越開，最後解出來
的兩張圖就差得越開——也就是**位移**，本專案的主讀數。

為什麼不能直接用攻擊指令
────────────────────────────────────────────────────────────────────
最直接的目標會是「讓 `編輯(原圖)` 與 `編輯(防禦圖)` 差很多」，但那需要把攻擊
的文字條件寫進損失，而**威脅模型的前提是防護對象已知、攻擊指令不是**
（`anti-purify/COLOR_AND_DECOY.md` 第二節）。用了攻擊指令就變成另一個威脅
模型，數字不可與既有批次並列。

故文字條件一律取**空字串**，與 `image_guidance` 同一個選擇。兩者的差別在
比較的對象：那一個比「有影像條件 vs 沒有影像條件」，本項比「防禦圖 vs 原圖」。

三個實作上的坑
────────────────────────────────────────────────────────────────────
1. **拼進 UNet 的影像 latent 不乘 `scaling_factor`**（`IP2PWrapper.image_latents`
   的 docstring）。用 `encode_image` 去拼會讓影像條件的強度整個跑掉，
   補錯不會拋錯。
2. **原圖那一支不依賴 x_def**，故在 `no_grad` 底下算並 detach。它是常數。
3. **符號**：`run_param_pgd` 一律**最小化**，故回傳的是負的散度。寫成正的不會
   拋錯，只會安靜地把防禦圖推回原圖，而報表上的曲線看起來仍然在收斂。

`z_t` 的抽法沿用 `image_guidance` 的兩個候選，且同樣**必填**：兩者都是近似。
`x_clean` 一律是原圖——它同時是軌跡的錨與比較的基準。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.defense.fixedpoint_loss import _null_embedding, _scheduler_of
from src.defense.image_guidance_loss import ZT_MODES, _INHERIT, _weighted_mean


def make_edit_divergence_loss(
    ip2p,
    *,
    zt_mode: str,
    x_clean: torch.Tensor,
    t_min: int = 1,
    t_max: int = 1000,
    samples: int = 1,
    seed: int = 0,
    weight: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x_def01) -> 純量`，**要最小化**（值越負代表推得越開）。

    `weight` 是像素域的 (1,1,H,W) 軟遮罩，把散度的空間平均改成加權平均；
    設成主體遮罩就是「只要求**主體那一塊**被推開」。`None` 時全域均勻。
    """
    if zt_mode not in ZT_MODES:
        raise ValueError(f"未知的 zt_mode：{zt_mode!r}，必須是 {ZT_MODES}")
    if not 1 <= t_min <= t_max:
        raise ValueError(f"需要 1 <= t_min <= t_max，收到 {t_min}／{t_max}")
    if samples < 1:
        raise ValueError(f"samples 必須為正整數，收到 {samples}")
    if x_clean is None:
        raise ValueError(
            "edit_divergence 需要 x_clean（**原圖**）：它同時是取樣軌跡的錨"
            "與散度的比較基準，沒有它這個損失沒有定義。")

    unet = ip2p.unet
    device = ip2p.device
    sched = _scheduler_of(ip2p)
    abar = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    if t_max > len(abar):
        raise ValueError(f"t_max={t_max} 超出排程長度 {len(abar)}")
    null_emb = _null_embedding(ip2p)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    with torch.no_grad():
        z_src = ip2p.encode_image(x_clean).detach()
        z_img_clean = ip2p.image_latents(x_clean).detach()

    def _x0(z_t, eps_hat, step):
        a = abar[step].to(z_t.dtype)
        return (z_t - (1.0 - a).sqrt() * eps_hat) / a.sqrt().clamp_min(1e-8)

    def _pair(z_t, step, z_img):
        tt = torch.tensor([step], device=device, dtype=torch.long)
        emb = null_emb.to(z_t.dtype)
        with torch.no_grad():
            e_cln = unet(torch.cat([z_t, z_img_clean.to(z_t.dtype)], dim=1), tt,
                         encoder_hidden_states=emb).sample.detach()
        e_def = unet(torch.cat([z_t, z_img], dim=1), tt,
                     encoder_hidden_states=emb).sample
        return _x0(z_t, e_def, step), _x0(z_t, e_cln, step)

    def _zt(ref, step, eps):
        if zt_mode == "noise":
            return eps
        a = abar[step].to(ref.dtype)
        return z_src.to(ref.dtype) * a.sqrt() + eps * (1.0 - a).sqrt()

    def make_fixed(n_draws: int, eval_seed: int):
        """決定性的評估函數。理由同 `image_guidance`：訓練用的損失每一步重抽
        `(t, ε)`，逐步值本來就會抖，拿它判收斂會判錯。"""
        g2 = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        steps_fixed = [int(torch.randint(t_min - 1, t_max, (1,), generator=g2))
                       for _ in range(n_draws)]
        eps_fixed = [None] * n_draws

        def fixed(x_def01: torch.Tensor, eval_weight=_INHERIT) -> torch.Tensor:
            w = weight if eval_weight is _INHERIT else eval_weight
            z_img = ip2p.image_latents(x_def01)
            total = None
            for k, step in enumerate(steps_fixed):
                if eps_fixed[k] is None:
                    eps_fixed[k] = torch.randn(
                        z_img.shape, generator=g2, dtype=torch.float32
                    ).to(device=z_img.device, dtype=z_img.dtype)
                z_t = _zt(z_img, step, eps_fixed[k])
                a, b = _pair(z_t, step, z_img)
                term = _weighted_mean((a - b).pow(2), w)
                total = term if total is None else total + term
            return -total / len(steps_fixed)

        return fixed

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        z_img = ip2p.image_latents(x_def01)
        total = None
        for _ in range(samples):
            step = int(torch.randint(t_min - 1, t_max, (1,), generator=gen))
            eps = torch.randn(z_img.shape, generator=gen, dtype=torch.float32
                              ).to(device=z_img.device, dtype=z_img.dtype)
            z_t = _zt(z_img, step, eps)
            a, b = _pair(z_t, step, z_img)
            term = _weighted_mean((a - b).pow(2), weight)
            total = term if total is None else total + term
        # **負號**：`run_param_pgd` 最小化，而我們要把散度推大。
        return -total / samples

    loss.make_fixed = make_fixed
    return loss
