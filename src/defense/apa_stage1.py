"""APA 官方階段一：denoising MSE 的字面重現。

由 `src/defense/optimize.py` 抽出（`docs/PLAN.md` §6.1a）。**抽出的理由是
依賴而非美觀**：主線（`scripts/apa_baseline.py`）只用 `align_apa_native`，
而 `optimize.py` 匯入 `generator`／`objective`／`purify.ops`／`calibration`，
一支就帶進 16 個模組。本函式只用 `sd` 與 `lora`，與該檔其餘部分零耦合。

`optimize.py` 仍以 `from src.defense.apa_stage1 import align_apa_native`
沿用同一份實作，不複製一份——FND-027 是拿這段程式量出來的，兩份實作分岔
之後那筆結論會失去對應的程式。
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.models.sd import SDWrapper


def align_apa_native(
    sd: SDWrapper,
    lora: nn.Module,
    x01: torch.Tensor,
    class_name: str,
    steps: int,
    lr: float,
    noise_offset: float,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 20,
) -> List[Dict]:
    """APA 官方階段一的字面重現：denoising MSE，`docs/reference/dia_apa.md` §1.2。

        R_s(Δθ) = E_{t,ε}[ −‖ε − ε_{θ+Δθ}(z_t, t, c)‖² ]

    這是 FND-016／FND-022（「原文階段一在本管線上是空操作」）尚未回答的一步：
    那兩筆記錄量的是 `align()` 誤用本專案自己的保真 hinge 當階段一損失的
    結果，不是官方目標本身。本函式改用官方目標，供並列比較。

    與 `align()` 的三個結構差異，全部是刻意的：

    1. **不經 DDIM/BDIA 鏈、不經 VAE decode。** 每步只對 `noisy_latents`
       跑一次 UNet、在 epsilon 空間算 MSE，反傳不觸及生成鏈或解碼器。
       這正是官方程式碼的路徑（`visual_alignment.py:158-192`），也是它比
       `align()`（端對端反傳到解碼影像）快得多的原因。
    2. **AdamW，不是 Adam。** `align()`／`run_stages` 全部用 Adam；官方用
       AdamW（`betas=(0.9,0.999)`、`weight_decay=1e-2`、`eps=1e-8`）。
    3. **學習率與步數是官方固定值，不查校準表。** `resolve_lr` 的規則
       （`docs/reference/CODE_CONTRACTS.md`）是「未校準的值一律拒絕」，
       但本函式的目的正是忠實重現官方超參數，不是本專案自訂的逐條件校準
       ——查表反而會把這次比較做成另一個「本專案版本」，答不了「原生做法
       多好」這個問題。呼叫端傳入 `src/residual/site_apa.py` 的
       `APA_STAGE1_LR`／`APA_STAGE1_STEPS`／`APA_NOISE_OFFSET`。

    不做 `align()` 的「保留軌跡最佳步」：官方程式碼本身也沒有這個機制
    （固定 200 步、直接用最後一步），加上去會製造第三個未查證的偏離。

    `class_name` 對應官方的 `data.json` 的 `class` 欄（ImageNet 類別名，
    無模板），不是本專案 prompt-free 協定下的攻擊 prompt——見呼叫端
    （`scripts/apa_baseline.py`）如何選取這個值。

    回傳逐步的 `{"step", "loss"}`（或發散時的 `{"step", "loss": inf,
    "diverged": True}`），不回傳影像——呼叫端在訓練結束後自行跑一次
    `DefenseGenerator` 取得 `x_lora_native`，理由是這裡的損失與最終保真度
    是两个不同的量，混在同一個函式裡會誤導成「這個損失曲線代表保真度」。
    """
    params = [p for p in lora.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "lora 模塊沒有可訓練參數；APA 原生階段一沒有東西可以優化"
        )
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                            weight_decay=1e-2, eps=1e-8)
    gen = torch.Generator(device=sd.device).manual_seed(seed)

    with torch.no_grad():
        latents = sd.encode_image(x01)
        cond = sd.encode_text(class_name)

    alphas_cumprod = sd.alphas_cumprod(sd.device).to(latents.dtype)
    num_train_timesteps = sd.num_train_timesteps

    lora.enable()
    history: List[Dict] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        noise = torch.randn(latents.shape, generator=gen, device=sd.device,
                            dtype=latents.dtype)
        if noise_offset:
            # 官方程式碼未在論文中說明的額外項（`visual_alignment.py:158-192`
            # 的 `noise_offset`），逐通道對 noise 加一個常數偏移。
            noise = noise + noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                generator=gen, device=sd.device, dtype=latents.dtype)

        t = torch.randint(0, num_train_timesteps, (latents.shape[0],),
                          generator=gen, device=sd.device).long()
        sqrt_ac = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1mac = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        noisy_latents = sqrt_ac * latents + sqrt_1mac * noise

        model_pred = sd.unet(noisy_latents, t, cond).sample
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

        if not torch.isfinite(loss):
            print(f"  [align_apa_native] step {step:>4d}  發散：loss 非有限值，"
                  "本候選就此停止", flush=True)
            history.append({"step": step, "loss": float("inf"),
                            "diverged": True})
            break

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        loss_value = float(loss.detach())
        history.append({"step": step, "loss": loss_value})
        if step % log_every == 0 or step == steps - 1:
            print(f"  [align_apa_native] step {step:>4d}  loss={loss_value:.4f}",
                  flush=True)

    return history
