"""encoder-targeted 損失：`‖E(x_def) − E(y_target)‖²`。

形式取自 PhotoGuard 的 encoder attack（Salman et al., ICML 2023, §3.1），
與弱 baseline 的 targeted reward（DEC-023）同源，差別只在後者量的是解碼後
的像素差、走完整條去噪軌跡。

選它當像素臂的共用損失有兩個理由：

1. **只跑 VAE 編碼器**，一次前向加反向在 512² 上是毫秒量級，使「對半徑
   二分搜尋以對齊預算」這件事負擔得起（`param_pgd.fit_to_budget`）。
   本專案的失敗方向多半很貴；這一條要能便宜地被否證。
2. **它不偏袒任何一個參數化。** 損失只看 latent，不看擾動長什麼樣子，
   故加性與相位在同一個目標下競爭。

`y_target` 沿用 `data/targets/gray.png`（DEC-023 的選擇）。FND-031 記過灰圖
的高位移有一部分來自「推向無內容」造成的全域降對比，那個偏差對三個條件
是共同的，不影響參數化之間的歸因。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def make_encoder_target_loss(sd, y_target01: torch.Tensor,
                             mask: Optional[torch.Tensor] = None) -> Callable:
    """回傳 `loss(x_def01) -> 純量`，要最小化。

    目標 latent 只編碼一次並 detach：它是常數，每次迭代重編碼除了浪費之外
    還會讓損失曲線帶上 VAE 的隨機性（`encode_image` 走 mode 而非 sample 時
    無此問題，但不應該依賴那個細節）。

    `mask` 給定時（inpainting 威脅模型，1 = 攻擊方要重畫的區域）改為編碼
    `x ⊙ (1 − mask)`。**這不是可選的細化，是正確性**：inpainting 的 UNet
    收到的後 4 個通道就是 `encode(x_def ⊙ (1 − mask))`
    （`SDWrapper.mask_latents`），對整張圖的編碼取損失等於把一大半梯度打在
    進不了模型的像素上。2026-08-14 的預檢實測到這個後果——`phase` 對
    `phase_rand` 只有 1.04×，即最佳化沒有找到任何相位特有的東西，剩下的
    只是擾動幅度本身。
    """
    with torch.no_grad():
        y = y_target01 if mask is None else y_target01 * (1.0 - mask.to(y_target01))
        z_target = sd.encode_image(y).detach()

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        x = x_def01 if mask is None else x_def01 * (1.0 - mask.to(x_def01))
        return (sd.encode_image(x) - z_target).pow(2).mean()

    return loss
