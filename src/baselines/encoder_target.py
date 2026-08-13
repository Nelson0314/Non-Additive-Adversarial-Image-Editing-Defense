"""encoder-targeted 損失：`‖E(x_def) − E(y_target)‖²`。

形式取自 PhotoGuard 的 encoder attack（Salman et al., ICML 2023, §3.1），
與弱 baseline 的 targeted reward（DEC-023）同源，差別只在後者量的是解碼後
的像素差、走完整條去噪軌跡。

選它當 A 臂的共用損失有兩個理由：

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

from typing import Callable

import torch


def make_encoder_target_loss(sd, y_target01: torch.Tensor) -> Callable:
    """回傳 `loss(x_def01) -> 純量`，要最小化。

    目標 latent 只編碼一次並 detach：它是常數，每次迭代重編碼除了浪費之外
    還會讓損失曲線帶上 VAE 的隨機性（`encode_image` 走 mode 而非 sample 時
    無此問題，但不應該依賴那個細節）。
    """
    with torch.no_grad():
        z_target = sd.encode_image(y_target01).detach()

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        return (sd.encode_image(x_def01) - z_target).pow(2).mean()

    return loss
