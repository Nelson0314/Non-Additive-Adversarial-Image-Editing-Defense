"""殘差模塊的統一介面 — spec §4.4。

介面判準：
- `optimize.py` 只呼叫 `module.parameters()`，不得知道自己在優化外積向量還是 LoRA 矩陣
- `generator.py` 只詢問模塊提供哪種能力，不得依 site 寫死分支邏輯

因此模塊以「能力」而非「型別」對外表達：像素側殘差實作 `pixel_residual`，
去噪側殘差實作 `eps_hook`，兩者的預設實作皆回傳 None 表示不提供。
"""

from typing import Optional

import torch
import torch.nn as nn


class ResidualModule(nn.Module):
    """可開關的殘差模塊。關閉時行為必須與模塊不存在完全一致。"""

    site: str = "?"

    def __init__(self):
        super().__init__()
        self._enabled = True

    # ---- 開關 ----

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    # ---- 能力（子類別擇一覆寫）----

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        """像素側：回傳防禦圖，或 None 表示不提供此能力。"""
        return None

    def eps_hook(self, ts: torch.Tensor, steps: int) -> Optional[callable]:
        """去噪側：回傳 eps_hook(eps, step_idx, t)，或 None 表示不提供。"""
        return None

    def emb_residual(self, emb: torch.Tensor) -> Optional[torch.Tensor]:
        """條件側：回傳要加到文字嵌入上的殘差，或 None 表示不提供。

        只作用於 `generate()` 的去噪段。DDIM inversion 必須使用未擾動的
        嵌入，否則 z_inv 依賴 φ、`prepare()` 的快取失效——那個快取每個
        iteration 省下一條 k_inv 步的 UNet 前向，是本迴圈最大的一項節省。
        """
        return None

    # ---- 診斷 ----

    def raw_residual(self) -> Optional[torch.Tensor]:
        """注入前、未經任何非線性變換的殘差，其秩等於設定值。

        存在理由是把「架構保證的秩」與「實際觀察到的像素秩」分開量測。
        site P 的 `x_def − x` 已經過 clamp，其數值秩**不等於**設定值：
        實測 r=2、128² 真實影像上為 84~87，但 99% 能量仍落在秩 2~3，
        因為只有 0.46% 的元素被 clamp 改動。沒有這個介面就無法區分
        「秩約束失效」與「clamp 造成的稀疏擾動」。

        回傳 None 表示該 site 沒有對應的像素空間量（如 site L）。
        """
        return None

    def rank_trace(self, ts: torch.Tensor, steps: int) -> list:
        """回傳每步實際使用的秩，供報告與診斷圖使用。"""
        return []

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
