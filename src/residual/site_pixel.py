"""site P — 像素空間 — spec §4.3。

    x_def = clamp(x + Δ, 0, 1)

此 site 為**加性**方法。注入的殘差 Δ 其秩精確等於 r，但**實際的像素殘差
`x_def − x` 的秩不等於 r**：`clamp` 是非線性變換，會在原圖的飽和像素處
產生稀疏擾動。實測（r=2、128² 真實影像、0.97% 飽和像素）：

    clamp 前 Δ          effective_rank = [2, 2, 2]    energy_rank99 = [2, 2, 2]
    clamp 後 x_def − x  effective_rank = [84, 87, 83] energy_rank99 = [3, 2, 2]

被 clamp 改動者僅 225/49152 個元素（0.46%），能量可忽略，故低秩結構在
**能量意義**下成立、在**精確秩**意義下不成立。報告須以 energy_rank 為主要
量度並同時列出 effective_rank，不得宣稱像素秩精確等於 r。

它與 site L 的對比構成 spec §7.4 的因果切割：

- P 亦耐淨化 ⟹ 機制是秩結構
- P 不耐而 L 耐 ⟹ 機制是非加性

沒有去噪鏈也沒有 VAE，是三個 site 中成本最低的。
"""

from typing import Optional

import torch

from src.residual.base import ResidualModule
from src.residual.lowrank import LowRankResidual


class PixelResidual(ResidualModule):
    site = "P"

    def __init__(
        self,
        size: int = 512,
        channels: int = 3,
        max_rank: int = 32,
        const_rank: int = 8,
        scale: float = 1.0,
        init_std: float = 0.02,
        seed: int = None,
    ):
        super().__init__()
        self.tensor = LowRankResidual(
            steps=1,
            channels=channels,
            height=size,
            width=size,
            max_rank=max_rank,
            init_std=init_std,
            seed=seed,
        )
        self.const_rank = const_rank
        self.scale = scale

    def delta(self) -> torch.Tensor:
        """回傳 (1,C,H,W) 的像素殘差，未經 clamp。"""
        d = self.tensor(step=0, rank=self.const_rank)
        return self.scale * d.unsqueeze(0)

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01
        return (x01 + self.delta().to(x01.dtype)).clamp(0.0, 1.0)

    def raw_residual(self) -> torch.Tensor:
        """clamp 前的 Δ，秩精確等於 const_rank。與 x_def−x 的差異見 base。"""
        return self.delta().detach()

    def clamped_fraction(self, x01: torch.Tensor) -> float:
        """被 clamp 改動的元素比例。此值決定像素秩偏離設定值的程度。"""
        with torch.no_grad():
            raw = x01 + self.delta().to(x01.dtype)
            return float((raw != raw.clamp(0.0, 1.0)).float().mean())

    def rank_trace(self, ts=None, steps=None) -> list:
        return [self.const_rank]
