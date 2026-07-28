"""site P — 像素空間 — spec §4.3。

    x_def = clamp(x + Δ, 0, 1)

此 site 為**加性**方法，且像素空間殘差的秩**精確等於** r（無任何後續非線性
變換）。它與 site L 的對比構成 spec §7.4 的因果切割：

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
    ):
        super().__init__()
        self.tensor = LowRankResidual(
            steps=1,
            channels=channels,
            height=size,
            width=size,
            max_rank=max_rank,
            init_std=init_std,
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

    def rank_trace(self, ts=None, steps=None) -> list:
        return [self.const_rank]
