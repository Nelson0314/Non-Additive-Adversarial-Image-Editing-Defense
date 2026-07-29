"""全秩像素殘差 —— 低秩約束的對照組。

    x_def = clamp(x + Δ, 0, 1)      Δ ∈ ℝ^{1×C×H×W}，不受任何秩約束

與 site P 唯一的差別是 Δ 的參數化：site P 以外積構造 `Σᵢ U[c,i,:] ⊗ V[c,i,:]ᵀ`
把秩限制在 r，此處則直接把每個像素當成一個自由參數。施加方式、clamp、
保真度約束全部相同，故兩者的差異可完全歸因於秩約束本身。

**存在理由**：截至目前，「低秩」在本專案沒有任何實驗支撐。r = 1/4/16 的
內部比較只顯示秩越大偏移越大，那證明的是 r 作為能量預算的作用，不是低秩
帶來任何好處。在相同的感知失真下比較全秩與低秩，是唯一能回答這個問題的
對照，三種可能結果都會改變論文的論點：

1. 全秩在偏移與淨化存活率上都贏 → 低秩論點不成立，須改寫賣點
2. 全秩偏移贏但存活率輸 → 低秩是一個 robustness prior，空間相干的擾動
   比逐像素擾動更耐淨化，這是可以支撐論文的結果
3. 兩者相當 → 低秩至少無害，但賣點須降級為效率而非效果

**初始化為零**：自由參數在 Δ=0 處梯度不為零，故不需要 site P 那種
「U 高斯、V 零」的安排；直接零初始化即可同時得到 `x_def = x` 逐元素相等
與可流動的梯度。這也讓「模塊停用時不改變任何計算結果」的不變量在此模塊
成立方式與 site P 一致。

**參數量**：C·H·W。512² 三通道為 786,432，相對 r=16 的 16·(512+512)·3
= 49,152 多 16 倍。逐圖優化下這個差距本身就是一個變因，報告時須併列。

**Δ=0 是防禦項的駐點**（煙霧測試實測，step 0 的 |∇| = 2.06e-08）。原因是
`d` 取 LPIPS，而 LPIPS(a, a) = 0 是極小值，故 x_def = x 處 ∂d/∂x_def = 0。
逃離靠的是 SDEdit 鏈的數值不對稱加上 Adam 每座標更新量與梯度大小無關的
性質，實測第 1 步 |∇| 已回到 4.43e-03。

這個駐點**兩個 arm 共有**，不是全秩獨有的劣勢：低秩側 Δ = Σ U⊗V 在 V=0
時亦為零，且 ∂L/∂V = (∂L/∂Δ)·U，只要 ∂L/∂Δ = 0 就同樣為零。spec §4.1 說
「U 高斯／V 零使 ∂L/∂V ≠ 0」解決的是另一個問題（U 也為零時梯度恆為零），
不是這個駐點。兩側起點條件相同，比較仍然公平。
"""

from typing import Optional

import torch
import torch.nn as nn

from src.residual.base import ResidualModule


class FullRankPixelResidual(ResidualModule):
    site = "PF"

    def __init__(
        self,
        size: int = 512,
        channels: int = 3,
        scale: float = 1.0,
        seed: int = None,   # 介面對齊 PixelResidual；零初始化下不使用
    ):
        super().__init__()
        # 零初始化：見模組 docstring。seed 之所以無作用，是因為此處沒有任何
        # 隨機來源——保留參數是為了讓呼叫端不必依 site 分支傳不同的引數。
        self.delta_param = nn.Parameter(torch.zeros(1, channels, size, size))
        self.scale = scale

    def delta(self) -> torch.Tensor:
        """回傳 (1,C,H,W) 的像素殘差，未經 clamp。"""
        return self.scale * self.delta_param

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01
        return (x01 + self.delta().to(x01.dtype)).clamp(0.0, 1.0)

    def raw_residual(self) -> torch.Tensor:
        return self.delta().detach()

    def clamped_fraction(self, x01: torch.Tensor) -> float:
        with torch.no_grad():
            raw = x01 + self.delta().to(x01.dtype)
            return float((raw != raw.clamp(0.0, 1.0)).float().mean())

    def rank_trace(self, ts=None, steps=None) -> list:
        """回報實際可達的最大秩，即 min(H, W)。

        不回報實測秩：實測要看奇異值譜，那是 metrics.spectrum 的職責。
        此處回報的是「架構允許到多少」，與 site P 回報 const_rank 的語意一致。
        """
        return [min(self.delta_param.shape[-2], self.delta_param.shape[-1])]
