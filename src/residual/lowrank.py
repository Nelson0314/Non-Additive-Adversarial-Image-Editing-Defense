"""外積參數化的低秩殘差張量與秩排程 — spec §4.1、§4.2。

秩不以懲罰項近似，而由參數化方式保證：

    Δ[c] = Σ_{i=1..r}  U[c,i,:] ⊗ V[c,i,:]ᵀ

矩陣的每一列都是 r 個固定向量 V[c,1..r,:] 的線性組合，故列空間維度至多為 r。
此為代數事實，不需事後驗證，訓練迴圈內亦不需執行 SVD。

秩排程 envelope 多項式與四種排程函數直接取自 LRDM（Tan & Yuan, CVPR 2026）
式 (15)–(19)；`const` 為本規格新增，作為排程 ablation 的基準線。
"""

import math

import torch
import torch.nn as nn

SCHEDULES = ("const", "LI", "LD", "PI", "PD")


def envelope(d: float, p: int = 3) -> float:
    """LRDM 式 (17)：envelope(d; p) = 1 + a·d^p + b·d^(p+1) + c·d^(p+2)。

    a = −(p+1)(p+2)/2,  b = p(p+2),  c = −p(p+1)/2
    在 d ∈ [0,1] 上由 1 單調遞減至 0（envelope(0)=1、envelope(1)=0）。
    """
    a = -(p + 1) * (p + 2) / 2.0
    b = p * (p + 2)
    c = -p * (p + 1) / 2.0
    return 1.0 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)


def rank_at(
    schedule: str,
    t: int,
    t_max: int,
    max_rank: int,
    const_rank: int = None,
    p: int = 3,
) -> int:
    """回傳 timestep t 上應使用的秩。

    `t` 為實際的 DDIM timestep、`t_max` 為排程上限，取 d = t/t_max。
    此對應沿用 LRDM 的 t/T 語意：d=0 為乾淨端、d=1 為高噪端，故 LI
    表示「噪聲越高秩越大」。spec §4.2 已載明此語意轉換。

    結果 clamp 至 [0, max_rank]。秩為 0 時該步不注入殘差，此為
    LI（d=0）與 LD（d=1）在端點的自然結果，非錯誤。
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"未知的秩排程 {schedule!r}，可用：{SCHEDULES}")

    if schedule == "const":
        if const_rank is None:
            raise ValueError("schedule='const' 需提供 const_rank")
        return max(0, min(int(const_rank), max_rank))

    d = 0.0 if t_max == 0 else t / t_max
    d = min(1.0, max(0.0, d))

    if schedule == "LI":
        frac = d
    elif schedule == "LD":
        frac = 1.0 - d
    elif schedule == "PI":
        frac = 1.0 - envelope(d, p)
    else:  # PD
        frac = envelope(d, p)

    r = math.ceil(min(1.0, max(0.0, frac)) * max_rank)
    return max(0, min(r, max_rank))


class LowRankResidual(nn.Module):
    """秩由外積參數化保證的殘差張量。

    U: (steps, channels, max_rank, height)
    V: (steps, channels, max_rank, width)

    `steps` 為需要獨立殘差的步數。site P 為 1（像素空間單一殘差）；
    site L 為 k_inv（去噪每步一組）。

    初始化採 U ~ N(0, init_std)、V = 0，故初始 Δ = 0（即 x_def = x），
    同時 ∂L/∂V ≠ 0 使梯度可流動。此慣例沿用 LoRA 的 A 高斯／B 零初始化。
    """

    def __init__(
        self,
        steps: int,
        channels: int,
        height: int,
        width: int,
        max_rank: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        if max_rank < 1:
            raise ValueError(f"max_rank 須 ≥ 1，收到 {max_rank}")
        if max_rank > min(height, width):
            raise ValueError(
                f"max_rank={max_rank} 超過 min(H,W)={min(height, width)}，"
                "秩不可能高於矩陣的較小維度"
            )
        self.steps = steps
        self.channels = channels
        self.height = height
        self.width = width
        self.max_rank = max_rank

        self.U = nn.Parameter(torch.randn(steps, channels, max_rank, height) * init_std)
        self.V = nn.Parameter(torch.zeros(steps, channels, max_rank, width))

    def forward(self, step: int = 0, rank: int = None) -> torch.Tensor:
        """回傳第 `step` 步、秩為 `rank` 的殘差，形狀 (channels, height, width)。

        `rank=None` 表示使用 max_rank；`rank=0` 回傳零張量（不注入）。
        僅切片前 `rank` 個分量，未使用的分量不參與計算。
        """
        r = self.max_rank if rank is None else int(rank)
        if r < 0 or r > self.max_rank:
            raise ValueError(f"rank={r} 超出 [0, {self.max_rank}]")
        if r == 0:
            return torch.zeros(
                self.channels,
                self.height,
                self.width,
                device=self.U.device,
                dtype=self.U.dtype,
            )
        u = self.U[step, :, :r, :]
        v = self.V[step, :, :r, :]
        return torch.einsum("cri,crj->cij", u, v)

    def num_parameters(self, rank: int = None) -> int:
        """回傳使用秩為 `rank` 時實際參與計算的參數量。"""
        r = self.max_rank if rank is None else int(rank)
        # U 貢獻 height、V 貢獻 width；H=W 時等於 2·steps·channels·r·H
        return self.steps * self.channels * r * (self.height + self.width)

    def extra_repr(self) -> str:
        return (
            f"steps={self.steps}, channels={self.channels}, "
            f"size={self.height}x{self.width}, max_rank={self.max_rank}"
        )
