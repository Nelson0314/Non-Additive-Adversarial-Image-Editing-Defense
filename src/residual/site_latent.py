"""site L — latent 空間、去噪逐步注入 — spec §4.3。

第 j 步的注入方式為修改預測噪聲：

    ε̂′ⱼ = ε̂ⱼ + Δⱼ

選擇修改 ε̂ 而非直接修改 z：ε̂ 同時進入 pred_x0 與下一步的 z，其影響會
沿軌跡傳播。此與 APA 的 step-level 注入位置一致。

**秩約束精確作用於注入的 latent 殘差 Δⱼ，而非最終像素殘差。**
x_def − x 還要經過 DDIM 更新與 VAE 解碼兩道非線性變換，其秩是湧現的，
必須由奇異值譜診斷實測（spec §4.3、§8.2）。此處不做任何像素秩的假設。
"""

from typing import Optional

import torch

from src.residual.base import ResidualModule
from src.residual.lowrank import LowRankResidual, rank_at


class LatentResidual(ResidualModule):
    site = "L"

    def __init__(
        self,
        steps: int,
        channels: int = 4,
        size: int = 64,
        max_rank: int = 32,
        schedule: str = "const",
        const_rank: int = 8,
        poly_p: int = 3,
        scale: float = 1.0,
        init_std: float = 0.02,
        seed: int = None,
    ):
        super().__init__()
        self.tensor = LowRankResidual(
            steps=steps,
            channels=channels,
            height=size,
            width=size,
            max_rank=max_rank,
            init_std=init_std,
            seed=seed,
        )
        self.steps = steps
        self.schedule = schedule
        self.const_rank = const_rank
        self.poly_p = poly_p
        # scale 是固定常數，不參與優化；用於讓 ε 的擾動量級與 ε 本身相稱
        self.scale = scale

    def _rank_for(self, t: torch.Tensor, t_max: int) -> int:
        return rank_at(
            self.schedule,
            t=int(t),
            t_max=t_max,
            max_rank=self.tensor.max_rank,
            const_rank=self.const_rank,
            p=self.poly_p,
        )

    def eps_hook(self, ts: torch.Tensor, steps: int) -> Optional[callable]:
        if not self.enabled:
            return None
        if steps > self.steps:
            raise ValueError(
                f"去噪步數 {steps} 超過模塊配置的 steps={self.steps}，"
                "殘差張量沒有對應的分量"
            )
        t_max = int(ts[-1])

        def hook(eps: torch.Tensor, step_idx: int, t: torch.Tensor) -> torch.Tensor:
            r = self._rank_for(t, t_max)
            if r == 0:
                return eps
            delta = self.tensor(step=step_idx, rank=r)  # (C,H,W)
            return eps + self.scale * delta.unsqueeze(0).to(eps.dtype)

        return hook

    def rank_trace(self, ts: torch.Tensor, steps: int) -> list:
        """每步實際使用的秩，順序與 denoise 的 step_idx 一致（降冪走訪 t）。

        **方向陷阱**：排程名稱的「遞增／遞減」是對 LRDM 的 timestep t 而言，
        t=0 為乾淨端、t=T 為高噪端。但去噪是由高噪走向乾淨，即 t 遞減，
        故沿著 step_idx：

            LD（對 t 遞減）→ 沿去噪步序**遞增**
            LI（對 t 遞增）→ 沿去噪步序**遞減**

        兩者在此處恰好反向。比對排程行為時務必指明是對 t 還是對 step_idx。
        """
        t_max = int(ts[-1])
        return [self._rank_for(ts[i + 1], t_max) for i in reversed(range(steps))]
