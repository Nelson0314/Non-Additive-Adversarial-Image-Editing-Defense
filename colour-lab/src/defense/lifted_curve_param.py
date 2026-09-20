"""在單調曲線外面加一段「輸出色階」：抬黑位與壓白位。

現行族到不了的方向
────────────────────────────────────────────────────────────────────
`ColorCurveParam` 的輸出恆滿足 `F(0) = 0`、`F(1) = 1`（第 0 段的累積和為 0，
末端除以總和）。也就是說**整類抬黑位／壓白位的操作不在可行域裡**，而那是
影像軟體裡的「輸出色階」，褪色底片感與陰影染色都從那裡來。本族把它補上：

    T_c(x) = lift_c + (1 − lift_c − drop_c) · F_c(x)

`F` 仍是原本那條 64 段單調分段線性曲線，`lift`、`drop` 逐通道各一個。
逐通道給的意思是黑位可以帶顏色——那正是這一族與純亮度操作不同的地方。

由構造保證的性質
────────────────────────────────────────────────────────────────────
1. **不需要鉗回**：`lift ≤ LIFT_MAX`、`drop ≤ DROP_MAX`，兩個上限相加小於 1，
   所以 `1 − lift − drop ≥ 1 − LIFT_MAX − DROP_MAX > 0`，輸出恆落在
   `[lift, 1 − drop] ⊂ [0,1]`。
2. **單調不變**：仿射的係數為正，`F` 的單調性直接傳下來。
3. **斜率仍有上下界**：只是整條乘上一個落在
   `[1 − LIFT_MAX − DROP_MAX, 1]` 的正係數。
4. **低頻不變**：仍然是一個只吃顏色的全域映射，沒有座標。

起點不是零梯度點
────────────────────────────────────────────────────────────────────
`lift = LIFT_MAX · u`，`u` 是直接的參數、每步之後投影回 `[0,1]`，不是
`sigmoid`。恆等起點 `u = 0` 在 sigmoid 下是梯度為零的點，一步都不會動。
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.defense.color_param import ColorCurveParam


class LiftedCurveParam(ColorCurveParam):
    """`ColorCurveParam` ＋ 逐通道的輸出色階。`lift_max`／`drop_max` 由設定檔給。"""

    name = "lifted_curve"

    def __init__(self, radius: float = 5.0, pieces: int = 64,
                 lift_max: float = 0.25, drop_max: float = 0.25,
                 bound_mode: str = "advcf",
                 apply_where: Optional[torch.Tensor] = None,
                 init_jitter: float = 0.0):
        super().__init__(radius=radius, pieces=pieces, bound_mode=bound_mode,
                         apply_where=apply_where, init_jitter=init_jitter)
        if float(lift_max) < 0.0 or float(drop_max) < 0.0:
            raise ValueError(
                f"lift_max 與 drop_max 不能是負的，收到 {lift_max}、{drop_max}")
        if float(lift_max) + float(drop_max) >= 1.0:
            raise ValueError(
                f"lift_max + drop_max 要小於 1，收到 {lift_max} + {drop_max}")
        self.lift_max = float(lift_max)
        self.drop_max = float(drop_max)
        self.u_lift: Optional[torch.Tensor] = None
        self.u_drop: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        shape = (1, 3, 1)
        zeros = torch.zeros(shape, device=x01.device, dtype=x01.dtype)
        if self.init_jitter > 0:
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7717)
            lift = self.init_jitter * torch.rand(shape, generator=generator)
            drop = self.init_jitter * torch.rand(shape, generator=generator)
            zeros_lift = lift.to(x01.device, x01.dtype)
            zeros_drop = drop.to(x01.device, x01.dtype)
        else:
            zeros_lift, zeros_drop = zeros, zeros.clone()
        self.u_lift = zeros_lift.clone().requires_grad_(True)
        self.u_drop = zeros_drop.clone().requires_grad_(True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        base = self._curve(x01, self.theta)
        lift = (self.lift_max * self.u_lift).unsqueeze(-1)
        drop = (self.drop_max * self.u_drop).unsqueeze(-1)
        out = lift + (1.0 - lift - drop) * base
        if self.apply_where is None:
            return out
        w = self.apply_where.to(device=out.device, dtype=out.dtype)
        return w * out + (1.0 - w) * x01

    def params(self) -> List[torch.Tensor]:
        return [self.theta, self.u_lift, self.u_drop]

    def state_dict(self):
        return {"theta": self.theta.detach().clone(),
                "u_lift": self.u_lift.detach().clone(),
                "u_drop": self.u_drop.detach().clone()}

    def load_state_dict(self, state):
        self.theta = state["theta"].detach().clone().requires_grad_(True)
        self.u_lift = state["u_lift"].detach().clone().requires_grad_(True)
        self.u_drop = state["u_drop"].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def project(self) -> None:
        super().project()
        self.u_lift.clamp_(0.0, 1.0)
        self.u_drop.clamp_(0.0, 1.0)

    def step_scale(self) -> float:
        """`θ` 的盒寬遠小於 `u` 的 1，取 `θ` 的——步長公式只吃一個數，
        取大的會讓 `θ` 一步撞邊界。`u` 走得慢一點由 `project` 兜著。"""
        return super().step_scale()

    def levels(self) -> dict:
        """回報用：目前的黑位與白位，逐通道。"""
        with torch.no_grad():
            lift = (self.lift_max * self.u_lift).reshape(-1).tolist()
            drop = (self.drop_max * self.u_drop).reshape(-1).tolist()
        return {"lift": [round(v, 5) for v in lift],
                "drop": [round(v, 5) for v in drop]}
