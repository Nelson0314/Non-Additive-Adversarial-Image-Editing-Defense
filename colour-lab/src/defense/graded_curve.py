"""一個色彩分級：色相旋轉的預算與「顏色放在哪一段」分成兩個結構常數。

要解決的是什麼
────────────────────────────────────────────────────────────────────
AdvCF 的三條通道曲線互相獨立，求解端沒有自然度項，於是預算全部花在最便宜的
大動作上——**整張圖的色相旋轉**。ΔE00 上限一抬高，臉就變綠。量到的天花板也
指向同一件事：把通道之間的差鎖住，`man_00` 的 ΔE00 從 21.40 掉到 9.85。

本族把「色相旋轉」與「顏色放在哪」分開，兩個都是設定檔給的常數，求解端動不了：

    F_c  = ChannelSpanCurveParam(span)     # 共用曲線 ＋ 有界的逐通道偏移
    T_c  = lift_c + (1 − lift_c − drop_c) · F_c

- `span` 是**中間調的色相預算**：`span = 0` 時三條曲線逐位元相同，中間調
  不可能偏色；`span = 1` 退回 AdvCF。
- `lift`／`drop` 是**兩端的顏色**：黑位與白位各自帶色，也就是分離色調
  （split-tone）與褪色片感。膚色落在中間調，所以兩端染色不會把臉轉走。

由構造保證的性質（各有測試釘住）
────────────────────────────────────────────────────────────────────
1. θ 進不了 AdvCF 盒子外面（`min(u,1−u)` 那個因子），每條曲線都單調、
   固定 0 與 1 兩端、斜率有上下界。
2. 通道之間的差有硬上界 `span·(hi−lo)`。
3. `lift_max + drop_max < 1`，輸出恆在 `[lift, 1−drop] ⊂ [0,1]`，不需鉗回。
4. 仍然是只吃顏色的全域映射，沒有座標、沒有遮罩。

參數量 `4K + 6`（K 個共用斜率 ＋ 3K 個逐通道偏移 ＋ 六個色階），比 AdvCF 多。
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.defense.channel_span_curve import ChannelSpanCurveParam


class GradedCurveParam(ChannelSpanCurveParam):
    """`ChannelSpanCurveParam` ＋ 逐通道的輸出色階。"""

    name = "graded_curve"

    def __init__(self, radius: float = 5.0, pieces: int = 64,
                 span: float = 0.25,
                 lift_max: float = 0.25, drop_max: float = 0.25,
                 apply_where: Optional[torch.Tensor] = None,
                 init_jitter: float = 0.0):
        super().__init__(radius=radius, pieces=pieces, span=span,
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
        if self.init_jitter > 0:
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7717)
            lift = self.init_jitter * torch.rand(shape, generator=generator)
            drop = self.init_jitter * torch.rand(shape, generator=generator)
        else:
            lift = torch.zeros(shape)
            drop = torch.zeros(shape)
        self.u_lift = lift.to(x01.device, x01.dtype).clone().requires_grad_(True)
        self.u_drop = drop.to(x01.device, x01.dtype).clone().requires_grad_(True)

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
        return [self.u, self.d, self.u_lift, self.u_drop]

    def state_dict(self):
        return {"u": self.u.detach().clone(), "d": self.d.detach().clone(),
                "u_lift": self.u_lift.detach().clone(),
                "u_drop": self.u_drop.detach().clone()}

    def load_state_dict(self, state):
        self.u = state["u"].detach().clone().requires_grad_(True)
        self.d = state["d"].detach().clone().requires_grad_(True)
        self.u_lift = state["u_lift"].detach().clone().requires_grad_(True)
        self.u_drop = state["u_drop"].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def project(self) -> None:
        super().project()
        self.u_lift.clamp_(0.0, 1.0)
        self.u_drop.clamp_(0.0, 1.0)

    def levels(self) -> dict:
        with torch.no_grad():
            lift = (self.lift_max * self.u_lift).reshape(-1).tolist()
            drop = (self.drop_max * self.u_drop).reshape(-1).tolist()
        return {"lift": [round(v, 5) for v in lift],
                "drop": [round(v, 5) for v in drop]}
