"""把「三條通道曲線可以差多少」做成一個結構上的旋鈕。

為什麼
────────────────────────────────────────────────────────────────────
AdvCF 的三條逐通道曲線完全獨立。振幅一大，三條曲線之間的差就是色偏，而
求解端沒有自然度項（自然要由結構保證，不是由損失保證），預算往哪個方向
花完全由目標函數決定。本族把「往色彩方向花多少」從求解端搬到參數化裡：

    θ_c = lo + (hi − lo) · [ u + span · d_c · min(u, 1 − u) ]

`u ∈ [0,1]` 是三通道**共用**的斜率剖面，`d_c ∈ [−1,1]` 是逐通道偏移，
`span ∈ [0,1]` 是設定檔給的常數，求解端動不了。

兩端各是什麼
────────────────────────────────────────────────────────────────────
- `span = 0`：三條曲線逐位元相同，等於影像軟體裡的 RGB 複合曲線
  （亮度與對比），**不可能產生色偏**。
- `span = 1`：逐通道偏移可以把 θ 推到盒子的任一端，能力與 AdvCF 相同。

由構造保證的性質
────────────────────────────────────────────────────────────────────
1. `min(u, 1−u)` 這個因子讓 `u + span·d·min(u,1−u)` 在 `span ≤ 1` 時**必定**
   落在 [0,1]，所以 θ 不必鉗回，每條曲線都在 AdvCF 的盒子裡：單調、固定
   0 與 1 兩端、斜率有上下限。這些性質與振幅無關。
2. 通道之間的差有硬上界：`|θ_c − θ_c'| ≤ span · (hi − lo)`。`span` 直接
   就是色偏的預算，與整體振幅分開。
3. 參數量是 4K（K 個共用 ＋ 3K 個逐通道），比 AdvCF 的 3K 多。

`project()` 只鉗 `u` 與 `d` 各自的盒子，不鉗 θ——θ 本來就進不了盒子外面。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.defense.color_param import ColorCurveParam


class ChannelSpanCurveParam(ColorCurveParam):
    """共用斜率剖面 ＋ 有界的逐通道偏移。`span` 是色偏預算。"""

    name = "channel_span_curve"

    def __init__(self, radius: float = 5.0, pieces: int = 64,
                 span: float = 0.25,
                 apply_where: Optional[torch.Tensor] = None,
                 init_jitter: float = 0.0):
        super().__init__(radius=radius, pieces=pieces, bound_mode="advcf",
                         apply_where=apply_where, init_jitter=init_jitter)
        if not 0.0 <= float(span) <= 1.0:
            raise ValueError(f"span 要落在 [0,1]，收到 {span}")
        self.span = float(span)
        self.u: Optional[torch.Tensor] = None
        self.d: Optional[torch.Tensor] = None

    # ---- 介面 ----

    #: 恆等起點放在盒子中央，不放在盒底。
    #:
    #: 曲線的輸出是 `(…)·K/Σθ`，所以 **θ 整體乘一個正的常數不改變曲線**：
    #: `θ_i = a`（任何 `a > 0`）都是恆等。盒底 `u = 0` 也是恆等，但那裡
    #: `min(u, 1−u) = 0`，`d` 的梯度恰為零——`init_jitter = 0` 的設定會讓逐通道
    #: 偏移一步都不動，而且不會報錯，整族靜默退化成一條共用曲線。
    #: 放在中央同樣是恆等，而且 `room = 0.5`，兩組參數都收得到梯度。
    IDENTITY_U = 0.5

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        """恆等起點 `u = IDENTITY_U`、`d = 0`。"""
        k = self.pieces
        u = torch.full((1, 1, k), self.IDENTITY_U,
                       device=x01.device, dtype=x01.dtype)
        d = torch.zeros((1, 3, k), device=x01.device, dtype=x01.dtype)
        if self.init_jitter > 0:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            ru = torch.rand((1, 1, k), generator=generator).to(x01.device, x01.dtype)
            rd = (2 * torch.rand((1, 3, k), generator=generator) - 1
                  ).to(x01.device, x01.dtype)
            u = (1.0 - self.init_jitter) * u + self.init_jitter * ru
            d = (1.0 - self.init_jitter) * d + self.init_jitter * rd
        self.u = u.clone().requires_grad_(True)
        self.d = d.clone().requires_grad_(True)

    @property
    def theta(self) -> torch.Tensor:                     # type: ignore[override]
        lo, hi = self.bounds()
        room = torch.minimum(self.u, 1.0 - self.u)
        t = self.u + self.span * self.d * room
        return lo + (hi - lo) * t

    @theta.setter
    def theta(self, value) -> None:
        # `ColorCurveParam.__init__` 會寫 `self.theta = None`；本族的 θ 是算出來的。
        if value is not None:
            raise AttributeError("θ 由 u 與 d 算出，不能直接指定")

    def params(self) -> List[torch.Tensor]:
        return [self.u, self.d]

    def state_dict(self):
        return {"u": self.u.detach().clone(), "d": self.d.detach().clone()}

    def load_state_dict(self, state):
        self.u = state["u"].detach().clone().requires_grad_(True)
        self.d = state["d"].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def project(self) -> None:
        self.u.clamp_(0.0, 1.0)
        self.d.clamp_(-1.0, 1.0)

    def step_scale(self) -> float:
        """`u` 與 `d` 都已經正規化到 O(1) 的盒子，步長尺度就是 1。"""
        return 1.0

    def channel_gap(self) -> float:
        """三條曲線之間目前的最大距離，回報用。上界是 `span·(hi−lo)`。"""
        with torch.no_grad():
            th = self.theta
            return float((th.max(dim=1).values - th.min(dim=1).values).max())

    def gap_bound(self) -> float:
        lo, hi = self.bounds()
        return self.span * (hi - lo)
