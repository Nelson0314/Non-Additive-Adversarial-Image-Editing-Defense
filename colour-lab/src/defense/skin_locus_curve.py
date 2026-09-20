"""在色彩空間裡把膚色那一帶保護起來，其餘的顏色自由。

要打破的是什麼
────────────────────────────────────────────────────────────────────
ΔE00 是逐像素平均，人像上膚色佔多數，所以**沒有結構的**全域映射一旦平均色差
做大，膚色一定被移走——現行操作點的紅臉、上限抬高之後的綠臉都是這樣來的。
但那個結論只對沒有結構的映射成立。映射如果在**色彩空間裡**是局部化的，
全圖振幅與臉的振幅就分得開：

    T(c) = c + w(c) · (F(c) − c)

`F` 是 AdvCF 那條 64 段單調分段線性曲線，`w(c) ∈ [0,1]` 是保護權重的補數，
只吃顏色、不吃座標。量過一次可行性：`man_00` 上全圖 ΔE00 仍有 17.71
（現行操作點 15.9），而中央框的 ΔE00 從 48.14 掉到 14.81。

膚色那一帶怎麼定
────────────────────────────────────────────────────────────────────
**固定的色相帶，不看影像**。CIELab 的色相角 `h = atan2(b*, a*)`，膚色不分
人種都落在橘紅那一帶（`HUE_CENTRE` 附近），差別主要在明度與彩度而不在色相。
保護度是兩個 smoothstep 的乘積：

- `hue`：`|h − HUE_CENTRE|` 在 `HUE_INNER` 以內全保護，到 `HUE_OUTER` 降到 0。
- `chroma`：彩度低於 `CHROMA_LOW` 的顏色色相沒有意義（近中性），不保護；
  在 `[CHROMA_LOW, CHROMA_HIGH]` 之間平滑上升。彩度趨近 0 時保護度也趨近 0，
  `atan2` 的奇點因此被這個因子壓掉，梯度不會爆。

三個常數都是設定檔給的結構常數，**求解端動不了**。`protect_scale = 0` 時
本族逐位元退回 AdvCF。

由構造保證的性質
────────────────────────────────────────────────────────────────────
1. **不需要鉗回**：`T` 是 `c` 與 `F(c)` 的凸組合，兩者都在 [0,1]。
2. **仍是純顏色全域映射**：`w` 與 `F` 都只是 `c` 的函數，同一個 RGB 三元組
   在整張圖得到同一個輸出。有測試把像素打亂後比對釘住這一點。
3. **膚色帶內幾乎不動**：帶心的顏色 `w = 0`，逐位元保留。
4. **低頻**：`w` 是色彩空間裡的 smoothstep，梯度有界，不會把空間上平滑的
   邊界放大成硬邊。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from src.defense.color_param import ColorCurveParam
from src.defense.ncf_param import rgb_to_lab

#: 膚色在 CIELab 上的色相角（度）。橘紅那一帶。
HUE_CENTRE = 50.0
#: 帶心的半寬：這以內全保護。
HUE_INNER = 25.0
#: 到這裡保護度降到 0。
HUE_OUTER = 55.0
#: 彩度低於這個值視為近中性，色相沒有意義，不保護。
CHROMA_LOW = 3.0
#: 彩度高於這個值，保護度的彩度因子達到 1。
CHROMA_HIGH = 12.0


def smoothstep(u: torch.Tensor) -> torch.Tensor:
    """`u` 夾到 [0,1] 之後的 3u²−2u³。兩端一階導數為零，所以不會出硬邊。"""
    u = u.clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def protection(x01: torch.Tensor, *, hue_centre: float = HUE_CENTRE,
               hue_inner: float = HUE_INNER, hue_outer: float = HUE_OUTER,
               chroma_low: float = CHROMA_LOW,
               chroma_high: float = CHROMA_HIGH) -> torch.Tensor:
    """(N,3,H,W) → (N,1,H,W)，1 = 完全保護（不動），0 = 完全自由。"""
    if not hue_outer > hue_inner >= 0.0:
        raise ValueError(f"要 hue_outer > hue_inner ≥ 0，收到 {hue_outer}、{hue_inner}")
    if not chroma_high > chroma_low >= 0.0:
        raise ValueError(
            f"要 chroma_high > chroma_low ≥ 0，收到 {chroma_high}、{chroma_low}")
    lab = rgb_to_lab(x01)
    a = lab[:, 1:2]
    b = lab[:, 2:3]
    chroma = (a * a + b * b).clamp_min(1e-12).sqrt()
    hue = torch.atan2(b, a) * (180.0 / math.pi)
    delta = (hue - hue_centre).abs()
    delta = torch.minimum(delta, 360.0 - delta)          # 角度是週期的
    near = 1.0 - smoothstep((delta - hue_inner) / (hue_outer - hue_inner))
    strong = smoothstep((chroma - chroma_low) / (chroma_high - chroma_low))
    return near * strong


class SkinLocusCurveParam(ColorCurveParam):
    """AdvCF 的曲線，套上色彩空間裡的膚色保護。"""

    name = "skin_locus_curve"

    def __init__(self, radius: float = 5.0, pieces: int = 64,
                 protect_scale: float = 1.0,
                 hue_centre: float = HUE_CENTRE,
                 hue_inner: float = HUE_INNER,
                 hue_outer: float = HUE_OUTER,
                 chroma_low: float = CHROMA_LOW,
                 chroma_high: float = CHROMA_HIGH,
                 bound_mode: str = "advcf",
                 apply_where: Optional[torch.Tensor] = None,
                 init_jitter: float = 0.0):
        super().__init__(radius=radius, pieces=pieces, bound_mode=bound_mode,
                         apply_where=apply_where, init_jitter=init_jitter)
        if not 0.0 <= float(protect_scale) <= 1.0:
            raise ValueError(f"protect_scale 要落在 [0,1]，收到 {protect_scale}")
        self.protect_scale = float(protect_scale)
        self.hue_centre = float(hue_centre)
        self.hue_inner = float(hue_inner)
        self.hue_outer = float(hue_outer)
        self.chroma_low = float(chroma_low)
        self.chroma_high = float(chroma_high)

    def weight(self, x01: torch.Tensor) -> torch.Tensor:
        """`1 − protect_scale · protection(c)`：可以動的比例。"""
        p = protection(x01, hue_centre=self.hue_centre,
                       hue_inner=self.hue_inner, hue_outer=self.hue_outer,
                       chroma_low=self.chroma_low, chroma_high=self.chroma_high)
        return 1.0 - self.protect_scale * p

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        base = self._curve(x01, self.theta)
        w = self.weight(x01)
        out = x01 + w * (base - x01)
        if self.apply_where is None:
            return out
        m = self.apply_where.to(device=out.device, dtype=out.dtype)
        return m * out + (1.0 - m) * x01

    def params(self) -> List[torch.Tensor]:
        return [self.theta]
