"""直接參數化 Lab 位移場本身，逐通道獨立。

與仿射場的差別
────────────────────────────────────────────────────────────────────
`color_field.ColorFieldParam` 參數化的是**矩陣**場：`y = T(x−μ) + μ_t + t`，
逐像素位移是 `d(p) = (T−I)(c(p)−μ) + (μ_t−μ) + t`，梯度 `∇d = (T−I)∇c`。
常數的是矩陣，不是位移——位移沿原圖的皺紋、陰影與顏色邊界而變，把矩陣場模糊
掉也消不掉這一項。實測 `global` 結構（每段空間常數）的位移場空間 std 仍有
19.47，而純全域色偏只有 0.33。

這裡的 `delta` 直接就是位移，`d(p)` 不含 `c(p)`，所以位移場的平滑度等於
參數場的平滑度，**均勻性是參數化的性質，不是最佳化要對抗的約束**。

為什麼逐通道
────────────────────────────────────────────────────────────────────
三個通道的不自然來源不同：`L` 的位移讀起來是說不出理由的明暗，人眼最敏感；
`a`／`b` 的位移讀起來是色偏或白平衡跑掉，寬容得多。`tv_L`／`tv_a`／`tv_b`
一直只在 CSV 裡回報，從未分開設過上限。三個通道各一道 CVaR 上限，預算因此
可以全部推到人眼寬容的那兩軸上。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .lowfreq_color import highfreq_report, soft_gamut
from .ncf_param import lab_to_rgb, rgb_to_lab

CHANNELS = ('L', 'a', 'b')


class LabOffsetFieldParam:
    """低頻的 Lab 位移場。介面與 `CompositeParam` 對各段的要求一致。"""

    name = 'lab_offset_field'

    def __init__(self, x01, *, grid: int = 8, support: Optional[torch.Tensor] = None,
                 amplitude: float = 1.0, box: Tuple[float, float, float] = (40.0, 80.0, 80.0)):
        if grid < 1:
            raise ValueError('控制網格至少要 1×1')
        self.grid = int(grid)
        self.amplitude = float(amplitude)
        self.box = tuple(float(v) for v in box)
        self.size = tuple(int(v) for v in x01.shape[-2:])
        self.support = (torch.ones_like(x01[:, :1]) if support is None
                        else support.to(device=x01.device, dtype=x01.dtype))
        self.delta = torch.zeros(1, 3, self.grid, self.grid,
                                 dtype=torch.float32, device=x01.device,
                                 requires_grad=True)

    @property
    def radius(self):
        return max(self.box)

    @property
    def stages(self):
        """單段載體也要有 `stages`，`immunise.fit_caps` 的退路是照段取幅度的。"""
        return [self]

    def set_amplitude(self, a):
        """純量或長度 1 的序列；後者是 `fit_caps` 逐段指定幅度的形狀。"""
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        self.amplitude = float(a)

    def reset(self, x01, seed: int = 0):
        self.size = tuple(int(v) for v in x01.shape[-2:])
        with torch.no_grad():
            self.delta.zero_()

    def params(self):
        return [self.delta]

    def project(self):
        with torch.no_grad():
            for k, b in enumerate(self.box):
                self.delta[:, k].clamp_(-b, b)

    def offset(self, size=None) -> torch.Tensor:
        size = tuple(size or self.size)
        if self.grid == 1:
            field = self.delta.expand(1, 3, *size)
        else:
            field = F.interpolate(self.delta, size=size, mode='bicubic',
                                  align_corners=False)
        return self.amplitude * field

    def channel_magnitude(self, k: int, size=None) -> torch.Tensor:
        return self.offset(size)[:, k:k + 1].abs()

    def raw_rgb(self, x):
        lab = rgb_to_lab(x).double()
        out = lab + self.offset(tuple(x.shape[-2:])).double()
        return lab_to_rgb(out).to(x.dtype)

    def render(self, x):
        w = self.support.to(device=x.device, dtype=x.dtype)
        out = w * soft_gamut(self.raw_rgb(x)) + (1 - w) * x
        return torch.where(w > 0, out, x)

    def state_dict(self):
        return {'delta': self.delta.detach().clone()}

    def load_state_dict(self, state):
        self.delta = state['delta'].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def diagnostics(self, x) -> Dict[str, float]:
        size = tuple(int(v) for v in x.shape[-2:])
        off = self.offset(size)
        raw = self.raw_rgb(x)
        row = {'name': self.name, 'grid': self.grid,
               'amplitude': self.amplitude,
               'clipping_fraction': float(((raw < 0) | (raw > 1)).double().mean()),
               'clipping_max': float((raw - raw.clamp(0, 1)).abs().max())}
        for k, tag in enumerate(CHANNELS):
            row[f'offset_{tag}_max'] = float(off[:, k].abs().max())
            row[f'offset_{tag}_mean'] = float(off[:, k].abs().mean())
            row[f'offset_{tag}_std'] = float(off[:, k].std())
        row.update(highfreq_report(x, self.render(x)))
        return row
