"""結構化的載體搜尋空間：一組旋鈕索引一整族顏色場。

為什麼旋鈕是結構而不是場本身
────────────────────────────────────────────────────────────────────
`ColorFieldParam` 在 `grid = 16` 時有上千個可學參數，無導數搜尋在 200 次評估
內碰不動它們。所以搜尋的是**結構**：控制點密度、帶寬、場的振幅、亮度鎖不鎖、
兩段各自的幅度，外加一個抽場用的種子。給定這組旋鈕，場本身由固定的亂數產生
器決定，因此整個映射對旋鈕是確定的、可重現的。

`grid = 1`、`sigma` 大、`scale = 0` 時退化成 Monge–Kantorovitch 的全域仿射，
也就是 `runs/color_scaleup_search/` 那一批的作法。搜尋因此**從現行方法出發**，
往外找。
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from .color_field import ColorFieldParam
from .color_search import Knob
from .composite import CompositeParam

GRID_CHOICES = (1, 2, 4, 8, 16, 32)

KNOBS = (
    Knob('frame_grid', 0.0, float(len(GRID_CHOICES) - 1)),
    Knob('frame_sigma', 0.0, 32.0),
    Knob('frame_scale', 0.0, 1.5),
    Knob('frame_amplitude', 0.05, 1.0),
    Knob('clothes_grid', 0.0, float(len(GRID_CHOICES) - 1)),
    Knob('clothes_sigma', 0.0, 32.0),
    Knob('clothes_scale', 0.0, 1.5),
    Knob('clothes_amplitude', 0.05, 1.0),
    Knob('lock_luminance', 0.0, 1.0),
    Knob('field_seed', 0.0, 997.0),
)

START = {'frame_grid': 0.0, 'frame_sigma': 24.0, 'frame_scale': 0.0,
         'frame_amplitude': 0.5, 'clothes_grid': 0.0, 'clothes_sigma': 24.0,
         'clothes_scale': 0.0, 'clothes_amplitude': 0.5,
         'lock_luminance': 1.0, 'field_seed': 0.0}


def grid_of(v: float) -> int:
    return GRID_CHOICES[min(len(GRID_CHOICES) - 1, max(0, int(round(float(v)))))]


def describe(values: Dict[str, float]) -> Dict[str, float]:
    """把旋鈕值翻成要寫進 CSV 的欄位，與實際建出來的載體一致。"""
    return {
        'knob_frame_grid': grid_of(values['frame_grid']),
        'knob_frame_sigma': round(float(values['frame_sigma']), 3),
        'knob_frame_scale': round(float(values['frame_scale']), 4),
        'knob_frame_amplitude': round(float(values['frame_amplitude']), 4),
        'knob_clothes_grid': grid_of(values['clothes_grid']),
        'knob_clothes_sigma': round(float(values['clothes_sigma']), 3),
        'knob_clothes_scale': round(float(values['clothes_scale']), 4),
        'knob_clothes_amplitude': round(float(values['clothes_amplitude']), 4),
        'knob_lock_luminance': int(round(float(values['lock_luminance']))),
        'knob_field_seed': int(round(float(values['field_seed']))),
    }


def _fill_field(param, scale: float, seed: int, offset: int) -> None:
    if scale <= 0:
        return
    gen = torch.Generator().manual_seed(int(seed) * 7919 + offset)
    noise = torch.randn(param.delta.shape, generator=gen,
                        dtype=param.delta.dtype)
    with torch.no_grad():
        param.delta.copy_(noise.to(param.delta.device) * float(scale))
    param.project()


def build_carrier(values: Dict[str, float], x01, *, frame_support,
                  clothes_support, frame_palette, clothes_palette,
                  radius=None, max_gain=None) -> CompositeParam:
    """依旋鈕值建出串接的載體，並把場填好。

    `radius` 與 `max_gain` 預設是 None，也就是**不設界**：L∞ 盒與奇異值上界
    都是可選的約束，不是這一族的定義。
    """
    lock = bool(round(float(values['lock_luminance'])))
    seed = int(round(float(values['field_seed'])))
    frame = ColorFieldParam(frame_palette[0], frame_palette[1],
                            support=frame_support,
                            grid=grid_of(values['frame_grid']),
                            sigma=float(values['frame_sigma']),
                            lock_luminance=lock, radius=radius,
                            max_gain=max_gain,
                            amplitude=float(values['frame_amplitude']))
    clothes = ColorFieldParam(clothes_palette[0], clothes_palette[1],
                              support=clothes_support,
                              grid=grid_of(values['clothes_grid']),
                              sigma=float(values['clothes_sigma']),
                              lock_luminance=lock, radius=radius,
                              max_gain=max_gain,
                              amplitude=float(values['clothes_amplitude']))
    carrier = CompositeParam([frame, clothes], tags=['frame', 'clothes'])
    carrier.reset(x01, seed)
    _fill_field(frame, float(values['frame_scale']), seed, 0)
    _fill_field(clothes, float(values['clothes_scale']), seed, 1)
    frame.set_amplitude(float(values['frame_amplitude']))
    clothes.set_amplitude(float(values['clothes_amplitude']))
    return carrier


def knob_list() -> List[Knob]:
    return list(KNOBS)


def spearman(a, b) -> float:
    """秩相關。長度不足或其中一側全部同分時回傳 nan，不回傳 0。"""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and x == x and y == y]
    n = len(pairs)
    if n < 3:
        return float('nan')

    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out

    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((r - ma) ** 2 for r in ra)
    vb = sum((r - mb) ** 2 for r in rb)
    if va <= 0 or vb <= 0:
        return float('nan')
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(va * vb)
