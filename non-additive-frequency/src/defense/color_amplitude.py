"""把顏色載體的幅度對齊到一個指定的 CIEDE2000。

跨臂比較一律要先對齊失真，否則只是在比誰付得多。幅度（`amplitude`）是空間
常數，ΔE00 在 `amplitude = 0` 時為 0 且隨它單調上升，所以二分法取得到。

**不可達時回報，不靜默。** 目標超過 `amplitude = 1` 能到的值時回傳
`reached = False` 與該臂實際到得了的 ΔE00；呼叫端要把這兩欄照報。

一個已知的偏移：`gamut='soft'` 的路徑上，`amplitude = 0` 不是逐位元恆等，
`soft_gamut` 在色域內也偏離恆等（knee = 0.06 時黑點被抬 0.0416）。因此那條
路徑上求出來的 ΔE00 含一份與顏色無關的色調壓縮，`tests/test_lowfreq_amplitude.py`
把它與載體的效果釘開。要一個乾淨的零點就用 `gamut='clip'` 或 `'scale'`。
"""
from __future__ import annotations

import torch

from src.metrics.suite import _delta_e00


def delta_e00(a01: torch.Tensor, b01: torch.Tensor) -> float:
    """兩張 [0,1] RGB 影像的平均 CIEDE2000。

    薄封裝：色差只有 `src/metrics/suite.py` 那一份實作（走 skimage 的
    `deltaE_ciede2000`），這裡不另寫，避免「量測用的色差」與「求解用的色差」
    悄悄變成兩個東西。
    """
    return _delta_e00(a01.detach(), b01.detach())


@torch.no_grad()
def solve_amplitude(param, x01: torch.Tensor, target_delta_e: float, *,
                    lo: float = 0.0, hi: float = 1.0,
                    tol: float = 1e-3, iters: int = 32) -> dict:
    """二分幅度，讓 `param.render(x01)` 的 ΔE00 落在 `target_delta_e`。

    回傳 `{'amplitude', 'delta_e00', 'reached'}`。求解結束後 `param` 留在解出
    的幅度上，呼叫端不需要再設一次。
    """
    if not (target_delta_e > 0):
        raise ValueError('target_delta_e 必須為正')
    param.set_amplitude(hi)
    top = delta_e00(x01, param.render(x01))
    if top <= target_delta_e:
        return {'amplitude': float(hi), 'delta_e00': float(top), 'reached': False}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        param.set_amplitude(mid)
        got = delta_e00(x01, param.render(x01))
        if got < target_delta_e:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    param.set_amplitude(hi)
    return {'amplitude': float(hi),
            'delta_e00': float(delta_e00(x01, param.render(x01))),
            'reached': True}
