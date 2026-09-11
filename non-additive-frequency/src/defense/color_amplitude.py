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


def delta_e00(a01: torch.Tensor, b01: torch.Tensor,
              support: torch.Tensor = None) -> float:
    """兩張 [0,1] RGB 影像的平均 CIEDE2000；給了 `support` 就取支撐加權平均。

    薄封裝：色差公式只有 skimage 的 `deltaE_ciede2000` 這一份，與
    `src/metrics/suite.py::_delta_e00` 同一個實作，這裡只換縮減方式，
    避免「量測用的色差」與「求解用的色差」悄悄變成兩個東西。

    **為什麼要支撐加權。** 全圖平均會被支撐面積稀釋：衣物載體只佔一兩成像素，
    把那塊完全換色，全圖平均的 ΔE00 也只有 4.5–5.2，於是任何 6 以上的目標
    都不可達，而整圖濾鏡在同一個數字上輕鬆到 24。兩者的等失真錨點因此對不起來。
    支撐加權問的是「載體真正作用的地方顏色走了多遠」，跨載體可比。
    全圖的那個數字仍然照報（CSV 的 `final_deltaE00`），只是不當錨點。
    """
    if support is None:
        return _delta_e00(a01.detach(), b01.detach())

    from skimage.color import deltaE_ciede2000, rgb2lab

    x = a01.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    y = b01.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    w = support.detach().cpu().float()[:, 0].numpy()
    total = float(w.sum())
    if total <= 0:
        raise ValueError('support 的總權重為零，支撐加權的色差沒有定義')
    return float((deltaE_ciede2000(rgb2lab(x), rgb2lab(y)) * w).sum() / total)


@torch.no_grad()
def solve_rotation(param, x01: torch.Tensor, *, lo: float = 0.0,
                   hi: float = 180.0, iters: int = 8,
                   limit: float = 1.0) -> dict:
    """在「高頻殘差比不超過 `limit`」的前提下，解出最大的色相旋轉角。

    為什麼要逐圖解而不是定一個角度
    ────────────────────────────────────────────────────────────────
    Lab 色度平面上的等距**不保證** RGB 的高通殘差不上升：Lab→RGB 是非線性的，
    同一個色度梯度轉到不同色相之後映進 RGB 的梯度可以變大，而**變多少隨影像
    內容而異**。實測在一張人像上 0–135 度都落在 0.966–0.980、180 度才到
    1.062，但在均勻隨機圖上 30 度就 1.079、90 度峰值 1.338。
    固定 90 度因此在某些影像上會違反約束（一批 45 格裡有 9 格超過 1，
    最大 1.0597），在另一些影像上又白白留著射程沒用。

    角度對高頻比**不是單調的**（隨機圖上 90 度是峰值、180 度反而回落），所以
    這裡走**粗掃再細分**：先在等距的格點上找最後一個仍然合格的角度，再在它與
    下一個格點之間二分。回傳的 `monotone` 記下粗掃時比值有沒有單調上升——
    沒有的話，解到的是「第一段合格區間的右端」，不是全域最大的合格角度，
    那是刻意的保守選擇，而且被記下來而不是藏起來。
    """
    from src.defense.lowfreq_color import highfreq_report

    if not getattr(param, 'isometric', False):
        raise ValueError('solve_rotation 只在等距臂上有定義：非等距臂會再過一次'
                         '奇異值上界，解出來的角度不會逐字生效')
    ratios, angles = [], []
    for k in range(iters + 1):
        a = lo + (hi - lo) * k / iters
        param.rotation_deg = a
        param.reset(x01, 0)
        ratios.append(highfreq_report(x01, param.render(x01))['hf_ratio_rgb_total'])
        angles.append(a)
    monotone = all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:]))
    ok = [k for k, r in enumerate(ratios) if r <= limit]
    if not ok:
        param.rotation_deg = angles[0]
        param.reset(x01, 0)
        return {'rotation_deg': angles[0], 'hf_ratio': ratios[0],
                'reached_limit': False, 'monotone': monotone}
    # 第一段合格區間的右端：從頭往後找到第一個不合格的格點。
    last = 0
    for k in range(len(ratios)):
        if ratios[k] > limit:
            break
        last = k
    a_lo = angles[last]
    a_hi = angles[last + 1] if last + 1 < len(angles) else angles[last]
    for _ in range(12):
        mid = .5 * (a_lo + a_hi)
        param.rotation_deg = mid
        param.reset(x01, 0)
        r = highfreq_report(x01, param.render(x01))['hf_ratio_rgb_total']
        if r <= limit:
            a_lo = mid
        else:
            a_hi = mid
    param.rotation_deg = a_lo
    param.reset(x01, 0)
    got = highfreq_report(x01, param.render(x01))['hf_ratio_rgb_total']
    return {'rotation_deg': round(a_lo, 3), 'hf_ratio': round(got, 5),
            'reached_limit': True, 'monotone': monotone}


@torch.no_grad()
def solve_amplitude(param, x01: torch.Tensor, target_delta_e: float, *,
                    support: torch.Tensor = None,
                    lo: float = 0.0, hi: float = 1.0,
                    tol: float = 1e-3, iters: int = 32) -> dict:
    """二分幅度，讓 `param.render(x01)` 的 ΔE00 落在 `target_delta_e`。

    `support` 給了就用支撐加權的色差當錨點（理由見 `delta_e00`）；沒給就用
    全圖平均。回傳 `{'amplitude', 'delta_e00', 'reached'}`。求解結束後 `param`
    留在解出的幅度上，呼叫端不需要再設一次。
    """
    if not (target_delta_e > 0):
        raise ValueError('target_delta_e 必須為正')
    param.set_amplitude(hi)
    top = delta_e00(x01, param.render(x01), support)
    if top <= target_delta_e:
        return {'amplitude': float(hi), 'delta_e00': float(top), 'reached': False}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        param.set_amplitude(mid)
        got = delta_e00(x01, param.render(x01), support)
        if got < target_delta_e:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    param.set_amplitude(hi)
    return {'amplitude': float(hi),
            'delta_e00': float(delta_e00(x01, param.render(x01), support)),
            'reached': True}
