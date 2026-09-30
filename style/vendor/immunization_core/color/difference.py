"""CIEDE2000 色差：skimage 量測路徑與可微求解路徑。

`delta_e00` 以 skimage `deltaE_ciede2000` 計算，為量測與 CSV 使用的正本；
`delta_e00_torch`、`cvar_torch` 保持可微，只供最佳化損失使用。兩者皆以支撐
加權平均縮減，支撐總權重為零時拒絕。

可微公式依 Sharma et al. (2005) 的更正版；給定同一組 Lab 時與 skimage 相差
4.4e-15，端到端差異約 0.003，來自 `color.space.rgb_to_lab` 與 skimage `rgb2lab`
的差異。所有 `sqrt` 先 `clamp_min(eps)`，使純灰像素（C1 = C2 = 0）的梯度有限。
"""
from __future__ import annotations

import math

import torch

from immunization_core.color.space import rgb_to_lab
from immunization_core.metrics.suite import _delta_e00


def delta_e00(a01: torch.Tensor, b01: torch.Tensor,
              support: torch.Tensor = None) -> float:
    """兩張 [0,1] RGB 影像的平均 CIEDE2000；給了 `support` 就取支撐加權平均。

    色差公式為 skimage `deltaE_ciede2000`，與 `metrics.suite._delta_e00` 相同，
    本函式只改變縮減方式。支撐加權平均只計入載體作用的像素，使支撐面積不同的
    載體可在同一 ΔE00 上比較；全圖平均另由 CSV 的 `final_deltaE00` 報告。
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


def cvar_e00(a01: torch.Tensor, b01: torch.Tensor,
             support: torch.Tensor = None, q: float = 0.95) -> float:
    """最差 `1−q` 比例像素的支撐加權平均色差；量測路徑，以 skimage 計算。

    逐像素色差與 `delta_e00` 相同，只改變縮減方式；可微版本為 `cvar_torch`。
    """
    import numpy as np
    from skimage.color import deltaE_ciede2000, rgb2lab

    x = a01.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    y = b01.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    d = deltaE_ciede2000(rgb2lab(x), rgb2lab(y)).ravel()
    if support is None:
        w = np.ones_like(d)
    else:
        w = support.detach().cpu().float()[:, 0].numpy().ravel()
    total = float(w.sum())
    if total <= 0:
        raise ValueError('support 的總權重為零，尾端色差沒有定義')
    order = np.argsort(d)
    ds, ws = d[order], w[order]
    keep = float((1.0 - q) * total)
    cum = np.cumsum(ws[::-1])
    idx = int(np.searchsorted(cum, keep))
    idx = min(idx, len(ds) - 1)
    take_d = ds[::-1][:idx + 1]
    take_w = ws[::-1][:idx + 1].copy()
    over = float(take_w.sum()) - keep
    if over > 0:
        take_w[-1] -= over
    return float((take_d * take_w).sum() / keep)


EPS = 1e-12
DEG = math.pi / 180.0


def _atan2_positive(y, x):
    h = torch.atan2(y, x)
    return torch.where(h < 0, h + 2 * math.pi, h)


def ciede2000(lab1: torch.Tensor, lab2: torch.Tensor,
              kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> torch.Tensor:
    """兩張 (B,3,H,W) 的 Lab 影像，回傳 (B,H,W) 的逐像素色差。"""
    L1, a1, b1 = lab1.unbind(1)
    L2, a2, b2 = lab2.unbind(1)
    C1 = (a1 * a1 + b1 * b1).clamp_min(EPS).sqrt()
    C2 = (a2 * a2 + b2 * b2).clamp_min(EPS).sqrt()
    Cbar = 0.5 * (C1 + C2)
    c7 = Cbar.pow(7)
    G = 0.5 * (1 - (c7 / (c7 + 25.0 ** 7)).clamp_min(EPS).sqrt())
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = (a1p * a1p + b1 * b1).clamp_min(EPS).sqrt()
    C2p = (a2p * a2p + b2 * b2).clamp_min(EPS).sqrt()
    h1p = _atan2_positive(b1, a1p)
    h2p = _atan2_positive(b2, a2p)

    prod = C1p * C2p
    zero = prod <= EPS
    dh = h2p - h1p
    dh = torch.where(dh > math.pi, dh - 2 * math.pi, dh)
    dh = torch.where(dh < -math.pi, dh + 2 * math.pi, dh)
    dh = torch.where(zero, torch.zeros_like(dh), dh)

    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2 * prod.clamp_min(EPS).sqrt() * torch.sin(0.5 * dh)

    Lbar = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = (h1p - h2p).abs()
    hbar = 0.5 * hsum
    hbar = torch.where((hdiff > math.pi) & (hsum < 2 * math.pi),
                       hbar + math.pi, hbar)
    hbar = torch.where((hdiff > math.pi) & (hsum >= 2 * math.pi),
                       hbar - math.pi, hbar)
    hbar = torch.where(zero, hsum, hbar)

    T = (1
         - 0.17 * torch.cos(hbar - 30 * DEG)
         + 0.24 * torch.cos(2 * hbar)
         + 0.32 * torch.cos(3 * hbar + 6 * DEG)
         - 0.20 * torch.cos(4 * hbar - 63 * DEG))
    dtheta = 30 * DEG * torch.exp(-(((hbar - 275 * DEG) / (25 * DEG)) ** 2))
    cp7 = Cbarp.pow(7)
    Rc = 2 * (cp7 / (cp7 + 25.0 ** 7)).clamp_min(EPS).sqrt()
    dL2 = (Lbar - 50) ** 2
    Sl = 1 + 0.015 * dL2 / (20 + dL2).clamp_min(EPS).sqrt()
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -torch.sin(2 * dtheta) * Rc

    tl = dLp / (kL * Sl)
    tc = dCp / (kC * Sc)
    th = dHp / (kH * Sh)
    return (tl * tl + tc * tc + th * th + Rt * tc * th).clamp_min(EPS).sqrt()


def delta_map(a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
    """逐像素色差圖 (B,H,W)。平均與尾端兩種縮減共用這一份。"""
    return ciede2000(rgb_to_lab(a01.clamp(0, 1)), rgb_to_lab(b01.clamp(0, 1)))


def delta_e00_torch(a01: torch.Tensor, b01: torch.Tensor,
                    support: torch.Tensor = None) -> torch.Tensor:
    """兩張 [0,1] RGB 影像的支撐加權平均色差，保持可微。

    縮減方式與 `delta_e00` 相同（支撐加權平均），結果保留在計算圖中。
    """
    d = delta_map(a01, b01)
    if support is None:
        return d.mean()
    w = support.to(device=d.device, dtype=d.dtype)[:, 0]
    total = w.sum()
    if float(total) <= 0:
        raise ValueError('support 的總權重為零，支撐加權的色差沒有定義')
    return (d * w).sum() / total


def cvar_from_map(d: torch.Tensor, support: torch.Tensor = None,
                  q: float = 0.95, eta_iters: int = 30) -> torch.Tensor:
    """最差 `1−q` 比例像素的（支撐加權）平均色差，即 CVaR。

    分位數限制尾端面積，CVaR 另計入尾端嚴重程度：96% 的像素 ΔE00 = 10、4% 為
    100 時，平均為 13.6、p95 為 10。CVaR 的梯度作用於整個尾端。

    用 Rockafellar–Uryasev 的變分形式

        CVaR_q(d; w) = min_η [ η + Σ_p w_p·[d_p − η]_+ / ((1−q)·Σ_p w_p) ]

    內層的 `η` 是加權的 `q` 分位數：二分到「權重高於 η 的部分」等於 `(1−q)·Σw`。
    它求出來之後 `detach`——那是一個純量的極小點，包絡定理下外層對參數的梯度
    不經過它。權重是羽化遮罩時這個形式仍然正確；**不可以先把 `d·w`
    當成色差圖再取整圖 topk**，那會把羽化的邊緣算成低色差而改變尾端的意義。
    """
    if not 0.0 < q < 1.0:
        raise ValueError('q 必須落在 (0,1)')
    flat = d.flatten()
    if support is None:
        w = torch.ones_like(flat)
    else:
        w = support.to(device=d.device, dtype=d.dtype)[:, 0].flatten()
    total = w.sum()
    if float(total) <= 0:
        raise ValueError('support 的總權重為零，尾端色差沒有定義')
    scale = (1.0 - q) * total
    with torch.no_grad():
        lo, hi = float(flat.min()), float(flat.max())
        target = float(scale)
        for _ in range(eta_iters):
            mid = 0.5 * (lo + hi)
            above = float((w * (flat > mid).to(w.dtype)).sum())
            if above > target:
                lo = mid
            else:
                hi = mid
        eta = flat.new_tensor(0.5 * (lo + hi))
    return eta + (w * (flat - eta).clamp_min(0)).sum() / scale


def cvar_torch(a01: torch.Tensor, b01: torch.Tensor,
               support: torch.Tensor = None, q: float = 0.95) -> torch.Tensor:
    return cvar_from_map(delta_map(a01, b01), support, q)
