"""可微的 CIEDE2000。**只給求解用，量測一律走 skimage 那一份。**

為什麼要有第二份實作
────────────────────────────────────────────────────────────────────
`color_amplitude.delta_e00` 走 skimage，必須 `detach()` 成 NumPy，所以色差只能
在最佳化**結束後**以整體幅度二分投影回上限。那等於解一個無約束問題再把解縮小，
不是解受約束的問題：`runs/immunise_both_budgets/` 的 30 列裡 28 列貼住整圖上限、
0 列貼住羽化臉上限，而選中的解身分框色差只有 8.4 到 14.3——預算沒有用完，
因為整體縮放只能收縮，不能把預算從整圖搬到臉上。

這一份把色差放進損失，讓最佳化在可行域附近工作。**它不取代量測**：
`tests/test_delta_e_torch.py` 逐點對照 skimage，交付前的檢查與 CSV 的 ΔE00
仍然由 skimage 那一份產生，兩者的差異由測試釘住。

數值
────────────────────────────────────────────────────────────────────
公式照 Sharma et al. (2005) 的更正版。**給同一組 Lab 時與 skimage 的
`deltaE_ciede2000` 差 4.4e-15**（測試釘住）。端到端還有約 0.003 的差，全部來自
專案的 `ncf_param.rgb_to_lab` 與 skimage 的 `rgb2lab` 相差最多 0.005 個 Lab
單位；載體本身用的就是專案那一份，所以求解端沿用它，差異寫成測試的容差。

所有 `sqrt` 都先 `clamp_min(eps)`：`sqrt` 在 0 的導數是無限大，而
`C1 = C2 = 0`（純灰）在人像上到處都是。
"""
from __future__ import annotations

import math

import torch

from .ncf_param import rgb_to_lab

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

    縮減方式與 `color_amplitude.delta_e00` 相同（支撐加權平均），
    差別只在這一份不離開計算圖。
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

    為什麼是 CVaR 而不是分位數
    ────────────────────────────────────────────────────────────────
    分位數與「超標像素比例」限制的是尾端的**面積**，不是尾端的**嚴重程度**：
    96% 的像素 ΔE00 = 10、4% 的像素 ΔE00 = 100，平均只有 13.6、p95 只有 10，
    極端色塊照樣藏得住。CVaR 取的是最差那一段的平均，嚴重程度直接進到數值裡，
    而且梯度作用於整個尾端，不像分位數只有少數位置有梯度。

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
