"""支撐區裡的東西**看起來像不像雜訊**。

為什麼不是用美學分數
────────────────────────────────────────────────────────────────────
NIMA-AVA 與 CNNIQA 在本專案的判別上實測不可用。校準用的是兩組已知答案的
產物——滿版彩色雜訊對分散圓點，拉圖判讀差很多：

    nima_drop    前者較差 7/10（中位 +0.153 對 +0.022，兩組分佈重疊）
    cnniqa_drop  前者較差 4/10，**比擲硬幣還差**；而且兩組的 drop 都是負的，
                 也就是它認為加了雜訊的圖「品質比原圖好」
    clip_sim     前者較低 10/10，但那量的是「跟原圖差多少」，與 DISTS 同義

原因是那兩個模型學的是「這張照片拍得好不好」（構圖、曝光、對焦），沒有理由
把「紙屑」與「印花」分開。

本模組改問一個可以直接量的問題：**這塊區域的統計像自然影像，還是像白噪聲。**

三個數
────────────────────────────────────────────────────────────────────
    spectral_slope   徑向平均功率譜的 log–log 斜率 α
    block_entropy    支撐內 8×8 區塊亮度直方圖的平均熵（bit）
    colour_count     支撐內量化到 4 bit/通道之後，覆蓋 95% 像素所需的色數

自然影像的功率譜大致隨頻率衰減（`|F(f)|² ∝ f^(−α)`，α 約 2），逐像素白噪聲的
譜是平的（α ≈ 0）。印花布的色數少、雜訊的色數多。三者都在**支撐內**量，並且
在**原圖的同一塊區域**上量一份當基準——單張影像的紋理差很多（毛衣本來就比
西裝雜），不扣掉基準就分不出「防禦讓它變雜」與「這塊布本來就雜」。這與
`identity.py` 的 `id_orig` 是同一條理由。

遮罩與譜洩漏
────────────────────────────────────────────────────────────────────
支撐是不規則的，而 FFT 要矩形。作法是乘上支撐再乘一個 Hann 窗，硬邊界因此
會洩漏能量到高頻。**這個偏差不做修正，而是靠對照消掉**：防禦圖與原圖走完全
相同的遮罩與窗，洩漏項是共同的，比較兩者的斜率差時大致抵銷。故本模組的主讀數
是 `*_def − *_orig` 的差，不是絕對值——絕對值帶著洩漏，不可跨影像比較。
"""

from __future__ import annotations

from typing import Optional

import torch

# BT.601，與 `color_param.LUMA_WEIGHTS` 同一組係數。兩處不一致的話「亮度」
# 在兩個模組裡講的就不是同一件事。
from src.defense.color_param import luma

# 擬合的頻帶。下界避開 DC 與遮罩本身的尺度（那一段由支撐的形狀決定，不是內容），
# 上界避開 Nyquist 附近的混疊。兩者都是本專案指定的，故是 CSV 欄位不是註解。
FIT_LO, FIT_HI = 0.05, 0.45
# 少於這麼多像素就不給值：樣本太少時斜率的擬合誤差比要量的差異還大。
MIN_PIXELS = 4096
# 顏色量化的位元數，以及「覆蓋多少比例的像素」。
COLOUR_BITS = 4
COLOUR_COVERAGE = 0.95


def support_from_pair(x_orig: torch.Tensor, x_def: torch.Tensor,
                      tol: float = 1.0 / 512) -> torch.Tensor:
    """(1,3,H,W) 兩張 → (1,1,H,W) 的 0/1 支撐：**防禦圖與原圖不同的地方**。

    為什麼由影像反推而不是重建載體：重建要再跑一次 ATR 與導引濾波，而那條路徑
    的任何旗標與當初跑的不一致，量到的就是另一塊區域，且不會有症狀。影像之間的
    差則是**交出去的那張圖的事實**。

    `tol` 取 1/512（8 bit 量化階的一半以下），擋掉 PNG 往返的捨入。
    """
    if x_orig.shape != x_def.shape:
        raise ValueError(f"兩張影像形狀不同：{tuple(x_orig.shape)} 與 "
                         f"{tuple(x_def.shape)}")
    return ((x_def - x_orig).abs().amax(dim=1, keepdim=True) > tol).to(x_orig.dtype)


def _hann(h: int, w: int, device, dtype) -> torch.Tensor:
    a = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    b = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return (a[:, None] * b[None, :])[None, None]


def spectral_slope(x01: torch.Tensor, support: torch.Tensor) -> Optional[float]:
    """支撐區裡的徑向平均功率譜斜率 α（`P(f) ∝ f^(−α)`）。

    白噪聲 α ≈ 0，自然影像 α 約 2。回傳的是 **−斜率**，故越大越像自然影像。
    支撐太小時回 `None`。
    """
    if float(support.sum()) < MIN_PIXELS:
        return None
    h, w = x01.shape[-2:]
    y = luma(x01)
    m = support.to(y.dtype)
    # 先扣掉支撐內的平均值再加窗：不扣的話 DC 分量會由區域亮度決定，
    # 而那與「像不像雜訊」無關。
    mean = (y * m).sum() / m.sum().clamp_min(1.0)
    field = (y - mean) * m * _hann(h, w, y.device, y.dtype)

    p = torch.fft.rfft2(field[0, 0]).abs().pow(2)
    fy = torch.fft.fftfreq(h, device=y.device).abs()[:, None]
    fx = torch.fft.rfftfreq(w, device=y.device)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2)

    keep = (r >= FIT_LO) & (r <= FIT_HI)
    if int(keep.sum()) < 32:
        return None
    # 徑向平均：把 [FIT_LO, FIT_HI] 切成等對數寬的桶，每桶取平均功率，
    # 再對 (log f, log P) 做最小平方。直接對所有格點迴歸會被高頻格點數壓過去
    # ——半徑 r 的格點數隨 r 線性增加，那等於給高頻加權。
    lr = torch.log10(r[keep])
    lp = torch.log10(p[keep].clamp_min(1e-20))
    nb = 24
    lo, hi = float(lr.min()), float(lr.max())
    idx = ((lr - lo) / max(hi - lo, 1e-9) * (nb - 1)).round().long().clamp(0, nb - 1)
    xs, ys = [], []
    for b in range(nb):
        sel = idx == b
        if int(sel.sum()) == 0:
            continue
        xs.append(float(lr[sel].mean()))
        ys.append(float(lp[sel].mean()))
    if len(xs) < 6:
        return None
    xt = torch.tensor(xs)
    yt = torch.tensor(ys)
    xm, ym = xt.mean(), yt.mean()
    denom = ((xt - xm) ** 2).sum()
    if float(denom) <= 0.0:
        return None
    slope = float(((xt - xm) * (yt - ym)).sum() / denom)
    return -slope


def block_entropy(x01: torch.Tensor, support: torch.Tensor,
                  block: int = 8, bins: int = 32) -> Optional[float]:
    """支撐內 `block`×`block` 區塊的亮度直方圖熵，取平均（bit）。

    只算**完全落在支撐內**的區塊：跨邊界的區塊一半是原圖，熵會被那一半帶著走。
    """
    if float(support.sum()) < MIN_PIXELS:
        return None
    y = (luma(x01).clamp(0, 1) * (bins - 1)).round().long()[0, 0]
    m = support[0, 0] > 0.5
    h, w = y.shape
    ent, n = 0.0, 0
    for i in range(0, h - block + 1, block):
        for j in range(0, w - block + 1, block):
            if not bool(m[i:i + block, j:j + block].all()):
                continue
            v = y[i:i + block, j:j + block].reshape(-1)
            cnt = torch.bincount(v, minlength=bins).to(torch.float32)
            p = cnt / cnt.sum()
            p = p[p > 0]
            ent += float(-(p * p.log2()).sum())
            n += 1
    return None if n == 0 else ent / n


def colour_count(x01: torch.Tensor, support: torch.Tensor) -> Optional[int]:
    """支撐內量化到 `COLOUR_BITS` 位元／通道之後，覆蓋 95% 像素所需的色數。

    印花布用少數幾個顏色，逐像素噪聲用很多。取「覆蓋 95%」而不是「相異色數」，
    是因為後者會被邊界上少量的過渡色灌爆。
    """
    if float(support.sum()) < MIN_PIXELS:
        return None
    q = (x01.clamp(0, 1) * (2 ** COLOUR_BITS - 1)).round().long()
    m = support[0, 0] > 0.5
    code = (q[0, 0] << (2 * COLOUR_BITS)) | (q[0, 1] << COLOUR_BITS) | q[0, 2]
    v = code[m].reshape(-1)
    cnt = torch.bincount(v).to(torch.float32)
    cnt = cnt[cnt > 0].sort(descending=True).values
    need = COLOUR_COVERAGE * float(cnt.sum())
    return int(torch.searchsorted(cnt.cumsum(0), torch.tensor(need)).item()) + 1


def naturalness_row(x_orig: torch.Tensor, x_def: torch.Tensor) -> dict:
    """兩張影像 → 逐列寫進 CSV 的九個欄位。

    `*_orig` 一律照報：單張影像的紋理差很多（毛衣本來就比西裝雜），不扣掉基準
    就分不出「防禦讓它變雜」與「這塊布本來就雜」。主讀數是 `*_drop`。
    """
    sup = support_from_pair(x_orig, x_def)
    area = float(sup.mean())
    out = {"support_area": round(area, 5)}
    for name, fn in (("slope", spectral_slope), ("entropy", block_entropy)):
        a, b = fn(x_orig, sup), fn(x_def, sup)
        out[f"{name}_orig"] = "" if a is None else round(a, 5)
        out[f"{name}_def"] = "" if b is None else round(b, 5)
        out[f"{name}_drop"] = ("" if a is None or b is None else round(a - b, 5))
    a, b = colour_count(x_orig, sup), colour_count(x_def, sup)
    out["colours_orig"] = "" if a is None else a
    out["colours_def"] = "" if b is None else b
    out["colours_ratio"] = ("" if not a or b is None else round(b / a, 4))
    return out
