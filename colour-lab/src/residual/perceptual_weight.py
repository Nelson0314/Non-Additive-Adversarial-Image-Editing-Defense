"""頻率格的知覺權重：每一格允許多少擾動，而不是能不能動。

存在的理由
────────────────────────────────────────────────────────────────────
`texture_rephase.radial_gate` 是**二值**的帶通遮罩：歸一化半徑落在
`[r_min, r_max]` 內的頻格全部拿到同一個 `theta_max`。人眼不是這樣看的——
對比敏感度在中低頻達到峰值、往高頻單調衰減，故同一個振幅的擾動放在
r = 0.15 與 r = 0.9 上，可見度差一個數量級。二值閘對兩者開同一個價。

實測「每單位 DISTS 換到多少位移」（IP2P 線，13 張服從影像）：

    DCT-Shield          13.2
    加性 delta           6.8
    紋理重相位（低強度）  6.6
    紋理重相位（帶邊緣）  3.7

`RESULTS.md` 已把 DCT-Shield 那 2 倍歸因給「JPEG 量化階的約束」，並註明
**那不是加性本身帶來的**。量化表就是一張以感知門檻定價的價目表，故這裡把
同一個約束搬進本方法的閘。

**這不改參數化。** 仍是頻譜重參數化、非加性，`theta = 0` 的逐位元恆等仍
成立（權重只縮放閘，閘只縮放 theta，theta = 0 時乘什麼都是 0）。
`DECISIONS.md` 的「頻譜加性項不做」明文寫「閘的開度不受此限，兩者都只改變
擾動被允許出現的位置，不改變參數化」。

座標對應
────────────────────────────────────────────────────────────────────
量化表是 8x8 DCT 的，本模組的格點是 `block x (block//2+1)` 的 rfft2。
接法是頻率而不是索引：

    8 點 DCT-II 的基底 u 對應 u/16 cycles/pixel，Nyquist（0.5）在 u = 8。
    `radial_gate` 用的歸一化座標 `fftfreq(block) * 2` 是 [-1, 1)，1 即
    Nyquist，故 cycles/pixel = f_norm / 2，於是 u = 16 * (f_norm / 2)
                                                 = 8 * f_norm。

表的索引只到 7，故 u、v 夾到 7 之後雙線性內插。夾取只影響最靠近 Nyquist
的那一圈，而那一圈的量化階本來就已經飽和。

出處
────────────────────────────────────────────────────────────────────
ITU-T T.81 (1992) Annex K, Table K.1，亮度量化表。這是該標準給的**範例**表
而非規範值，JPEG 實作普遍採用它，DCT-Shield 的 `jpeg_codec` 也是。
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch

# ITU-T T.81 Annex K 表 K.1（亮度）。列索引是垂直頻率，行索引是水平頻率。
JPEG_LUMA_TABLE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float64)


def _binary(block: int, device, dtype) -> torch.Tensor:
    """全 1。預設值，逐位元等於加這個模組之前的行為。"""
    return torch.ones(block, block // 2 + 1, device=device, dtype=dtype)


def _bilinear(table: torch.Tensor, u: torch.Tensor,
              v: torch.Tensor) -> torch.Tensor:
    """在 8x8 表上對 (u, v) 做雙線性內插。u、v 已夾在 [0, 7]。"""
    u0 = u.floor().clamp(0, 6)
    v0 = v.floor().clamp(0, 6)
    du = u - u0
    dv = v - v0
    i0 = u0.long()
    j0 = v0.long()
    t = table.to(device=u.device, dtype=u.dtype)
    return (t[i0, j0] * (1 - du) * (1 - dv)
            + t[i0 + 1, j0] * du * (1 - dv)
            + t[i0, j0 + 1] * (1 - du) * dv
            + t[i0 + 1, j0 + 1] * du * dv)


def _jpeg_luma(block: int, device, dtype) -> torch.Tensor:
    """JPEG 亮度量化階，正規化到最大值 1。

    量化階大 = 人眼在該頻率上看不見 = 允許較多擾動，方向與「權重越大越
    放行」一致，不需要取倒數。
    """
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0   # [-1, 1)
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0  # [0, 1]
    u = (fy.abs() * 8.0).clamp(0.0, 7.0)[:, None].expand(block, block // 2 + 1)
    v = (fx * 8.0).clamp(0.0, 7.0)[None, :].expand(block, block // 2 + 1)
    q = _bilinear(JPEG_LUMA_TABLE, u, v)
    return q / q.max()


FREQ_WEIGHTS: Dict[str, Callable[[int, object, object], torch.Tensor]] = {
    "binary": _binary,
    "jpeg_luma": _jpeg_luma,
}


def freq_weight(name: str, block: int, device, dtype,
                power: float = 1.0) -> torch.Tensor:
    """`(block, block//2+1)` 的知覺權重，值域 (0, 1]。

    名字打錯要拋錯而不是回退到 `binary`：靜默回退會讓一整批掃描跑成基準
    的重複，而報表上的 `freq_weight` 欄仍寫著它以為跑的那個名字。

    `power` 是定價的力道：`w ** power`。0 使權重恆為 1，即退回二值閘；
    1 是量化表的原始定價。**兩端都不是操作點**——二值閘的位移／DISTS 只有
    3.3–4.3，完整加權把它拉到 8–14.5 但通帶有效容量掉到 0.544，要摸到會擋下
    的強度就得把半徑推過 theta 的封頂（pi），之後只有增益在長而增益是振幅，
    PSNR 直接被打掉。中間值讓效率與可達性可以取捨。本值無出處，是本專案指定。
    """
    if name not in FREQ_WEIGHTS:
        raise ValueError(
            f"未知的 freq_weight：{name!r}，可用的是 {sorted(FREQ_WEIGHTS)}")
    if power < 0:
        raise ValueError(f"freq_weight 的 power 不可為負，收到 {power}")
    w = FREQ_WEIGHTS[name](block, device, dtype)
    return w if power == 1.0 else w ** power


# ---------------------------------------------------------------- 存活加權
#
# 為什麼要有這一層
# ────────────────────────────────────────────────────────────────────
# `jpeg_luma` 定價的是**人眼在該頻率看不看得見**，而量化階隨頻率遞增，
# 所以它把預算往高頻推。但攻擊方會先淨化：高斯模糊在頻域乘的是實正數
# `|H_σ(f)| = exp(-2π²σ²f²)`，把高頻的振幅整個拿掉。最佳化迴圈看不到這件事
# ——它跑的是未淨化的前向——所以「哪一帶的擾動活得下來」必須寫進閘裡。
#
# 這一層只補**最佳化看不到的那一半**：期望存活振幅。編碼器對哪一帶敏感由
# 最佳化自己找，不必也不應該把六張圖量到的敏感度曲線焊進方法。
#
#     w_surv(ω) = (1/(1+|S|)) · [ 1 + Σ_{σ∈S} |H_σ(ω)| ]
#
# 第一項的 1 是 identity（攻擊方不淨化），與 `eot_broad` 把 identity 當成一個
# 類別的作法一致。`S` 由名字決定，逐個列出，不吃參數——填得進 CSV 才查得回來。
#
# 座標：`radial_gate` 用的歸一化半徑 1 = Nyquist，故 cycles/pixel = r / 2。
#
# **本值無出處，是本專案指定。** `|H_σ|` 是高斯核的解析傅立葉轉換，
# `runs/fixedpoint_blur` 實測與它一致到小數第三位。

def _blur_survival(block: int, device, dtype,
                   sigmas: tuple) -> torch.Tensor:
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    f = r / 2.0                                    # cycles/pixel
    total = torch.ones_like(f)                     # identity 這一項
    for s in sigmas:
        total = total + torch.exp(-2.0 * (math.pi ** 2) * (s ** 2) * f ** 2)
    return total / (1.0 + len(sigmas))


def _surv_none(block: int, device, dtype) -> torch.Tensor:
    """全 1。預設值，逐位元等於加這一層之前的行為。"""
    return torch.ones(block, block // 2 + 1, device=device, dtype=dtype)


SURVIVAL_WEIGHTS: Dict[str, Callable[[int, object, object], torch.Tensor]] = {
    "none": _surv_none,
    # 評測用的兩個模糊 σ。σ2 目前是最差的一欄（`runs/readout_ceiling`）。
    "blur12": lambda b, d, t: _blur_survival(b, d, t, (1.0, 2.0)),
    # 只押 σ1，用來分辨「往低頻搬」的收益是不是全部來自 σ2 那一項。
    "blur1": lambda b, d, t: _blur_survival(b, d, t, (1.0,)),
}


def survival_weight(name: str, block: int, device, dtype) -> torch.Tensor:
    """`(block, block//2+1)` 的期望存活振幅，值域 (0, 1]。

    名字打錯要拋錯而不是回退到 `none`：靜默回退會讓一整批掃描跑成基準的
    重複，而報表上的 `survival_weight` 欄仍寫著它以為跑的那個名字。
    """
    if name not in SURVIVAL_WEIGHTS:
        raise ValueError(
            f"未知的 survival_weight：{name!r}，"
            f"可用的是 {sorted(SURVIVAL_WEIGHTS)}")
    return SURVIVAL_WEIGHTS[name](block, device, dtype)
