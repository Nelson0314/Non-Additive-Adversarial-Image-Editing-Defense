"""擾動本身的兩個頻域讀數：低頻佔比與模糊殘存率。

為什麼量的是擾動而不是影像
────────────────────────────────────────────────────────────────────
ΔE00／PSNR／LPIPS 說的是「防禦圖離原圖多遠」，說不出那段距離**擺在哪個
頻段**。抗淨化的行為由後者決定：低通式的淨化（模糊、JPEG、擴散重建）先吃
高頻，擾動落在低頻的載體因此殘存得多。這兩欄把「擾動住在哪裡」量出來。

直流分量一律先去掉
────────────────────────────────────────────────────────────────────
整體提亮或整體壓暗是常數項，它在頻譜上全部落在 `f = 0`，會讓低頻佔比逼近 1
而與載體的空間結構無關；模糊也動不了它，殘存率同樣逼近 1。兩欄都先減掉
逐通道的平均值，量的才是**非直流**的那一部分。

單位
────────────────────────────────────────────────────────────────────
徑向頻率 `f = sqrt(fx² + fy²)`，`fx = kx/W`、`fy = ky/H`，單位是 cycle/pixel，
上限 0.5（Nyquist）。門檻 `1/8` 即 `≤ 0.125 cycle/pixel`。
"""
from __future__ import annotations

import numpy as np

#: 低頻的徑向門檻，cycle/pixel。
LOW_BAND = 0.125
#: 模糊殘存率用的高斯標準差，像素。
BLUR_SIGMA = 2.0


def remove_dc(delta: np.ndarray) -> np.ndarray:
    """逐通道減掉平均值。`delta` 為 (C,H,W)。"""
    return delta - delta.mean(axis=(-2, -1), keepdims=True)


def low_frequency_share(delta: np.ndarray, band: float = LOW_BAND) -> float:
    """非直流擾動裡，徑向頻率 ≤ `band` 的那一段佔多少能量。

    `delta` 為 (C,H,W)。回傳 `Σ_{0 < f ≤ band} |F|² / Σ_{f > 0} |F|²`，
    三個通道的能量先加總再取比值（逐通道取比值再平均會給振幅小的通道
    同樣的票數）。
    """
    d = remove_dc(np.asarray(delta, dtype=np.float64))
    spectrum = np.fft.fft2(d, axes=(-2, -1))
    power = (spectrum.real ** 2 + spectrum.imag ** 2).sum(axis=0)
    h, w = power.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radial = np.sqrt(fy ** 2 + fx ** 2)
    non_dc = radial > 0
    total = power[non_dc].sum()
    if total <= 0:
        # 擾動去掉直流之後逐位元為零：佔比沒有定義，不用 0 或 1 冒充。
        return float("nan")
    return float(power[non_dc & (radial <= band)].sum() / total)


def blur_retention(delta: np.ndarray, sigma: float = BLUR_SIGMA) -> float:
    """非直流擾動經過 σ 的高斯模糊之後，L2 範數還剩多少。

    模糊是線性的，所以「先模糊兩張影像再相減」與「模糊差值」等價，這裡取
    後者。邊界用 `reflect`，與 `src/purify/ops.py` 的模糊淨化同一種延拓。
    """
    from scipy.ndimage import gaussian_filter

    d = remove_dc(np.asarray(delta, dtype=np.float64))
    base = float(np.sqrt((d ** 2).sum()))
    if base <= 0:
        return float("nan")
    blurred = np.stack([gaussian_filter(c, sigma=sigma, mode="reflect",
                                        truncate=4.0) for c in d])
    return float(np.sqrt((blurred ** 2).sum()) / base)
