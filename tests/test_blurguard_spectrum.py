"""`scripts/blurguard_spectrum.py`：BlurGuard 的 `sigma_loss` 移植是否忠實。

釘住的是三件會靜默錯掉的事：分箱是**求和**不是平均、`shift` 兩種讀法確實不同、
以及恆等輸入必須給零。
"""

import math

import pytest
import torch

from scripts.blurguard_spectrum import (
    fps, hist, radial_bins, sigma_gap,
)


def test_相同影像的_sigma_gap_為零():
    """`x' = x` 時兩個直方圖逐位元相同，差必須是 0。

    若分箱或 log 的 epsilon 寫錯，這裡會漏出一個小的非零值，而報表上看起來
    仍然「很小」——那正是會被讀成「本方法偏離很小」的假訊號。
    """
    x = torch.rand(3, 64, 64, dtype=torch.float64)
    for shift in (False, True):
        assert sigma_gap(x, x, shift=shift) == pytest.approx(0.0, abs=1e-12)


def test_分箱是求和不是平均():
    """原始碼是 `(masks * magnitude_spectrum).sum(dim=(-1,-2))`。

    改成平均不會拋錯，但每一環會被自己的面積除掉，低半徑的環（面積小）與
    高半徑的環（面積大）之間的比例整個變掉。
    """
    f = torch.ones(8, 8, dtype=torch.float64)
    idx = radial_bins(8, 8)
    n = int(idx.max()) + 1
    h = hist(f, idx, n)
    counts = torch.bincount(idx.reshape(-1), minlength=n).to(torch.float64)
    assert torch.allclose(h, counts)          # 全 1 的譜，求和 = 該環的格數
    assert h.sum() == pytest.approx(64.0)     # 總和守恆


def test_shift_與_unshifted_是不同的量():
    """原始碼由陣列中心起算半徑卻沒有 fftshift，兩種讀法必須分開報。

    若 `shift` 這個參數被接錯（例如兩邊都走同一條路），這裡會相等而沒有症狀。
    """
    torch.manual_seed(0)
    x = torch.rand(3, 64, 64, dtype=torch.float64)
    xd = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    a = sigma_gap(x, xd, shift=False)
    b = sigma_gap(x, xd, shift=True)
    assert a != pytest.approx(b, rel=1e-6)


def test_fps_是逐通道平方和():
    x = torch.zeros(3, 4, 4, dtype=torch.float64)
    x[:, 0, 0] = 1.0
    f = fps(x, shift=False)
    # 單一脈衝的 |FFT|² 在每一格都是 1，三通道相加得 3
    assert torch.allclose(f, torch.full((4, 4), 3.0, dtype=torch.float64))


def test_擾動越大_sigma_gap_越大():
    """單調性。不單調表示分箱或 log 有錯。"""
    torch.manual_seed(1)
    x = torch.rand(3, 64, 64, dtype=torch.float64)
    n = torch.randn_like(x)
    gaps = [sigma_gap(x, (x + s * n).clamp(0, 1), shift=True)
            for s in (0.005, 0.02, 0.08)]
    assert gaps[0] < gaps[1] < gaps[2]
