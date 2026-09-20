"""四條結構性質各一個測試。

這個族的設計意圖全部寫在參數化裡（見 `src/defense/tone_desat_param.py`
的 docstring），而參數化壞掉不會有症狀：影像仍然產得出來，只是偏色、
過飽和或出現色帶。四條性質因此逐條釘住，隨機 θ 與隨機影像各抽一批。

`m = 0` 的像素單獨檢查：那是 `f(m)/m` 唯一的奇異點，實作走多項式繞開了
除法，這裡驗它的輸出逐位元為 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defense.tone_desat_param import (  # noqa: E402
    S_HI, S_LO, SLOPE_HI, SLOPE_LO, apply_filter, coeffs_to_theta, feasible,
    slope, tone)

SAMPLES = 24
SIZE = 48


def random_theta(gen: torch.Generator):
    """在 `(d0, d1, d2, s)` 的盒子裡均勻抽，再映射回 θ：抽出來一定可行。"""
    u = torch.rand(4, generator=gen, dtype=torch.float64)
    d = [SLOPE_LO + (SLOPE_HI - SLOPE_LO) * float(v) for v in u[:3]]
    t0, t1, t2 = coeffs_to_theta(*d)
    return (t0, t1, t2, S_LO + (S_HI - S_LO) * float(u[3]))


def random_image(gen: torch.Generator):
    """一半均勻、一半含大量 0 與 1，把端點與 `m = 0` 都抽到。"""
    x = torch.rand(1, 3, SIZE, SIZE, generator=gen, dtype=torch.float64)
    hard = (torch.rand(1, 3, SIZE, SIZE, generator=gen,
                       dtype=torch.float64) * 3).floor() / 2.0
    keep = torch.rand(1, 1, SIZE, SIZE, generator=gen, dtype=torch.float64) < 0.5
    return torch.where(keep, x, hard)


@pytest.fixture(scope="module")
def cases():
    gen = torch.Generator().manual_seed(20260101)
    out = []
    for _ in range(SAMPLES):
        theta = random_theta(gen)
        assert feasible(theta), theta
        x = random_image(gen)
        out.append((theta, x, apply_filter(x, theta)))
    return out


def hsv_parts(x: torch.Tensor):
    """回傳 `(max, min, max − min)`。HSV 的色相與飽和度都只用到這三個量。"""
    hi = x.amax(dim=1, keepdim=True)
    lo = x.amin(dim=1, keepdim=True)
    return hi, lo, hi - lo


def test_hue_angle_is_unchanged(cases):
    """色相不變：`(c_i − c_min)/(c_max − c_min)` 逐通道保持。

    直接比色相角會在灰像素上除以 0；比的是色相角的定義式本身，對所有
    `c_max > c_min` 的像素都成立，灰像素本來就沒有色相。
    """
    for theta, x, y in cases:
        _, x_lo, x_span = hsv_parts(x)
        _, y_lo, y_span = hsv_parts(y)
        live = (x_span > 1e-6) & (y_span > 1e-6)
        mask = live.expand_as(x)
        a = ((x - x_lo) / x_span.clamp_min(1e-12))[mask]
        b = ((y - y_lo) / y_span.clamp_min(1e-12))[mask]
        assert torch.allclose(a, b, atol=1e-9), (theta, float((a - b).abs().max()))


def test_hsv_saturation_never_increases(cases):
    """飽和度 `(max − min)/max` 不增。`max = 0` 的像素飽和度定義為 0。"""
    for theta, x, y in cases:
        x_hi, _, x_span = hsv_parts(x)
        y_hi, _, y_span = hsv_parts(y)
        sx = torch.where(x_hi > 0, x_span / x_hi.clamp_min(1e-12),
                         torch.zeros_like(x_hi))
        sy = torch.where(y_hi > 0, y_span / y_hi.clamp_min(1e-12),
                         torch.zeros_like(y_hi))
        assert float((sy - sx).max()) <= 1e-9, (theta, float((sy - sx).max()))


def test_max_channel_never_increases(cases):
    """最大通道不增：這個族不可能把影像變亮。"""
    for theta, x, y in cases:
        x_hi = x.amax(dim=1, keepdim=True)
        y_hi = y.amax(dim=1, keepdim=True)
        assert float((y_hi - x_hi).max()) <= 1e-9, (theta,
                                                    float((y_hi - x_hi).max()))
        assert float(y.min()) >= -1e-12 and float(y.max()) <= 1.0 + 1e-9


def test_curve_slope_stays_inside_the_bounds(cases):
    """`f'(m) ∈ [0.75, 1]`：色帶與平坦段都不可達。

    解析的 `slope` 與有限差分各驗一次——係數算錯而曲線對、或曲線算錯而
    係數對，兩種錯法都要擋。
    """
    m = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    h = 1e-5
    for theta, _, _ in cases:
        s = slope(theta, m)
        assert float(s.min()) >= SLOPE_LO - 1e-9, (theta, float(s.min()))
        assert float(s.max()) <= SLOPE_HI + 1e-9, (theta, float(s.max()))
        mid = m[1:-1]
        fd = (tone(theta, mid + h) - tone(theta, mid - h)) / (2 * h)
        assert float((fd - slope(theta, mid)).abs().max()) < 1e-6, theta


def test_black_pixels_stay_black(cases):
    """`m = 0` 的像素輸出逐位元為 0；實作以多項式繞開除法，不靠 clamp。"""
    for theta, x, y in cases:
        black = x.amax(dim=1, keepdim=True) == 0
        if bool(black.any()):
            assert float(y[black.expand_as(y)].abs().max()) == 0.0, theta
    x = torch.zeros(1, 3, 4, 4, dtype=torch.float64)
    for theta, _, _ in cases:
        assert float(apply_filter(x, theta).abs().max()) == 0.0
