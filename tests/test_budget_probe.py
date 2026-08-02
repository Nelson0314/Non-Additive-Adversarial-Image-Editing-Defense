"""E31 Task 3：放大係數二分搜尋的純函數測試。

這一段決定「在哪一個 LPIPS 上量 acut 與 chroma」。搜尋若靜默回傳上界，
後續量到的兩道約束值對應的其實是別的 LPIPS，整份預估失效而不報錯。
"""

import pytest

from scripts.p13_budget_probe import find_scale


def test_線性函數的二分搜尋():
    s = find_scale(lambda k: 0.01 * k, target=0.28, lo=1.0, hi=64.0)
    assert abs(0.01 * s - 0.28) < 1e-3


def test_目標低於下界時回傳下界():
    s = find_scale(lambda k: 0.5 * k, target=0.28, lo=1.0, hi=64.0)
    assert s == pytest.approx(1.0)


def test_上界不足時必須報錯而非回傳上界():
    with pytest.raises(ValueError):
        find_scale(lambda k: 1e-6 * k, target=0.28, lo=1.0, hi=64.0)


def test_飽和函數也能收斂到容差內():
    # 實際的 LPIPS(x + kδ) 隨 k 遞增但會飽和；二分只需單調性。
    import math

    s = find_scale(lambda k: 0.5 * math.tanh(0.05 * k), target=0.28,
                   lo=1.0, hi=64.0)
    assert abs(0.5 * math.tanh(0.05 * s) - 0.28) < 1e-3
