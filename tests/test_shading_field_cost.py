"""明暗場探針的守門測試（`scripts/shading_field_cost.py`）。

這支探針的**全部**論證是「兩邊的殘差 RMS 相同，所以剩下的差異只能是乘性對
加性」。等 RMS 一旦沒對上，比值就同時混進了強度差，而報表上看不出來——兩邊
仍各自是一個合理的數字。故把等 RMS 與場的頻帶逐條釘住。

全部在 CPU 上執行。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from shading_field_cost import (  # noqa: E402
    field_radius,
    low_freq_field,
    residual_rms,
    solve_scale,
)

DEV = torch.device("cpu")


def _img(seed=0, size=128):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


@pytest.mark.parametrize("multiplicative", [True, False])
@pytest.mark.parametrize("target", [0.01, 0.06])
def test_二分法真的解到目標_RMS(multiplicative, target):
    x = _img()
    f = low_freq_field(0, 16, 128, DEV)
    _, y = solve_scale(x, f, target, multiplicative)
    assert residual_rms(x, y) == pytest.approx(target, abs=1e-5)


def test_乘性與加性在同一個目標上對齊():
    """兩邊 RMS 若不同，PSNR 就會不同——報表用 PSNR 當這件事的守門。"""
    x = _img(1)
    f = low_freq_field(2, 16, 128, DEV)
    _, y_mul = solve_scale(x, f, 0.04, True)
    _, y_add = solve_scale(x, f, 0.04, False)
    assert residual_rms(x, y_mul) == pytest.approx(residual_rms(x, y_add), abs=1e-5)


def test_輸出留在值域內():
    x = _img(3)
    f = low_freq_field(1, 16, 128, DEV)
    for mult in (True, False):
        _, y = solve_scale(x, f, 0.06, mult)
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_達不到目標時拋錯而不是回傳一個較小的值():
    """全黑影像乘任何數還是全黑；靜默回傳會讓那一列的等 RMS 是假的。"""
    x = torch.zeros(1, 3, 64, 64)
    f = low_freq_field(0, 16, 64, DEV)
    with pytest.raises(RuntimeError, match="解不出係數"):
        solve_scale(x, f, 0.05, multiplicative=True)


def test_場是單位標準差且可重現():
    f = low_freq_field(5, 16, 128, DEV)
    assert float(f.std()) == pytest.approx(1.0, abs=1e-4)
    assert torch.equal(f, low_freq_field(5, 16, 128, DEV))
    assert not torch.equal(f, low_freq_field(6, 16, 128, DEV))


def test_場落在候選二宣稱的低頻帶內():
    """候選二的全部依據是它落在對放大與對模糊都免疫的子空間（f_n 約 0.03）。

    粗網格放大一倍，頻率半徑就要跟著大約放大一倍——不成立表示上採樣寫錯了。
    """
    r16 = field_radius(low_freq_field(0, 16, 512, DEV))
    r32 = field_radius(low_freq_field(0, 32, 512, DEV))
    assert 0.01 < r16 < 0.04
    assert r32 > 1.5 * r16


def test_純直流場的頻率半徑為零():
    f = torch.ones(1, 1, 64, 64)
    assert field_radius(f) == pytest.approx(0.0, abs=1e-6)
