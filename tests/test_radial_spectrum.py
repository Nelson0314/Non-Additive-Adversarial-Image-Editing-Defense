"""`scripts/radial_spectrum.py` 的驗收 — FND-060 的數字就是它算的。

**這一組測試是補寫的。** FND-060 的頭條（加性方法 42–65% 的擾動能量在半
Nyquist 以上、紋理重相位只有 22.6%）在 2026-08-19 寫入 `FINDINGS.md` 時，
產生那些數字的程式一項測試都沒有。分箱寫錯、半徑正規化寫錯、或高低頻的門檻
取反，結果都仍然是一張看起來合理的表。

這裡用**解析上已知答案**的輸入來釘：單一頻率的正弦波應該只落在對應的箱裡，
棋盤格是純 Nyquist，寬緩的梯度是純低頻。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "radial_spectrum",
    Path(__file__).resolve().parent.parent / "scripts" / "radial_spectrum.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def _sine(h, w, ku, kv, amp=1.0):
    """空間頻率為 (ku, kv) 週期／全圖的餘弦波，三通道相同。

    以 `fftfreq` 的尺度看，它的頻率是 `(2ku/h, 2kv/w)`——即 `ku = h/2` 就是
    垂直方向的 Nyquist。
    """
    y = torch.arange(h, dtype=torch.float64).view(-1, 1)
    x = torch.arange(w, dtype=torch.float64).view(1, -1)
    p = amp * torch.cos(2 * math.pi * (ku * y / h + kv * x / w))
    return p.view(1, 1, h, w).repeat(1, 3, 1, 1)


# ---- 徑向功率譜本身 ----

def test_profile_sums_to_one():
    """回傳值已正規化成總和為 1，否則跨條件的佔比不可比。"""
    p = rs.radial_power(_sine(64, 64, 5, 3))
    assert abs(float(p.sum()) - 1.0) < 1e-12


def test_zero_perturbation_gives_all_zeros_without_dividing_by_zero():
    p = rs.radial_power(torch.zeros(1, 3, 32, 32, dtype=torch.float64))
    assert np.all(p == 0.0)


def test_single_frequency_lands_in_one_bin():
    """單一餘弦波只有兩個非零頻格（±k），兩者半徑相同，故只佔一個箱。"""
    p = rs.radial_power(_sine(64, 64, 8, 0), bins=64)
    nz = np.flatnonzero(p > 1e-9)
    assert len(nz) == 1, f"應只有一個非零箱，實際 {len(nz)} 個"
    assert abs(float(p[nz[0]]) - 1.0) < 1e-9


def test_higher_spatial_frequency_lands_in_a_higher_bin():
    """箱的索引必須隨頻率單調上升——取反的話整張表的判讀會顛倒。"""
    prev = -1
    for k in (2, 4, 8, 16, 32):
        p = rs.radial_power(_sine(64, 64, k, 0), bins=64)
        idx = int(np.argmax(p))
        assert idx > prev, f"k={k} 的箱索引沒有比前一個高"
        prev = idx


def test_nyquist_is_at_radius_one_not_at_the_maximum_radius():
    """半徑以 Nyquist 為 1，而角落可達 √2。垂直方向的 Nyquist（k = h/2）
    對應的箱中心應落在 1.0 附近，**不是**在最後一箱。"""
    bins = 64
    p = rs.radial_power(_sine(64, 64, 32, 0), bins=bins)
    idx = int(np.argmax(p))
    edges = np.arange(bins + 1) / bins * math.sqrt(2.0)
    centre = 0.5 * (edges[idx] + edges[idx + 1])
    assert abs(centre - 1.0) < 2.0 / bins * math.sqrt(2.0)
    assert idx < bins - 1, "Nyquist 不該落在最後一箱（那是角落）"


def test_checkerboard_is_the_corner_frequency():
    """棋盤格是 (h/2, w/2)，即半徑 √2 的角落，應落在最後一箱。"""
    bins = 64
    p = rs.radial_power(_sine(64, 64, 32, 32), bins=bins)
    assert int(np.argmax(p)) == bins - 1


def test_rejects_non_square_and_wrong_rank():
    with pytest.raises(ValueError):
        rs.radial_power(torch.zeros(1, 3, 32, 64, dtype=torch.float64))
    with pytest.raises(ValueError):
        rs.radial_power(torch.zeros(3, 32, 32, dtype=torch.float64))


# ---- 三個摘要 ----

def test_summary_of_a_pure_low_frequency_perturbation():
    """寬緩的正弦（k=2 於 64²，即 0.0625 Nyquist）應該幾乎全在低頻。"""
    s = rs.summarise(rs.radial_power(_sine(64, 64, 2, 0)))
    assert s["lo_frac"] > 0.99
    assert s["hi_frac"] < 0.01
    assert s["f50"] < 0.125


def test_summary_of_a_pure_high_frequency_perturbation():
    s = rs.summarise(rs.radial_power(_sine(64, 64, 28, 0)))
    assert s["hi_frac"] > 0.99
    assert s["lo_frac"] < 0.01
    assert s["f50"] > 0.5


def test_thresholds_are_where_the_docstring_says():
    """`hi_frac` 的門檻是 0.5·Nyquist、`lo_frac` 是 0.125。取反或取錯會讓
    FND-060 的「加性 42–65% vs 相位 22.6%」整段失去意義。"""
    # 0.4 Nyquist：低於 hi 門檻、高於 lo 門檻，故兩者都應為 0
    s = rs.summarise(rs.radial_power(_sine(64, 64, 13, 0)))   # 13/32 ≈ 0.406
    assert s["hi_frac"] < 0.01 and s["lo_frac"] < 0.01


def test_fifty_percent_frequency_sits_between_two_equal_components():
    """兩個等能量的成分，f50 應落在兩者之間。"""
    p = _sine(64, 64, 4, 0) + _sine(64, 64, 24, 0)
    s = rs.summarise(rs.radial_power(p))
    assert 4 / 32 < s["f50"] < 24 / 32


def test_fractions_never_exceed_one():
    for k in (1, 5, 12, 20, 31):
        s = rs.summarise(rs.radial_power(_sine(64, 64, k, k // 2)))
        assert 0.0 <= s["hi_frac"] <= 1.0
        assert 0.0 <= s["lo_frac"] <= 1.0
        assert s["hi_frac"] + s["lo_frac"] <= 1.0 + 1e-12


# ---- 通道處理 ----

def test_power_is_summed_over_channels_not_averaged_to_luma():
    """逐通道做 DFT 再把功率相加。先合成亮度會把只出現在單一通道的擾動抹掉
    ——加性 baseline 的擾動正是逐通道獨立的。"""
    h = w = 32
    only_red = torch.zeros(1, 3, h, w, dtype=torch.float64)
    only_red[:, 0] = _sine(h, w, 8, 0)[0, 0]
    p = rs.radial_power(only_red)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    assert float(p.max()) > 0.99, "單通道的擾動不該被稀釋掉"
