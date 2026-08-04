"""E31：`p16_criterion_correlation.spearman` 的正確性。

這支手寫的等級相關現在撐著一項標題級的結論（三類判準之間 ρ = 0.140／
−0.207／0.014，見 `docs/RESULTS_E25-E31.md` §6）。手寫實作沒有測試就
不該讓結論建立在它上面——特別是平手（tie）的處理，那是等級相關最常見的
實作錯誤：不用平均等級會讓 ρ 系統性偏高。
"""

import pytest

from scripts.p16_criterion_correlation import _ranks, spearman


def test_完全單調遞增為正一():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_完全單調遞減為負一():
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_只看等級不看數值():
    # 非線性但單調的變換不改變等級相關，這正是選它而非 Pearson 的理由。
    xs = [1, 2, 3, 4, 5]
    assert spearman(xs, [v ** 3 for v in xs]) == pytest.approx(1.0)


def test_平手以平均等級處理():
    # [10, 20, 20, 30] 的等級應為 [1, 2.5, 2.5, 4]。
    assert _ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_已知案例():
    # 手算：x 等級 [1,2,3,4,5]、y 等級 [2,1,4,3,5]，d = [-1,1,-1,1,0]，
    # Σd² = 4，n = 5 → ρ = 1 − 6·4/(5·24) = 0.8。
    assert spearman([1, 2, 3, 4, 5], [20, 10, 40, 30, 50]) == pytest.approx(0.8)


def test_長度不符必須報錯():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_樣本太少必須報錯():
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2])


def test_其中一軸無變異必須報錯():
    # 靜默回傳 0 會被讀成「兩者無關」，那與「其中一軸根本沒有資訊」是
    # 完全不同的兩件事。
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [7, 7, 7])
