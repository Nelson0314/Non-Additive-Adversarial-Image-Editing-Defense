"""E31 Task 2：ISR 判定的純函數測試。

判定邏輯是本專案全部結論的依據，必須能在沒有任何 run 資料的情況下驗證。
E25 曾因缺 n≥2 這個條件在兩個步數掃描的 run 上產生 24 格假陽性，那是
「判定邏輯沒有被單獨測過」的直接後果。
"""

import pytest

from scripts.p12_isr_rejudge import judge_cell


def test_語意失敗需要平均為負且絕對值大於標準差():
    r = judge_cell([-0.05, -0.05], [0.0, 0.0], degrade_tau=1.0)
    assert r["semantic_fail"] is True
    assert r["isr"] is True


def test_平均為負但小於標準差不算語意失敗():
    r = judge_cell([-0.01, +0.03], [0.0, 0.0], degrade_tau=1.0)
    assert r["semantic_fail"] is False
    assert r["isr"] is False


def test_單一樣本不得判為語意失敗():
    # n=1 時 pstdev 恆為 0，|mean| > sd 對任何負值自動成立。
    r = judge_cell([-0.5], [0.0], degrade_tau=1.0)
    assert r["semantic_fail"] is False
    assert r["n"] == 1


def test_單一樣本也不得判為感知劣化():
    r = judge_cell([0.0], [99.0], degrade_tau=1.0)
    assert r["degrade_fail"] is False


def test_感知劣化單獨也能使ISR成立():
    r = judge_cell([+0.02, +0.02], [1.6, 1.5], degrade_tau=1.0)
    assert r["semantic_fail"] is False
    assert r["degrade_fail"] is True
    assert r["isr"] is True


def test_兩者都不成立時ISR為假():
    r = judge_cell([+0.02, +0.02], [0.1, 0.2], degrade_tau=1.0)
    assert r["isr"] is False


def test_空輸入必須報錯而非回傳假():
    # 靜默回傳 False 會把「沒有資料」與「判定為未擋下」混為一談。
    with pytest.raises(ValueError):
        judge_cell([], [], degrade_tau=1.0)


def test_兩軸樣本數不符必須報錯():
    with pytest.raises(ValueError):
        judge_cell([0.1, 0.2], [1.0], degrade_tau=1.0)
