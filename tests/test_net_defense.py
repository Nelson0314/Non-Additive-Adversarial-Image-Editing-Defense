"""淨額防禦項的正確性測試。**分支上的提案，未進 main。**

檢驗的是一條性質：hinge 內減去未防禦對照後，「淨化自己造成的偏移」不再
被計入防禦效果。
"""

import pytest
import torch

from src.defense.objective import DefenseObjective, LossConfig

DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def obj():
    return DefenseObjective(LossConfig(), DEV)


def _img(seed, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def test_對照等於防禦時淨額為零(obj):
    """φ 什麼也沒加時，防禦項必須回到 hinge 的最大值 margin。

    構造：y_def 與 y_ctrl 相同，代表偏移全部來自淨化。此時淨額為 0，
    hinge 應為 max(0, m − 0) = m，即「完全沒有防禦效果」。
    """
    y_orig = _img(1)
    y_same = _img(2)
    d = float(obj.distance(y_same, y_orig))

    without = float(obj.defense_term([y_same], [y_orig]))
    with_ctrl = float(obj.defense_term([y_same], [y_orig], [d]))

    assert without == pytest.approx(max(0.0, obj.cfg.margin - d), abs=1e-5)
    assert with_ctrl == pytest.approx(obj.cfg.margin, abs=1e-5), (
        "偏移全部來自淨化時，防禦項不得因此獲得任何credit"
    )
    assert with_ctrl > without, "減去對照後施力必須變大，否則修正沒有生效"


def test_防禦超出對照的部分才算數(obj):
    y_orig = _img(3)
    y_def = _img(4)
    d = float(obj.distance(y_def, y_orig))
    half = d / 2

    term = float(obj.defense_term([y_def], [y_orig], [half]))
    assert term == pytest.approx(max(0.0, obj.cfg.margin - half), abs=1e-5)


def test_對照數與取樣數不符時報錯(obj):
    y = _img(5)
    with pytest.raises(ValueError, match="對照數"):
        obj.defense_term([y, y], [y, y], [0.1])


def test_不提供對照時行為與原式相同(obj):
    """向後相容：`d_ctrl_list=None` 必須還原成 spec §5.1 的原始形式。"""
    y_orig, y_def = _img(6), _img(7)
    a = float(obj.defense_term([y_def], [y_orig]))
    b = float(obj.defense_term([y_def], [y_orig], None))
    assert a == b
