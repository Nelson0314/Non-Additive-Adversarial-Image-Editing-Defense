import pytest
import torch

from src.defense.victim_classifier import RegularisedVictim, VictimClassifier


class _Stub(VictimClassifier):
    """不下載權重的替身，只驗 margin／anchor／report 的邏輯。"""

    def __init__(self, table, kappa=0.0):
        self.name = 'stub'
        self.kappa = float(kappa)
        self.device = 'cpu'
        self.label = None
        self._table = table

    def logits(self, x01):
        return self._table(x01)


def _table(vec):
    return lambda _x: torch.tensor([vec], dtype=torch.float32)


def test_anchor_記住乾淨影像的_top1():
    v = _Stub(_table([1.0, 5.0, 2.0]))
    row = v.anchor(torch.zeros(1, 3, 8, 8))
    assert row['clean_label'] == 1
    assert 0.0 < row['clean_prob'] < 1.0


def test_margin_未翻轉時為正翻轉後飽和在負kappa():
    """C&W 的 margin 是 max(z_true − max_other, −kappa)：翻過邊界就停止施壓。

    kappa = 0 時翻轉後為 0，不是負值——這是論文的定義，不是截斷錯誤。
    """
    v = _Stub(_table([1.0, 5.0, 2.0]))
    v.anchor(torch.zeros(1, 3, 8, 8))
    assert float(v.margin(torch.zeros(1, 3, 8, 8))) == pytest.approx(3.0)
    v._table = _table([1.0, 2.0, 6.0])
    assert float(v.margin(torch.zeros(1, 3, 8, 8))) == pytest.approx(0.0)
    assert v.report(torch.zeros(1, 3, 8, 8))['flipped'] == 1


def test_kappa_擋住負得太多的_margin():
    v = _Stub(_table([1.0, 5.0, 2.0]), kappa=1.0)
    v.anchor(torch.zeros(1, 3, 8, 8))
    v._table = _table([1.0, 2.0, 9.0])
    assert float(v.margin(torch.zeros(1, 3, 8, 8))) == pytest.approx(-1.0)


def test_沒有_anchor_就算_margin_要報錯():
    v = _Stub(_table([1.0, 2.0]))
    with pytest.raises(RuntimeError, match='anchor'):
        v.margin(torch.zeros(1, 3, 8, 8))


def test_report_的_flipped_與_top1_一致():
    v = _Stub(_table([1.0, 5.0, 2.0]))
    v.anchor(torch.zeros(1, 3, 8, 8))
    assert v.report(torch.zeros(1, 3, 8, 8))['flipped'] == 0
    v._table = _table([9.0, 5.0, 2.0])
    r = v.report(torch.zeros(1, 3, 8, 8))
    assert r['flipped'] == 1 and r['pred_label'] == 0


def test_正則項與_margin_一起進同一個純量():
    class _Carrier:
        def smoothness(self):
            return torch.tensor(2.0)

    v = _Stub(_table([1.0, 5.0, 2.0]))
    v.anchor(torch.zeros(1, 3, 8, 8))
    reg = RegularisedVictim(v, _Carrier(), weight=0.5)
    x = torch.zeros(1, 3, 8, 8)
    assert float(reg.score(x)) == pytest.approx(3.0 + 0.5 * 2.0)
    assert set(reg.terms(x)) == {'margin', 'smoothness'}


def test_沒有正則項的載體也接得上():
    v = _Stub(_table([1.0, 5.0, 2.0]))
    v.anchor(torch.zeros(1, 3, 8, 8))
    reg = RegularisedVictim(v, object(), weight=1.0)
    assert float(reg.score(torch.zeros(1, 3, 8, 8))) == pytest.approx(3.0)
    assert set(reg.terms(torch.zeros(1, 3, 8, 8))) == {'margin'}
