import pytest
import torch

from src.defense.outside_terms import (OutsideDisplacement, PrintObjective,
                                       band_masks)


class _StubObjective:
    """只提供 `OutsideDisplacement` 會用到的三樣東西。

    `null_edit` 把輸入乘上一個固定的增益場，所以位移的分佈是已知的，
    帶的加權有沒有算對可以逐帶檢查。
    """

    def __init__(self, gain):
        self.draws = 0
        self.seed = 7
        self.gain = gain
        self.calls = 0

    def null_edit(self, x01):
        self.calls += 1
        return x01 * self.gain


def _face_patch(monkeypatch, box):
    import src.metrics.identity as identity

    monkeypatch.setattr(identity, 'face_boxes', lambda x, dev: [box])


def test_bands_are_disjoint_and_cover_the_outside(monkeypatch):
    _face_patch(monkeypatch, (4, 4, 20, 20))
    sup = torch.zeros(1, 1, 64, 64)
    sup[0, 0, 28:36, 28:36] = 1.0
    x = torch.rand(1, 3, 64, 64)
    b = band_masks(sup, x, margin=2, near_width=6)
    assert float((b['near'] * b['far']).sum()) == 0.0
    assert torch.allclose(b['near'] + b['far'], b['outside'])
    assert float((b['face'] * b['outside']).sum()) == float(b['face'].sum())


def test_bands_reject_an_empty_outside(monkeypatch):
    _face_patch(monkeypatch, (0, 0, 2, 2))
    sup = torch.ones(1, 1, 16, 16)
    with pytest.raises(ValueError, match='帶是空的'):
        band_masks(sup, torch.rand(1, 3, 16, 16), margin=1, near_width=2)


def test_displacement_ignores_changes_inside_the_support(monkeypatch):
    """支撐內改得再多也不該進分數，那正是要拿掉的捷徑。"""
    _face_patch(monkeypatch, (2, 2, 6, 6))
    sup = torch.zeros(1, 1, 64, 64)
    sup[0, 0, 28:36, 28:36] = 1.0
    x = torch.full((1, 3, 64, 64), 0.5)
    term = OutsideDisplacement(_StubObjective(1.0), x, sup, margin=2,
                               near_width=6, weight_near=1.0)
    loud = x.clone()
    loud[:, :, 28:36, 28:36] = 0.0
    assert float(term(loud)) == pytest.approx(0.0, abs=1e-7)


def test_displacement_sees_changes_outside(monkeypatch):
    _face_patch(monkeypatch, (2, 2, 6, 6))
    sup = torch.zeros(1, 1, 64, 64)
    sup[0, 0, 28:36, 28:36] = 1.0
    x = torch.full((1, 3, 64, 64), 0.5)
    term = OutsideDisplacement(_StubObjective(1.0), x, sup, margin=2,
                               near_width=6, weight_far=1.0, weight_face=0.0)
    moved = x.clone()
    moved[:, :, 0:8, 0:8] = 0.25
    assert float(term(moved)) > 0.0


def test_reference_is_cached_until_the_draw_changes(monkeypatch):
    _face_patch(monkeypatch, (2, 2, 6, 6))
    sup = torch.zeros(1, 1, 32, 32)
    sup[0, 0, 12:20, 12:20] = 1.0
    x = torch.rand(1, 3, 32, 32)
    stub = _StubObjective(1.0)
    term = OutsideDisplacement(stub, x, sup, margin=1, near_width=4)
    term(x)
    after_first = stub.calls
    term(x)
    assert stub.calls == after_first + 1
    stub.draws += 1
    term(x)
    assert stub.calls == after_first + 3


def test_band_weights_are_applied(monkeypatch):
    _face_patch(monkeypatch, (0, 0, 8, 8))
    sup = torch.zeros(1, 1, 64, 64)
    sup[0, 0, 28:36, 28:36] = 1.0
    x = torch.full((1, 3, 64, 64), 0.5)
    moved = x.clone()
    moved[:, :, 0:8, 0:8] = 0.0
    single = OutsideDisplacement(_StubObjective(1.0), x, sup, margin=2,
                                 near_width=6, weight_far=1.0,
                                 weight_face=1.0)
    doubled = OutsideDisplacement(_StubObjective(1.0), x, sup, margin=2,
                                  near_width=6, weight_far=1.0,
                                  weight_face=3.0)
    assert float(doubled(moved)) > float(single(moved))


def test_all_zero_weights_are_rejected(monkeypatch):
    _face_patch(monkeypatch, (2, 2, 6, 6))
    sup = torch.zeros(1, 1, 32, 32)
    sup[0, 0, 12:20, 12:20] = 1.0
    with pytest.raises(ValueError, match='權重全為零'):
        OutsideDisplacement(_StubObjective(1.0), torch.rand(1, 3, 32, 32),
                            sup, margin=1, near_width=4, weight_far=0.0,
                            weight_face=0.0, weight_near=0.0)


def test_report_lists_every_band(monkeypatch):
    _face_patch(monkeypatch, (2, 2, 6, 6))
    sup = torch.zeros(1, 1, 64, 64)
    sup[0, 0, 28:36, 28:36] = 1.0
    x = torch.full((1, 3, 64, 64), 0.5)
    term = OutsideDisplacement(_StubObjective(0.9), x, sup, margin=2,
                               near_width=6)
    got = term.report(x)
    assert set(got) == {'outside_near', 'outside_far', 'outside_face',
                        'outside_outside'}


class _StubBase:
    weights = {'enc': 0.5, 'cond': 1.0, 'id': 1.0}
    ip2p = None
    timesteps = []
    val = None

    def terms(self, x_def):
        return {'enc': torch.tensor(0.2), 'cond': torch.tensor(0.1),
                'id': torch.tensor(0.9)}


def test_outside_enters_the_score_with_a_negative_sign():
    """分數一律是要最小化的方向，所以越大越好的項以負號進。"""
    base = _StubBase()
    quiet = PrintObjective(base, outside_term=lambda y: torch.tensor(0.01),
                           weights={'outside': 10.0})
    loud = PrintObjective(base, outside_term=lambda y: torch.tensor(0.05),
                          weights={'outside': 10.0})
    x = torch.zeros(1, 3, 8, 8)
    assert float(loud.score(x)) < float(quiet.score(x))


def test_outside_is_skipped_when_its_weight_is_zero():
    base = _StubBase()
    obj = PrintObjective(base, outside_term=lambda y: torch.tensor(1.0),
                         weights={'outside': 0.0})
    assert 'outside' not in obj.terms(torch.zeros(1, 3, 8, 8))
