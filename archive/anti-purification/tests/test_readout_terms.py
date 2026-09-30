import pytest
import torch

from src.defense.instruction_free import FreeObjective
from src.defense.mainstream_terms import texture_target
from src.defense.readout_terms import (CompositeObjective, OutputDisplacement,
                                       TargetedOutputDisplacement)
from tests.test_instruction_free import FakeIP2P, _image


def _l2(a, b):
    return (a - b).flatten().norm()


def _setup(size=16, **kw):
    x = _image(0, size)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=3, steps=20, **kw)
    tex = texture_target(x)
    return x, obj, tex


def test_目標側是目標圖的_null_edit_而不是原圖的():
    x, obj, tex = _setup()
    free = OutputDisplacement(obj, x, _l2)
    tgt = TargetedOutputDisplacement(obj, tex, _l2)
    assert not torch.allclose(free.reference(), tgt.reference())
    with torch.no_grad():
        expect = obj.null_edit(tex).float()
    assert torch.allclose(tgt.reference(), expect)


def test_把目標本身代進去時項為零():
    x, obj, tex = _setup()
    tgt = TargetedOutputDisplacement(obj, tex, _l2)
    assert float(tgt(tex)) == pytest.approx(0.0, abs=1e-9)
    assert float(tgt(x)) > 0.0


def test_重抽之後目標側跟著重算():
    """`null_edit` 的取樣噪聲由 `seed + draws` 決定；目標側凍結會讓差值變噪聲差。"""
    x, obj, tex = _setup(resample=True)
    tgt = TargetedOutputDisplacement(obj, tex, _l2)
    first = tgt.reference().clone()
    obj._redraw()
    second = tgt.reference()
    assert not torch.allclose(first, second)


def test_驗證抽樣下的目標側與訓練抽樣下的不同():
    """`eval_terms` 自己進出驗證狀態，外掛項必須落在那個狀態上。"""
    x, obj, tex = _setup(resample=True)
    tgt = TargetedOutputDisplacement(obj, tex, _l2)
    c = CompositeObjective(obj, out_target_term=tgt,
                           weights={'id': 0.0, 'enc': 0.5, 'cond': 1.0,
                                    'out_target': 2.0})
    train = float(c.terms(x)['out_target'])
    assert int(obj.draws) != int(obj.val[4])
    ev = float(c.eval_terms(x)['out_target'])
    assert train != pytest.approx(ev, abs=1e-12)
    assert int(tgt._cache_key[0]) == int(obj.val[4])


def test_符號是要被最小化的方向():
    """`out` 越大越好、`out_target` 越小越好，兩者的符號必須相反。"""
    x, obj, tex = _setup()
    tgt = TargetedOutputDisplacement(obj, tex, _l2)
    base = {'id': 0.0, 'enc': 0.0, 'cond': 0.0}
    c = CompositeObjective(obj, out_target_term=tgt,
                           weights=dict(base, out_target=2.0))
    t = c.terms(x)
    assert float(c.score(x)) == pytest.approx(2.0 * float(t['out_target']),
                                              abs=1e-9)
    assert float(c.score(x)) > 0.0


def test_接得上求解器並寫出逐項欄位():
    from src.defense.color_param import ColorCurveParam
    from src.defense.immunise import optimise_carrier

    x, obj, tex = _setup(size=24)
    carrier = ColorCurveParam(radius=1.0, pieces=8)
    carrier.reset(x, seed=0)
    tgt = TargetedOutputDisplacement(obj, tex, _l2)
    c = CompositeObjective(obj, carrier=carrier, out_target_term=tgt,
                           weights={'id': 0.0, 'enc': 0.5, 'cond': 1.0,
                                    'out_target': 2.0})
    stats = optimise_carrier(carrier, x, c, steps=6, lr=0.3, caps=[],
                             check_every=3)
    assert stats['free_score_end'] < stats['free_score_start']
    assert 'free_term_out_target' in stats
