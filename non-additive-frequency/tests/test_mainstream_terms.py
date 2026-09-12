import pytest
import torch

from src.defense.instruction_free import FreeObjective
from src.defense.mainstream_terms import (CombinedObjective, MainstreamTerms,
                                          grey_target)
from tests.test_instruction_free import FakeIP2P, _image


def _pair(seed=0, size=16):
    x = _image(seed, size)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=3, steps=20)
    return x, obj, MainstreamTerms(obj, x, use_ckpt=False)


def test_中性灰目標與輸入同形狀且無結構():
    x = _image(0, 16)
    t = grey_target(x)
    assert t.shape == x.shape
    assert float(t.std()) == 0.0
    assert float(t.mean()) == pytest.approx(0.5)


def test_把目標本身代進去時目標項為零():
    x, obj, extra = _pair()
    assert float(extra.enc_target(grey_target(x))) == pytest.approx(0.0, abs=1e-12)


def test_目標項對原圖不為零():
    """無目標的 `enc` 對原圖恆為零，有目標的這一項不是——方向不同就量得出來。"""
    x, obj, extra = _pair()
    assert float(obj.terms(x)['enc']) == pytest.approx(0.0, abs=1e-12)
    assert float(extra.enc_target(x)) > 0.1


def test_擴散損失是非負的且可微():
    x, obj, extra = _pair()
    y = x.clone().requires_grad_(True)
    loss = extra.diffusion(y)
    assert float(loss) >= 0.0
    loss.backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    assert float(y.grad.abs().sum()) > 0.0


def test_擴散損失用的是防禦圖自己的兩側():
    """兩側都換成防禦圖，所以換一張圖就會換一個值；否則這一項量不到載體。"""
    x, obj, extra = _pair()
    a = float(extra.diffusion(x))
    b = float(extra.diffusion((x * 0.7 + 0.15).clamp(0, 1)))
    assert abs(a - b) > 1e-9


def test_權重為零的項不計算():
    x, obj, extra = _pair()
    calls = {'n': 0}
    original = extra.diffusion

    def counted(v):
        calls['n'] += 1
        return original(v)

    extra.diffusion = counted
    combined = CombinedObjective(obj, extra,
                                 {'id': 1.0, 'enc': 0.5, 'cond': 1.0,
                                  'enc_target': 0.0, 'diffusion': 0.0})
    terms = combined.terms(x)
    assert calls['n'] == 0
    assert 'diffusion' not in terms and 'enc_target' not in terms


def test_兩個項的符號方向相反():
    """`enc_target` 越小越好、`diffusion` 越大越好，分數一律是要最小化的方向。"""
    x, obj, extra = _pair()
    w = {'id': 0.0, 'enc': 0.0, 'cond': 0.0,
         'enc_target': 1.0, 'diffusion': 1.0}
    combined = CombinedObjective(obj, extra, w)
    t = combined.terms(x)
    expect = (torch.tanh(t['enc_target']) - torch.tanh(t['diffusion']))
    assert float(combined.score(x)) == pytest.approx(float(expect), abs=1e-9)


def test_合成目標仍然接得上求解器():
    from src.defense.geometry_field import FlowFieldParam
    from src.defense.immunise import optimise_carrier

    x, obj, extra = _pair(size=24)
    combined = CombinedObjective(obj, extra,
                                 {'id': 1.0, 'enc': 0.5, 'cond': 1.0,
                                  'enc_target': 1.0, 'diffusion': 1.0})
    carrier = FlowFieldParam(x, grid=4, taper=4.0)
    carrier.reset(x)
    stats = optimise_carrier(carrier, x, combined, steps=6, lr=0.3,
                             caps=[], check_every=3)
    assert stats['free_score_end'] < stats['free_score_start']
    assert 'free_term_enc_target' in stats
    assert 'free_term_diffusion' in stats


def test_合成目標不碰文字編碼器以外的任何指令():
    """`FakePipe._encode_prompt` 對非空字串直接拋錯，這裡要能建起來就代表沒有指令。"""
    x, obj, extra = _pair()
    combined = CombinedObjective(obj, extra, {'enc_target': 1.0})
    assert combined.timesteps == obj.timesteps
    assert float(combined.score(x)) == pytest.approx(
        float(torch.tanh(extra.enc_target(x))), abs=1e-9)
