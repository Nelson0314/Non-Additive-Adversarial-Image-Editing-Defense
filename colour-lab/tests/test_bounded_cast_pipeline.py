"""以 CPU 釘住 factory、設定檔、起點梯度與 cap 二分的相容性。"""

import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_baseline import COLUMNS, build_carrier, load_guard
from src.defense import bounded_cast_curve as bc
from src.defense.immunise import Cap, cap_violations, fit_caps, quantise

CONFIG = json.loads((ROOT / 'configs/bounded_cast.json').read_text(encoding='utf-8'))
ARMS = [v for v in CONFIG['variants'] if v['carrier'] == 'bounded_cast']


@pytest.mark.parametrize('arm', ARMS, ids=lambda arm: arm['name'])
def test_every_config_bound_pair_constructs(arm):
    """僅建構物件，不 reset、不 render，也不執行 optimisation。"""
    carrier = bc.BoundedCastCurveParam(**arm['options'])
    assert carrier.offset_radius > 0
    assert carrier.theta_l is carrier.theta_a is carrier.theta_b is None


def test_config_keeps_operating_point_controls_and_weights():
    load_guard()(CONFIG)
    assert CONFIG['free']['weights'] == {'id': 1.0, 'enc': 0.5, 'cond': 0.0, 'sds': 1.0}
    assert CONFIG['free']['use_identity'] is True
    controls = [v for v in CONFIG['variants'] if v['carrier'] == 'advcf']
    assert {v['options']['init_jitter'] for v in controls} == {0.0, 0.25}
    assert len(ARMS) in (2, 3)
    assert all(v['deltae_cap'] == 16.0 for v in CONFIG['variants'])
    assert {'C_max', 'd_max'} <= set(COLUMNS)


@pytest.mark.parametrize('opts', [
    {},
    {'C_max': 24, 'd_max': 38, 'pieces': 12, 'skin_width': 7, 'init_jitter': 0.4},
])
def test_factory_builds_usable_bounded_cast_and_round_trips(opts):
    x = torch.rand((1, 3, 4, 5), generator=torch.Generator().manual_seed(8))
    carrier = build_carrier('bounded_cast', x, opts, CONFIG, 'cpu')
    assert isinstance(carrier, bc.BoundedCastCurveParam)
    expected = dict(C_max=32.0, d_max=24.0, pieces=64, skin_width=8.0, init_jitter=0.25)
    expected.update(opts)
    assert all(getattr(carrier, key) == value for key, value in expected.items())
    carrier.reset(x, 11)
    original = carrier.render(x).detach()
    assert original.shape == x.shape and original.dtype == x.dtype
    assert torch.isfinite(original).all() and original.min() >= 0 and original.max() <= 1
    params = carrier.params()
    assert len(params) == 3 and all(p.is_leaf and p.requires_grad for p in params)
    assert all(p.device.type == 'cpu' for p in params)
    assert carrier.stages == [carrier] and carrier.step_scale() == 1.0
    assert carrier.bounds() == (-math.inf, math.inf)
    saved = [p.detach().clone() for p in params]
    carrier.project()
    assert all(torch.equal(p, old) for p, old in zip(params, saved))
    carrier.reset(x, 11)
    assert torch.equal(carrier.render(x), original)
    carrier.set_amplitude([0.4])
    rendered = carrier.render(x).detach()
    state = carrier.state_dict()
    carrier.reset(x, 12)
    carrier.load_state_dict(state)
    assert carrier.amplitude == 0.4
    assert torch.equal(carrier.render(x), rendered)
    assert all(p.is_leaf and p.requires_grad for p in carrier.params())


@pytest.mark.parametrize('arm', ARMS, ids=lambda arm: arm['name'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('zero_start', [True, False], ids=['zero', 'configured'])
def test_config_zero_and_actual_starts_reach_every_parameter_tensor(arm, dtype, zero_start):
    """以不對稱 RGB 損失測梯度，避免把 objective 自身的駐點誤判為參數退化。"""
    opts = dict(arm['options'])
    if zero_start:
        opts['init_jitter'] = 0.0
    x = torch.rand((2, 3, 12, 16), dtype=dtype,
                   generator=torch.Generator().manual_seed(101))
    carrier = build_carrier('bounded_cast', x, opts, CONFIG, 'cpu')
    carrier.reset(x, 0)
    if zero_start:
        assert all(torch.count_nonzero(p) == 0 for p in carrier.params())
    weights = x.new_tensor((0.7, 1.3, 2.1)).view(1, 3, 1, 1)
    (carrier.render(x).square() * weights).mean().backward()
    maxima = []
    for name, param in zip(('theta_l', 'theta_a', 'theta_b'), carrier.params()):
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        maxima.append(float(param.grad.abs().max()))
        assert maxima[-1] > 0, name
    print(arm['name'], str(dtype), f"jitter={opts['init_jitter']}",
          dict(zip(('theta_l', 'theta_a', 'theta_b'), maxima)))


def test_amplitude_scales_lab_offset_and_preserves_certificate_precision():
    x = torch.rand((1, 3, 4, 5), dtype=torch.float64,
                   generator=torch.Generator().manual_seed(9))
    carrier = build_carrier('bounded_cast', x, {}, CONFIG, 'cpu')
    carrier.reset(x, 2)
    prepared = carrier.prepare(x)
    assert prepared.base_lab.dtype == prepared.radius.dtype == torch.float64
    carrier.set_amplitude(1)
    full = bc.rgb_to_lab(carrier.render_prepared(prepared))
    carrier.set_amplitude([0.3])
    partial = bc.rgb_to_lab(carrier.render_prepared(prepared))
    assert torch.allclose(partial - prepared.base_lab,
                          0.3 * (full - prepared.base_lab), rtol=0, atol=2e-12)
    carrier.set_amplitude(0)
    assert torch.allclose(bc.rgb_to_lab(carrier.render(x)), prepared.base_lab,
                          rtol=0, atol=2e-12)
    assert not torch.allclose(carrier.render(x), x)


def test_cap_bisection_returns_a_feasible_amplitude_for_palette():
    x = torch.full((1, 3, 2, 3), 0.4, dtype=torch.float64)
    carrier = build_carrier('bounded_cast', x, {}, CONFIG, 'cpu')
    carrier.reset(x, 0)
    with torch.no_grad():
        carrier.theta_l.fill_(2)
        carrier.theta_a.zero_()
        carrier.theta_b.zero_()
        full = float(quantise(carrier.render(x)).mean())
        carrier.set_amplitude(0)
        base = float(quantise(carrier.render(x)).mean())
        assert base < full
        cap = Cap('mean', lambda y: y.mean(), lambda y: float(y.mean()), (base + full) / 2)
        carrier.set_amplitude(1)
        shrink = fit_caps(carrier, x, [cap])
        assert 0 < shrink < 1 and carrier.amplitude == shrink
        assert cap_violations(carrier, x, [cap]) == 0


def test_cap_bisection_rejects_infeasible_palette_base_and_restores_amplitude():
    x = torch.zeros((1, 3, 2, 3), dtype=torch.float64)
    x[:, 0] = 1
    carrier = build_carrier('bounded_cast', x, {}, CONFIG, 'cpu')
    carrier.reset(x, 0)
    carrier.set_amplitude(0.7)
    cap = Cap('mse', lambda y: (y - x).square().mean(),
              lambda y: float((y - x).square().mean()), 1e-4)
    with torch.no_grad(), pytest.raises(ValueError, match='amplitude=0'):
        fit_caps(carrier, x, [cap])
    assert carrier.amplitude == 0.7
