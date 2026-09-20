"""釘住 LiftedCurveParam：輸出不溢出、單調、而且真的到得了父族到不了的地方。

最後一條是這一族存在的理由：`ColorCurveParam` 恆滿足 F(0)=0、F(1)=1，
抬黑位那一類操作在它的可行域外。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.color_param import ColorCurveParam  # noqa: E402
from src.defense.lifted_curve_param import LiftedCurveParam  # noqa: E402

K = 64


def image(seed: int = 0, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 8, 8), generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def carrier(seed: int = 0, jitter: float = 1.0, lift=0.25, drop=0.25):
    c = LiftedCurveParam(radius=5.0, pieces=K, lift_max=lift, drop_max=drop,
                         init_jitter=jitter)
    c.reset(image(seed=0), seed=seed)
    return c


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_output_stays_in_range_without_clamping(seed):
    c = carrier(seed)
    y = c.render(image(seed=seed + 50))
    assert float(y.min()) >= -1e-6 and float(y.max()) <= 1.0 + 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_curves_stay_monotone(seed):
    c = carrier(seed)
    q = torch.linspace(0, 1, 513).view(1, 1, 1, -1).expand(1, 3, 1, 513).contiguous()
    r = c.render(q)[0, :, 0]
    assert bool((r[:, 1:] - r[:, :-1] >= -1e-6).all())


def test_identity_at_zero_jitter():
    c = carrier(jitter=0.0)
    x = image(seed=7)
    assert float((c.render(x) - x).abs().max()) < 1e-6


def test_endpoints_move_where_the_parent_cannot():
    """父族的 0 與 1 兩端釘死，本族要能把它們挪開。"""
    black = torch.zeros((1, 3, 2, 2))
    white = torch.ones((1, 3, 2, 2))
    parent = ColorCurveParam(radius=5.0, pieces=K, bound_mode="advcf",
                             init_jitter=1.0)
    parent.reset(black, seed=3)
    assert float(parent.render(black).abs().max()) < 1e-6
    assert float((parent.render(white) - white).abs().max()) < 1e-5

    c = carrier(seed=3)
    with torch.no_grad():
        c.u_lift.fill_(1.0)
        c.u_drop.fill_(1.0)
    assert float(c.render(black).min()) == pytest.approx(0.25, abs=1e-5)
    assert float(c.render(white).max()) == pytest.approx(0.75, abs=1e-5)


def test_identity_start_is_not_a_zero_gradient_point():
    """恆等起點上 lift 與 drop 都要收得到非零梯度。"""
    c = carrier(jitter=0.0)
    x = image(seed=11)
    c.render(x).square().mean().backward()
    for name, p in (("theta", c.theta), ("u_lift", c.u_lift), ("u_drop", c.u_drop)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0, f"{name} 在恆等起點上梯度是零"


def test_black_level_can_carry_colour():
    c = carrier(jitter=0.0)
    with torch.no_grad():
        c.u_lift[0, 0].fill_(1.0)
    black = torch.zeros((1, 3, 2, 2))
    y = c.render(black)
    assert float(y[0, 0].max()) == pytest.approx(0.25, abs=1e-5)
    assert float(y[0, 1].max()) == pytest.approx(0.0, abs=1e-6)


def test_construction_rejects_an_impossible_budget():
    with pytest.raises(ValueError):
        LiftedCurveParam(lift_max=0.6, drop_max=0.5)
    with pytest.raises(ValueError):
        LiftedCurveParam(lift_max=-0.1, drop_max=0.1)


def test_state_round_trip():
    c = carrier(seed=2)
    x = image(seed=13)
    before = c.render(x).detach().clone()
    other = carrier(seed=9)
    other.load_state_dict(c.state_dict())
    assert float((other.render(x) - before).abs().max()) < 1e-7


def test_levels_report_matches_the_parameters():
    c = carrier(jitter=0.0, lift=0.2, drop=0.1)
    with torch.no_grad():
        c.u_lift.fill_(0.5)
        c.u_drop.fill_(1.0)
    got = c.levels()
    assert got["lift"] == [0.1, 0.1, 0.1] and got["drop"] == [0.1, 0.1, 0.1]
