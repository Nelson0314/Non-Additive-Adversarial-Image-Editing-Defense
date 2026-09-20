"""GradedCurveParam：兩個結構常數各自守住自己那一半。

`span` 管中間調的色相、`lift`／`drop` 管兩端的顏色，兩個都不是求解端動得到的。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.graded_curve import GradedCurveParam  # noqa: E402

K = 64


def image(seed: int = 0, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 8, 8), generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def carrier(span=0.25, lift=0.25, drop=0.25, seed=0, jitter=1.0):
    c = GradedCurveParam(radius=5.0, pieces=K, span=span,
                         lift_max=lift, drop_max=drop, init_jitter=jitter)
    c.reset(image(seed=0), seed=seed)
    return c


@pytest.mark.parametrize("span", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_output_stays_in_range_without_clamping(span, seed):
    c = carrier(span=span, seed=seed)
    y = c.render(image(seed=seed + 60))
    assert float(y.min()) >= -1e-6 and float(y.max()) <= 1.0 + 1e-6


@pytest.mark.parametrize("span", [0.0, 0.25, 1.0])
def test_curves_stay_monotone(span):
    for seed in range(3):
        c = carrier(span=span, seed=seed)
        q = torch.linspace(0, 1, 513).view(1, 1, 1, -1).expand(1, 3, 1, 513).contiguous()
        r = c.render(q)[0, :, 0]
        assert bool((r[:, 1:] - r[:, :-1] >= -1e-6).all()), f"span={span} seed={seed}"


def test_span_zero_leaves_the_midtones_neutral_in_hue():
    """span=0 時三條曲線相同，所以中間調的偏色只能來自 lift／drop。

    把 lift 與 drop 都關掉，同一個灰階輸入的三個通道輸出必須完全相同。
    """
    c = carrier(span=0.0, lift=0.0, drop=0.0, seed=4)
    q = torch.linspace(0, 1, 257).view(1, 1, 1, -1).expand(1, 3, 1, 257).contiguous()
    r = c.render(q)[0, :, 0]
    assert float((r[0] - r[1]).abs().max()) < 1e-6
    assert float((r[0] - r[2]).abs().max()) < 1e-6


def test_channel_gap_bound_still_holds():
    for span in (0.0, 0.2, 0.5):
        for seed in range(4):
            c = carrier(span=span, seed=seed)
            assert c.channel_gap() <= c.gap_bound() + 1e-9


def test_identity_at_zero_jitter():
    c = carrier(jitter=0.0)
    x = image(seed=71)
    assert float((c.render(x) - x).abs().max()) < 1e-6


def test_identity_start_is_not_a_zero_gradient_point():
    c = carrier(jitter=0.0)
    c.render(image(seed=73)).square().mean().backward()
    for name, p in (("u", c.u), ("d", c.d),
                    ("u_lift", c.u_lift), ("u_drop", c.u_drop)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0, f"{name} 在恆等起點上梯度是零"


def test_levels_reach_their_bounds():
    c = carrier(span=0.0, lift=0.2, drop=0.1, jitter=0.0)
    with torch.no_grad():
        c.u_lift.fill_(1.0)
        c.u_drop.fill_(1.0)
    black = torch.zeros((1, 3, 2, 2))
    white = torch.ones((1, 3, 2, 2))
    assert float(c.render(black).max()) == pytest.approx(0.2, abs=1e-5)
    assert float(c.render(white).max()) == pytest.approx(0.9, abs=1e-5)


def test_construction_rejects_an_impossible_budget():
    with pytest.raises(ValueError):
        GradedCurveParam(lift_max=0.7, drop_max=0.4)
    with pytest.raises(ValueError):
        GradedCurveParam(span=1.5)


def test_state_round_trip():
    c = carrier(seed=5)
    x = image(seed=77)
    before = c.render(x).detach().clone()
    other = carrier(seed=12)
    other.load_state_dict(c.state_dict())
    assert float((other.render(x) - before).abs().max()) < 1e-7
