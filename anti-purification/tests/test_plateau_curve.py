import pytest
import torch

from src.defense.color_param import ColorCurveParam, luma
from src.defense.plateau_curve import (PlateauCurveParam, centres_from_luma,
                                       smoothstep01)


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param(x, hw=0.08, slope=0.0, **kw):
    p = PlateauCurveParam(centres_from_luma(x), **kw)
    p.reset(x, half_width=hw, slope=slope)
    p.project()
    return p


def test_identity_start_is_bit_exact():
    x = _image()
    p = PlateauCurveParam(centres_from_luma(x))
    p.reset(x)
    assert float((p.render(x) - x).abs().max().detach()) == 0.0


def test_zero_slope_is_a_feasible_point_not_a_limit():
    """這是本參數化存在的理由，見模組 docstring。"""
    x = _image()
    p = _param(x)
    assert float(p.slope_profile().min().detach()) == 0.0
    assert p.flat_coverage(x) > 0.0


def test_slope_floor_matches_the_colour_curve_bound_and_kills_the_plateau():
    """`ColorCurveParam` 的有效斜率下限是 `1/(1+radius)`，零覆蓋是它的必然結果。"""
    radius = 5.0
    floor = 1.0 / (1.0 + radius)
    ref = ColorCurveParam(radius=radius, bound_mode='advcf', pieces=64)
    lo, hi = ref.bounds()
    assert lo / hi == pytest.approx(floor, abs=1e-6)

    x = _image()
    p = _param(x, slope_floor=floor)
    assert float(p.slope_profile().min().detach()) == pytest.approx(floor, abs=1e-6)
    assert p.flat_coverage(x) == 0.0


def test_curve_is_monotone_and_pinned_at_both_ends():
    x = _image()
    p = _param(x, hw=0.13)
    f = p.curve(torch.linspace(0.0, 1.0, 2001))
    assert float(f.diff().min()) >= -1e-7
    assert float(f[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(f[-1]) == pytest.approx(1.0, abs=1e-5)


def test_normalisation_cannot_lift_the_plateau_off_zero():
    """`F = ∫m / ∫m` 是乘一個常數，0 乘任何常數仍是 0。"""
    x = _image()
    c = centres_from_luma(x)
    lo, hi = c[0] - 0.05, c[0] + 0.05
    p = _param(x, hw=0.05)
    a = p.curve(torch.tensor([lo + 1e-3]))
    b = p.curve(torch.tensor([hi - 1e-3]))
    assert float((a - b).abs().max()) < 1e-5


def test_identity_start_freezes_the_half_width():
    """`∂m/∂h` 帶因子 `(1−s)`，`s = 1` 時恰為零，所以恆等起點只動得了斜率。"""
    x = _image()
    p = PlateauCurveParam(centres_from_luma(x))
    p.reset(x)
    p.render(x).pow(2).sum().backward()
    assert float(p.half_width.grad.abs().max()) == 0.0
    assert float(p.slope.grad.abs().max()) > 0.0


def test_a_non_identity_slope_unfreezes_the_half_width():
    """避開上面那個點的方式：起點給 `slope < 1`，不需要 `init_jitter`。"""
    x = _image()
    p = PlateauCurveParam(centres_from_luma(x))
    p.reset(x, half_width=0.05, slope=0.5)
    p.render(x).pow(2).sum().backward()
    assert float(p.half_width.grad.abs().max()) > 0.0
    assert float(p.slope.grad.abs().max()) > 0.0


def test_mask_core_gets_the_curve_and_outside_is_bit_exact():
    x = _image()
    m = torch.zeros(1, 1, 64, 64)
    m[..., :32, :] = 1.0
    p = PlateauCurveParam(centres_from_luma(x), apply_where=m)
    p.reset(x, half_width=0.1, slope=0.0)
    y = p.render(x)
    assert float((y[..., 32:, :] - x[..., 32:, :]).abs().max()) == 0.0
    assert float((y[..., :32, :] - x[..., :32, :]).abs().max()) > 0.0

    bare = PlateauCurveParam(p.centres)
    bare.reset(x, half_width=0.1, slope=0.0)
    full = bare.render(x)
    assert float((y[..., :32, :] - full[..., :32, :]).abs().max()) == 0.0


def test_flat_coverage_ignores_plateaus_that_are_not_flat():
    x = _image()
    p = _param(x, hw=0.1, slope=0.3, slope_floor=0.2)
    assert p.flat_coverage(x) == 0.0


def test_coverage_rises_with_half_width():
    x = _image()
    got = [_param(x, hw=h).flat_coverage(x) for h in (0.02, 0.06, 0.13)]
    assert got[0] < got[1] < got[2]


def test_centres_follow_the_weighted_luminance_distribution():
    x = _image()
    m = torch.zeros(1, 1, 64, 64)
    m[..., :16, :] = 1.0
    c_all = centres_from_luma(x)
    c_top = centres_from_luma(x, weight=m)
    assert c_all != c_top
    lum = luma(x)[..., :16, :]
    assert float(lum.min()) <= c_top[0] <= float(lum.max())


def test_per_channel_mode_never_clips():
    x = _image()
    p = _param(x, hw=0.13, mode='per_channel')
    y = p.render(x)
    assert p.clip_fraction(x) == 0.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_smoothstep_is_clamped_at_both_ends():
    z = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.0])
    got = smoothstep01(z)
    assert got.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_state_round_trip_reproduces_the_render():
    x = _image()
    p = _param(x, hw=0.07)
    y = p.render(x)
    q = PlateauCurveParam(p.centres)
    q.reset(x)
    q.load_state_dict(p.state_dict())
    assert float((q.render(x) - y).abs().max()) == 0.0


def test_bad_configuration_raises_instead_of_being_accepted():
    x = _image()
    with pytest.raises(ValueError):
        PlateauCurveParam([])
    with pytest.raises(ValueError):
        PlateauCurveParam([0.5], mode='nope')
    with pytest.raises(ValueError):
        PlateauCurveParam([0.5], transition=0.0)
    with pytest.raises(ValueError):
        PlateauCurveParam([0.5], slope_floor=1.0)
    with pytest.raises(ValueError):
        PlateauCurveParam([0.5], pieces=4)
    with pytest.raises(ValueError):
        centres_from_luma(x, weight=torch.zeros(1, 1, 64, 64))
