import math

import pytest
import torch

from src.defense.print_patch import (ARCHETYPES, BASIS_COUNT,
                                     GeometricPrintParam, archetype_weights,
                                     basis_bank, place_shapes, support_frame)


def _support(h=96, w=96, y0=20, y1=76, x0=16, x1=80):
    s = torch.zeros(1, 1, h, w)
    s[0, 0, y0:y1, x0:x1] = 1.0
    return s


def _image(h=96, w=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, h, w), generator=g) * 0.4 + 0.3


def test_basis_bank_shape_and_range():
    b = basis_bank((32, 40), (16.0, 20.0), 16.0, torch.device('cpu'),
                   torch.float32)
    assert b.shape == (1, BASIS_COUNT, 32, 40)
    assert float(b.min()) >= -1.0001 and float(b.max()) <= 1.0001


def test_basis_last_field_is_constant():
    b = basis_bank((16, 16), (8.0, 8.0), 8.0, torch.device('cpu'),
                   torch.float32)
    assert torch.allclose(b[0, -1], torch.ones(16, 16))


def test_support_frame_centre_and_scale():
    centre, scale = support_frame(_support())
    assert centre == pytest.approx((47.5, 47.5), abs=0.5)
    assert scale == pytest.approx(0.5 * 63, abs=0.5)


def test_support_frame_rejects_empty():
    with pytest.raises(ValueError, match='支撐全為零'):
        support_frame(torch.zeros(1, 1, 8, 8))


@pytest.mark.parametrize('kind', [k for k in ARCHETYPES if k != 'random'])
@pytest.mark.parametrize('colours', [2, 3, 4])
def test_every_colour_is_used(kind, colours):
    """單一相位的線性場只有兩端勝得出，K−2 個色塊會永遠是零面積。

    正交相位對修掉的就是這件事，所以這是回歸測試而不是裝飾。
    """
    sup = _support()
    x = _image()
    carrier = GeometricPrintParam(sup, colours=colours, archetype=kind)
    carrier.reset(x, 0)
    assert carrier.readout()['print_share_min'] > 0.01


def test_random_archetype_requires_generator():
    with pytest.raises(ValueError, match='generator'):
        archetype_weights('random', 3, None)


def test_unknown_archetype_rejected():
    with pytest.raises(ValueError, match='未知的原型'):
        archetype_weights('paisley', 3, torch.Generator())


def test_outside_support_is_bit_exact():
    sup = _support()
    x = _image()
    carrier = GeometricPrintParam(sup, colours=3)
    carrier.reset(x, 0)
    y = carrier.render(x)
    outside = sup.expand_as(x) <= 0.0
    assert torch.equal(y[outside], x[outside])


def test_chroma_cap_is_structural():
    """彩度上限不是懲罰項：把參數推到極端仍然越不過去。"""
    sup = _support()
    x = _image()
    carrier = GeometricPrintParam(sup, colours=3, chroma_max=10.0)
    carrier.reset(x, 0)
    with torch.no_grad():
        carrier.pc.fill_(50.0)
    assert carrier.readout()['print_chroma_used'] <= 10.0 + 1e-3


def test_gradients_reach_every_parameter():
    sup = _support()
    x = _image()
    carrier = GeometricPrintParam(sup, colours=3)
    carrier.reset(x, 0)
    carrier.render(x).pow(2).mean().backward()
    for p in carrier.params():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0


def test_amplitude_zero_removes_the_pattern():
    sup = _support()
    x = _image()
    carrier = GeometricPrintParam(sup, colours=3)
    carrier.reset(x, 0)
    carrier.set_amplitude(0.0)
    with torch.no_grad():
        flat = carrier.pattern()
    spread = (flat.amax((0, 2, 3)) - flat.amin((0, 2, 3))).abs().max()
    assert float(spread) < 1e-5


def test_state_dict_roundtrip():
    sup = _support()
    x = _image()
    a = GeometricPrintParam(sup, colours=3)
    a.reset(x, 0)
    with torch.no_grad():
        a.pl.add_(0.4)
    state = a.state_dict()
    b = GeometricPrintParam(sup, colours=3)
    b.reset(x, 1)
    b.load_state_dict(state)
    with torch.no_grad():
        assert torch.allclose(a.render(x), b.render(x))
    assert all(p.requires_grad for p in b.params())


def test_keep_shading_preserves_the_fabric_shading():
    sup = _support()
    x = _image()
    lit = GeometricPrintParam(sup, colours=3, keep_shading=True)
    flat = GeometricPrintParam(sup, colours=3, keep_shading=False)
    lit.reset(x, 0)
    flat.reset(x, 0)
    with torch.no_grad():
        inside = (sup > 0.5).expand_as(x)
        var_lit = float(lit.render(x)[inside].var())
        var_flat = float(flat.render(x)[inside].var())
    assert var_lit > var_flat


@pytest.mark.parametrize('bad,message', [
    (dict(colours=1), '色盤至少兩色'),
    (dict(tau=0.0), 'tau 必須為正'),
    (dict(chroma_max=-1.0), 'chroma_max 必須為正'),
    (dict(archetype='paisley'), 'archetype'),
    (dict(shade_radius=0), 'shade_radius'),
    (dict(lightness_span=0.0), 'lightness_span'),
])
def test_constructor_rejects_bad_settings(bad, message):
    with pytest.raises(ValueError, match=message):
        GeometricPrintParam(_support(), **bad)


def test_reset_rejects_wrong_lightness_count():
    carrier = GeometricPrintParam(_support(), colours=3,
                                  lightness_init=[10.0, 50.0])
    with pytest.raises(ValueError, match='lightness_init'):
        carrier.reset(_image(), 0)


def test_place_shapes_gives_a_round_disc():
    mask = torch.zeros(1, 1, 128, 128)
    mask[0, 0, 20:108, 20:108] = 1.0
    sup = place_shapes(mask, shape='disc', area=0.05, count=1, feather=0)
    hard = (sup > 0.5).float()
    got = float(hard.mean())
    assert got == pytest.approx(0.05, rel=0.05)
    ys, xs = torch.nonzero(hard[0, 0], as_tuple=True)
    cy, cx = float(ys.float().mean()), float(xs.float().mean())
    radius = math.sqrt(0.05 * 128 * 128 / math.pi)
    d = ((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).sqrt()
    assert float(d.max()) <= radius + 1.5


def test_place_shapes_square_is_axis_aligned():
    mask = torch.zeros(1, 1, 128, 128)
    mask[0, 0, 20:108, 20:108] = 1.0
    sup = place_shapes(mask, shape='square', area=0.05, count=1, feather=0)
    hard = (sup > 0.5).float()[0, 0]
    rows = hard.sum(1)
    filled = rows[rows > 0]
    assert float(filled.min()) == float(filled.max())


def test_place_shapes_refuses_when_it_does_not_fit():
    mask = torch.zeros(1, 1, 128, 128)
    mask[0, 0, 60:68, 10:118] = 1.0
    with pytest.raises(ValueError, match='內接半徑|放不下'):
        place_shapes(mask, shape='disc', area=0.05, count=1)


def test_place_shapes_keeps_shapes_apart():
    mask = torch.zeros(1, 1, 128, 128)
    mask[0, 0, 10:118, 10:118] = 1.0
    sup = place_shapes(mask, shape='disc', area=0.04, count=2, feather=0)
    hard = (sup > 0.5).float()
    assert float(hard.mean()) == pytest.approx(0.04, rel=0.08)


def test_place_shapes_rejects_unknown_shape():
    mask = torch.ones(1, 1, 32, 32)
    with pytest.raises(ValueError, match='shape 要是'):
        place_shapes(mask, shape='triangle', area=0.1)
