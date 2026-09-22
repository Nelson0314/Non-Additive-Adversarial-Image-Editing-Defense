import pytest
import torch

from src.defense.patch_canvas import PatchCanvasParam


def _image(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=g) * 0.5 + 0.25
    return (base + torch.linspace(0, 0.3, size).view(1, 1, size, 1)).clamp(0, 1)


def _support(size=64, lo=16, hi=48):
    m = torch.zeros(1, 1, size, size)
    m[..., lo:hi, lo:hi] = 1.0
    return m


def _pattern(seed=5, size=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g).clamp(0, 1)


def _param(x, sup, **kw):
    kw.setdefault('canvas', _pattern())
    kw.setdefault('canvas_size', 32)
    p = PatchCanvasParam(sup, **kw)
    p.reset(x, seed=0)
    return p


def test_outside_the_support_is_bit_exact():
    x, sup = _image(), _support()
    y = _param(x, sup).render(x)
    outside = sup.expand_as(x) <= 0
    assert float((y - x).detach()[outside].abs().max()) == 0.0
    assert float((y - x).detach().abs().max()) > 0.0


def test_canvas_stays_in_range_under_both_parameterisations():
    x, sup = _image(), _support()
    bound = _param(x, sup, epsilon=0.3)
    free = _param(x, sup, canvas=None, epsilon=None)
    for p in (bound, free):
        p.u.data.mul_(50.0)
        c = p.canvas().detach()
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0


def test_epsilon_bounds_the_drift_from_the_pattern():
    x, sup = _image(), _support()
    p = _param(x, sup, epsilon=0.2)
    p.u.data.fill_(100.0)
    assert p.drift() <= 0.2 + 1e-6
    q = _param(x, sup, epsilon=0.05)
    q.u.data.fill_(100.0)
    assert q.drift() <= 0.05 + 1e-6


def test_starting_on_the_pattern_is_not_a_zero_gradient_point():
    """`d/du [ε·tanh(u)] = ε(1 − tanh²u)`，在 0 處是 ε，非零。"""
    x, sup = _image(), _support()
    p = _param(x, sup, epsilon=0.25, init_jitter=0.0)
    assert float(p.u.abs().max()) == 0.0
    assert p.drift() == pytest.approx(0.0, abs=1e-7)
    p.render(x).pow(2).sum().backward()
    assert float(p.u.grad.abs().max()) > 0.0


def test_epsilon_without_a_pattern_raises_instead_of_silently_going_free():
    x, sup = _image(), _support()
    p = PatchCanvasParam(sup, canvas=None, epsilon=0.3, canvas_size=32)
    with pytest.raises(ValueError):
        p.reset(x)


def test_keep_shading_preserves_the_fabric_detail():
    x, sup = _image(), _support()
    on = _param(x, sup, keep_shading=True).render(x).detach()
    off = _param(x, sup, keep_shading=False).render(x).detach()
    m = (sup > 0.5).expand_as(x)

    def hi(t):
        pad = torch.nn.functional.pad(t, (12,) * 4, mode='reflect')
        return (t - torch.nn.functional.avg_pool2d(pad, 25, 1))[m]
    dx = hi(x)
    assert float(torch.corrcoef(torch.stack([dx, hi(on)]))[0, 1]) > \
        float(torch.corrcoef(torch.stack([dx, hi(off)]))[0, 1])


def test_amplitude_returns_the_canvas_to_the_pattern():
    x, sup = _image(), _support()
    p = _param(x, sup, epsilon=0.3)
    p.u.data.fill_(2.0)
    assert p.drift() > 0.1
    p.set_amplitude(0.0)
    assert p.drift() == pytest.approx(0.0, abs=1e-7)
    p.set_amplitude([1.0])
    assert p.drift() > 0.1


def test_canvas_size_sets_the_feature_scale():
    x, sup = _image(), _support()

    def tv(p):
        c = torch.nn.functional.interpolate(p.canvas(), size=(64, 64),
                                            mode='bicubic', align_corners=False)
        return float(c.diff(dim=-1).abs().mean() + c.diff(dim=-2).abs().mean())
    coarse = _param(x, sup, canvas=_pattern(size=8), canvas_size=8)
    fine = _param(x, sup, canvas=_pattern(size=48), canvas_size=48)
    assert tv(coarse) < tv(fine)


def test_state_round_trip_reproduces_the_render():
    x, sup = _image(), _support()
    p = _param(x, sup)
    p.u.data.normal_()
    y = p.render(x).detach()
    q = _param(x, sup)
    q.load_state_dict(p.state_dict())
    assert float((q.render(x).detach() - y).abs().max()) == 0.0


def test_bad_configuration_raises_instead_of_being_accepted():
    sup = _support()
    with pytest.raises(ValueError):
        PatchCanvasParam(torch.zeros(1, 1, 64, 64))
    with pytest.raises(ValueError):
        PatchCanvasParam(sup, canvas_size=2)
    with pytest.raises(ValueError):
        PatchCanvasParam(sup, epsilon=0.0)
    with pytest.raises(ValueError):
        PatchCanvasParam(sup, epsilon=1.5)
    with pytest.raises(ValueError):
        PatchCanvasParam(sup, canvas=torch.rand(1, 3, 8, 8) * 3)
    with pytest.raises(ValueError):
        PatchCanvasParam(sup[0])
