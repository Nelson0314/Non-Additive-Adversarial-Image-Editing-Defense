import torch

from src.defense.color_amplitude import delta_e00, solve_amplitude
from src.defense.lowfreq_color import ChromaAffineParam


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param():
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.2, max_gain=1.0)


def test_delta_e00_of_identical_images_is_zero():
    x = _image()
    assert delta_e00(x, x) < 1e-6


def test_solve_hits_a_reachable_target():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    full = delta_e00(x, p.render(x))
    want = full * 0.5
    out = solve_amplitude(p, x, want)
    assert out['reached'] is True
    assert abs(out['delta_e00'] - want) < 0.2
    assert 0.0 < out['amplitude'] < 1.0


def test_solve_reports_an_unreachable_target_instead_of_pretending():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    full = delta_e00(x, p.render(x))
    out = solve_amplitude(p, x, full * 10.0)
    assert out['reached'] is False
    assert out['amplitude'] == 1.0
    assert abs(out['delta_e00'] - full) < 1e-6


def test_solve_leaves_the_param_at_the_solved_amplitude():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    out = solve_amplitude(p, x, delta_e00(x, p.render(x)) * 0.5)
    assert p.amplitude == out['amplitude']


def test_solve_rejects_a_nonpositive_target():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    try:
        solve_amplitude(p, x, 0.0)
    except ValueError as e:
        assert 'target_delta_e' in str(e)
    else:
        raise AssertionError('非正的目標色差必須拋錯')
