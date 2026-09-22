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


def test_support_weighted_delta_e_ignores_pixels_outside_the_support():
    """支撐加權只看支撐內；支撐外改多少都不影響。"""
    x = _image()
    y = x.clone()
    y[:, :, 32:] = 0.0
    support = torch.zeros(1, 1, 64, 64)
    support[:, :, :32] = 1.0
    assert delta_e00(x, y, support) < 1e-6
    assert delta_e00(x, y) > 1.0


def test_support_weighting_raises_on_an_empty_support():
    x = _image()
    try:
        delta_e00(x, x, torch.zeros(1, 1, 64, 64))
    except ValueError as e:
        assert 'support' in str(e)
    else:
        raise AssertionError('空支撐必須拋錯，回傳 0/0 是靜默失效')


def test_solve_uses_the_support_weighted_anchor_when_given_one():
    """同一個目標在支撐加權下解出的幅度**較小**——全圖平均的稀釋被拿掉了。"""
    x = _image()
    support = torch.zeros(1, 1, 64, 64)
    support[:, :, :16] = 1.0
    p = ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=support, radius=0.2, max_gain=1.0)
    p.reset(x, seed=0)
    whole = solve_amplitude(p, x, 3.0)
    p.reset(x, seed=0)
    weighted = solve_amplitude(p, x, 3.0, support=support)
    assert weighted['amplitude'] < whole['amplitude']


def test_solve_rotation_requires_the_isometric_arm():
    """非等距臂會再過一次奇異值上界，解出來的角度不會逐字生效。"""
    from src.defense.color_amplitude import solve_rotation
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    try:
        solve_rotation(p, x)
    except ValueError as e:
        assert '等距臂' in str(e)
    else:
        raise AssertionError('非等距臂必須拋錯')


def test_solve_rotation_returns_an_angle_within_the_limit():
    from src.defense.color_amplitude import solve_rotation
    from src.defense.lowfreq_color import ChromaAffineParam, highfreq_report
    x = _image()
    p = ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.2, max_gain=1.0,
        isometric=True, rotation_deg=90.0)
    p.reset(x, seed=0)
    out = solve_rotation(p, x, limit=1.0)
    assert 0.0 <= out['rotation_deg'] <= 180.0
    assert 'monotone' in out
    if out['reached_limit']:
        p.rotation_deg = out['rotation_deg']
        p.reset(x, seed=0)
        assert highfreq_report(x, p.render(x))['hf_ratio_rgb_total'] <= 1.0 + 1e-6
