import pytest
import torch

from src.defense.material_patch import (MaterialPatchParam, garment_palette)


def _image(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=g) * 0.5 + 0.25
    ramp = torch.linspace(0.0, 0.3, size).view(1, 1, size, 1)
    return (base + ramp).clamp(0, 1)


def _support(size=64, lo=16, hi=40):
    m = torch.zeros(1, 1, size, size)
    m[..., lo:hi, lo:hi] = 1.0
    return m


def _param(x, sup, **kw):
    pal = garment_palette(x, sup, kw.pop('count', 5))
    p = MaterialPatchParam(sup, pal, **kw)
    p.reset(x, seed=0)
    return p


def test_outside_the_support_is_bit_exact():
    x, sup = _image(), _support()
    y = _param(x, sup).render(x)
    outside = sup.expand_as(x) <= 0
    assert float((y - x).detach()[outside].abs().max()) == 0.0
    assert float((y - x).abs().max()) > 0.0


def test_every_patch_pixel_stays_in_the_palette_convex_hull():
    """`softmax` 的權重非負且和為 1，所以反射色落在色盤的凸包內。"""
    x, sup = _image(), _support()
    p = _param(x, sup)
    rho = p.reflectance(x.shape[-2:])
    lo = p.palette.min(0).values.view(1, 3, 1, 1)
    hi = p.palette.max(0).values.view(1, 3, 1, 1)
    assert float((rho - lo).min()) >= -1e-6
    assert float((hi - rho).min()) >= -1e-6


def test_shading_and_detail_are_frozen_by_reset():
    x, sup = _image(), _support()
    p = _param(x, sup)
    before = (p.shade.clone(), p.detail.clone())
    p.theta.data.add_(torch.randn_like(p.theta))
    p.render(x)
    assert torch.equal(p.shade, before[0])
    assert torch.equal(p.detail, before[1])


def test_the_frozen_detail_survives_into_the_patch():
    """換掉的只有顏色，布料的織紋要留著。"""
    x, sup = _image(), _support()
    p = _param(x, sup)
    y = p.render(x)
    m = (sup > 0.5).expand_as(x)
    dx = (x - torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x, (12,) * 4, mode='reflect'), 25, 1))[m]
    dy = (y - torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(y, (12,) * 4, mode='reflect'), 25, 1))[m]
    corr = torch.corrcoef(torch.stack([dx, dy]))[0, 1]
    assert float(corr) > 0.5


def test_uniform_start_is_not_a_zero_gradient_point():
    x, sup = _image(), _support()
    p = _param(x, sup, init_jitter=0.0)
    assert float(p.theta.abs().max()) == 0.0
    p.render(x).pow(2).sum().backward()
    assert float(p.theta.grad.abs().max()) > 0.0


def test_project_caps_the_logits_so_edges_cannot_go_hard():
    x, sup = _image(), _support()
    p = _param(x, sup, logit_cap=2.0)
    p.theta.data.mul_(100.0)
    p.project()
    assert float(p.theta.abs().max()) <= 2.0 + 1e-6


def test_grid_sets_the_spatial_scale_not_the_optimiser():
    """粗網格的反射色場必須比細網格平滑。"""
    x, sup = _image(), _support()
    rough = _param(x, sup, grid=4).reflectance(x.shape[-2:])
    fine = _param(x, sup, grid=32).reflectance(x.shape[-2:])

    def tv(t):
        return float((t.diff(dim=-1).abs().mean() + t.diff(dim=-2).abs().mean()))
    assert tv(rough) < tv(fine)


def test_different_seeds_start_in_different_places():
    x, sup = _image(), _support()
    pal = garment_palette(x, sup, 5)
    a = MaterialPatchParam(sup, pal); a.reset(x, seed=1)
    b = MaterialPatchParam(sup, pal); b.reset(x, seed=2)
    assert not torch.allclose(a.theta, b.theta)
    assert float((a.render(x) - b.render(x)).abs().max()) > 0.0


def test_state_round_trip_reproduces_the_render():
    x, sup = _image(), _support()
    p = _param(x, sup)
    y = p.render(x)
    q = MaterialPatchParam(sup, p.palette)
    q.reset(x, seed=99)
    q.load_state_dict(p.state_dict())
    assert float((q.render(x) - y).abs().max()) == 0.0


def test_palette_spans_the_garment_and_is_ordered_by_its_own_spread():
    x, sup = _image(), _support()
    pal = garment_palette(x, sup, 6)
    assert pal.shape == (6, 3)
    px = x[0].permute(1, 2, 0)[(sup > 0.5)[0, 0]]
    assert float(pal.min()) >= float(px.min()) - 1e-6
    assert float(pal.max()) <= float(px.max()) + 1e-6
    assert float((pal[-1] - pal[0]).abs().max()) > 0.0


def test_chroma_gain_pushes_the_palette_out_of_the_garment_gamut():
    x, sup = _image(), _support()
    plain = garment_palette(x, sup, 5, chroma_gain=1.0)
    wide = garment_palette(x, sup, 5, chroma_gain=2.5)
    assert not torch.allclose(plain, wide)
    assert float((wide.max(0).values - wide.min(0).values).sum()) \
        > float((plain.max(0).values - plain.min(0).values).sum())


def test_render_before_reset_raises():
    x, sup = _image(), _support()
    p = MaterialPatchParam(sup, garment_palette(x, sup, 4))
    with pytest.raises(ValueError):
        p.render(x)


def test_bad_configuration_raises_instead_of_being_accepted():
    x, sup = _image(), _support()
    pal = garment_palette(x, sup, 4)
    with pytest.raises(ValueError):
        MaterialPatchParam(torch.zeros(1, 1, 64, 64), pal)
    with pytest.raises(ValueError):
        MaterialPatchParam(sup, pal, grid=1)
    with pytest.raises(ValueError):
        MaterialPatchParam(sup, pal, logit_cap=0.0)
    with pytest.raises(ValueError):
        MaterialPatchParam(sup, pal[:, :2])
    with pytest.raises(ValueError):
        MaterialPatchParam(sup[0], pal)
    with pytest.raises(ValueError):
        garment_palette(x, sup, 1)
    with pytest.raises(ValueError):
        garment_palette(x, torch.zeros(1, 1, 64, 64), 4)


def test_print_mode_rescues_a_garment_with_no_colour_of_its_own():
    """白襯衫的色域幾乎是一個點，`chroma_gain` 乘不出振幅；`print` 模式可以。"""
    g = torch.Generator().manual_seed(3)
    white = (0.88 + 0.06 * torch.rand(1, 3, 64, 64, generator=g)).clamp(0, 1)
    sup = _support()
    gain = garment_palette(white, sup, 5, chroma_gain=3.0)
    pr = garment_palette(white, sup, 5, mode='print', chroma=40.0)

    def spread(p):
        return float((p.max(0).values - p.min(0).values).sum())
    assert spread(pr) > 2.5 * spread(gain)
    assert float(pr.min()) >= 0.0 and float(pr.max()) <= 1.0


def test_print_mode_keeps_the_garment_luminance_profile():
    """彩度先擬進 sRGB 色域再轉，亮度才留得住。

    直接設成半徑 40 再 clip 的話，最亮與最暗那兩個條目的亮度會被裁掉 0.52。
    """
    x, sup = _image(), _support()
    pal = garment_palette(x, sup, 5)
    pr = garment_palette(x, sup, 5, mode='print')
    from src.defense.color_param import luma
    a = luma(pal.t().reshape(1, 3, 5, 1)).flatten()
    b = luma(pr.t().reshape(1, 3, 5, 1)).flatten()
    assert float((a - b).abs().max()) < 0.15


def test_print_chroma_is_a_ceiling_not_an_exact_radius():
    """亮度極端處放不下滿彩度，寧可不足也不裁出色域。"""
    from skimage.color import rgb2lab
    x, sup = _image(), _support()
    pr = garment_palette(x, sup, 6, mode='print', chroma=60.0)
    lab = rgb2lab(pr.numpy()[None])
    c = (lab[0, :, 1] ** 2 + lab[0, :, 2] ** 2) ** 0.5
    assert c.max() <= 60.0 + 1e-3
    assert c.min() < 60.0 - 1e-3
    assert float(pr.min()) >= 0.0 and float(pr.max()) <= 1.0


def test_bad_palette_mode_raises():
    x, sup = _image(), _support()
    with pytest.raises(ValueError):
        garment_palette(x, sup, 4, mode='nope')
    with pytest.raises(ValueError):
        garment_palette(x, sup, 4, mode='print', chroma=0.0)


def test_amplitude_is_a_real_knob_not_a_bookkeeping_field():
    """`optimise_carrier` 還原 checkpoint 後會呼叫它；`fit_caps` 用它找可行點。"""
    x, sup = _image(), _support()
    p = _param(x, sup)
    full = p.render(x)
    p.set_amplitude(0.0)
    flat = p.render(x)
    assert float((full - flat).abs().max().detach()) > 0.0
    rho = p.reflectance(x.shape[-2:])
    assert float(rho.std(dim=(2, 3)).max()) < 1e-5
    p.set_amplitude([1.0])
    assert float((p.render(x) - full).abs().max().detach()) == 0.0


def test_amplitude_moves_the_patch_monotonically():
    x, sup = _image(), _support()
    p = _param(x, sup)
    m = (sup > 0.5).expand_as(x)
    p.set_amplitude(0.0)
    base = p.render(x).detach()
    prev = 0.0
    for a in (0.25, 0.5, 1.0):
        p.set_amplitude(a)
        d = float((p.render(x).detach() - base)[m].abs().mean())
        assert d > prev
        prev = d


def test_bad_amplitude_raises():
    x, sup = _image(), _support()
    p = _param(x, sup)
    with pytest.raises(ValueError):
        p.set_amplitude(-1.0)
    with pytest.raises(ValueError):
        p.set_amplitude([1.0, 2.0])


def test_palette_order_is_reproducible_across_calls():
    """`torch.pca_lowrank` 走隨機化 SVD，主方向的正負號是任意的。

    `print` 模式按索引配色相，號一翻就換成另一組色盤，整批不可重現。
    符號釘在「投影與亮度正相關」上之後，重複呼叫必須逐位元相同。
    """
    x, sup = _image(), _support()
    runs = [garment_palette(x, sup, 6, mode='print') for _ in range(5)]
    for r in runs[1:]:
        assert torch.equal(r, runs[0])
    plain = [garment_palette(x, sup, 6) for _ in range(5)]
    for r in plain[1:]:
        assert torch.equal(r, plain[0])


def test_palette_axis_points_the_same_way_as_luminance():
    """釘住的是**方向**不是逐項排序。

    沿第一主方向取分位數不保證亮度逐項遞增（主方向同時含色調變化），
    保證的是這條軸與亮度正相關，所以最暗與最亮兩端不會對調。
    """
    from src.defense.color_param import luma
    x, sup = _image(), _support()
    pal = garment_palette(x, sup, 6)
    lum = luma(pal.t().reshape(1, 3, 6, 1)).flatten()
    assert float(lum[-1]) > float(lum[0])
    idx = torch.arange(6, dtype=lum.dtype)
    assert float(torch.corrcoef(torch.stack([idx, lum]))[0, 1]) > 0.5
