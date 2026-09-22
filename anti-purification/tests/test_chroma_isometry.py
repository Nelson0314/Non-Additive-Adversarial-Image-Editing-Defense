import torch

from src.defense.lowfreq_color import ChromaAffineParam, _project_orthogonal
from src.defense.ncf_param import rgb_to_lab


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param(isometric, support=None, amplitude=1.0):
    if support is None:
        support = torch.ones(1, 1, 64, 64)
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=support, radius=0.2, max_gain=1.0,
        amplitude=amplitude, isometric=isometric)


def _chroma_spread(x01):
    """色度的標準差，兩個通道合起來。去飽和會把它壓下去。"""
    ab = rgb_to_lab(x01)[:, 1:]
    return float(ab.reshape(2, -1).std(dim=1).norm())


def test_projection_returns_singular_values_of_one():
    m = torch.tensor([[2.0, 0.3], [-0.1, 0.4]], dtype=torch.float64)
    s = torch.linalg.svdvals(_project_orthogonal(m))
    assert torch.allclose(s, torch.ones(2, dtype=torch.float64), atol=1e-9)


def test_isometric_arm_has_unit_gain():
    x = _image()
    p = _param(True)
    p.reset(x, seed=0)
    s = torch.linalg.svdvals(p.chroma_matrix())
    assert torch.allclose(s, torch.ones(2, dtype=s.dtype), atol=1e-9)


def test_isometric_arm_adds_no_high_frequency():
    x = _image()
    p = _param(True)
    p.reset(x, seed=0)
    assert p.diagnostics(x)['hf_ratio_rgb_total'] <= 1.0


def test_isometric_arm_keeps_the_chroma_spread_the_bounded_arm_loses():
    """等距臂不去飽和；奇異值上界那一臂會。

    這是等距臂存在的理由：`_clamp_singular_values` 只設上界，起點的 `T0_ab`
    常常是收縮映射，色度被壓扁，付了顏色位移卻換到比較小的色度變化。
    """
    x = _image()
    iso, bounded = _param(True), _param(False)
    iso.reset(x, seed=0)
    bounded.reset(x, seed=0)
    base = _chroma_spread(x)
    assert _chroma_spread(iso.render(x)) > _chroma_spread(bounded.render(x))
    assert _chroma_spread(bounded.render(x)) < base


def test_support_zero_pixels_are_bit_exact():
    support = torch.zeros(1, 1, 64, 64)
    support[:, :, :32] = 1.0
    x = _image()
    p = _param(True, support=support)
    p.reset(x, seed=0)
    out = p.render(x)
    assert torch.equal(out[:, :, 32:], x[:, :, 32:])


def test_isometric_flag_appears_in_diagnostics():
    x = _image()
    p = _param(True)
    p.reset(x, seed=0)
    assert p.diagnostics(x)['isometric'] == 1


def test_clamping_equal_singular_values_does_not_poison_the_gradient():
    """夾取會製造重根，而 SVD 的反向傳播在重根上除以零。

    兩個奇異值都超過上界時 `clamp(max=g)` 把它們夾成**完全相等**的 g；
    SVD 的 backward 含 `1/(s_i^2 - s_j^2)`，於是第一次 backward 就是 nan。
    實測 `RegionPaletteParam` 的一塊區域被夾成 1.000000/1.000000 之後，
    碰撞目標由 87.66 直接變 nan。約束因此搬到 `project()`，前向不做 SVD。
    """
    from src.defense.lowfreq_color import RegionPaletteParam
    x = _image()
    r = torch.zeros(1, 1, 64, 64)
    r[:, :, 16:48, 8:32] = 1.0
    # blur_sigma 要小於影像尺寸，否則高斯核比圖還寬。
    regions = [(r, [60.0, 25.0, -20.0],
                [[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]]),
               ((1 - r).clamp(0, 1), [45.0, -18.0, 30.0],
                [[60.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]])]
    p = RegionPaletteParam(regions, support=torch.ones(1, 1, 64, 64),
                           blur_sigma=6.0, radius=0.2, epsilon_lab=5,
                           max_gain=1.0, gamut='soft')
    p.reset(x, seed=0)
    p.render(x).pow(2).mean().backward()
    g = p.delta.grad
    assert g is not None
    assert not bool(torch.isnan(g).any()), '前向做 SVD 會讓重根的梯度變成 nan'


def test_projection_enforces_the_gain_bound_after_a_step():
    """約束搬到 `project()` 之後，每一步之後仍然成立。"""
    from src.defense.lowfreq_color import ChromaAffineParam
    x = _image()
    p = ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.5, max_gain=1.0)
    p.reset(x, seed=0)
    with torch.no_grad():
        p.delta.add_(torch.full_like(p.delta, 0.4))
    p.project()
    s = torch.linalg.svdvals(p.chroma_matrix())
    assert float(s.max()) <= 1.0 + 1e-6


def test_untrained_render_is_unchanged_by_moving_the_projection():
    """delta 為零時前向等同投影後的值——未訓練的臂逐位元不變。"""
    from src.defense.lowfreq_color import ChromaAffineParam, _clamp_singular_values
    x = _image()
    p = ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.2, max_gain=1.0)
    p.reset(x, seed=0)
    assert torch.allclose(p.chroma_matrix(),
                          _clamp_singular_values(p.chroma_matrix(), 1.0),
                          atol=1e-9)
