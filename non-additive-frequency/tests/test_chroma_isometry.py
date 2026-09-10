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
