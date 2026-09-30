import torch

from src.defense.lowfreq_color import ChromaAffineParam, soft_gamut


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param(amplitude, support=None, gamut='soft'):
    if support is None:
        support = torch.ones(1, 1, 64, 64)
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=support, radius=0.2, max_gain=1.0, amplitude=amplitude,
        gamut=gamut)


def test_amplitude_zero_is_identity_up_to_the_lab_round_trip():
    """幅度為零時色彩映射不動任何東西；殘差只有 RGB↔Lab 往返的浮點誤差。

    走 `gamut='clip'` 是為了把 `soft_gamut` 隔離掉——它在色域內就偏離恆等
    （見下一筆測試），與幅度無關。
    """
    x = _image()
    p = _param(0.0, gamut='clip')
    p.reset(x, seed=0)
    assert float((p.render(x) - x).abs().max()) < 1e-4


def test_amplitude_zero_on_the_soft_path_shows_only_the_gamut_softening():
    """`gamut='soft'` 的殘差全部來自 `soft_gamut`，不是來自載體。

    `soft_gamut` 的導數恆落在 [0,1]（不放大梯度），代價是它在色域**內**也
    偏離恆等：knee = 0.06 時黑點被抬 0.0416、白點被壓到 0.9584。所以在這條
    路徑上「幅度為零」不是逐位元恆等，而該偏離不是幅度造成的。這一筆把兩者
    釘開，避免以後把色調壓縮誤讀成載體的效果。
    """
    x = _image()
    p = _param(0.0, gamut='soft')
    p.reset(x, seed=0)
    residual = (p.render(x) - x).abs().max()
    gamut_only = (soft_gamut(x) - x).abs().max()
    assert torch.allclose(residual, gamut_only, atol=1e-4)
    assert float(gamut_only) > 1e-2


def test_amplitude_is_monotone_in_colour_displacement():
    x = _image()
    d = []
    for a in (0.25, 0.5, 1.0):
        p = _param(a)
        p.reset(x, seed=0)
        d.append(float((p.render(x) - x).abs().mean()))
    assert d[0] < d[1] < d[2]


def test_amplitude_preserves_the_gain_bound():
    x = _image()
    for a in (0.25, 0.5, 1.0):
        p = _param(a)
        p.reset(x, seed=0)
        s = torch.linalg.svdvals(p.effective_chroma_matrix())
        assert float(s.max()) <= 1.0 + 1e-9


def test_amplitude_appears_in_diagnostics():
    x = _image()
    p = _param(0.5)
    p.reset(x, seed=0)
    assert p.diagnostics(x)['amplitude'] == 0.5
