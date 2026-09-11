import torch

from src.defense.color_field import ColorFieldParam
from src.defense.lowfreq_color import ChromaAffineParam, highfreq_report


MEAN = [71.3, -5.9, 1.8]
COV = [[210.0, 4.0, -3.0], [4.0, 36.0, 2.0], [-3.0, 2.0, 30.0]]


def _image(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=g)
    ramp = torch.linspace(0, 1, size)[None, None, None, :].expand(1, 3, size, size)
    return (0.5 * base + 0.5 * ramp).clamp(0, 1)


def _full(x):
    return torch.ones_like(x[:, :1])


def test_單一控制點且重度模糊時退化成全域仿射():
    x = _image()
    a = ColorFieldParam(MEAN, COV, support=_full(x), grid=1, sigma=64.0)
    b = ChromaAffineParam(MEAN, COV, support=_full(x), radius=0.2,
                          epsilon_lab=None, max_gain=1e9, gamut='soft')
    a.reset(x, 0)
    b.reset(x, 0)
    assert torch.allclose(a.T0, b.T0_ab, atol=1e-9)
    assert torch.allclose(a.render(x), b.render(x), atol=1e-6)


def test_鎖住亮度且未出色域時_L_通道幾乎不動():
    x = 0.35 + 0.3 * _image(1)
    p = ColorFieldParam(MEAN, COV, support=_full(x), grid=4, sigma=4.0,
                        lock_luminance=True, gamut='clip')
    p.reset(x, 0)
    with torch.no_grad():
        p.delta.mul_(0).add_(0.02)
    from src.defense.ncf_param import rgb_to_lab
    assert p.diagnostics(x)['clipping_fraction'] == 0.0
    before = rgb_to_lab(x)[:, :1]
    after = rgb_to_lab(p.render(x))[:, :1]
    assert (before - after).abs().max() < 1e-3


def test_放開亮度會讓可學參數多一個通道():
    x = _image(2)
    locked = ColorFieldParam(MEAN, COV, support=_full(x), grid=3, sigma=2.0)
    free = ColorFieldParam(MEAN, COV, support=_full(x), grid=3, sigma=2.0,
                           lock_luminance=False)
    locked.reset(x, 0)
    free.reset(x, 0)
    assert locked.delta.shape[1] == 2 * 2 + 2
    assert free.delta.shape[1] == 3 * 3 + 3


def test_帶寬不設限_sigma_為零時高頻比可以超過一():
    x = _image(3)
    sharp = ColorFieldParam(MEAN, COV, support=_full(x), grid=16, sigma=0.0)
    smooth = ColorFieldParam(MEAN, COV, support=_full(x), grid=16, sigma=16.0)
    out = {}
    for tag, p in (('sharp', sharp), ('smooth', smooth)):
        p.reset(x, 0)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            p.delta.add_(torch.randn(p.delta.shape, generator=g,
                                     dtype=p.delta.dtype))
        out[tag] = highfreq_report(x, p.render(x))['hf_ratio_rgb_total']
    assert out['sharp'] > out['smooth']


def test_支撐外逐位元不動():
    x = _image(4)
    w = torch.zeros_like(x[:, :1])
    w[..., :32, :] = 1.0
    p = ColorFieldParam(MEAN, COV, support=w, grid=4, sigma=1.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.delta.add_(0.5)
    y = p.render(x)
    assert torch.equal(y[..., 32:, :], x[..., 32:, :])
    assert not torch.equal(y[..., :32, :], x[..., :32, :])


def test_幅度為零時輸出只差一個色調壓縮():
    x = _image(5)
    p = ColorFieldParam(MEAN, COV, support=_full(x), grid=4, sigma=4.0,
                        amplitude=0.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.delta.add_(1.0)
    from src.defense.lowfreq_color import soft_gamut
    assert torch.allclose(p.render(x), soft_gamut(x), atol=1e-5)


def test_奇異值上界可以關掉也可以打開():
    x = _image(6)
    p = ColorFieldParam(MEAN, COV, support=_full(x), grid=2, sigma=2.0,
                        radius=10.0, max_gain=1.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.delta[0, :4].add_(3.0)
    p.project()
    m, _ = p._maps((8, 8))
    sv = torch.linalg.svdvals(m.reshape(2, 2, -1).permute(2, 0, 1))
    assert float(sv.max()) <= 1.0 + 1e-6

    free = ColorFieldParam(MEAN, COV, support=_full(x), grid=2, sigma=0.0,
                           radius=None, max_gain=None)
    free.reset(x, 0)
    with torch.no_grad():
        free.delta.add_(3.0)
    free.project()
    assert float(free.delta.abs().max()) >= 3.0
