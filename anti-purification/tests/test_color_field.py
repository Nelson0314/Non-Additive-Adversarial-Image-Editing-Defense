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


def test_單一控制點且重度模糊時整片場是空間常數():
    """退化的性質是**空間常數**，不是與舊類別逐值相同。

    `ChromaAffineParam.T0_ab` 取的是 3×3 MK 解的 a/b 子區塊，那不是 a/b 平面上
    的 MK 解（見 `color_field` 的模組說明）。`ColorFieldParam` 已改成直接解 2D，
    兩者因此不再逐值相同；舊類別只有 `scripts/color_ceiling.py` 在用，維持原樣
    以便重現既有批次。
    """
    x = _image()
    a = ColorFieldParam(MEAN, COV, support=_full(x), grid=1, sigma=64.0)
    a.reset(x, 0)
    m, t = a._maps(tuple(x.shape[-2:]))
    assert torch.allclose(m, m[..., :1, :1].expand_as(m), atol=1e-12)
    assert torch.allclose(t, t[..., :1, :1].expand_as(t), atol=1e-12)


def test_鎖亮度時_T0_是_ab_平面上的_MK_解():
    x = _image(3)
    a = ColorFieldParam(MEAN, COV, support=_full(x), grid=1, sigma=64.0,
                        lock_luminance=True)
    a.reset(x, 0)
    from src.defense.ncf_param import regularize_covariance, rgb_to_lab
    lab = rgb_to_lab(x).double()[0].flatten(1)
    centred = lab - lab.mean(1, keepdim=True)
    src, _ = regularize_covariance(centred @ centred.T / lab.shape[1], 1e-4)
    tgt, _ = regularize_covariance(
        torch.as_tensor(COV, dtype=torch.float64), 1e-4)
    mapped = a.T0 @ src[1:, 1:] @ a.T0.T
    assert torch.allclose(mapped, tgt[1:, 1:], atol=1e-8)


def test_舊的全域仿射取的是_3x3_解的子區塊():
    """把差異釘住：不是兩邊都對，是舊的那一條在數學上不滿足 2D 的 MK 條件。"""
    x = _image()
    b = ChromaAffineParam(MEAN, COV, support=_full(x), radius=0.2,
                          epsilon_lab=None, max_gain=1e9, gamut='soft')
    b.reset(x, 0)
    assert torch.allclose(b.T0_ab, b.T0[1:, 1:], atol=1e-12)


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
