"""不加高頻的兩個色彩載體。全部 CPU、float64。

這一組測試要守住的是**構造上的保證**，不是經驗觀察：亮度逐像素不變、色度
增益不超過上界、空間變化的頻寬由高斯 sigma 決定。三者都可以直接斷言。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.lowfreq_color import (ChromaAffineParam,  # noqa: E402
                                       RegionPaletteParam, _to_like,
                                       highfreq_report, soft_gamut)
from src.defense.ncf_param import rgb_to_lab  # noqa: E402


def _image(n=48, seed=5):
    g = torch.Generator().manual_seed(seed)
    # 低頻底色加上高頻紋理：沒有高頻的話「高頻有沒有被動」測不出來。
    base = torch.rand(1, 3, 4, 4, generator=g).double()
    base = torch.nn.functional.interpolate(base, size=(n, n), mode='bicubic', align_corners=False)
    detail = torch.rand(1, 3, n, n, generator=g).double() * .12
    return (base * .7 + detail).clamp(.02, .98)


TARGET_MEAN = [58., 12., -9.]
TARGET_COV = [[120., 4., -3.], [4., 60., 2.], [-3., 2., 44.]]


def _support(n=48, hole=False):
    s = torch.ones(1, 1, n, n, dtype=torch.float64)
    if hole:
        s[..., : n // 4, :] = 0
    return s


def test_chroma_affine_leaves_luminance_bit_for_bit():
    """映射不碰 L：在不做色域軟裁的模式下，輸出的亮度逐像素等於輸入。

    `gamut='soft'` 在 RGB 域動作，會輕微改動 L，所以那個更強的性質只在
    `'clip'` 與 `'scale'` 成立。真正要守住的「不放大亮度高頻」由
    `test_chroma_affine_does_not_add_high_frequency` 負責。
    """
    x = _image()
    p = ChromaAffineParam(TARGET_MEAN, TARGET_COV, support=_support(),
                          epsilon_lab=5., gamut='clip')
    p.reset(x)
    y = p.render(x)
    assert not torch.allclose(y, x)                       # 確實有換色
    lx, ly = rgb_to_lab(x)[:, 0], rgb_to_lab(y.detach())[:, 0]
    # 只剩 Lab→RGB→Lab 往返的浮點誤差；門檻對照 fabric 那條 1e-5 的先例。
    assert float((ly - lx).abs().max()) < 1e-4, float((ly - lx).abs().max())


def test_chroma_gain_is_bounded_after_projection():
    """增益的上界由 `project()` 保證，不由前向保證。

    **這是一個語意的改變。** 原本前向每次都做一次 SVD 夾取，於是任何時刻的
    `chroma_matrix()` 都在界內；但那個夾取會把兩個都超界的奇異值壓成**完全
    相等**的上界值，而 SVD 的 backward 在重根上除以零——實測第一次 backward
    就是 nan。約束因此搬到 `project()`（那本來就是 PGD 的形狀）。

    現在的保證是：**每一個被記錄下來的 render 都在界內**，因為訓練迴圈每步
    之後都投影，而輸出的圖與讀數都取在投影後的點。梯度步與投影之間的中間
    狀態可以超界，那個狀態不會被記錄。
    """
    x = _image()
    p = ChromaAffineParam(TARGET_MEAN, TARGET_COV, support=_support(),
                          epsilon_lab=50., radius=10., max_gain=1.)
    p.reset(x)
    with torch.no_grad():                                  # 硬推到很大的增量
        p.delta.copy_(torch.full((2, 2), 9., dtype=torch.float64))
    p.project()
    s = torch.linalg.svdvals(p.chroma_matrix())
    assert float(s.max()) <= 1. + 1e-6, float(s.max())
    # 起點本身也已經被壓進上界內。
    p.reset(x)
    assert float(torch.linalg.svdvals(p.T0_ab).max()) <= 1. + 1e-9
    assert float(torch.linalg.svdvals(p.chroma_matrix()).max()) <= 1. + 1e-9


def test_chroma_affine_does_not_add_high_frequency():
    x = _image()
    p = ChromaAffineParam(TARGET_MEAN, TARGET_COV, support=_support(), epsilon_lab=5.)
    p.reset(x)
    r = highfreq_report(x, p.render(x))
    # 要求是「不放大」，不是「完全不動」：軟裁只會衰減梯度。
    # RGB 那組是操作性的量（淨化在像素域作用）；Lab 的色度比值在接近無彩的
    # 影像上會從極小的基準放大，只供診斷。
    assert r['hf_ratio_rgb_total'] <= 1.0
    assert r['hf_ratio_lab_L'] <= 1.02 and r['hf_ratio_lab_a'] <= 1.05
    assert r['hf_ratio_lab_b'] <= 1.05


def test_gradient_reaches_the_two_by_two_parameter():
    x = _image()
    p = ChromaAffineParam(TARGET_MEAN, TARGET_COV, support=_support(), epsilon_lab=5.)
    p.reset(x)
    assert p.params()[0].shape == (2, 2)
    g, = torch.autograd.grad(p.render(x).pow(2).mean(), p.params())
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0


def test_support_outside_is_the_original_bit_for_bit():
    x = _image()
    s = _support(hole=True)
    p = ChromaAffineParam(TARGET_MEAN, TARGET_COV, support=s, epsilon_lab=5.)
    p.reset(x)
    frozen = (s == 0).expand_as(x)
    assert torch.equal(p.render(x)[frozen], x[frozen])


def _two_regions(n=48):
    top = torch.zeros(1, 1, n, n, dtype=torch.float64)
    top[..., : n // 2, :] = 1.
    return [(top, TARGET_MEAN, TARGET_COV),
            (1 - top, [45., -14., 20.], [[90., 1., 1.], [1., 50., 3.], [1., 3., 55.]])]


def test_region_palette_blurs_the_boundary_instead_of_cutting_it():
    """兩塊區域給不同配色時，交界的高頻不能比原圖多。"""
    x = _image()
    p = RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=12., epsilon_lab=5.)
    p.reset(x)
    y = p.render(x)
    assert not torch.allclose(y, x)
    r = highfreq_report(x, y)
    assert r['hf_ratio_rgb_total'] <= 1.0
    assert r['hf_ratio_lab_L'] <= 1.02
    assert r['hf_ratio_lab_a'] <= 1.05 and r['hf_ratio_lab_b'] <= 1.05


def test_sharp_weight_map_is_what_the_blur_is_protecting_against():
    """把 sigma 調到很小，色度高頻就會上升——證明那個保證真的來自模糊。"""
    x = _image()
    sharp = RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=.35,
                               epsilon_lab=5.)
    sharp.reset(x)
    smooth = RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=12.,
                                epsilon_lab=5.)
    smooth.reset(x)
    a = highfreq_report(x, sharp.render(x))
    b = highfreq_report(x, smooth.render(x))
    chroma = lambda r: r['hf_ratio_lab_a'] + r['hf_ratio_lab_b']   # noqa: E731
    assert chroma(a) > chroma(b)


def test_region_palette_rejects_degenerate_configurations():
    with pytest.raises(ValueError, match='至少要兩塊區域'):
        RegionPaletteParam(_two_regions()[:1], support=_support(), blur_sigma=8.)
    with pytest.raises(ValueError, match='blur_sigma 必須為正'):
        RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=0.)
    with pytest.raises(NotImplementedError, match='只實作鎖亮度'):
        RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=8.,
                           lock_luminance=False)


def test_region_weights_sum_to_one_per_pixel():
    x = _image()
    p = RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=9.)
    p.reset(x)
    torch.testing.assert_close(p.weights.sum(1), torch.ones_like(p.weights[:, 0]))


def test_highfreq_report_detects_added_detail():
    """把高頻加回去時比值必須大於 1，否則這個量測抓不到我們要防的東西。"""
    x = _image()
    g = torch.Generator().manual_seed(11)
    noisy = (x + torch.randn(x.shape, generator=g).double() * .05).clamp(0, 1)
    r = highfreq_report(x, noisy)
    assert r['hf_ratio_rgb_total'] > 1.5
    same = highfreq_report(x, x)
    assert same['hf_ratio_rgb_total'] == pytest.approx(1.)


def test_gamut_scaling_removes_the_clipping_that_added_high_frequency():
    """    構造一張色度已經接近色域邊界的影像，再套一組會把色度推更遠的配色。

    `clip` 版本會裁掉大量像素並造出新的邊；`scale` 版本不裁，所以色度的
    高通能量只可能持平或下降。
    """
    x = _image()
    # 推到高飽和：紅綠通道拉開，藍壓低，映射後很容易越界。
    x = torch.stack([x[:, 0] * .95 + .04, x[:, 1] * .5, x[:, 2] * .12], dim=1).clamp(.02, .98)
    saturated = [72., 46., 52.]
    cov = [[140., 6., -4.], [6., 90., 5.], [-4., 5., 80.]]
    hard = ChromaAffineParam(saturated, cov, support=_support(), epsilon_lab=5., gamut='clip')
    soft = ChromaAffineParam(saturated, cov, support=_support(), epsilon_lab=5., gamut='scale')
    hard.reset(x)
    soft.reset(x)
    dh, ds = hard.diagnostics(x), soft.diagnostics(x)
    assert dh['clipping_fraction'] > .02, dh['clipping_fraction']
    # 看的是裁切**幅度**不是比例：`gamut_scale` 的 1e-4 容差讓貼著色域邊界的
    # 像素仍算在內，`clamp` 因此會削掉至多那個量級。那造不出邊，而造不造得出邊
    # 由下面的高頻比負責把關。
    assert ds['clipping_max'] <= 2e-4, ds['clipping_max']
    assert ds['clipping_fraction'] < dh['clipping_fraction'] / 100
    chroma = lambda d: max(d['hf_ratio_lab_a'], d['hf_ratio_lab_b'])   # noqa: E731
    assert chroma(ds) <= 1.02, chroma(ds)
    # 保證是有代價的，而且代價要看得見：這張刻意做成高飽和的影像上，整體係數
    # 被少數貼著色域邊界的像素壓到很小，載體的幅度跟著縮水。不比較兩者的高頻
    # 比孰高孰低——`clip` 的色度映射本身也可能是壓縮的，那個比較沒有意義；
    # 這個修正要保證的是 `scale` 永遠不超過 1，不是它一定比 `clip` 低。
    assert 0 < ds['gamut_scale'] < .5, ds['gamut_scale']
    assert dh['gamut_scale'] == 1.0


def test_gamut_option_is_explicit():
    with pytest.raises(ValueError, match='gamut'):
        ChromaAffineParam(TARGET_MEAN, TARGET_COV, support=_support(), gamut='hard')
    with pytest.raises(ValueError, match='gamut'):
        RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=8., gamut='hard')


def test_region_palette_also_scales_into_gamut():
    x = _image()
    p = RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=12.,
                           epsilon_lab=5., gamut='scale')
    p.reset(x)
    d = p.diagnostics(x)
    assert d['gamut'] == 'scale' and 0 < d['gamut_scale'] <= 1
    assert d['clipping_max'] <= 2e-4


def test_constants_are_moved_to_the_image_device():
    """實際崩過的那條：目標配色留在 CPU、來源統計在 GPU，`mk_matrix` 就炸。

    本機沒有 GPU 可測，所以用 meta 裝置代替——它同樣不是 CPU，搬與不搬看得出來。
    """
    like = torch.empty(0, device='meta')
    assert _to_like([1., 2., 3.], like).device.type == 'meta'
    assert _to_like([[1., 0.], [0., 1.]], like).dtype == torch.float64
    # 參數本身也必須跟著影像走，否則 autograd 會在跨裝置那一步失敗。
    x = _image()
    p = RegionPaletteParam(_two_regions(), support=_support(), blur_sigma=9.)
    p.reset(x)
    assert p.delta.device == x.device
    assert all(t.device == x.device for t in p.T0)
    assert all(t.device == x.device for t in p.target_means)


def test_soft_gamut_never_amplifies_a_gradient():
    """導數落在 [0,1] 是這個式子存在的全部理由，直接對它做數值驗證。"""
    v = torch.linspace(-.4, 1.4, 4001, dtype=torch.float64).requires_grad_(True)
    y = soft_gamut(v.view(1, 1, 1, -1), knee=.06).view(-1)
    g, = torch.autograd.grad(y.sum(), v)
    assert float(g.min()) >= -1e-12, float(g.min())
    assert float(g.max()) <= 1. + 1e-12, float(g.max())
    # 中段幾乎是恆等：軟裁不應該把沒有越界的顏色也拉走。
    # knee=0.06 時 v=0.2 處的偏移約 2.1e-3，所以中段的門檻取 4e-3。
    mid = (v.detach() > .2) & (v.detach() < .8)
    assert float((y.detach()[mid] - v.detach()[mid]).abs().max()) < 4e-3
    assert float(y.min()) >= 0. and float(y.max()) <= 1.
    with pytest.raises(ValueError, match='knee'):
        soft_gamut(v.view(1, 1, 1, -1), knee=0.)


def test_soft_gamut_keeps_amplitude_where_global_scaling_collapses():
    """整體縮放會被少數飽和像素綁架；軟裁不會。"""
    x = _image()
    x = x.clone()
    x[:, 0, :3, :3] = .995                      # 少數極端像素
    x[:, 1, :3, :3] = .01
    x[:, 2, :3, :3] = .01
    saturated = [70., 44., 50.]
    cov = [[140., 6., -4.], [6., 90., 5.], [-4., 5., 80.]]
    scaled = ChromaAffineParam(saturated, cov, support=_support(), epsilon_lab=5.,
                               gamut='scale')
    soft = ChromaAffineParam(saturated, cov, support=_support(), epsilon_lab=5.,
                             gamut='soft')
    scaled.reset(x)
    soft.reset(x)
    a = float((scaled.render(x).detach() - x).abs().mean())
    b = float((soft.render(x).detach() - x).abs().mean())
    assert b > 5 * a, (a, b)                    # 軟裁保住了幅度
    d = soft.diagnostics(x)
    assert max(d['hf_ratio_lab_a'], d['hf_ratio_lab_b']) <= 1.02
    assert d['hf_ratio_lab_L'] <= 1.02
