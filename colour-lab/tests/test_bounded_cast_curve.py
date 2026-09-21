"""把解析參數域、純顏色映射與實際起點的梯度逐條釘住。

取樣不是全域證明的替代品；全域證明寫在載體 docstring。這裡刻意取整個
RGB 立方體、各面與頂點，以及固定膚色管帶的邊界，不只取人像支撐。
"""

import inspect
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defense import bounded_cast_curve as bc
from src.defense.bounded_cast_curve import BoundedCastCurveParam
from src.defense.delta_e_torch import ciede2000
from scripts import bounded_cast_preview as preview


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    """小張量單元測試只用一個 CPU thread，結束後還原設定。"""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def image(seed=0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((2, 3, 12, 16), generator=generator, dtype=dtype)


def cube():
    grid = torch.linspace(0, 1, 17, dtype=torch.float64)
    return torch.cartesian_prod(grid, grid, grid).T.reshape(1, 3, 1, -1)


def randomise(carrier, seed, scale):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in carrier.params():
            p.copy_(scale * torch.randn(p.shape, generator=generator, dtype=p.dtype))


def skin_samples():
    """取軌跡的明暗兩端、管帶中心與接近半徑 8 的整圈邊界。"""
    l, angle, radius = torch.meshgrid(
        torch.linspace(2, 98, 25, dtype=torch.float64),
        torch.linspace(0, 2 * math.pi, 25, dtype=torch.float64),
        torch.tensor((0.0, 4.0, 8.0 - 1e-8), dtype=torch.float64), indexing='ij')
    reference = bc.reference_lab(l.reshape(1, 1, 1, -1))
    reference[:, 1] += (radius * angle.cos()).reshape(1, 1, -1)
    reference[:, 2] += (radius * angle.sin()).reshape(1, 1, -1)
    x = bc.lab_to_rgb(reference)
    valid = ((x >= 0) & (x <= 1)).all(1)
    # 膚色帶定義是管帶與 RGB 立方體的交集，不能先把帶外顏色裁回來。
    return x.reshape(1, 3, 1, -1)[..., valid.reshape(-1)]


@pytest.mark.parametrize('pair', bc.DEFAULT_BOUNDS)
@pytest.mark.parametrize('scale', [0.0, 1.0, 1000.0])
def test_global_chroma_and_gamut_on_rgb_cube(pair, scale):
    random_pixels = torch.rand((1, 3, 1, 1024), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(97))
    x = torch.cat((cube(), random_pixels), dim=-1)
    carrier = BoundedCastCurveParam(C_max=pair[0], d_max=pair[1])
    carrier.reset(x, 0)
    for seed in (0, 11):
        randomise(carrier, seed, scale)
        y = carrier.render(x).detach()
        assert torch.isfinite(y).all()
        assert float(y.min()) >= -1e-12
        assert float(y.max()) <= 1.0 + 1e-12
        c = bc.rgb_to_lab(y)[:, 1:].square().sum(1).sqrt()
        assert float(c.max()) <= pair[0]


@pytest.mark.parametrize('pair', bc.DEFAULT_BOUNDS)
@pytest.mark.parametrize('seed', [0, 3, 19])
def test_skin_band_displacement_is_bounded_but_not_zero(pair, seed):
    x = skin_samples()
    carrier = BoundedCastCurveParam(C_max=pair[0], d_max=pair[1])
    carrier.reset(x, seed)
    randomise(carrier, seed, 100.0)
    assert x.shape[-1] > 500
    assert carrier.skin_band(x).all()
    y = carrier.render(x).detach()
    d = ciede2000(bc.rgb_to_lab(x), bc.rgb_to_lab(y))
    assert torch.isfinite(d).all()
    assert float(d.max()) <= pair[1]
    assert float(d.max()) > 0.1


@pytest.mark.parametrize('pair', bc.DEFAULT_BOUNDS)
def test_independent_skimage_measurement(pair):
    """用正式讀數的 skimage 再檢查，不能只讓自己的色彩轉換互相對答案。"""
    x = torch.cat((cube(), skin_samples()), dim=-1)
    carrier = BoundedCastCurveParam(C_max=pair[0], d_max=pair[1])
    carrier.reset(x, 0)
    randomise(carrier, 7, 100.0)
    y = carrier.render(x)
    delta, chroma = preview.measured_maps(x, y)
    assert float(chroma.max()) <= pair[0]
    skin = carrier.skin_band(x).numpy()
    assert float(delta[skin].max()) <= pair[1]


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_is_a_pure_colour_map(seed, dtype):
    """像素打亂後輸出跟著打亂；也不能偷偷使用 batch 或影像統計。"""
    x = image(seed, dtype)
    carrier = BoundedCastCurveParam(init_jitter=1.0)
    carrier.reset(x, seed)
    y = carrier.render(x)
    flat = x.reshape(2, 3, -1)
    order = torch.randperm(flat.shape[-1], generator=torch.Generator().manual_seed(seed))
    shuffled = flat[:, :, order].reshape_as(x)
    want = y.reshape(2, 3, -1)[:, :, order].reshape_as(x)
    assert torch.equal(carrier.render(shuffled), want)
    assert torch.equal(carrier.render(x[:1]), y[:1])
    # 相同 RGB 放到另一張尺寸不同的影像，仍要得到相同輸出。
    repeated = x[:1, :, :1, :1].expand(1, 3, 7, 9).contiguous()
    # 不同尺寸可能切換 BLAS kernel；容許 float64 末位捨入，不容許內容相依。
    tolerance = 2e-14 if dtype == torch.float64 else 2e-7
    assert torch.allclose(carrier.render(repeated),
                          y[:1, :, :1, :1].expand_as(repeated), rtol=0, atol=tolerance)


def test_no_clamp_no_projection_even_for_large_unprojected_parameters(monkeypatch):
    x = cube()
    carrier = BoundedCastCurveParam()
    carrier.reset(x, 0)
    randomise(carrier, 3, 1e6)
    saved = [p.detach().clone() for p in carrier.params()]

    def forbidden(*args, **kwargs):
        raise AssertionError('載體不得依賴 clamp、clip 或投影')

    for owner in (torch, torch.Tensor):
        for name in ('clamp', 'clamp_', 'clamp_min', 'clamp_min_',
                     'clamp_max', 'clamp_max_', 'clip', 'clip_'):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
    y = carrier.render(x).detach()
    carrier.project()
    assert float(y.min()) >= -1e-12 and float(y.max()) <= 1.0 + 1e-12
    assert all(torch.equal(p, old) for p, old in zip(carrier.params(), saved))
    assert 'skin_band(' not in inspect.getsource(carrier.prepare)
    assert 'skin_band(' not in inspect.getsource(carrier.render_prepared)


@pytest.mark.parametrize('pair', bc.DEFAULT_BOUNDS)
@pytest.mark.parametrize('restart', [0, 1, 2])
def test_every_parameter_tensor_has_gradient_at_actual_preview_start(pair, restart):
    """直接使用 CLI 預設值；零抖動與實際重啟點都必須把梯度送到三組參數。"""
    args = preview.build_parser().parse_args([])
    carrier = preview.make_carrier(args, pair, restart)
    x = image(101, torch.float32)
    carrier.reset(x, args.seed + 1009 * restart)
    assert sum(p.numel() for p in carrier.params()) == 192
    if restart == 0:
        assert carrier.init_jitter == 0
        assert all(torch.count_nonzero(p) == 0 for p in carrier.params())
    weights = x.new_tensor((0.7, 1.3, 2.1)).view(1, 3, 1, 1)
    (carrier.render(x).square() * weights).mean().backward()
    for name, p in zip(('theta_l', 'theta_a', 'theta_b'), carrier.params()):
        assert p.shape == (1, 1, 64)
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0, f'{name} 在實際起點收不到梯度'


def test_true_identity_start_has_nonzero_gradient_in_every_tensor():
    """容得下全 RGB 彩度時，真正的恆等起點也不能變成零梯度盒角。"""
    x = image(101)
    carrier = BoundedCastCurveParam(C_max=500.0, d_max=12.0, base='identity')
    carrier.reset(x, 0)
    assert torch.allclose(carrier.render(x), x, rtol=0, atol=2e-14)
    carrier.render(x).square().mean().backward()
    for name, p in zip(('theta_l', 'theta_a', 'theta_b'), carrier.params()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0, name


def test_low_chroma_start_is_explicitly_a_palette_not_an_identity_claim():
    x = cube()
    carrier = BoundedCastCurveParam()
    carrier.reset(x, 0)
    assert not torch.allclose(carrier.render(x), x)
    with pytest.raises(ValueError, match='恆等'):
        BoundedCastCurveParam(C_max=32, base='identity')


def test_identity_mode_displacement_bound_holds_for_entire_cube():
    x = cube()
    carrier = BoundedCastCurveParam(C_max=500, d_max=8, base='identity')
    carrier.reset(x, 0)
    randomise(carrier, 7, 100)
    y = carrier.render(x).detach()
    delta, chroma = preview.measured_maps(x, y)
    assert delta.max() <= 8 and chroma.max() <= 500
    assert float(y.min()) >= -1e-12 and float(y.max()) <= 1 + 1e-12


def test_neutral_endpoints_have_finite_backward():
    x = torch.linspace(0, 1, 65).reshape(1, 1, 1, -1).expand(1, 3, 1, 65).clone()
    x.requires_grad_(True)
    carrier = BoundedCastCurveParam()
    carrier.reset(x, 0)
    carrier.render(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for p in carrier.params():
        assert torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0


def test_prepared_render_and_reset_reproducibility():
    x = image()
    carrier = BoundedCastCurveParam(init_jitter=0.5)
    carrier.reset(x, 17)
    first = carrier.render(x)
    assert torch.equal(first, carrier.render_prepared(carrier.prepare(x)))
    carrier.reset(image(9), 17)
    assert torch.equal(first, carrier.render(x))
    carrier.reset(x, 18)
    assert not torch.equal(first, carrier.render(x))


def test_state_round_trip_and_checkpoint_is_not_aliased():
    x = image()
    carrier = BoundedCastCurveParam(init_jitter=0.5)
    carrier.reset(x, 7)
    carrier.set_amplitude([0.3])
    expected = carrier.render(x).detach()
    state = carrier.state_dict()
    randomise(carrier, 19, 10)
    other = BoundedCastCurveParam()
    other.load_state_dict(state)
    assert torch.equal(other.render(x), expected)
    assert other.amplitude == 0.3
    assert all(p.is_leaf and p.requires_grad for p in other.params())
    state['params'][0].fill_(0)
    assert torch.equal(other.render(x), expected)
    with pytest.raises(ValueError, match='結構'):
        BoundedCastCurveParam(C_max=48).load_state_dict(other.state_dict())


@pytest.mark.parametrize('amplitude', [0, 0.25, 1])
def test_amplitude_stays_inside_the_same_domain(amplitude):
    x = cube()
    carrier = BoundedCastCurveParam(init_jitter=1)
    carrier.reset(x, 4)
    carrier.set_amplitude(amplitude)
    y = carrier.render(x)
    assert preview.measured_maps(x, y)[1].max() <= carrier.C_max
    if amplitude == 0:
        zero = BoundedCastCurveParam()
        zero.reset(x, 0)
        assert torch.equal(y, zero.render(x))
        assert not torch.equal(y, x)


@pytest.mark.parametrize('kwargs', [
    {'pieces': 3}, {'pieces': 4.5}, {'C_max': 0}, {'d_max': -1},
    {'C_max': float('nan')}, {'d_max': float('inf')}, {'skin_width': 0},
    {'init_jitter': -0.1}, {'init_jitter': 1.1}, {'base': 'unknown'},
    {'C_max': 16, 'd_max': 2}, {'apply_where': torch.ones(1, 1, 2, 2)},
])
def test_invalid_or_uncertified_domains_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundedCastCurveParam(**kwargs)


def test_optimise_carrier_interface_without_caps():
    """只跑兩步的小型 CPU 單元測試，檢查真實求解介面與 checkpoint。"""
    from src.defense.immunise import optimise_carrier

    class Objective:
        def score(self, y):
            return y.square().mean()

        def terms(self, y):
            return {'probe': self.score(y)}

    x = image(12)[:, :, :3, :4]
    carrier = BoundedCastCurveParam()
    carrier.reset(x, 0)
    assert carrier.stages == [carrier]
    assert carrier.bounds() == (-math.inf, math.inf)
    assert carrier.step_scale() == 1.0
    result = optimise_carrier(carrier, x, Objective(), steps=2, check_every=1, caps=[])
    assert result['free_cap_violations'] == 0
    assert result['free_steps'] == 2
    assert math.isfinite(result['free_score_end'])


def test_preview_png_readouts_match_written_pixels(tmp_path):
    """小張量 I/O 測試：PNG 像素與各項 PNG 讀數均與磁碟內容一致。"""
    x = image(8, torch.float32)[:1, :, :4, :5]
    carrier = BoundedCastCurveParam(init_jitter=0.5)
    carrier.reset(x, 7)
    y = carrier.render(x).detach()
    path = tmp_path / 'preview.png'
    png = preview.save_png(path, y)
    with preview.Image.open(path) as saved:
        disk = torch.from_numpy(np.array(saved)).permute(2, 0, 1)[None].float() / 255
    assert torch.equal(png, disk)
    row = preview.readout(x, y, png, carrier)
    actual = disk - x
    delta, chroma = preview.measured_maps(x, disk)
    skin = carrier.skin_band(x).numpy()
    assert row['png_L2'] == pytest.approx(float(actual.norm()))
    assert row['png_RGB_RMS'] == pytest.approx(float(actual.square().mean().sqrt()))
    assert row['png_deltaE00'] == pytest.approx(float(delta.mean()))
    assert row['png_chroma_max'] == pytest.approx(float(chroma.max()))
    assert row['png_skin_deltaE00_max'] == pytest.approx(
        float(delta[skin].max()) if skin.any() else float('nan'), nan_ok=True)
    assert row['png_L2'] > 0


def test_preview_multistart_checkpoint_loop_on_tiny_cpu_tensor():
    """只用 20 個像素、兩步測試搜尋流程，包含黑白端點與最後一步候選。"""
    args = preview.build_parser().parse_args(['--steps', '2', '--restarts', '2'])
    x = image(7, torch.float32)[:1, :, :4, :5].clone()
    x[:, :, 0, 0] = 0
    x[:, :, 0, 1] = 1
    best = preview.ascend(x, args, (32, 24), args.seed)
    assert best['restart'] in (0, 1) and 0 <= best['step'] <= 2
    assert torch.isfinite(best['image']).all() and math.isfinite(best['value'])
    assert torch.equal(best['image'], best['carrier'].render(x))
    assert best['value'] == pytest.approx(float(preview.measured_maps(x, best['image'])[0].mean()))


def test_certificate_matrix_constants_and_positive_parameter_budgets():
    """檢查解析推導使用的矩陣係數與數值餘裕，不能換矩陣後沿用舊證書。"""
    forward = torch.tensor(bc.RGB_TO_XYZ, dtype=torch.float64)
    forward = forward / torch.tensor(bc.WHITE, dtype=torch.float64)[:, None]
    white = forward.sum(1)
    assert white.min() >= 0.999 and white.max() <= 1
    bias = math.hypot(500 * float(white[0] - 1), 200 * float(white[2] - 1))
    bias /= 3 * 0.999 ** (2 / 3)
    assert bias < 0.006 < bc.MARGIN
    cast = torch.linalg.inv(forward) @ forward.new_tensor((0.2, 0.0, -0.4))
    assert float(cast.abs().max()) < 1
    for c_max, d_max in bc.DEFAULT_BOUNDS:
        carrier = BoundedCastCurveParam(C_max=c_max, d_max=d_max)
        radius, scale = carrier.offset_radius, carrier.palette_scale
        assert radius > 0
        assert bc.PALETTE_BOUND * scale + radius + bias <= c_max
        assert bc.DE_LIPSCHITZ * (carrier.skin_width + bc.PALETTE_BOUND * (1 - scale)
                                  + radius) <= d_max
