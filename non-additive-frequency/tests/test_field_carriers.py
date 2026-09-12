import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.defense.geometry_field import (FlowFieldParam, affine_residual,
                                        base_grid, border_window,
                                        det_jacobian)
from src.defense.immunise import Cap, optimise_carrier
from src.defense.lab_offset_field import LabOffsetFieldParam
from src.defense.uniformity import lab_offset

ROOT = Path(__file__).resolve().parents[1]


def _image(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g, dtype=torch.float32)


class _Objective:
    """把影像推向固定目標；與載體無關，只用來驗求解器接得上。"""

    def __init__(self, target):
        self.target = target

    def score(self, y):
        return ((y - self.target) ** 2).mean()

    def terms(self, y):
        return {'enc': self.score(y), 'cond': self.score(y),
                'id': self.score(y)}


def test_零場是恆等映射():
    x = _image(32)
    c = FlowFieldParam(x, grid=4, taper=8.0)
    c.reset(x)
    assert torch.allclose(c.render(x), x, atol=1e-5)


def test_邊界窗在畫幅邊緣為零():
    w = border_window((32, 32), 8.0, dtype=torch.float32, device='cpu')
    assert float(w[0, 0, 0, :].abs().max()) == 0.0
    assert float(w[0, 0, -1, :].abs().max()) == 0.0
    assert float(w[0, 0, :, 0].abs().max()) == 0.0
    assert float(w[0, 0, :, -1].abs().max()) == 0.0
    assert float(w[0, 0, 16, 16]) == pytest.approx(1.0)


def test_邊界窗讓位移在畫幅邊緣為零():
    x = _image(32)
    c = FlowFieldParam(x, grid=4, taper=8.0)
    c.reset(x)
    with torch.no_grad():
        c.theta.fill_(5.0)
    f = c.flow()
    assert float(f[..., 0, :].abs().max()) == 0.0
    assert float(f[..., -1].abs().max()) == 0.0
    assert float(f.abs().max()) > 1.0


def test_恆等映射的行列式是一():
    flow = torch.zeros(1, 2, 16, 16)
    assert torch.allclose(det_jacobian(flow), torch.ones(1, 1, 16, 16))


def test_行列式抓得到摺疊():
    n = 32
    xs = torch.arange(n, dtype=torch.float32)
    flow = torch.zeros(1, 2, n, n)
    flow[:, 0] = (-2.0 * xs)[None, None, :]
    det = det_jacobian(flow)
    assert float(det.min()) < 0.0


def test_純仿射的位移場殘差為零():
    n = 32
    ys = torch.arange(n, dtype=torch.float32) / (n - 1)
    xs = torch.arange(n, dtype=torch.float32) / (n - 1)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    flow = torch.stack([3.0 * gx - 1.5 * gy + 0.7,
                        0.4 * gx + 2.0 * gy - 1.1])[None]
    support = torch.ones(1, 1, n, n)
    assert float(affine_residual(flow, support).abs().max()) < 1e-3


def test_局部隆起的殘差遠大於整體形變的殘差():
    """臉框內的自然度約束要能分開這兩件事。

    臉整體被壓扁或拉長讀起來只是另一個人，那是**仿射**；鼻子被局部拉長讀起來
    是壞掉的，那是仿射以外的殘差。同樣的最大位移量，兩者的殘差要差一個數量級，
    這道約束才綁得住該綁的東西。
    """
    n = 64
    xs = torch.arange(n, dtype=torch.float32) / (n - 1)
    gx = xs[None, :].expand(n, n)
    gy = xs[:, None].expand(n, n)
    support = torch.ones(1, 1, n, n)

    stretch = torch.stack([8.0 * (gx - 0.5), 8.0 * (gy - 0.5)])[None]
    centre = ((gx - 0.5) ** 2 + (gy - 0.5) ** 2) / 0.02
    bulge = torch.stack([8.0 * torch.exp(-centre), torch.zeros(n, n)])[None]

    r_stretch = float(affine_residual(stretch, support).abs().max())
    r_bulge = float(affine_residual(bulge, support).abs().max())
    assert r_stretch < 1e-3
    assert r_bulge > 10 * max(r_stretch, 1e-3)


def test_彎曲的位移場殘差不為零():
    n = 32
    xs = torch.arange(n, dtype=torch.float32) / (n - 1)
    gx = xs[None, :].expand(n, n)
    flow = torch.stack([torch.sin(6.0 * gx), torch.zeros(n, n)])[None]
    support = torch.ones(1, 1, n, n)
    assert float(affine_residual(flow, support).abs().max()) > 0.1


def test_取樣格與恆等對齊():
    g = base_grid((8, 8), dtype=torch.float32, device='cpu')
    x = _image(8)
    y = torch.nn.functional.grid_sample(x, g, mode='bilinear',
                                        padding_mode='border',
                                        align_corners=False)
    assert torch.allclose(y, x, atol=1e-6)


def test_求解器接得上位移場載體():
    x = _image(32)
    c = FlowFieldParam(x, grid=4, taper=4.0)
    c.reset(x)
    obj = _Objective(torch.roll(x, shifts=3, dims=-1))
    caps = [Cap('flow_face', lambda _y, c=c: c.magnitude().mean(),
                lambda _y, c=c: float(c.magnitude().mean().detach()), 2.0)]
    stats = optimise_carrier(c, x, obj, steps=12, lr=0.5, caps=caps,
                             check_every=4)
    assert stats['free_score_end'] < stats['free_score_start']
    assert float(c.theta.abs().max()) > 0.0


def test_摺疊約束量得到嚴重程度():
    x = _image(32)
    c = FlowFieldParam(x, grid=4, taper=0.0)
    c.reset(x)
    assert float(c.fold_measure()) == 0.0
    with torch.no_grad():
        c.theta[:, 0] = torch.linspace(0.0, -40.0, 4)[None, None, :]
    assert float(c.fold_measure()) > 0.0


def test_位移場的均勻性不跟著影像內容():
    """直接參數化的位移場，`d(p)` 不含 `c(p)`，所以位移場的空間 std 由參數決定。

    這是與仿射場的分野：仿射的 `∇d = (T−I)∇c` 讓位移沿原圖的邊界變化，
    空間常數的矩陣仍給出極不均勻的位移場。
    """
    x = 0.3 + 0.4 * _image(64)
    c = LabOffsetFieldParam(x, grid=1)
    c.reset(x)
    with torch.no_grad():
        c.delta[:, 1] = 12.0
        c.delta[:, 2] = -9.0
    off = lab_offset(x, c.render(x))
    assert float(off[:, 1].std()) < 1.0
    assert float(off[:, 2].std()) < 1.0
    assert float(off[:, 1].mean()) > 5.0


def test_色域壓縮會把不均勻加回位移場():
    """交付圖上的位移場**不等於**參數場：出了 sRGB 的像素被壓回來，壓的幅度
    跟原像素的顏色有關，於是內容相依的不均勻從 gamut 這一步回來。

    這不是實作的瑕疵而是色彩載體的性質，`clipping_fraction` 逐列回報它。
    中間調的圖上位移場空間 std < 1（見上一則），而通篇飽和的圖上同樣的參數
    量到 2.6——差別全部來自越界的比例。
    """
    x = _image(64)
    c = LabOffsetFieldParam(x, grid=1)
    c.reset(x)
    with torch.no_grad():
        c.delta[:, 1] = 12.0
    off = lab_offset(x, c.render(x))
    assert float(off[:, 1].std()) > 1.0
    assert c.diagnostics(x)['clipping_fraction'] > 0.0


def test_逐通道位移只動指定的通道():
    x = _image(32)
    c = LabOffsetFieldParam(x, grid=2)
    c.reset(x)
    with torch.no_grad():
        c.delta[:, 0] = 6.0
    off = c.offset((32, 32))
    assert float(off[:, 0].mean()) == pytest.approx(6.0, abs=1e-4)
    assert float(off[:, 1].abs().max()) == 0.0
    assert float(off[:, 2].abs().max()) == 0.0


def test_逐通道上限各自獨立():
    from src.defense.delta_e_torch import cvar_from_map
    x = _image(32)
    c = LabOffsetFieldParam(x, grid=2)
    c.reset(x)
    with torch.no_grad():
        c.delta[:, 1] = 20.0
    support = torch.ones(1, 1, 32, 32)
    assert float(cvar_from_map(c.channel_magnitude(0), support, 0.99)) == 0.0
    assert float(cvar_from_map(c.channel_magnitude(1), support, 0.99)) \
        == pytest.approx(20.0, abs=1e-3)


def test_投影把參數壓回盒內():
    x = _image(16)
    c = LabOffsetFieldParam(x, grid=2, box=(5.0, 7.0, 9.0))
    c.reset(x)
    with torch.no_grad():
        c.delta.fill_(100.0)
    c.project()
    assert float(c.delta[:, 0].max()) == pytest.approx(5.0)
    assert float(c.delta[:, 1].max()) == pytest.approx(7.0)
    assert float(c.delta[:, 2].max()) == pytest.approx(9.0)


def test_狀態存取可還原():
    x = _image(16)
    c = FlowFieldParam(x, grid=4)
    c.reset(x)
    with torch.no_grad():
        c.theta.normal_()
    state = c.state_dict()
    before = c.render(x)
    with torch.no_grad():
        c.theta.zero_()
    c.load_state_dict(state)
    assert torch.allclose(c.render(x), before, atol=1e-6)
    assert c.params()[0].requires_grad


def _guard():
    spec = importlib.util.spec_from_file_location(
        'immunise_script', ROOT / 'scripts' / 'immunise.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assert_no_instructions


FIELD_CONFIGS = ('immunise_field.json', 'immunise_field_affine.json',
                 'immunise_field_objective.json',
                 'immunise_field_eot.json',
                 'immunise_field_margin.json',
                 'immunise_field_coarse_margin.json')


@pytest.mark.parametrize('name', FIELD_CONFIGS)
def test_位移場的設定檔不含任何指令(name):
    guard = _guard()
    spec = json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))
    guard(spec)


@pytest.mark.parametrize('name', FIELD_CONFIGS)
def test_位移場設定檔的每個變體都掛得上載體(name):
    spec = json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))
    x = _image(32)
    for variant in spec['variants']:
        knobs = variant['knobs']
        if variant['carrier'] == 'flow':
            c = FlowFieldParam(x, grid=int(knobs['grid']),
                               taper=float(knobs['taper']),
                               box=float(knobs['box']))
            assert set(variant['caps']) >= {'face_px', 'rigid_px', 'fold'}
        else:
            c = LabOffsetFieldParam(x, grid=int(knobs['grid']),
                                    box=tuple(knobs['box']))
            assert set(variant['caps']) >= {'offset_L', 'offset_a', 'offset_b'}
        c.reset(x)
        assert c.params()[0].requires_grad
        assert c.render(x).shape == x.shape


def test_膨脹讓遮罩變大且不縮小():
    from src.defense.geometry_field import dilate
    m = torch.zeros(1, 1, 32, 32)
    m[..., 14:18, 14:18] = 1.0
    d = dilate(m, 4)
    assert float(d.sum()) > float(m.sum())
    assert float((d - m).clamp_max(0).abs().sum()) == 0.0
    assert float(d[..., 10, 16]) == 1.0
    assert float(d[..., 5, 16]) == 0.0


def test_膨脹半徑為零是恆等():
    from src.defense.geometry_field import dilate
    m = torch.zeros(1, 1, 16, 16)
    m[..., 4:8, 4:8] = 1.0
    assert torch.equal(dilate(m, 0), m)


def test_膨脹之後背景的支撐仍非空():
    """`rigid_margin` 太大會把背景整個吃掉，那時約束沒有定義。"""
    from src.defense.geometry_field import dilate
    m = torch.zeros(1, 1, 64, 64)
    m[..., 20:44, 20:44] = 1.0
    assert float((1.0 - dilate(m, 8)).sum()) > 0.0
    assert float((1.0 - dilate(m, 40)).sum()) == 0.0
