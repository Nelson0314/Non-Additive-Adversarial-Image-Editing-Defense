import pytest
import torch

from src.defense.color_amplitude import delta_e00
from src.defense.delta_e_torch import ciede2000, delta_e00_torch
from src.defense.ncf_param import rgb_to_lab


def _pair(seed=0, size=24):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(1, 3, size, size, generator=g)
    b = (a + 0.25 * (torch.rand(1, 3, size, size, generator=g) - 0.5)).clamp(0, 1)
    return a.double(), b.double()


def _skimage_map(a, b):
    from skimage.color import deltaE_ciede2000, rgb2lab
    x = a.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    y = b.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return torch.as_tensor(deltaE_ciede2000(rgb2lab(x), rgb2lab(y)))


def test_給同一組_Lab_時公式與_skimage_逐位元等價():
    """把色差公式與色彩空間轉換分開釘。公式這一層不允許有偏差。"""
    from skimage.color import deltaE_ciede2000, rgb2lab
    a, b = _pair()
    la = torch.as_tensor(rgb2lab(a.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2)
    lb = torch.as_tensor(rgb2lab(b.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2)
    got = ciede2000(la, lb)
    want = torch.as_tensor(deltaE_ciede2000(
        la.permute(0, 2, 3, 1).numpy(), lb.permute(0, 2, 3, 1).numpy()))
    assert float((got - want).abs().max()) < 1e-9


def test_端到端的差異全部來自_rgb_to_lab():
    """專案的 `rgb_to_lab` 與 skimage 的 `rgb2lab` 相差最多約 0.005 個 Lab 單位，
    換算到 ΔE00 是 0.003。載體本身用的就是專案那一份，所以求解端沿用它，
    這條測試把容差寫明而不是假裝沒有。"""
    a, b = _pair()
    got = ciede2000(rgb_to_lab(a), rgb_to_lab(b))
    want = _skimage_map(a, b)
    assert float((got - want).abs().max()) < 1e-2


def test_支撐加權的平均與量測路徑一致():
    a, b = _pair(1)
    w = torch.zeros_like(a[:, :1])
    w[..., 6:18, 4:20] = 1.0
    got = float(delta_e00_torch(a, b, w))
    want = delta_e00(a, b, w)
    assert abs(got - want) < 1e-2


def test_純灰與同一張圖的色差是零且梯度有限():
    x = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float64, requires_grad=True)
    y = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float64)
    d = delta_e00_torch(x, y)
    assert float(d) < 1e-5
    d.backward()
    assert torch.isfinite(x.grad).all()


def test_大色差的梯度指向縮小色差():
    a, _ = _pair(2)
    b = a.clone().requires_grad_(True)
    tgt = (a + 0.3).clamp(0, 1)
    d = delta_e00_torch(tgt, b)
    d.backward()
    assert torch.isfinite(b.grad).all()
    assert float(b.grad.abs().sum()) > 0


def test_色差隨幅度單調上升():
    a, _ = _pair(3)
    vals = [float(delta_e00_torch(a, (a + s).clamp(0, 1))) for s in
            (0.02, 0.05, 0.1, 0.2)]
    assert vals == sorted(vals)


@pytest.mark.parametrize('seed', [4, 5, 6])
def test_多組隨機影像都對得上(seed):
    a, b = _pair(seed)
    assert abs(float(delta_e00_torch(a, b)) - delta_e00(a, b)) < 1e-2


def test_CVaR_與量測路徑一致():
    from src.defense.color_amplitude import cvar_e00
    from src.defense.delta_e_torch import cvar_torch
    a, b = _pair(7, size=48)
    assert abs(float(cvar_torch(a, b)) - cvar_e00(a, b)) < 1e-2
    w = torch.zeros_like(a[:, :1])
    w[..., 10:40, 8:44] = 1.0
    assert abs(float(cvar_torch(a, b, w)) - cvar_e00(a, b, w)) < 1e-2


def test_CVaR_抓得到平均與分位數抓不到的色塊():
    """96% 的像素小色差、4% 的像素極端色差：平均與 p95 都合格，CVaR 不合格。"""
    import numpy as np
    from src.defense.delta_e_torch import cvar_torch, delta_map
    a = torch.full((1, 3, 50, 50), 0.5, dtype=torch.float64)
    b = a.clone()
    b[..., :2, :] = 0.0
    d = delta_map(a, b)
    mean = float(d.mean())
    p95 = float(np.percentile(d.numpy(), 95))
    cvar = float(cvar_torch(a, b))
    assert mean < 16 and p95 < 16
    assert cvar > 4 * mean


def test_CVaR_的梯度作用在整個尾端():
    from src.defense.delta_e_torch import cvar_torch
    a, _ = _pair(8, size=32)
    b = (a + 0.1).clamp(0, 1).requires_grad_(True)
    cvar_torch(a, b).backward()
    touched = (b.grad.abs().sum(1) > 0).double().mean()
    assert float(touched) > 0.04
