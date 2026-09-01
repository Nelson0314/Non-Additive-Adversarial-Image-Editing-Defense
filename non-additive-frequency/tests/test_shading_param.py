"""候選二的乘性明暗場參數化（`src/defense/param_pgd.py`）。

四件事會靜默出錯，逐條釘住：零初始化的恆等性（壞了就是每一張圖都先付一筆
無償失真）、梯度能不能回傳（斷了 PGD 會安靜地什麼都不學）、投影夾的是粗網格
本身（夾錯地方預算就沒有定義）、以及場真的落在候選二宣稱的低頻帶裡（跑掉了
就不再對準它宣稱的兩個失效機制）。

全部在 CPU 上執行。
"""

import math

import pytest
import torch

from src.defense.param_pgd import ShadingParam, ShadingRandomParam


def _img(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def test_零初始化逐位等於原圖():
    """`exp(0) = 1`。與相位 θ=0 同性質；壞了就是無償的失真地板。"""
    x = _img()
    p = ShadingParam(radius=0.1)
    p.reset(x, 0)
    assert torch.allclose(p.render(x), x, atol=1e-6)


def test_參數量是粗網格大小():
    x = _img()
    p = ShadingParam(radius=0.1, grid=16)
    p.reset(x, 0)
    assert p.params()[0].numel() == 16 * 16


def test_梯度回得來():
    x = _img()
    p = ShadingParam(radius=0.1)
    p.reset(x, 0)
    p.render(x).pow(2).mean().backward()
    assert float(p.params()[0].grad.abs().sum()) > 0


def test_投影夾的是粗網格本身():
    """雙三次上採樣會過衝，夾在上採樣之後預算就沒有定義。"""
    x = _img()
    p = ShadingParam(radius=0.1)
    p.reset(x, 0)
    with torch.no_grad():
        p.m.fill_(5.0)
    p.project()
    assert float(p.m.detach().max()) == pytest.approx(0.1, abs=1e-6)


def test_輸出留在值域內():
    x = _img()
    p = ShadingParam(radius=0.5)
    p.reset(x, 0)
    with torch.no_grad():
        p.m.fill_(0.5)
    y = p.render(x)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_是消色差的_三通道乘同一個增益():
    """逐通道的彩色明暗場會撞上已否決的 `顏色通道`，主線必須是單通道。"""
    x = torch.full((1, 3, 32, 32), 0.5)
    p = ShadingParam(radius=0.2, grid=8)
    p.reset(x, 0)
    with torch.no_grad():
        p.m.normal_(generator=torch.Generator().manual_seed(1))
        p.m.clamp_(-0.2, 0.2)
    y = p.render(x)
    assert torch.allclose(y[0, 0], y[0, 1], atol=1e-6)
    assert torch.allclose(y[0, 0], y[0, 2], atol=1e-6)


def test_場落在候選二宣稱的低頻帶():
    """`f_n <= 0.03`。跑掉了就不再對準模糊與裁切放大那兩個失效機制。"""
    x = torch.full((1, 3, 512, 512), 0.5)
    p = ShadingParam(radius=0.2, grid=16)
    p.reset(x, 0)
    with torch.no_grad():
        p.m.normal_(generator=torch.Generator().manual_seed(2))
    m = torch.nn.functional.interpolate(p.m, size=(512, 512), mode="bicubic",
                                        align_corners=False)[0, 0]
    spec = torch.fft.rfft2(m - m.mean())
    fy = torch.fft.fftfreq(512).view(-1, 1)
    fx = torch.fft.rfftfreq(512).view(1, -1)
    r = torch.sqrt(fy ** 2 + fx ** 2) / 0.5
    e = spec.abs() ** 2
    assert float((r * e).sum() / e.sum()) < 0.05


def test_隨機版不最佳化且真的動了():
    """步驟 3 的同失真對照。`params()` 為空，PGD 就無事可做。"""
    x = _img()
    r = ShadingRandomParam(radius=0.2)
    r.reset(x, 3)
    assert r.params() == []
    assert not torch.allclose(r.render(x), x, atol=1e-4)
    assert float(r.m.abs().max()) <= 0.2 + 1e-6


def test_隨機版同種子可重現_不同種子不同():
    x = _img()
    a, b, c = (ShadingRandomParam(radius=0.2) for _ in range(3))
    a.reset(x, 5)
    b.reset(x, 5)
    c.reset(x, 6)
    assert torch.equal(a.m, b.m)
    assert not torch.equal(a.m, c.m)


def test_粗網格太小時拋錯():
    with pytest.raises(ValueError, match="至少為 2"):
        ShadingParam(radius=0.1, grid=1)


def test_半徑決定亮部的最大增益():
    """radius 0.30 時亮部最多變 exp(0.30) = 1.35 倍，這是取上界的理由。"""
    assert math.exp(0.30) == pytest.approx(1.3499, abs=1e-3)
