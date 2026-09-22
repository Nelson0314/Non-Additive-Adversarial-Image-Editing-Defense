import pytest
import torch

from src.defense.recoloradv_param import (ReColorAdvParam, luv_to_rgb,
                                          rgb_to_luv)


def _image(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g, dtype=torch.float32)


def test_luv_來回轉換是恆等():
    x = _image(24)
    back = luv_to_rgb(rgb_to_luv(x))
    assert torch.allclose(back, x, atol=2e-4)


def test_luv_對照已知值():
    """sRGB 純白與中灰的 LUV，對 skimage 的定義。"""
    white = torch.ones(1, 3, 1, 1)
    luv = rgb_to_luv(white)[0, :, 0, 0]
    assert luv[0] == pytest.approx(100.0, abs=1e-3)
    assert luv[1].abs() < 1e-2 and luv[2].abs() < 1e-2
    black = torch.zeros(1, 3, 1, 1)
    assert rgb_to_luv(black)[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)


def test_零位移逐位元恆等():
    x = _image(32)
    c = ReColorAdvParam(x, resolution=(16, 32, 32), radius=0.06)
    c.reset(x)
    assert torch.allclose(c.render(x), x, atol=2e-4)


def test_投影守住逐通道的_epsilon():
    x = _image(16)
    c = ReColorAdvParam(x, resolution=(4, 8, 8), radius=0.06)
    c.reset(x)
    with torch.no_grad():
        c.delta.add_(0.5)
    c.project()
    assert float(c.delta.abs().max()) == pytest.approx(0.06, abs=1e-6)


def test_位移會改變輸出而且梯度回得到查表():
    x = _image(32)
    c = ReColorAdvParam(x, resolution=(8, 16, 16), radius=0.06)
    c.reset(x)
    with torch.no_grad():
        c.delta[:, 1] += 0.05
    y = c.render(x)
    assert not torch.allclose(y, x, atol=1e-3)
    loss = ((c.render(x) - x) ** 2).mean()
    loss.backward()
    assert c.delta.grad is not None and float(c.delta.grad.abs().max()) > 0


def test_同色像素得到同一個變換():
    """查表是純逐點的顏色映射，空間位置不進表。"""
    x = torch.zeros(1, 3, 8, 8)
    x[:, 0] = 0.4; x[:, 1] = 0.6; x[:, 2] = 0.2
    c = ReColorAdvParam(x, resolution=(8, 16, 16), radius=0.06)
    c.reset(x)
    with torch.no_grad():
        c.delta.uniform_(-0.05, 0.05, generator=torch.Generator().manual_seed(3))
    y = c.render(x)
    assert torch.allclose(y, y[..., :1, :1].expand_as(y), atol=1e-5)


def test_平滑正則在恆等時為零而且非負():
    x = _image(16)
    c = ReColorAdvParam(x, resolution=(4, 8, 8))
    c.reset(x)
    assert float(c.smoothness()) == pytest.approx(0.0, abs=1e-9)
    with torch.no_grad():
        c.delta[:, 0, ::2] += 0.03
    assert float(c.smoothness()) > 0


def test_apply_where_為零的地方逐位元保留原圖():
    x = _image(16)
    w = torch.zeros(1, 1, 16, 16)
    w[..., :8, :] = 1.0
    c = ReColorAdvParam(x, resolution=(4, 8, 8), apply_where=w)
    c.reset(x)
    with torch.no_grad():
        c.delta.add_(0.04)
    y = c.render(x)
    assert torch.equal(y[..., 8:, :], x[..., 8:, :])
    assert not torch.allclose(y[..., :8, :], x[..., :8, :], atol=1e-3)


def test_解析度不合法要報錯():
    x = _image(8)
    with pytest.raises(ValueError, match='resolution'):
        ReColorAdvParam(x, resolution=(1, 8, 8))
