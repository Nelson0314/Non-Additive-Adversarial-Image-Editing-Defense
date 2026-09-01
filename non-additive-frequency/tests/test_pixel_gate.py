"""逐像素紋理閘（2026-08-20，使用者裁定的修正方向 A）。

起因：θ=1.30 的防禦圖在人臉上出現抹開與波紋。診斷顯示 `texture_gate` 回傳的是
**每個 32×32 區塊一個純量**，而擾動是逐像素可見的——一個同時蓋到鬍鬚與臉頰的
區塊，會讓平滑的臉頰也被全強度旋轉相位。模組 docstring 原本就宣稱要壓掉平坦
區，逐區塊的解析度沒有兌現。

要釘四件事：

1. **關閉時逐位元不變**——SDEdit 那條凍結的線必須能重跑；
2. 打開時平坦區確實不再被改動；
3. 遮罩是逐像素的形狀，不是逐區塊；
4. 非法 sigma 被擋下。
"""

import math

import pytest
import torch

from src.residual.texture_rephase import PhaseResidual, pixel_texture_mask


def _img(size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, size, size, generator=g, dtype=torch.float64) * 0.15 + 0.4
    # 左半平坦、右半高紋理：遮罩應該只點亮右半
    x[..., size // 2:] += torch.rand(1, 3, size, size // 2, generator=g,
                                     dtype=torch.float64) * 0.5
    return x.clamp(0, 1)


def _module(size, **kw):
    m = PhaseResidual(size=size, block=16, hop=8, theta_max=math.pi, **kw)
    return m.to(torch.float64)


def test_關閉時與加這個選項之前逐位元相同():
    x = _img()
    a = _module(64)
    b = _module(64, pixel_gate_sigma=0.0)
    for m in (a, b):
        m.prepare_gates(x)
        with torch.no_grad():
            m.theta.fill_(1.0)
    ya, yb = a.pixel_residual(x), b.pixel_residual(x)
    assert torch.equal(ya, yb)
    assert a.pixel_mask is None


def test_打開時平坦區不再被改動():
    x = _img()
    m = _module(64, pixel_gate_sigma=2.0)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.fill_(1.0)
    y = m.pixel_residual(x)
    d = (y - x).abs()
    flat, tex = d[..., :32].mean(), d[..., 32:].mean()
    assert flat < tex * 0.25, f"平坦區改動 {flat:.2e} 相對紋理區 {tex:.2e} 沒有被壓下來"


def test_遮罩是逐像素的形狀():
    x = _img()
    m = pixel_texture_mask(x, sigma=2.0)
    assert m.shape == (1, 1, 64, 64)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    # 右半（高紋理）的平均應明顯高於左半（平坦）
    assert float(m[..., 32:].mean()) > 3.0 * float(m[..., :32].mean())


def test_sigma非正時拒絕():
    with pytest.raises(ValueError, match="sigma"):
        pixel_texture_mask(_img(), sigma=0.0)
    with pytest.raises(ValueError, match="pixel_gate_sigma"):
        _module(64, pixel_gate_sigma=-1.0)
