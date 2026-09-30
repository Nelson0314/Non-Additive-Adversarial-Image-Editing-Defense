"""位移場均勻性與端點讀數；來源為 anti-purification/tests/test_uniformity.py。"""
import torch

from immunization_core.color.space import lab_to_rgb, rgb_to_lab
from immunization_core.color.uniformity import (endpoint_ramp, lab_offset,
                                                new_endpoint_fraction, raw_excursion,
                                                tv_offset, u16_offset)

from carrier_stub import OffsetCarrier


def _img(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return (0.3 + 0.4 * torch.rand(1, 3, size, size, generator=g)).clamp(0, 1)


def test_同一張圖的位移與兩個均勻度讀數都是零():
    x = _img()
    off = lab_offset(x, x)
    assert float(off.abs().max()) < 1e-9
    assert float(tv_offset(off)) < 1e-9
    assert float(u16_offset(off)) < 1e-5


def test_全域色偏的_TV_與_U16_都很小():
    x = _img(1)
    lab = rgb_to_lab(x)
    lab[:, 1] += 9.0
    lab[:, 2] -= 9.0
    y = lab_to_rgb(lab).clamp(0, 1)
    off = lab_offset(x, y)
    assert float(tv_offset(off)) < 0.05
    assert float(u16_offset(off)) < 0.5


def test_半圖色偏騙得過_TV_但騙不過_U16():
    """TV 付的是邊界的錢；大片區域之間的色偏差異要靠 U16 才量得到。

    直接在位移場上做，不經 RGB 往返——往返的色域裁切本身會製造逐像素的變化，
    那會把要驗的性質蓋掉。
    """
    off = torch.zeros(1, 3, 256, 256)
    off[:, 1, :, :128] = -30.0
    off[:, 1, :, 128:] = 30.0
    assert float(tv_offset(off)) < 0.05
    assert float(u16_offset(off)) > 5.0


def test_新增端點不計原圖本來就有的端點():
    x = torch.full((1, 3, 8, 8), 0.5)
    x[..., :2, :] = 0.0
    y = x.clone()
    y[..., 4:6, :] = 1.0
    assert new_endpoint_fraction(x, x) == 0.0
    assert abs(new_endpoint_fraction(x, y) - 2 / 8) < 1e-6


def test_端點斜坡可微且原圖自己為零():
    x = _img(3)
    assert float(endpoint_ramp(x, x)) == 0.0
    y = x.clone().requires_grad_(True)
    r = endpoint_ramp(x, (y * 1.6).clamp(0, 1))
    r.backward()
    assert torch.isfinite(y.grad).all()
    assert float(r) > 0


def test_越界幅度量在_gamut_壓縮之前():
    x = _img(4, size=32)
    c = OffsetCarrier(x, amplitude=4.0)
    vals = raw_excursion(c, x)
    assert len(vals) == len(c.stages)
    assert all(float(v) >= 0 for v in vals)
    assert float(vals[0]) > 0
    assert vals[0].requires_grad
