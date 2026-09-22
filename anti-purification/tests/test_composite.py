import pytest
import torch

from src.defense.color_field import ColorFieldParam
from src.defense.composite import CompositeParam


MEAN = [71.3, -5.9, 1.8]
COV = [[210.0, 4.0, -3.0], [4.0, 36.0, 2.0], [-3.0, 2.0, 30.0]]
SECOND = [60.6, 8.9, 33.9]


def _image(seed=0, size=48):
    g = torch.Generator().manual_seed(seed)
    return (0.3 + 0.4 * torch.rand(1, 3, size, size, generator=g)).clamp(0, 1)


def _frame(x):
    return torch.ones_like(x[:, :1])


def _clothes(x):
    w = torch.zeros_like(x[:, :1])
    w[..., 24:, 8:40] = 1.0
    return w


def _pair(x, **kw):
    frame = ColorFieldParam(MEAN, COV, support=_frame(x), grid=1, sigma=0.0, **kw)
    clothes = ColorFieldParam(SECOND, COV, support=_clothes(x), grid=4, sigma=2.0, **kw)
    return CompositeParam([frame, clothes], tags=['frame', 'clothes'])


def test_至少要兩段():
    x = _image()
    with pytest.raises(ValueError):
        CompositeParam([ColorFieldParam(MEAN, COV, support=_frame(x))])


def test_參數是各段參數的聯集():
    x = _image()
    p = _pair(x)
    p.reset(x, 0)
    assert len(p.params()) == 2
    assert p.params()[0].shape[-1] == 1
    assert p.params()[1].shape[-1] == 4


def test_串接的輸出等於逐段套用():
    x = _image(1)
    p = _pair(x)
    p.reset(x, 0)
    with torch.no_grad():
        for t in p.params():
            t.add_(0.1)
    step = p.stages[1].render(p.stages[0].render(x))
    assert torch.equal(p.render(x), step)


def test_衣物段只改它自己的支撐():
    x = _image(2)
    clothes = ColorFieldParam(SECOND, COV, support=_clothes(x), grid=4, sigma=1.0)
    clothes.reset(x, 0)
    with torch.no_grad():
        clothes.delta.add_(0.4)
    y = clothes.render(x)
    assert torch.equal(y[..., :24, :], x[..., :24, :])
    assert not torch.equal(y[..., 24:, 8:40], x[..., 24:, 8:40])


def test_整圖濾鏡在串裡時沒有逐位元不動的像素():
    x = _image(3)
    p = _pair(x)
    p.reset(x, 0)
    with torch.no_grad():
        p.stages[0].delta.add_(0.3)
    y = p.render(x)
    assert not torch.equal(y[..., :24, :], x[..., :24, :])


def test_幅度可以廣播也可以逐段指定():
    x = _image(4)
    p = _pair(x)
    p.reset(x, 0)
    p.set_amplitude(0.25)
    assert p.amplitude == [0.25, 0.25]
    p.set_amplitude([0.1, 0.9])
    assert p.amplitude == [0.1, 0.9]
    with pytest.raises(ValueError):
        p.set_amplitude([0.1])


def test_兩段的幅度都為零時只剩色調壓縮():
    x = _image(5)
    p = _pair(x)
    p.reset(x, 0)
    with torch.no_grad():
        for t in p.params():
            t.add_(1.0)
    p.set_amplitude(0.0)
    from src.defense.lowfreq_color import soft_gamut
    w = _clothes(x)
    once = soft_gamut(x)
    expect = w * soft_gamut(once) + (1 - w) * once
    assert torch.allclose(p.render(x), expect, atol=1e-5)


def test_讀數逐段加前綴且整體高頻比照報():
    x = _image(6)
    p = _pair(x)
    p.reset(x, 0)
    d = p.diagnostics(x)
    assert d['stages'] == 'frame|clothes'
    assert 'frame_grid' in d and 'clothes_grid' in d
    assert d['frame_grid'] == 1 and d['clothes_grid'] == 4
    assert 'hf_ratio_rgb_total' in d
