import pytest
import torch

from src.defense.carrier_search import START, build_carrier
from src.defense.inner_optimise import _scale_into_cap, optimise_field


MEAN = [71.3, -5.9, 1.8]
COV = [[210.0, 4.0, -3.0], [4.0, 36.0, 2.0], [-3.0, 2.0, 30.0]]
SECOND = [60.6, 8.9, 33.9]


def _image(seed=0, size=32):
    g = torch.Generator().manual_seed(seed)
    return (0.3 + 0.4 * torch.rand(1, 3, size, size, generator=g)).clamp(0, 1)


def _carrier(x, **over):
    frame = torch.ones_like(x[:, :1])
    clothes = torch.zeros_like(x[:, :1])
    clothes[..., 16:, :] = 1.0
    return build_carrier(dict(START, **over), x, frame_support=frame,
                         clothes_support=clothes, frame_palette=(MEAN, COV),
                         clothes_palette=(SECOND, COV))


class FakeIP2P:
    """把 latent 當成一個固定線性投影，梯度路徑與真的一樣可微。"""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(4, 3, generator=g, dtype=torch.float64)

    def encode_image(self, x01):
        return torch.einsum('ij,bjhw->bihw', self.w.to(x01.dtype), x01)


def test_內層會把_latent_距離推大():
    x = _image()
    c = _carrier(x, frame_grid=2.0, frame_scale=0.2, frame_amplitude=0.5)
    out = optimise_field(c, x, FakeIP2P(), steps=25, lr=0.08)
    assert out['inner_latent_end'] > out['inner_latent_start']
    assert out['inner_steps'] == 25


def test_沒有可學參數時直接拋錯():
    x = _image()
    c = _carrier(x)
    for p in c.params():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        optimise_field(c, x, FakeIP2P(), steps=2)


def test_縮放會把量測壓回上限內且不動場的形狀():
    x = _image(1)
    c = _carrier(x, frame_grid=2.0, frame_scale=0.6, frame_amplitude=1.0)
    before = c.stages[0].delta.detach().clone()

    def measure(y):
        return float((y - x).abs().mean() * 100)

    cap = measure(c.render(x)) * 0.4
    s = _scale_into_cap(c, x, measure, cap)
    assert 0.0 < s < 1.0
    assert measure(c.render(x)) <= cap + 1e-6
    assert torch.equal(c.stages[0].delta.detach(), before)


def test_已經在上限內時不縮放():
    x = _image(2)
    c = _carrier(x, frame_amplitude=0.1)

    def measure(y):
        return float((y - x).abs().mean())

    assert _scale_into_cap(c, x, measure, 1e3) == 1.0
    assert c.stages[0].amplitude == pytest.approx(0.1)


def test_上限會在內層結束後套用():
    x = _image(3)
    c = _carrier(x, frame_grid=2.0, frame_scale=0.3, frame_amplitude=1.0)

    def measure(y):
        return float((y - x).abs().mean() * 100)

    cap = 0.1
    out = optimise_field(c, x, FakeIP2P(), steps=10, lr=0.05,
                         caps=[(measure, cap)])
    assert measure(c.render(x)) <= cap + 1e-6
    assert out['inner_amplitude_shrink'] < 1.0
