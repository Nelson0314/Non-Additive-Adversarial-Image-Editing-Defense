"""釘住 ValueCurveParam 的四個結構保證。

這一族的賣點是「振幅再大也不會改到色相與飽和度」，所以這些性質不能靠看圖
確認，要逐條釘在隨機 θ × 隨機影像上。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.value_curve_param import ValueCurveParam  # noqa: E402

K = 64


def image(seed: int = 0, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 8, 8), generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def carrier(seed: int, jitter: float = 1.0, radius: float = 5.0):
    c = ValueCurveParam(radius=radius, pieces=K, init_jitter=jitter)
    c.reset(image(seed=0), seed=seed)
    return c


def reference_curve(m: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """直接照定義算 f(m)，當作 `_ratio` 的對照。"""
    th = theta[0, 0]
    idx = (m * K).floor().clamp(0, K - 1).long()
    cum = torch.cat([th.new_zeros(1), th.cumsum(0)[:-1]])
    return (cum[idx] / K + (m - idx.to(m.dtype) / K) * th[idx]) * (K / th.sum())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ratio_times_m_is_the_curve(seed):
    c = carrier(seed)
    x = image(seed=seed + 10)
    m = x.max(dim=1, keepdim=True).values
    got = c._ratio(x, c.theta) * m
    want = reference_curve(m, c.theta)
    assert torch.allclose(got, want, atol=1e-6), \
        f"最大偏差 {float((got - want).abs().max()):.2e}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_hue_is_preserved(seed):
    c = carrier(seed)
    x = image(seed=seed + 20)
    y = c.render(x)
    m_in = x.max(dim=1, keepdim=True).values
    m_out = y.max(dim=1, keepdim=True).values
    lit = (m_in > 1e-3) & (m_out > 1e-3)
    lit3 = lit.expand_as(x)
    # 色相由「除以最大通道後的三元組」決定；比例相同則這個三元組不變。
    a = (x / m_in.clamp(min=1e-6))[lit3]
    b = (y / m_out.clamp(min=1e-6))[lit3]
    assert float((a - b).abs().max()) < 1e-4


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_hsv_saturation_is_preserved(seed):
    c = carrier(seed)
    x = image(seed=seed + 30)
    y = c.render(x)

    def sat(z):
        mx = z.max(dim=1, keepdim=True).values
        mn = z.min(dim=1, keepdim=True).values
        return (mx - mn) / mx.clamp(min=1e-6), mx

    s_in, m_in = sat(x)
    s_out, _ = sat(y)
    lit = m_in > 1e-3
    assert float((s_in - s_out)[lit].abs().max()) < 1e-4


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_output_stays_in_range_without_clamping(seed):
    c = carrier(seed)
    x = image(seed=seed + 40)
    y = c.render(x)
    assert float(y.min()) >= -1e-6, float(y.min())
    assert float(y.max()) <= 1.0 + 1e-6, float(y.max())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_value_order_is_preserved(seed):
    c = carrier(seed)
    grid = torch.linspace(0, 1, 513).view(1, 1, 1, -1).expand(1, 3, 1, 513).contiguous()
    y = c.render(grid)
    m = y.max(dim=1).values.reshape(-1)
    assert bool((m[1:] - m[:-1] >= -1e-6).all()), "明度不單調"


def test_uniform_theta_is_the_identity():
    c = ValueCurveParam(radius=5.0, pieces=K, init_jitter=0.0)
    x = image(seed=99)
    c.reset(x, seed=0)
    y = c.render(x)
    assert float((y - x).abs().max()) < 1e-6


def test_black_pixels_are_finite():
    c = carrier(7)
    x = torch.zeros((1, 3, 4, 4))
    y = c.render(x)
    assert torch.isfinite(y).all()
    assert float(y.abs().max()) == 0.0


def test_gradient_reaches_theta():
    c = carrier(11)
    x = image(seed=123)
    y = c.render(x)
    y.square().mean().backward()
    assert c.theta.grad is not None
    assert torch.isfinite(c.theta.grad).all()
    assert float(c.theta.grad.abs().max()) > 0


def test_endpoints_are_fixed():
    c = carrier(13)
    white = torch.ones((1, 3, 2, 2))
    assert float((c.render(white) - white).abs().max()) < 1e-5
