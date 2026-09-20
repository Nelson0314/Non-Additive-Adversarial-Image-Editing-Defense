"""釘住 ChannelSpanCurveParam 的三件事：θ 不出盒、通道差有硬上界、span=0 無色偏。

這一族的用處是把「色偏預算」從求解端搬進參數化，所以「求解端動不了它」
必須是構造上的事實，不能只是設定檔的慣例。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.channel_span_curve import ChannelSpanCurveParam  # noqa: E402

K = 64


def image(seed: int = 0, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 8, 8), generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def carrier(span: float, seed: int = 0, jitter: float = 1.0):
    c = ChannelSpanCurveParam(radius=5.0, pieces=K, span=span, init_jitter=jitter)
    c.reset(image(seed=0), seed=seed)
    return c


@pytest.mark.parametrize("span", [0.0, 0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_theta_never_leaves_the_box(span, seed):
    c = carrier(span, seed=seed)
    lo, hi = c.bounds()
    th = c.theta
    assert float(th.min()) >= lo - 1e-6, f"{float(th.min())} < {lo}"
    assert float(th.max()) <= hi + 1e-6, f"{float(th.max())} > {hi}"


@pytest.mark.parametrize("span", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_channel_gap_respects_the_bound(span):
    worst = 0.0
    for seed in range(8):
        c = carrier(span, seed=seed)
        worst = max(worst, c.channel_gap() / max(c.gap_bound(), 1e-12)
                    if span > 0 else c.channel_gap())
    if span == 0:
        assert worst < 1e-7, f"span=0 仍有通道差 {worst:.2e}"
    else:
        assert worst <= 1.0 + 1e-6, f"通道差超過上界 {worst:.4f} 倍"


def test_span_zero_gives_one_curve_for_all_channels():
    c = carrier(0.0, seed=5)
    th = c.theta
    assert torch.allclose(th[0, 0], th[0, 1], atol=0) and \
        torch.allclose(th[0, 0], th[0, 2], atol=0)
    # 複合曲線是同一個純量函數套在三個通道上：同一個輸入值三通道必得同一個輸出。
    q = torch.linspace(0, 1, 257).view(1, 1, 1, -1).expand(1, 3, 1, 257).contiguous()
    r = c.render(q)
    assert float((r[0, 0] - r[0, 1]).abs().max()) < 1e-6
    assert float((r[0, 0] - r[0, 2]).abs().max()) < 1e-6


def test_identity_at_zero_jitter():
    c = carrier(0.5, seed=0, jitter=0.0)
    x = image(seed=21)
    assert float((c.render(x) - x).abs().max()) < 1e-6


def test_curves_stay_monotone():
    for span in (0.0, 0.25, 1.0):
        for seed in range(4):
            c = carrier(span, seed=seed)
            q = torch.linspace(0, 1, 513).view(1, 1, 1, -1).expand(1, 3, 1, 513).contiguous()
            r = c.render(q)[0, :, 0]
            assert bool((r[:, 1:] - r[:, :-1] >= -1e-6).all()), \
                f"span={span} seed={seed} 曲線不單調"


def test_gradient_reaches_both_parameters():
    c = carrier(0.25, seed=3)
    x = image(seed=31)
    c.render(x).square().mean().backward()
    for name, p in (("u", c.u), ("d", c.d)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0, f"{name} 收不到梯度"


def test_span_one_reaches_the_box_corners():
    c = carrier(1.0, seed=0, jitter=0.0)
    lo, hi = c.bounds()
    with torch.no_grad():
        c.u.fill_(0.5)
        c.d[0, 0].fill_(1.0)
        c.d[0, 1].fill_(-1.0)
    th = c.theta
    assert float(th[0, 0].max()) == pytest.approx(hi, rel=1e-6)
    assert float(th[0, 1].min()) == pytest.approx(lo, rel=1e-6)


def test_theta_cannot_be_assigned():
    c = carrier(0.25)
    with pytest.raises(AttributeError):
        c.theta = torch.zeros((1, 3, K))


def test_state_round_trip():
    c = carrier(0.3, seed=2)
    x = image(seed=41)
    before = c.render(x).detach().clone()
    state = c.state_dict()
    c2 = carrier(0.3, seed=9)
    c2.load_state_dict(state)
    assert float((c2.render(x) - before).abs().max()) < 1e-7


def test_identity_start_is_not_a_zero_gradient_point():
    """`init_jitter = 0` 的設定檔從恆等出發，那一點上 `d` 必須收得到梯度。

    盒底 `u = 0` 也是恆等，但 `min(u, 1−u) = 0` 會讓 `d` 的梯度恰為零，
    整族靜默退化成一條共用曲線。起點改放盒子中央就是為了這個。
    """
    c = carrier(0.5, seed=0, jitter=0.0)
    x = image(seed=101)
    assert float((c.render(x) - x).abs().max()) < 1e-6, "起點必須仍是恆等"
    c.render(x).square().mean().backward()
    for name, p in (("u", c.u), ("d", c.d)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) > 0, f"{name} 在恆等起點上梯度是零"
