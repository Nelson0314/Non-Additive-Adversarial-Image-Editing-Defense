"""幅度相依的相位上限（Perturbing the Phase, arXiv:2602.06577 的約束）。

把係數 `X` 的相位轉 `theta`，係數本身移動 `2|X|·sin(theta/2)`。要把那個位移
界在 `eps` 以內，相位必須滿足

    |theta| <= 2·arcsin( eps / (2|X|) )      （2|X| > eps）
    相位自由                                  （2|X| <= eps）

它處理的是本專案記過的缺陷：**固定的 theta 不等於固定的失真**（FND-038，
同一個 theta 在 24 張圖上 PSNR 由 23.15 漂到 39.54）。

`0` 必須逐位元等於加這個旗標之前，否則既有批次不可重跑。
"""

import math

import pytest
import torch

from src.defense.param_pgd import PhaseParam
from src.residual.texture_rephase import PhaseResidual

BLOCK = 32
DT = torch.float64
SIZE = 128


def _image(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, SIZE, SIZE, generator=g, dtype=DT)
    x[:, :, :, : SIZE // 2] = 0.45          # 左半平坦，幅度小
    return x


def _module(budget: float) -> PhaseResidual:
    return PhaseResidual(size=SIZE, block=BLOCK, hop=8, r_min=0.12,
                         theta_max=math.pi, gain_max=0.0, energy_quantile=0.0,
                         freq_weight="jpeg_luma", freq_weight_power=0.25,
                         theta_budget=budget).to(DT)


def test_關閉時不建上限且行為與加旗標之前相同():
    x = _image()
    off, ref = _module(0.0), PhaseResidual(
        size=SIZE, block=BLOCK, hop=8, r_min=0.12, theta_max=math.pi,
        gain_max=0.0, energy_quantile=0.0, freq_weight="jpeg_luma",
        freq_weight_power=0.25).to(DT)
    off.prepare_gates(x); ref.prepare_gates(x)
    assert off.theta_cap is None and ref.theta_cap is None
    with torch.no_grad():
        off.theta.normal_(0.0, 0.5)
        ref.theta.copy_(off.theta)
    assert torch.equal(off.pixel_residual(x), ref.pixel_residual(x))


def test_負的預算直接拋錯():
    with pytest.raises(ValueError, match="theta_budget"):
        PhaseResidual(size=64, block=16, theta_budget=-0.1)


def test_上限就是那條閉式公式():
    x = _image()
    eps = 0.02
    m = _module(eps)
    m.prepare_gates(x)
    mag = m.analyze(x).abs().mean(dim=1)
    want = 2.0 * torch.asin((eps / (2.0 * mag).clamp_min(1e-12)).clamp(max=1.0))
    assert torch.allclose(m.theta_cap, want, atol=1e-12)


def test_幅度小的頻格相位自由():
    """`2|X| <= eps` 的位置上限應恰為 pi，不是某個接近 pi 的數。"""
    x = _image()
    eps = 0.05
    m = _module(eps)
    m.prepare_gates(x)
    mag = m.analyze(x).abs().mean(dim=1)
    free = 2.0 * mag <= eps
    assert bool(free.any())
    assert torch.allclose(m.theta_cap[free], torch.full_like(m.theta_cap[free], math.pi))


def test_上限隨幅度單調遞減():
    x = _image()
    m = _module(0.02)
    m.prepare_gates(x)
    mag = m.analyze(x).abs().mean(dim=1).flatten()
    cap = m.theta_cap.flatten()
    order = torch.argsort(mag)
    sorted_cap = cap[order]
    # 允許浮點上的平手，但不可上升
    assert bool((sorted_cap[1:] - sorted_cap[:-1] <= 1e-12).all())


def test_預算越大上限越鬆():
    x = _image()
    lo, hi = _module(0.01), _module(0.04)
    lo.prepare_gates(x); hi.prepare_gates(x)
    assert bool((hi.theta_cap >= lo.theta_cap - 1e-12).all())
    assert float(hi.theta_cap.mean()) > float(lo.theta_cap.mean())


def test_theta為零時仍然逐位元恆等():
    x = _image()
    m = _module(0.02)
    m.prepare_gates(x)
    assert float((m.pixel_residual(x) - x).abs().max()) < 1e-12


def test_投影把參數拉回可行集而不是只在前向夾():
    """只在前向夾的話，界外座標的梯度是零，PGD 會把參數推出去再也回不來，
    而報表上看不出來。"""
    x = _image()
    p = PhaseParam(size=SIZE, block=BLOCK, hop=8, r_min=0.12, radius=math.pi,
                   energy_quantile=0.0, freq_weight="jpeg_luma",
                   freq_weight_power=0.25, theta_budget=0.02)
    p.reset(x.to(torch.float32), 0)
    with torch.no_grad():
        p.module.theta.fill_(10.0)
    p.project()
    cap = p.module.theta_cap
    assert bool((p.module.theta.abs() <= cap + 1e-6).all())


def test_同一個預算下不同影像拿到不同的上限():
    """這正是這條約束存在的理由——固定的 theta 不等於固定的失真。"""
    a, b = _image(0), _image(7)
    ma, mb = _module(0.02), _module(0.02)
    ma.prepare_gates(a); mb.prepare_gates(b)
    assert not torch.allclose(ma.theta_cap, mb.theta_cap)
