"""紋理重相位的構造性質。

每一條都在釘住模組 docstring 宣稱「由構造保證」的東西。這些性質若破掉，
症狀不會出現在訓練曲線上——輸出仍是一張合理的防禦圖，只是它不再是相位
擾動，而報告會照樣宣稱它是。
"""

import math

import pytest
import torch

from src.residual.texture_rephase import (
    PhaseResidual,
    hann2d,
    radial_gate,
    rephase_blocks,
    texture_gate,
)


def _image(size: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g, dtype=torch.float64)


def test_identity_when_theta_is_zero():
    """theta=0 時輸出逐位等於原圖。恆等由 OLA(w^2) 正規化保證，只需 NOLA 不需 COLA。"""
    x = _image()
    m = PhaseResidual(size=64, block=16).to(torch.float64)
    m.prepare_gates(x)
    out = m.pixel_residual(x)
    assert torch.allclose(out, x, atol=1e-12), float((out - x).abs().max())


def test_identity_holds_for_non_cola_hop():
    """換成不滿足 COLA 的 hop，恆等仍須成立——否則恆等是靠巧合而非構造。"""
    x = _image(size=64)
    m = PhaseResidual(size=64, block=16, hop=8).to(torch.float64)
    m.prepare_gates(x)
    assert torch.allclose(m.pixel_residual(x), x, atol=1e-12)


def test_output_is_real_and_finite():
    x = _image()
    m = PhaseResidual(size=64, block=16, init_std=0.5, seed=1).to(torch.float64)
    m.prepare_gates(x)
    out = m.pixel_residual(x)
    assert out.dtype == x.dtype and not out.is_complex()
    assert torch.isfinite(out).all()


def test_single_block_preserves_amplitude_exactly():
    """單區塊、無重疊時，幅度譜逐位保留。

    這是模組的核心恆等式 |X * e^{i*theta}| = |X|。徑向閘已把 fx=0 與
    fx=N/2 兩行歸零，故半平面仍是合法的 Hermitian 表示，round trip 精確。
    """
    g = torch.Generator().manual_seed(3)
    blk = torch.rand(1, 1, 1, 16, 16, generator=g, dtype=torch.float64)
    gate = radial_gate(16, 0.25, blk.device, blk.dtype)
    shift = torch.rand(1, 1, 1, 16, 9, generator=g, dtype=torch.float64) * 2 - 1
    out = rephase_blocks(blk, shift * gate)

    a = torch.fft.rfft2(blk, norm="ortho").abs()
    b = torch.fft.rfft2(out, norm="ortho").abs()
    assert torch.allclose(a, b, atol=1e-12), float((a - b).abs().max())


def test_gradient_reaches_theta():
    x = _image()
    m = PhaseResidual(size=64, block=16, init_std=0.3, seed=2).to(torch.float64)
    m.prepare_gates(x)
    loss = m.pixel_residual(x).pow(2).sum()
    loss.backward()
    assert m.theta.grad is not None
    assert float(m.theta.grad.abs().max()) > 0.0


def test_theta_is_clamped_to_theta_max():
    x = _image()
    m = PhaseResidual(size=64, block=16, theta_max=0.2).to(torch.float64)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.fill_(10.0)
    shift = torch.clamp(m.theta.detach(), -m.theta_max, m.theta_max)
    assert float(shift.abs().max()) == pytest.approx(0.2)


def test_texture_gate_low_on_edges_high_on_noise():
    """邊緣的 coherence 高 -> 閘接近 0；雜訊紋理的 coherence 低 -> 閘接近 1。"""
    size, block = 64, 16
    edge = torch.zeros(1, 3, size, size, dtype=torch.float64)
    edge[..., size // 2:] = 1.0
    g = torch.Generator().manual_seed(5)
    noise = torch.rand(1, 3, size, size, generator=g, dtype=torch.float64)

    ge = texture_gate(edge, block, block // 2)
    gn = texture_gate(noise, block, block // 2)
    assert float(ge.max()) < 0.2, float(ge.max())
    assert float(gn.mean()) > 0.5, float(gn.mean())


def test_radial_gate_zeroes_dc_and_hermitian_columns():
    m = radial_gate(16, 0.25, torch.device("cpu"), torch.float64)
    assert float(m[0, 0]) == 0.0                      # DC
    assert float(m[:, 0].abs().max()) == 0.0          # fx = 0
    assert float(m[:, -1].abs().max()) == 0.0         # fx = N/2
    assert float(m.max()) == 1.0                      # 仍有可用的頻格


def test_hann_periodic_sums_to_one_at_half_overlap():
    w = hann2d(16, torch.device("cpu"), torch.float64)
    w1d = 0.5 - 0.5 * torch.cos(2.0 * math.pi * torch.arange(16, dtype=torch.float64) / 16)
    assert torch.allclose(w1d[:8] + w1d[8:], torch.ones(8, dtype=torch.float64))
    assert w.shape == (16, 16)


def test_amplitude_deviation_is_zero_at_identity():
    x = _image()
    m = PhaseResidual(size=64, block=16).to(torch.float64)
    m.prepare_gates(x)
    assert m.amplitude_deviation(x) == pytest.approx(0.0, abs=1e-12)


def test_amplitude_deviation_is_small_but_nonzero_when_active():
    """重疊相加使整圖層級的幅度保留成為近似。要量它，不要假設它為零。"""
    x = _image()
    m = PhaseResidual(size=64, block=16, init_std=1.0, seed=7).to(torch.float64)
    m.prepare_gates(x)
    dev = m.amplitude_deviation(x)
    assert dev > 0.0
    assert dev < 0.5, dev


def test_disabled_module_is_a_no_op():
    x = _image()
    m = PhaseResidual(size=64, block=16, init_std=1.0, seed=8).to(torch.float64)
    m.prepare_gates(x)
    m.disable()
    assert torch.equal(m.pixel_residual(x), x)


def test_forward_before_prepare_gates_raises():
    x = _image()
    m = PhaseResidual(size=64, block=16).to(torch.float64)
    with pytest.raises(RuntimeError, match="prepare_gates"):
        m.pixel_residual(x)
