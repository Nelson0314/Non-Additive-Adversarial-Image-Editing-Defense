"""參數化 PGD 的介面契約與預算對齊。

這裡不載入 Stable Diffusion——`run_param_pgd` 只吃一個 `loss_fn(x_def)`,
故整條迴圈可以用一個合成損失在 CPU 上釘住。
"""

import math

import pytest
import torch

from src.defense.param_pgd import (
    AdditiveParam,
    PhaseParam,
    RandomPhaseParam,
    fit_to_budget,
    run_param_pgd,
)


def _image(size: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def _l2_to(target: torch.Tensor):
    return lambda x: (x - target).pow(2).mean()


def _linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def test_additive_respects_radius():
    x = _image()
    p = AdditiveParam(radius=4.0 / 255.0)
    res = run_param_pgd(x, p, _l2_to(torch.zeros_like(x)), steps=40)
    assert _linf(res.x_def, x) <= 4.0 / 255.0 + 1e-6


def test_additive_reduces_loss():
    x = _image()
    tgt = torch.zeros_like(x)
    p = AdditiveParam(radius=16.0 / 255.0)
    p.reset(x, 0)
    before = float(_l2_to(tgt)(p.render(x)).detach())
    res = run_param_pgd(x, p, _l2_to(tgt), steps=60)
    assert float(_l2_to(tgt)(res.x_def)) < before


def test_phase_param_reduces_loss_and_is_bounded():
    x = _image()
    tgt = torch.zeros_like(x)
    p = PhaseParam(size=64, block=16, radius=math.pi)
    p.reset(x, 0)
    before = float(_l2_to(tgt)(p.render(x)).detach())
    res = run_param_pgd(x, p, _l2_to(tgt), steps=60)
    assert float(_l2_to(tgt)(res.x_def)) < before
    assert float(p.module.theta.abs().max()) <= math.pi + 1e-6


def test_phase_at_zero_radius_limit_is_near_identity():
    """半徑趨近 0 時輸出趨近原圖——構造上的恆等在迴圈裡仍然成立。"""
    x = _image()
    p = PhaseParam(size=64, block=16, radius=1e-6)
    res = run_param_pgd(x, p, _l2_to(torch.zeros_like(x)), steps=10)
    assert _linf(res.x_def, x) < 1e-4


def test_random_phase_is_not_optimized():
    """對照組不得有可優化參數，否則它就不是對照組。"""
    x = _image()
    p = RandomPhaseParam(size=64, block=16, radius=1.0)
    p.reset(x, 0)
    assert p.params() == []
    res = run_param_pgd(x, p, _l2_to(torch.zeros_like(x)), steps=50)
    assert not torch.allclose(res.x_def, x)


def test_random_phase_is_reproducible_across_seeds():
    x = _image()
    a = RandomPhaseParam(size=64, block=16, radius=1.0)
    b = RandomPhaseParam(size=64, block=16, radius=1.0)
    ra = run_param_pgd(x, a, _l2_to(x), steps=1, seed=3)
    rb = run_param_pgd(x, b, _l2_to(x), steps=1, seed=3)
    rc = run_param_pgd(x, RandomPhaseParam(size=64, block=16, radius=1.0),
                       _l2_to(x), steps=1, seed=4)
    assert torch.equal(ra.x_def, rb.x_def)
    assert not torch.equal(ra.x_def, rc.x_def)


def test_fit_to_budget_hits_target():
    x = _image()
    p = AdditiveParam()
    res = fit_to_budget(
        x, p, _l2_to(torch.zeros_like(x)), lambda a, b: _linf(a, b),
        target=0.02, lo=1e-4, hi=0.2, steps=20, rounds=12, tol=0.002,
    )
    assert res.history[-1]["unreachable"] is False
    assert abs(res.history[-1]["reached"] - 0.02) <= 0.002


def test_fit_to_budget_reports_unreachable_instead_of_lying():
    """目標高於天花板時必須標 unreachable，不得靜默回傳一個離目標很遠的結果。"""
    x = _image()
    p = AdditiveParam()
    res = fit_to_budget(
        x, p, _l2_to(torch.zeros_like(x)), lambda a, b: _linf(a, b),
        target=0.9, lo=1e-4, hi=0.05, steps=10, rounds=4, tol=0.002,
    )
    assert res.history[-1]["unreachable"] is True
    assert res.history[-1]["reached"] < 0.9


def test_phase_ceiling_is_reported_as_unreachable():
    """相位的失真有構造上的天花板（|θ| ≤ π），要求超過它必須被誠實標記。"""
    x = _image()
    p = PhaseParam(size=64, block=16, r_min=0.25)
    res = fit_to_budget(
        x, p, _l2_to(torch.zeros_like(x)), lambda a, b: _linf(a, b),
        target=5.0, lo=0.05, hi=math.pi, steps=10, rounds=3, tol=0.002,
    )
    assert res.history[-1]["unreachable"] is True


def test_set_radius_clamps_phase_to_pi():
    p = PhaseParam(size=64, block=16)
    p.set_radius(10.0)
    assert p.radius == pytest.approx(math.pi)
