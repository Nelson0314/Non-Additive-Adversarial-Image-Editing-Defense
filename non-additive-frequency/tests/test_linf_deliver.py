"""交付前的像素域 L∞ 投影：界限真的守住，且預設關閉時行為不變。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.defense.linf_deliver import clamp_residual, clamp_residual_ste


def test_夾取之後_linf_不超過界限():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 16, 16)
    d = (torch.rand_like(x) - 0.5) * 2.0          # 遠超過界限
    out = clamp_residual(x + d, x, 0.05)
    assert float((out - x).abs().max()) <= 0.05 + 1e-6


def test_夾取同時守住值域():
    x = torch.zeros(1, 1, 4, 4)
    out = clamp_residual(x - 1.0, x, 0.5)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_界限夠大時是恆等():
    torch.manual_seed(1)
    x = torch.rand(1, 3, 8, 8)
    d = (torch.rand_like(x) - 0.5) * 0.01
    out = clamp_residual((x + d).clamp(0, 1), x, 1.0)
    assert torch.allclose(out, (x + d).clamp(0, 1), atol=1e-6)


def test_ste_的前向等於硬夾取_反傳是恆等():
    x = torch.rand(1, 1, 4, 4)
    raw = (x + 0.4).clone().requires_grad_(True)
    out = clamp_residual_ste(raw, x, 0.05)
    assert torch.allclose(out.detach(), clamp_residual(raw.detach(), x, 0.05))
    out.sum().backward()
    # 硬夾取在球外梯度為零；STE 必須是 1，否則被夾住的座標再也拿不到訊號
    assert torch.allclose(raw.grad, torch.ones_like(raw))


def test_eps_非正時報錯而不是靜默放行():
    x = torch.rand(1, 1, 4, 4)
    with pytest.raises(ValueError):
        clamp_residual(x, x, 0.0)


def test_run_param_pgd_不給_deliver_時逐位元不變():
    from src.defense.param_pgd import run_param_pgd

    class Toy:
        name = "toy"
        radius = 1.0

        def reset(self, x01, seed):
            self.p = torch.zeros_like(x01).requires_grad_(True)

        def params(self):
            return [self.p]

        def render(self, x01):
            return x01 + self.p

        def project(self):
            with torch.no_grad():
                self.p.clamp_(-self.radius, self.radius)

    x = torch.full((1, 1, 4, 4), 0.5)

    def loss(y):
        return ((y - 1.0) ** 2).mean()

    a = run_param_pgd(x, Toy(), loss, steps=5, step_size=0.1)
    b = run_param_pgd(x, Toy(), loss, steps=5, step_size=0.1, deliver=None)
    assert torch.equal(a.x_def, b.x_def)


def test_run_param_pgd_的_deliver_同時作用在輸出上():
    from src.defense.param_pgd import run_param_pgd

    class Toy:
        name = "toy"
        radius = 1.0

        def reset(self, x01, seed):
            self.p = torch.zeros_like(x01).requires_grad_(True)

        def params(self):
            return [self.p]

        def render(self, x01):
            return x01 + self.p

        def project(self):
            with torch.no_grad():
                self.p.clamp_(-self.radius, self.radius)

    x = torch.full((1, 1, 4, 4), 0.5)

    def loss(y):
        return ((y - 1.0) ** 2).mean()

    def deliver(y):
        return clamp_residual_ste(y, x, 0.02)

    out = run_param_pgd(x, Toy(), loss, steps=20, step_size=0.1,
                        deliver=deliver)
    assert float((out.x_def - x).abs().max()) <= 0.02 + 1e-6
