"""Reproducible CPU checks; no model construction, downloads or CUDA calls.

The legacy carrier class is compiled verbatim from its AST to exercise all its
methods without importing its unrelated diffusion / learned-metric front end.
Full-file byte identity separately covers that front end and its CLI defaults.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import torch

from ab_prism_defence import AbPrismParam, make_caps, read_targets
from src.defense.immunise import optimise_carrier, quantise
from src.defense.ncf_param import lab_to_rgb, rgb_to_lab

ROOT = Path(__file__).resolve().parents[2]
BEFORE_SHA256 = "555a9e1b44582fcb7bdc924d7ded7991340e5e3854c2d8f393a0a47c2ca61753"
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def synthetic():
    gen = torch.Generator(device="cpu").manual_seed(314159)
    x = 0.15 + 0.7 * torch.rand((1, 3, 64, 64), generator=gen, device="cpu")
    x[:, :, 24:40, 24:40] = torch.tensor([0.64, 0.45, 0.35]).view(1, 3, 1, 1)
    x[:, :, 0, :3] = torch.tensor([0.0, 0.5, 1.0])[None, None, :]
    face = torch.zeros_like(x[:, :1])
    face[:, :, 24:40, 24:40] = 1.0
    return x, face


class TensorObjective:
    def __init__(self, x):
        self.x = x

    def terms(self, y):
        return {"synthetic_mse": (y - self.x).square().mean()}

    def score(self, y):
        return -self.terms(y)["synthetic_mse"]


def smoke():
    x, face = synthetic()
    carrier = AbPrismParam()
    carrier.reset(x, seed=7)
    y = carrier.render(x)
    assert y.device.type == "cpu" and y.shape == (1, 3, 64, 64)
    assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    assert not torch.equal(y, x)
    assert sum(p.numel() for p in carrier.params()) == 112
    replica = AbPrismParam()
    replica.reset(x, seed=7)
    assert torch.equal(y, replica.render(x))
    print("render: CPU 1x3x64x64; finite [0,1]; non-identity; seeded repeat exact; dof=112")

    # Tensor-only distance is an explicit test dependency, NOT a learned LPIPS
    # approximation or a reported fidelity / editor result.
    distance = lambda z: (z-x).square().mean()
    target = float(distance(quantise(y)))
    caps, supports = make_caps(x, face, distance, target, tolerance=0.4*target)
    assert len(caps) == 6 and float(supports["skin_colour"].sum()) >= 256
    soft = [c.soft(y) for c in caps]
    for value in soft:
        grads = torch.autograd.grad(value, carrier.params(), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    gaps = [s/c.value-1 for s, c in zip(soft, caps)]
    loss = -distance(y) + sum(0.1*g + 5*g.clamp_min(0).square() for g in gaps)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in carrier.params())
    hard = [c.hard(quantise(y)) for c in caps]
    assert all(math_isfinite(v) and v <= c.value for v, c in zip(hard, caps))
    assert all(abs(float(s.detach())-c.hard(y.detach())) < 0.002 for s, c in zip(soft[:3], caps[:3]))
    assert caps[-2].soft(x) < caps[-2].value and caps[-1].soft(x) > caps[-1].value
    print("constraints: all 6 soft/hard paths finite; all 6 gradients nonzero; LPIPS-band signs correct")
    print("distance dependency: synthetic MSE only; no learned LPIPS or editor measurements")

    # Spatial permutation equivariance: a colour has one output everywhere.
    perm = torch.randperm(64*64, generator=torch.Generator().manual_seed(17))
    shuffled = x.flatten(2)[:, :, perm].reshape_as(x)
    ys = carrier.render(shuffled).flatten(2)
    assert torch.allclose(ys, y.detach().flatten(2)[:, :, perm], atol=1e-6, rtol=0)
    state = carrier.state_dict()
    carrier.project()
    assert all(torch.equal(state[k], carrier.state_dict()[k]) for k in state)
    replica.load_state_dict(state)
    assert torch.equal(y, replica.render(x))
    print("global map: permutation error <=1e-6; project has no mutation; state round-trip exact")

    l = torch.linspace(0, 1, 257).requires_grad_()
    ab = torch.tensor([[12.0, 18.0]]).expand(257, 2)
    out = carrier.lightness(l, ab)
    derivative = torch.autograd.grad(out.sum(), l)[0]
    assert out[0] == 0 and out[-1] == 1 and (out[1:] > out[:-1]).all()
    assert derivative.min() >= 1/2.25-1e-5 and derivative.max() <= 2.25+1e-5
    lab = torch.tensor([[[[53.0]], [[12.0]], [[18.0]]]], requires_grad=True)
    jac = torch.autograd.functional.jacobian(carrier.map_lab, lab).reshape(3, 3)
    assert torch.isfinite(jac).all() and torch.linalg.det(jac) > 0
    print("Lab structure: lightness endpoints/monotonicity/slope bounds; positive Jacobian determinant")

    # Directional finite difference checks the differentiable render itself.
    for p in carrier.params():
        p.grad = None
    scalar = carrier.render(x).square().mean()
    grad = torch.autograd.grad(scalar, carrier.tone_raw)[0][12, 1]
    eps = 0.002
    with torch.no_grad():
        carrier.tone_raw[12, 1] += eps
        plus = carrier.render(x).square().mean()
        carrier.tone_raw[12, 1] -= 2*eps
        minus = carrier.render(x).square().mean()
        carrier.tone_raw[12, 1] += eps
    finite_diff = (plus-minus)/(2*eps)
    assert torch.isclose(grad, finite_diff, atol=2e-5, rtol=0.05)
    carrier.load_state_dict(state)
    # Start fresh so the strict no-checkpoint guard is also active in this run.
    carrier.reset(x, seed=7)
    stats = optimise_carrier(carrier, x, TensorObjective(x), caps=caps,
                             steps=4, lr=0.0001, rho=10, lam_every=1,
                             check_every=1, lr_final_ratio=0.2)
    assert stats["free_feasible_step"] > 0 and stats["free_cap_violations"] == 0
    assert stats["free_amplitude_shrink"] == 1
    assert all(c.hard(quantise(carrier.render(x))) <= c.value for c in caps)
    print("solver: 4 real augmented-Lagrangian CPU steps; feasible checkpoint; violations=0; shrink=1")
    print("render gradient: central finite difference within atol=2e-5, rtol=0.05")

    rejected = subprocess.run([sys.executable, "-B", str(Path(__file__)), "no_feasible"],
                              cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "RuntimeError: No feasible checkpoint; post-hoc fit_caps is forbidden" in rejected.stderr
    print("infeasible solve: subprocess exit=1; RuntimeError explicitly forbids post-hoc fit_caps")
    targets = read_targets(ROOT / "lab/results/fidelity.csv", "ab_warp")
    assert len(targets) == 8
    assert not any(n == "diffusers" or n.startswith("diffusers.") or n == "piq"
                   for n in sys.modules)
    print("targets: 8 unique ab_warp rows; model packages not imported; smoke PASS")


def math_isfinite(value):
    return bool(torch.isfinite(torch.tensor(value)))


def no_feasible():
    x, face = synthetic()
    carrier = AbPrismParam()
    carrier.reset(x, seed=7)
    distance = lambda y: (y-x).square().mean()
    caps, _ = make_caps(x, face, distance, target=0.9, tolerance=0.001)
    optimise_carrier(carrier, x, TensorObjective(x), caps=caps,
                     steps=1, lr=0.0001, check_every=1)
    raise AssertionError("An impossible LPIPS band unexpectedly returned")


def legacy_class(source, filename):
    module = ast.parse(source, filename=filename)
    nodes = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AbWarpParam"]
    assert len(nodes) == 1
    namespace = {"torch": torch, "rgb_to_lab": rgb_to_lab, "lab_to_rgb": lab_to_rgb}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace["AbWarpParam"]


def defaults():
    rel = "lab/code/ab_warp_defence.py"
    current = (ROOT / rel).read_bytes()
    before = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                            check=True, capture_output=True).stdout
    assert current == before
    assert hashlib.sha256(current).hexdigest() == BEFORE_SHA256
    print("ab_warp full file: byte-identical to pre-edit SHA256 and git HEAD")
    print(f"SHA256={BEFORE_SHA256}")
    old = legacy_class(before, "ab_warp_before.py")()
    now = legacy_class(current, "ab_warp_after.py")()
    x, _ = synthetic()
    old.reset(x, seed=0)
    now.reset(x, seed=0)
    assert torch.equal(old.render(x), now.render(x))
    optimisers = [torch.optim.Adam(c.params(), lr=0.02) for c in (old, now)]
    for _ in range(4):
        for c, opt in zip((old, now), optimisers):
            opt.zero_grad(set_to_none=True)
            y = c.render(x)
            (y.square().mean()).backward()
            assert all(torch.isfinite(p.grad).all() for p in c.params())
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(old.params(), now.params()))
        for c, opt in zip((old, now), optimisers):
            opt.step()
            c.project()
        assert all(torch.equal(a, b) for a, b in zip(old.params(), now.params()))
        assert torch.equal(old.render(x), now.render(x))
    assert torch.equal(quantise(old.render(x)), quantise(now.render(x)))
    shell = (ROOT / "lab/scripts/defence_cmd.sh").read_bytes()
    shell_before = subprocess.run(["git", "show", "HEAD:lab/scripts/defence_cmd.sh"],
                                  cwd=ROOT, check=True, capture_output=True).stdout
    case = lambda s: s.split(b"  ab_warp)\n", 1)[1].split(b"    ;;", 1)[0]
    assert case(shell) == case(shell_before)
    print("default carrier: init + 4 Adam steps; render/gradients/parameters/quantisation max_abs_diff=0")
    print("legacy class executed verbatim via AST; no diffusion front end imported")
    print("ab_warp shell case: byte-identical; defaults PASS")


if __name__ == "__main__":
    {"smoke": smoke, "defaults": defaults, "no_feasible": no_feasible}[sys.argv[1]]()
