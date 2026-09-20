"""三個座標臂的測試。要釘住的是「換座標沒有偷偷換掉問題」。

三件事會靜默地把診斷變成另一個問題的答案，全部在這裡釘住：

1. **白化截斷。** `M` 若丟掉小特徵值方向，B 臂就不是在 250 維裡走，而是
   在 30–50 維裡走；幾何瓶頸會被偽裝成座標問題，而且不會報錯。
2. **起點不同。** 三臂的 θ 起點只要差一點，`ΔJ` 的比值就不再是座標的
   效果。
3. **限制沒有真的成立。** C 的「非暗化」若只是損失項或事後投影，增量就
   不是硬性落在子空間裡。

影像與 `tests/test_reachability.py` 同一條線：真的肖像縮到 256²（再小低頻
錐對常數係數場已經沒有嚴格內點）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense import bernstein_colour as bc  # noqa: E402
from src.defense import coordinate_arms as ca  # noqa: E402
from src.defense import reachability as rb  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

SIZE = 256
IMAGE = ROOT / 'data' / 'portraits' / 'man' / 'man_00.png'


@pytest.fixture(scope='module')
def setup():
    x = load_image_tensor(IMAGE, torch.device('cpu'), size=SIZE)
    domain = bc.BernsteinDomain(x)
    theta0, _ = bc.interior_point(domain)
    gram = bc.perturbation_gram(x).numpy()
    whiten = ca.RegularisedWhitening(gram)
    arms = ca.build_arms(whiten)
    radial = ca.RadialTheta(domain, theta0.double())
    return x, domain, theta0, gram, whiten, arms, radial


def unit(seed: int) -> torch.Tensor:
    gen = torch.Generator(device='cpu').manual_seed(seed)
    v = torch.randn(bc.DIM, generator=gen, dtype=torch.float64)
    return v / v.norm()


# ---- 白化：滿秩、不截斷 ----

def test_whitening_keeps_every_direction(setup):
    """250 維全部保留，一個都不丟。"""
    *_, whiten, _, _ = setup
    assert whiten.dim == bc.DIM
    assert whiten.inv_sqrt.shape == (bc.DIM, bc.DIM)
    assert np.linalg.matrix_rank(whiten.inv_sqrt) == bc.DIM


def test_small_eigenvalues_are_floored_not_dropped(setup):
    """地板是 `τ = 1e−4·λ_max`，而且真的有方向被抬起來。

    被抬起來的方向數 > 0 才代表這個設定有意義；若一個都沒有，`M = G`，
    A 與 B 的差別就只剩浮點誤差。
    """
    *_, whiten, _, _ = setup
    assert whiten.tau == pytest.approx(ca.TAU_REL * whiten.lam_max)
    assert whiten.lam.min() == pytest.approx(whiten.tau)
    assert whiten.floored > 0
    assert whiten.lam_min_raw < whiten.tau


def test_regularised_whitening_is_wider_than_the_truncating_one(setup):
    """與 `reachability.Whitening` 並排：那一支的有效維度遠小於 250。

    這一條是為了讓「不要截斷」這個規格在測試裡有一個可查的對照，而不是
    只寫在註解裡。
    """
    _, _, _, gram, whiten, _, _ = setup
    truncating = rb.Whitening(gram)
    assert truncating.dim < whiten.dim
    assert whiten.dim == bc.DIM


def test_sqrt_and_inverse_are_mutually_inverse(setup):
    *_, whiten, _, _ = setup
    prod = whiten.sqrt @ whiten.inv_sqrt
    assert np.allclose(prod, np.eye(bc.DIM), atol=1e-9)


def test_inv_sqrt_squared_is_m_inverse(setup):
    """`M^{−1/2}M^{−1/2} = M^{−1}`：預條件化的確是 `M⁻¹` 而不是別的東西。"""
    *_, whiten, _, _ = setup
    m = (whiten.basis * whiten.lam) @ whiten.basis.T
    assert np.allclose(whiten.inv_sqrt @ whiten.inv_sqrt @ m,
                       np.eye(bc.DIM), atol=1e-8)


# ---- 座標：可逆、起點相同 ----

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_each_arm_roundtrips(setup, seed):
    *_, arms, _ = setup
    v = unit(seed)
    for arm in arms.values():
        back = arm.to_v(arm.to_z(v))
        assert torch.allclose(back, v, atol=1e-9)


@pytest.mark.parametrize('seed', [0, 3])
def test_all_arms_start_at_the_same_theta(setup, seed):
    """起點的 θ 必須逐位相同，否則 `ΔJ` 的比值量的不只是座標。"""
    *_, arms, radial = setup
    v_start = 0.01 * unit(seed)
    thetas = [radial(arm.to_v(arm.to_z(v_start))).detach()
              for arm in arms.values()]
    for t in thetas[1:]:
        assert torch.allclose(t, thetas[0], atol=1e-10)


# ---- C 的限制 ----

def test_nondark_projection_is_exact_in_the_g_metric(setup):
    """投影後的更新量與 `DARK` 在 `G` 內積下正交。

    `z` 空間的那一條（`nᵀΔz = 0`）是**逐位**成立的，因為投影就是照它寫的；
    θ 空間的餘弦則要重算一次 `DARKᵀG·M^{−1/2}p`，結合順序不同、又多了
    `M^{−1/2}` 的 100 倍條件數，所以門檻比 `test_reachability.py` 的 1e−10
    鬆三個數量級。兩個都測：前者釘實作，後者釘它的意思。
    """
    _, _, _, gram, whiten, arms, _ = setup
    arm = arms['whitened_nondark']
    rng = np.random.default_rng(11)
    for _ in range(5):
        u = torch.as_tensor(rng.standard_normal(bc.DIM))
        p = arm.project(u)
        assert arm.violation(p) < 1e-12
        dv = arm.to_v(p).numpy()
        assert abs(rb._cos_g(dv, rb.DARK, gram)) < 1e-7


def test_accumulated_increment_stays_in_the_subspace(setup):
    """連走 20 步，`Δz` 仍然滿足 `UᵀG M^{−1/2}Δz = 0`。

    限制是恆等成立的（每一步的更新量都在零空間裡），不是每步再投影一次
    才勉強成立。
    """
    _, _, _, gram, _, arms, _ = setup
    arm = arms['whitened_nondark']
    rng = np.random.default_rng(5)
    z0 = arm.to_z(0.01 * unit(7))
    z = z0.clone()
    for _ in range(20):
        u = torch.as_tensor(rng.standard_normal(bc.DIM))
        z = z + 0.03 * arm.project(u)
    dv = arm.to_v(z - z0).numpy()
    assert abs(rb._cos_g(dv, rb.DARK, gram)) < 1e-7
    assert arm.violation(z - z0) < 1e-12


def test_unprojected_arm_does_move_along_darkening(setup):
    """對照：B 臂不投影，隨機更新在 `G` 內積下**不**與暗化正交。"""
    _, _, _, gram, _, arms, _ = setup
    arm = arms['whitened']
    rng = np.random.default_rng(3)
    u = torch.as_tensor(rng.standard_normal(bc.DIM))
    dv = arm.to_v(arm.project(u)).numpy()
    assert abs(rb._cos_g(dv, rb.DARK, gram)) > 1e-3


# ---- 前向參數化：域仍然精確 ----

@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('scale', [0.01, 0.5, 3.0, 40.0])
def test_theta_is_feasible_for_every_arm_and_scale(setup, seed, scale):
    """任何 `z`（包含把 `tanh` 推到飽和的）映出來的 θ 都在域內。"""
    _, domain, _, _, _, arms, radial = setup
    for arm in arms.values():
        z = scale * unit(seed)
        theta = radial(arm.to_v(z)).detach()
        slacks = ca.slack_report(domain, theta)
        assert slacks['min_slack'] > 0, (arm.name, seed, scale, slacks)


def test_radial_map_matches_the_existing_parameterisation(setup):
    """A 臂的 `v → θ` 與 `BernsteinColourParam.theta()` 是同一條式子。

    釘這一條是因為診斷宣稱「用既有的前向參數化」；兩邊若分岔，硬約束的
    保證就不是既有那一份。
    """
    x, domain, theta0, _, _, arms, radial = setup
    param = bc.BernsteinColourParam(x, domain=domain)
    assert torch.allclose(param.theta0, theta0)
    with torch.no_grad():
        param.z.copy_(unit(9).float() * 0.7)
    expected = param.theta().detach().double()
    got = radial(arms['theta'].to_v(unit(9) * 0.7)).detach()
    assert torch.allclose(got, expected, atol=1e-6)


def test_gradient_reaches_z_through_the_whitening(setup):
    """梯度接得回 `z`，而且 B 的梯度是 A 的梯度經 `M^{−1/2}` 變換。

    用一個便宜的替身目標（θ 的線性泛函）釘鏈式規則，不必載 VAE。
    """
    *_, whiten, arms, radial = setup
    w = torch.as_tensor(np.random.default_rng(2).standard_normal(bc.DIM))
    grads = {}
    for name, arm in arms.items():
        z = arm.to_z(0.3 * unit(4)).clone().requires_grad_(True)
        value = (radial(arm.to_v(z)) * w).sum()
        g, = torch.autograd.grad(value, z)
        grads[name] = g
    expected = torch.as_tensor(whiten.inv_sqrt) @ grads['theta']
    assert torch.allclose(grads['whitened'], expected, atol=1e-8)
