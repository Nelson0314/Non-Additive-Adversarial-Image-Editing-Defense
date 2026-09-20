"""六條結構限制各一個測試，隨機 θ × 隨機影像。

釘的是**原式**不是充分條件
────────────────────────────────────────────────────────────────────
`src/defense/bernstein_colour.py` 用 Bernstein 係數界把六條限制寫成 θ 的
凸限制，那些界比原式保守。這裡不查係數，只在隨機取到的 θ 上直接量
`a`、`b`、`a + c·∇a`、Jacobian 與 Hessian（全部用 autograd），確認原式
成立——界寫錯了而係數檢查照過，只有這一層擋得住。

θ 從哪裡來
────────────────────────────────────────────────────────────────────
`BernsteinColourParam` 的徑向映射：`‖z‖` 取得大（`tanh` 逼近 1）時 θ 會貼
到邊界上，那正是限制最緊、最容易破的地方。低頻錐與影像有關且建一次要
數秒，而六條結構限制與影像無關，所以這裡用 `lowfreq=False` 的域。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defense import bernstein_colour as bc  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
TOL = 1e-6


def random_theta(seed: int) -> torch.Tensor:
    """域內一個貼近邊界的 θ。`‖z‖ = 3` 時 `tanh ≈ 0.995`。"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, 24, 24, generator=gen, dtype=torch.float64)
    domain = bc.BernsteinDomain(x, lowfreq=False)
    param = bc.BernsteinColourParam(x, domain=domain, seed=seed)
    with torch.no_grad():
        param.z.mul_(3.0 / param.z.norm())
    return param.theta().detach().double()


def sample_points(seed: int, count: int = 1536) -> torch.Tensor:
    """立方體內的隨機點 ＋ 八個角 ＋ 一條灰軸，(N,3)、float64。"""
    gen = torch.Generator().manual_seed(1000 + seed)
    rand = torch.rand(count, 3, generator=gen, dtype=torch.float64)
    corners = torch.tensor([[i, j, k] for i in (0.0, 1.0)
                            for j in (0.0, 1.0) for k in (0.0, 1.0)],
                           dtype=torch.float64)
    grey = torch.linspace(0, 1, 33, dtype=torch.float64).unsqueeze(1).repeat(1, 3)
    return torch.cat([rand, corners, grey], dim=0)


def as_image(points: torch.Tensor) -> torch.Tensor:
    """(N,3) → (1,3,N,1)，好餵進 `apply_filter`。"""
    return points.T.reshape(1, 3, -1, 1)


def fields(theta: torch.Tensor, points: torch.Tensor):
    """`a(c)` 與 `b(c)`，形狀 (N,)。"""
    img = as_image(points)
    basis = bc.basis_of(img)
    a_coeff, b_coeff = bc.split(theta)
    a = bc.evaluate_field(a_coeff, basis).reshape(-1)
    b = bc.evaluate_field(b_coeff, basis).reshape(-1)
    return a, b


def transform(theta: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """`T_θ` 作用在 (N,3) 的點集上，回傳 (N,3)。"""
    out = bc.apply_filter(as_image(points), theta)
    return out.reshape(3, -1).T


def jacobian(theta: torch.Tensor, points: torch.Tensor, create_graph=False):
    """`J_{T−I}`，形狀 (N,3,3)，`[n,i,j] = ∂_j(T−I)_i`。"""
    c = points.clone().requires_grad_(True)
    y = transform(theta, c) - c
    rows = []
    for i in range(3):
        grad, = torch.autograd.grad(y[:, i].sum(), c, create_graph=True)
        rows.append(grad)
    jac = torch.stack(rows, dim=1)
    return (jac, c) if create_graph else (jac.detach(), c)


# ── 限制 1：`0 ≤ a ≤ 0.25`、`0 ≤ b ≤ 0.1` ───────────────────────────

@pytest.mark.parametrize('seed', SEEDS)
def test_amplitude_bounds(seed):
    theta = random_theta(seed)
    a, b = fields(theta, sample_points(seed))
    assert float(a.min()) >= -TOL
    assert float(a.max()) <= bc.A_CAP + TOL
    assert float(b.min()) >= -TOL
    assert float(b.max()) <= bc.B_CAP + TOL


# ── 限制 2：三通道共用乘數與中性灰加量 ──────────────────────────────

@pytest.mark.parametrize('seed', SEEDS)
def test_shared_multiplier_keeps_hue(seed):
    """`T(c) = m·c + k·1`，`m > 0`、`k ≥ 0`：色相不變、飽和度不增、不變亮。

    色相只取決於 `(c_i − c_min)/(c_max − c_min)`，同乘再同加不改變它；
    這一條因此是形式本身的性質，不靠任何係數限制。
    """
    theta = random_theta(seed)
    points = sample_points(seed)
    y = transform(theta, points)
    a, b = fields(theta, points)
    mult = (1.0 - a - b).unsqueeze(1)
    add = (b * (points * torch.tensor(bc.LUMA, dtype=torch.float64)).sum(1)
           ).unsqueeze(1)
    assert torch.allclose(y, mult * points + add, atol=1e-10)
    assert float(mult.min()) > 0.0
    assert float(add.min()) >= -TOL

    spread = points.max(1).values - points.min(1).values
    new_spread = y.max(1).values - y.min(1).values
    live = spread > 1e-6
    # 色相：三個通道的相對位置不變。
    ratio = ((points - points.min(1, keepdim=True).values)[live]
             / spread[live].unsqueeze(1))
    new_ratio = ((y - y.min(1, keepdim=True).values)[live]
                 / new_spread[live].unsqueeze(1))
    assert float((ratio - new_ratio).abs().max()) < 1e-8
    # 飽和度（HSV）不增、最大通道不增。
    sat = spread[live] / points.max(1).values[live].clamp_min(1e-12)
    new_sat = new_spread[live] / y.max(1).values[live].clamp_min(1e-12)
    assert float((new_sat - sat).max()) <= TOL
    assert float((y.max(1).values - points.max(1).values).max()) <= TOL
    assert float(y.min()) >= -TOL and float(y.max()) <= 1.0 + TOL


# ── 限制 3：`0 ≤ a + c·∇a ≤ 0.25` ───────────────────────────────────

@pytest.mark.parametrize('seed', SEEDS)
def test_radial_derivative_of_a(seed):
    theta = random_theta(seed)
    c = sample_points(seed).clone().requires_grad_(True)
    a, _ = fields(theta, c)
    grad, = torch.autograd.grad(a.sum(), c)
    radial = a.detach() + (c.detach() * grad).sum(1)
    assert float(radial.min()) >= -TOL
    assert float(radial.max()) <= bc.A_CAP + TOL
    # 等價說法：沿色相射線 `t ↦ t(1 − a)` 的斜率落在 0.75–1。
    slope = 1.0 - radial
    assert float(slope.min()) >= 1.0 - bc.A_CAP - TOL
    assert float(slope.max()) <= 1.0 + TOL


# ── 限制 4：`0 ≤ b + c·∇b ≤ 0.1` ────────────────────────────────────

@pytest.mark.parametrize('seed', SEEDS)
def test_radial_derivative_of_b(seed):
    theta = random_theta(seed)
    c = sample_points(seed).clone().requires_grad_(True)
    _, b = fields(theta, c)
    grad, = torch.autograd.grad(b.sum(), c)
    radial = b.detach() + (c.detach() * grad).sum(1)
    assert float(radial.min()) >= -TOL
    assert float(radial.max()) <= bc.B_CAP + TOL


# ── 限制 5：`sup‖J_{T−I}‖_∞ ≤ 0.35`，以及等價的雙邊 Lipschitz ───────

@pytest.mark.parametrize('seed', SEEDS)
def test_jacobian_infinity_norm(seed):
    theta = random_theta(seed)
    jac, _ = jacobian(theta, sample_points(seed))
    row_sums = jac.abs().sum(dim=2)
    assert float(row_sums.max()) <= bc.JAC_CAP + TOL


@pytest.mark.parametrize('seed', SEEDS)
def test_two_sided_lipschitz(seed):
    """`0.65‖c−d‖_∞ ≤ ‖T(c)−T(d)‖_∞ ≤ 1.35‖c−d‖_∞`。

    近距離的點對（色塊與折疊都發生在那裡）另外抽一組：均勻取點兩兩之間
    的距離集中在 0.3 附近，只用它會漏掉窄色階。
    """
    theta = random_theta(seed)
    gen = torch.Generator().manual_seed(500 + seed)
    p = torch.rand(4096, 3, generator=gen, dtype=torch.float64)
    q = torch.rand(4096, 3, generator=gen, dtype=torch.float64)
    near = (p + 0.01 * (torch.rand(4096, 3, generator=gen,
                                   dtype=torch.float64) - 0.5)).clamp(0, 1)
    for other in (q, near):
        gap = (p - other).abs().max(1).values
        moved = (transform(theta, p) - transform(theta, other)).abs().max(1).values
        live = gap > 1e-12
        ratio = moved[live] / gap[live]
        assert float(ratio.min()) >= 1.0 - bc.JAC_CAP - 1e-9
        assert float(ratio.max()) <= 1.0 + bc.JAC_CAP + 1e-9


# ── 限制 6：`sup max_i Σ_jk |∂_jk(T−I)_i| ≤ 1` ──────────────────────

@pytest.mark.parametrize('seed', SEEDS)
def test_second_derivative_row_sum(seed):
    theta = random_theta(seed)
    jac, c = jacobian(theta, sample_points(seed, count=512), create_graph=True)
    total = torch.zeros(jac.shape[0], 3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            grad, = torch.autograd.grad(jac[:, i, j].sum(), c,
                                        retain_graph=True)
            total[:, i] += grad.abs().sum(1)
    assert float(total.max()) <= bc.HESS_CAP + TOL


# ── 參數化本身：不需要鉗回 ──────────────────────────────────────────

@pytest.mark.parametrize('seed', SEEDS)
def test_forward_map_lands_inside(seed):
    """任何 `z` 都落在域內，而且 `project()` 不會改變 θ。"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, 24, 24, generator=gen)
    domain = bc.BernsteinDomain(x, lowfreq=False)
    param = bc.BernsteinColourParam(x, domain=domain, seed=seed)
    for scale in (1e-3, 0.5, 3.0, 30.0):
        with torch.no_grad():
            param.z.mul_(scale / param.z.norm())
        theta = param.theta().detach()
        assert not bc.structural_violations(theta), \
            f'scale {scale} 的 θ 違反 {bc.structural_violations(theta)}'
        before = theta.clone()
        param.project()
        assert torch.equal(param.theta().detach(), before)


def test_radius_is_tight():
    """`max_radius` 給的是**邊界**：內一點可行、外一點不可行。"""
    gen = torch.Generator().manual_seed(7)
    x = torch.rand(1, 3, 24, 24, generator=gen)
    domain = bc.BernsteinDomain(x, lowfreq=False)
    theta0, _ = bc.interior_point(domain)
    for k in range(5):
        d = torch.randn(bc.DIM, generator=gen)
        d = d / d.norm()
        r = domain.max_radius(theta0, d)
        assert r > 0
        assert domain.feasible(theta0 + 0.999 * r * d)
        assert not domain.feasible(theta0 + 1.01 * r * d)
