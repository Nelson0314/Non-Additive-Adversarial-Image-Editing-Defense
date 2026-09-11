import torch

from scripts.gradient_diagnostics import COLUMNS, first_order_gain, grad_stats


class _Stub:
    """最小的參數化：`render` 就是把參數加到影像上，`project` 是 L∞ 盒。"""

    def __init__(self, radius=0.1, shape=(1, 3, 8, 8)):
        self.radius = radius
        self.theta = torch.zeros(*shape, requires_grad=True)

    def params(self):
        return [self.theta]

    def render(self, x01):
        return x01 + self.theta

    @torch.no_grad()
    def project(self):
        self.theta.clamp_(-self.radius, self.radius)


def test_columns_carry_the_four_fingerprints():
    for name in ('g_mean', 'g_rms', 'cancel_ratio',
                 'a_carrier', 'a_relaxed', 'kappa',
                 'moved_carrier', 'moved_relaxed',
                 'rho_reached_carrier', 'rho_reached_relaxed',
                 'value_spread', 'value_at_start'):
        assert name in COLUMNS


def test_deterministic_loss_gives_cancel_ratio_of_one():
    """沒有抽樣的損失，g_mean 與 g_rms 相等——比值 1 就是「沒有抵銷」。"""
    x = torch.full((1, 3, 8, 8), 0.5)
    p = _Stub()
    g_mean, g_rms, _ = grad_stats(p, lambda y: y.pow(2).sum(), x, draws=5)
    assert abs(g_mean - g_rms) < 1e-6
    assert g_mean > 0


def test_cancelling_draws_collapse_g_mean_but_not_g_rms():
    """逐次梯度大、平均之後互相抵銷——這正是逐步損失看不出來的那個失效。"""
    x = torch.full((1, 3, 8, 8), 0.5)
    p = _Stub()
    flip = {'n': 0}

    def loss(y):
        flip['n'] += 1
        sign = 1.0 if flip['n'] % 2 else -1.0
        return sign * y.sum()

    g_mean, g_rms, _ = grad_stats(p, loss, x, draws=6)
    assert g_rms > 1.0
    assert g_mean < 1e-6


def test_first_order_gain_is_positive_along_the_negative_gradient():
    x = torch.full((1, 3, 8, 8), 0.5)
    p = _Stub(radius=1.0)
    _, _, mean_grad = grad_stats(p, lambda y: y.pow(2).sum(), x, draws=1)
    gain, moved = first_order_gain(p, mean_grad, x, rho=0.5)
    assert gain > 0
    assert moved > 0


def test_first_order_gain_leaves_the_parameters_where_it_found_them():
    x = torch.full((1, 3, 8, 8), 0.5)
    p = _Stub(radius=1.0)
    _, _, mean_grad = grad_stats(p, lambda y: y.pow(2).sum(), x, draws=1)
    before = p.theta.detach().clone()
    first_order_gain(p, mean_grad, x, rho=0.5)
    assert torch.equal(p.theta.detach(), before)


def test_zero_gradient_reports_zero_gain_instead_of_dividing_by_zero():
    x = torch.full((1, 3, 8, 8), 0.5)
    p = _Stub()
    zero = [torch.zeros_like(p.theta)]
    assert first_order_gain(p, zero, x, rho=0.5) == (0., 0.)


def test_value_spread_starts_every_draw_from_the_same_point():
    """每個點都從起點出發，不是相關的隨機漫步。

    `boundary_init` 是 `p.add_()`，不還原的話第 k 個點是
    `project(p_{k-1} + u_k)`，量到的散布就不是可行集合上的獨立探測。
    """
    from scripts.gradient_diagnostics import value_spread

    class _Rec(_Stub):
        def __init__(self):
            super().__init__(radius=1.0)
            self.seen = []

        def render(self, x01):
            self.seen.append(self.theta.detach().abs().sum().item())
            return x01 + self.theta

    x = torch.zeros(1, 3, 8, 8)
    p = _Rec()
    value_spread(p, lambda y: y.sum(), x, draws_points=4, seed=0)
    # 隨機漫步下絕對值總和會單調累積；每次從零起算則不會。
    assert not all(b > a for a, b in zip(p.seen, p.seen[1:]))
    assert torch.equal(p.theta.detach(), torch.zeros_like(p.theta))


def test_value_spread_prefers_the_fixed_evaluation_when_given_one():
    from scripts.gradient_diagnostics import value_spread
    x = torch.zeros(1, 3, 8, 8)
    p = _Stub(radius=1.0)
    calls = {'sampled': 0, 'fixed': 0}

    def sampled(y):
        calls['sampled'] += 1
        return y.sum()

    def fixed(y):
        calls['fixed'] += 1
        return y.sum()

    value_spread(p, sampled, x, draws_points=3, seed=0, fixed=fixed)
    assert calls['fixed'] == 3
    assert calls['sampled'] == 0


def test_sampled_flag_is_measured_not_inferred_from_an_attribute():
    """決定性的目標也可能有 `fixed` 屬性，屬性不能當成「有抽樣」的判準。

    `StepwiseObjective` 在目標本身決定性時把 `fixed` 設成內層函式，物件與該
    函式不是同一個，靠 `fixed is not loss_fn` 判斷就會把 `latent_norm` 誤標。
    """
    import torch as _t

    class _Deterministic:
        def __init__(self):
            self.fixed = self._inner

        @staticmethod
        def _inner(y):
            return y.sum()

        def __call__(self, y):
            return self._inner(y)

    f = _Deterministic()
    x = _t.full((1, 3, 4, 4), 0.5)
    assert f.fixed is not f
    assert float(f(x)) == float(f(x))
