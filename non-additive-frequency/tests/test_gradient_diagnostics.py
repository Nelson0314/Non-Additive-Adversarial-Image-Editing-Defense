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
