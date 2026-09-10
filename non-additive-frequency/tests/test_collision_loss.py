import torch

from src.defense.collision_loss import make_collision_loss, ring_of


def _split_image():
    x = torch.zeros(1, 3, 64, 64)
    x[:, 0] = 0.9          # 左半偏紅
    x[:, :, :, 32:] = 0.2  # 右半壓暗
    return x


def _region():
    m = torch.zeros(1, 1, 64, 64)
    m[:, :, 16:48, 4:28] = 1.0
    return m


def test_ring_is_outside_the_region_and_non_empty():
    r = _region()
    ring = ring_of(r, width=6)
    assert float(ring.sum()) > 0
    assert float((ring * r).sum()) == 0


def test_loss_is_zero_when_region_and_ring_match():
    x = torch.full((1, 3, 64, 64), 0.5)
    r = _region()
    loss = make_collision_loss(r, ring_of(r, width=6))
    assert float(loss(x)) < 1e-6


def test_loss_is_positive_when_they_differ():
    x = _split_image()
    r = _region()
    loss = make_collision_loss(r, ring_of(r, width=6))
    assert float(loss(x)) > 1e-3


def test_loss_is_differentiable_and_points_downhill():
    x = _split_image().requires_grad_(True)
    r = _region()
    loss = make_collision_loss(r, ring_of(r, width=6))
    value = loss(x)
    value.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0


def test_a_gradient_step_reduces_the_loss():
    """梯度真的指向合流的方向，不只是非零。"""
    x = _split_image()
    r = _region()
    ring = ring_of(r, width=6)
    loss = make_collision_loss(r, ring)
    before = float(loss(x))
    z = x.clone().requires_grad_(True)
    loss(z).backward()
    after = float(loss((z - 1e-3 * z.grad.sign()).clamp(0, 1)))
    assert after < before


def test_empty_region_raises_instead_of_returning_nan():
    x = _split_image()
    empty = torch.zeros(1, 1, 64, 64)
    try:
        make_collision_loss(empty, ring_of(_region(), width=6))(x)
    except ValueError as e:
        assert 'region' in str(e)
    else:
        raise AssertionError('空區域必須拋錯，回傳 nan 是靜默失效')
