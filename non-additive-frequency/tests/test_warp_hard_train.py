"""位移場硬訓練那一批新加的三個旋鈕：`init_std`、`step_size`、`update`。

三個都是**加選項、預設行為不變**。這裡釘的就是那個「不變」——一旦破了，
既有的 `runs/ip2p_warp/opt_r*` 與所有相位臂的批次都不能重跑。
"""

from __future__ import annotations

import pytest
import torch

from src.defense.param_pgd import WarpParam, run_param_pgd


def _image(seed: int = 0, n: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g)


def _loss(x: torch.Tensor) -> torch.Tensor:
    """一個可微、有明確極小值的替身損失，不需要擴散模型。"""
    return (x - 0.5).pow(2).mean()


def test_init_std為零時起點仍是全零():
    p = WarpParam(radius=8.0, grid=8)
    p.reset(_image(), seed=3)
    assert torch.count_nonzero(p.c) == 0
    assert p.c.requires_grad


def test_init_std非零時起點隨seed改變且夾在半徑內():
    x = _image()
    a, b = WarpParam(radius=2.0, grid=8, init_std=1.5), WarpParam(
        radius=2.0, grid=8, init_std=1.5)
    a.reset(x, seed=1); b.reset(x, seed=2)
    assert not torch.allclose(a.c, b.c), "不同 seed 必須抽到不同起點"
    same = WarpParam(radius=2.0, grid=8, init_std=1.5)
    same.reset(x, seed=1)
    assert torch.allclose(a.c, same.c), "同 seed 必須逐位元相同"
    assert float(a.c.detach().abs().max()) <= 2.0 + 1e-6


def test_零位移時輸出逐位元等於原圖():
    """加了 `init_std` 之後這條構造保證仍要成立，否則失真有一個無償的地板。"""
    x = _image(5, n=64)
    p = WarpParam(radius=8.0, grid=8)
    p.reset(x, seed=0)
    assert torch.equal(p.render(x), x)


def test_預設更新規則與step_size不變時逐位元等於舊行為():
    """舊行為 = sign PGD、步長 radius/(steps·saturate_at)。手寫一遍對照。"""
    x = _image(7)
    steps, sat = 6, 0.25

    p = WarpParam(radius=4.0, grid=8)
    got = run_param_pgd(x, p, _loss, steps=steps, saturate_at=sat, seed=0)

    ref = WarpParam(radius=4.0, grid=8)
    ref.reset(x, seed=0)
    alpha = ref.radius / (steps * sat)
    for _ in range(steps):
        loss = _loss(ref.render(x))
        (g,) = torch.autograd.grad(loss, [ref.c])
        with torch.no_grad():
            ref.c.sub_(alpha * torch.sign(g))
        ref.project()
    with torch.no_grad():
        assert torch.equal(got.x_def, ref.render(x).detach())


def test_step_size會取代半徑推出來的步長():
    x = _image(8)
    p = WarpParam(radius=4.0, grid=8)
    a = run_param_pgd(x, p, _loss, steps=4, seed=0)
    q = WarpParam(radius=4.0, grid=8)
    b = run_param_pgd(x, q, _loss, steps=4, seed=0, step_size=0.01)
    assert not torch.equal(a.x_def, b.x_def)


def test_adam走的路與sign不同且參數確實在動():
    x = _image(9)
    p = WarpParam(radius=8.0, grid=8, init_std=0.5)
    r = run_param_pgd(x, p, _loss, steps=12, seed=1, update="adam",
                      step_size=0.05, log_every=1)
    mags = [h["param_absmean"] for h in r.history]
    assert len(mags) == 12
    assert mags[-1] != mags[0], "參數沒有移動的話這一批問不出任何事"
    q = WarpParam(radius=8.0, grid=8, init_std=0.5)
    s = run_param_pgd(x, q, _loss, steps=12, seed=1, update="sign",
                      step_size=0.05)
    assert not torch.equal(r.x_def, s.x_def)


def test_未知更新規則要拋出而不是靜默退回sign():
    x = _image(10)
    p = WarpParam(radius=4.0, grid=8)
    with pytest.raises(ValueError, match="未知的更新規則"):
        run_param_pgd(x, p, _loss, steps=2, seed=0, update="rmsprop")


def test_history記錄了參數量級供收斂判定():
    x = _image(11)
    p = WarpParam(radius=4.0, grid=8)
    r = run_param_pgd(x, p, _loss, steps=5, seed=0, log_every=1)
    assert all("param_absmean" in h and "loss" in h for h in r.history)
