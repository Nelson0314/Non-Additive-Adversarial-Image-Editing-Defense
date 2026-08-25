"""`--coarsen`：三個空間場改存在粗視窗網格上、前向雙線性升取樣。

釘住四件事：

1. **`coarsen=1` 逐位元等於加這個旋鈕之前**——否則所有既有批次都不能重跑。
2. **凸包界**：`|expand(p)| <= max|p|`。可行集因此可以只定義在參數上，
   `project()` 夾粗網格就夠了，不必另外夾升取樣後的每一個視窗。
3. **參數量真的變少**，而且視窗網格的邊長是 `ceil(side/k)`。
4. **與 `theta_budget` 併用時拒絕啟動**，因為 `theta_cap` 定義在細網格上。
"""
from __future__ import annotations

import math

import pytest
import torch

from src.defense.param_pgd import PhaseParam
from src.residual.texture_rephase import PhaseResidual

SIZE, BLOCK, HOP = 64, 16, 8


def make(coarsen: int, **kw) -> PhaseResidual:
    return PhaseResidual(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                         theta_max=math.pi, coarsen=coarsen, **kw)


def image() -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return torch.rand(1, 3, SIZE, SIZE, generator=g)


def test_coarsen_1_是恆等的升取樣():
    m = make(1)
    p = torch.randn(1, m.n_blocks, BLOCK, BLOCK // 2 + 1)
    assert m.expand(p) is p


def test_coarsen_1_的前向完全不呼叫插值(monkeypatch):
    """「逐位元等於加這個旋鈕之前」的直接證據。

    拿兩個 `coarsen=1` 的模組互比是套套邏輯——兩邊走的是同一條新程式。
    真正要證的是**新程式在關閉時沒有多做任何事**，所以把
    `F.interpolate` 換成會爆的樁：三個場全開仍跑得完，就表示插值沒被碰到。
    """
    import src.residual.texture_rephase as tr

    def boom(*a, **k):
        raise AssertionError("coarsen=1 不該呼叫 F.interpolate")

    x = image()
    m = make(1, spectral_floor=0.04, gain_max=1.0)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.normal_(generator=torch.Generator().manual_seed(3))
        m.gain.normal_(generator=torch.Generator().manual_seed(4))
        m.floor.uniform_(-1, 1, generator=torch.Generator().manual_seed(5))
    monkeypatch.setattr(tr.F, "interpolate", boom)
    out = m.pixel_residual(x)
    assert torch.isfinite(out).all()


def test_coarsen_大於1_的前向確實會插值(monkeypatch):
    """上一項的對照：換成會爆的樁時，k > 1 必須爆。否則那個樁根本沒裝上。"""
    import src.residual.texture_rephase as tr

    def boom(*a, **k):
        raise AssertionError("樁有裝上")

    x = image()
    m = make(2)
    m.prepare_gates(x)
    monkeypatch.setattr(tr.F, "interpolate", boom)
    with pytest.raises(AssertionError, match="樁有裝上"):
        m.pixel_residual(x)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_粗網格邊長是天花板除法(k: int):
    m = make(k)
    side = (SIZE + BLOCK - BLOCK) // HOP + 1  # padded = SIZE + block
    side = (SIZE + 2 * (BLOCK // 2) - BLOCK) // HOP + 1
    assert m.side == side
    assert m.side_c == -(-side // k)
    assert m.theta.shape[1] == m.side_c ** 2
    assert m.theta.shape[1] < m.n_blocks


@pytest.mark.parametrize("k", [2, 4])
def test_升取樣後的形狀是逐視窗(k: int):
    m = make(k)
    out = m.expand(m.theta)
    assert out.shape == (1, m.n_blocks, BLOCK, BLOCK // 2 + 1)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_升取樣不會超出粗網格的極值(k: int):
    """雙線性是四個鄰居的凸組合，`align_corners=True` 保證不外插。"""
    m = make(k)
    g = torch.Generator().manual_seed(7)
    p = torch.randn(1, m.side_c ** 2, BLOCK, BLOCK // 2 + 1, generator=g) * 2.0
    out = m.expand(p)
    assert out.max() <= p.max() + 1e-6
    assert out.min() >= p.min() - 1e-6
    assert out.abs().max() <= p.abs().max() + 1e-6


def test_常數場升取樣後仍是同一個常數():
    m = make(4)
    p = torch.full((1, m.side_c ** 2, BLOCK, BLOCK // 2 + 1), 0.37)
    assert torch.allclose(m.expand(p), torch.full_like(m.expand(p), 0.37))


def test_升取樣的角落精確等於粗網格的角落():
    """`align_corners=True` 的定義。角落對不上就表示插值有半格偏移。"""
    m = make(2)
    g = torch.Generator().manual_seed(9)
    p = torch.randn(1, m.side_c ** 2, BLOCK, BLOCK // 2 + 1, generator=g)
    out = m.expand(p)
    pc = p.reshape(1, m.side_c, m.side_c, -1)
    oc = out.reshape(1, m.side, m.side, -1)
    for (a, b), (c, d) in (((0, 0), (0, 0)), ((0, -1), (0, -1)),
                           ((-1, 0), (-1, 0)), ((-1, -1), (-1, -1))):
        assert torch.allclose(pc[0, a, b], oc[0, c, d], atol=1e-6)


def test_粗網格讓相鄰視窗的角度變平滑():
    """判準的核心：同樣的參數分布下，逐視窗的變異必須明顯下降。"""
    fine, coarse = make(1), make(4)
    g = torch.Generator().manual_seed(13)
    tf = torch.randn(1, fine.n_blocks, BLOCK, BLOCK // 2 + 1, generator=g)
    g = torch.Generator().manual_seed(13)
    tc = torch.randn(1, coarse.side_c ** 2, BLOCK, BLOCK // 2 + 1, generator=g)

    def neighbour_var(m, t):
        grid = m.expand(t).reshape(1, m.side, m.side, -1)
        return float((grid[:, 1:] - grid[:, :-1]).pow(2).mean())

    assert neighbour_var(coarse, tc) < 0.35 * neighbour_var(fine, tf)


def test_theta_budget_併用時拒絕啟動():
    with pytest.raises(ValueError, match="theta_cap"):
        make(2, theta_budget=0.5)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_不合法的倍率要拋錯(bad):
    with pytest.raises(ValueError, match="coarsen"):
        make(bad)


def test_PhaseParam_把倍率轉交下去():
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12, coarsen=3)
    par.reset(image(), seed=0)
    assert par.module.coarsen == 3
    assert par.module.theta.shape[1] == par.module.side_c ** 2


def test_project_夾粗網格就足以夾住全部視窗():
    """凸包界的實際用途：可行集只定義在參數上。"""
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     radius=0.9, coarsen=3)
    par.reset(image(), seed=0)
    with torch.no_grad():
        par.module.theta.normal_(generator=torch.Generator().manual_seed(17))
        par.module.theta.mul_(5.0)
    par.project()
    assert par.module.theta.abs().max() <= 0.9 + 1e-6
    assert par.module.expand(par.module.theta).abs().max() <= 0.9 + 1e-6


def test_梯度傳得回粗網格():
    x = image()
    m = make(2)
    m.prepare_gates(x)
    m.pixel_residual(x).pow(2).sum().backward()
    assert m.theta.grad is not None
    assert m.theta.grad.shape == m.theta.shape
    assert float(m.theta.grad.abs().sum()) > 0


def test_加密後的_jpeg_格點可以被選到():
    """計畫零：交叉點落在 jpeg75 與 jpeg50 之間，四點畫不出它在哪。"""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    from phase_retention import label, purifier_set

    want = ["identity", "jpeg90", "jpeg75", "jpeg60",
            "jpeg50", "jpeg40", "jpeg30", "jpeg20"]
    got = {label(p) for p in purifier_set(None, seed=0, only=want)}
    assert got == set(want)
