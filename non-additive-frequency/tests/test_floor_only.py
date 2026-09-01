"""只有加性下限、相位與幅度都不動的那一格。

`DECISIONS.md` 撤回「不做加性項」時，站得住的唯一理由是「非加性那一半買的是
**感知代價**」，而那句話的證據是一次性探針裡 `radius 0.1` 的近似——theta_max
與 gain_max 都還有 0.1，不是真的關掉，而且程式已刪除。真正的「只有加性」
從未在主線程式上跑過，因為 `PhaseParam` 有一道**在加性下限存在之前寫的**
自由度檢查把它擋住了。
"""

import math

import pytest
import torch

from src.defense.param_pgd import PhaseParam

SIZE = 128


def _image() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, SIZE, SIZE, generator=g)
    x[:, :, :, : SIZE // 2] = 0.45
    return x


def _param(**kw) -> PhaseParam:
    base = dict(size=SIZE, block=32, hop=8, r_min=0.12, radius=2.0,
                energy_quantile=0.0, freq_weight="jpeg_luma",
                freq_weight_power=0.25, gain_ratio=0.0, phase_on=False)
    base.update(kw)
    return PhaseParam(**base)


def test_三個自由度全關才拋錯():
    with pytest.raises(ValueError, match="沒有任何自由度"):
        _param(spectral_floor=0.0)


def test_只留下限時可以建起來且恰有一個可學參數():
    p = _param(spectral_floor=0.04)
    p.reset(_image(), 0)
    params = p.params()
    assert len(params) == 1
    assert params[0] is p.module.floor


def test_相位與幅度都真的不動():
    p = _param(spectral_floor=0.04)
    x = _image()
    p.reset(x, 0)
    assert p.module.gain_max == 0.0
    assert float(p.module.theta.abs().max()) == 0.0
    # 就算有人把 theta 寫進去，phase_on=False 也不會把它交給最佳化
    with torch.no_grad():
        p.module.theta.fill_(1.0)
    assert not any(t is p.module.theta for t in p.params())


def test_下限拉滿時確實動得了平坦區():
    """加法項存在的理由就是要進到乘法碰不到的地方。"""
    p = _param(spectral_floor=0.04)
    x = _image()
    p.reset(x, 0)
    assert float((p.render(x) - x).abs().max()) < 1e-5      # 初始化即恆等
    with torch.no_grad():
        p.module.floor.uniform_(-1.0, 1.0)
    flat = (p.render(x) - x)[:, :, :, : SIZE // 2]
    assert float(flat.abs().max()) > 1e-3


def test_radius在這一格上完全沒有作用():
    x = _image()
    a, b = _param(spectral_floor=0.04, radius=0.5), _param(spectral_floor=0.04,
                                                           radius=3.0)
    a.reset(x, 0); b.reset(x, 0)
    with torch.no_grad():
        a.module.floor.uniform_(-1.0, 1.0)
        b.module.floor.copy_(a.module.floor)
    assert torch.equal(a.render(x), b.render(x))


def test_投影只夾下限係數():
    p = _param(spectral_floor=0.04)
    p.reset(_image(), 0)
    with torch.no_grad():
        p.module.floor.fill_(5.0)
    p.project()
    assert float(p.module.floor.max()) == pytest.approx(1.0)


def test_build_在沒給下限時拒絕這個條件():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from phase_ablation import build
    with pytest.raises(ValueError, match="floor_only"):
        build("floor_only", 0, spectral_floor=0.0)
    param, lo, hi = build("floor_only", 0, block=32, hop=8, quantile=0.0,
                          spectral_floor=0.04)
    assert param.phase_on is False and param.gain_ratio == 0.0


def test_驅動認得這個條件名():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ip2p_run import PHASE_CONDS
    assert "floor_only" in PHASE_CONDS
