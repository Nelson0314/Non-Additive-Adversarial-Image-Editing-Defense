"""一致性懲罰：預設關閉時行為不變，開啟時梯度真的通得過。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.defense.consistency_loss import (
    amplitude_deviation, make_consistency_term,
)
from src.residual.texture_rephase import PhaseResidual


def _module(**kw):
    return PhaseResidual(size=64, block=16, hop=4, **kw)


def test_權重為零時回傳_None_而不是恆零的函式():
    """回傳 None 讓呼叫端整條路徑不變；回傳恆零函式仍會多一次前向。"""
    assert make_consistency_term(lambda: _module(), torch.rand(1, 3, 64, 64), 0.0) is None
    assert make_consistency_term(lambda: _module(), torch.rand(1, 3, 64, 64), -1.0) is None


def test_theta_為零時懲罰為零():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    m = _module()
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_()
    term = make_consistency_term(lambda: m, x, 1.0)
    assert float(term(m._rephase(x))) < 1e-4


def test_旋轉越大懲罰越大():
    torch.manual_seed(1)
    x = torch.rand(1, 3, 64, 64)
    vals = []
    for s in (0.05, 0.3, 1.0):
        m = _module()
        m.prepare_gates(x)
        with torch.no_grad():
            m.theta.normal_(0.0, s)
        term = make_consistency_term(lambda: m, x, 1.0)
        vals.append(float(term(m._rephase(x))))
    assert vals[0] < vals[1] < vals[2]


def test_梯度通得到_theta():
    """兩條路（rot 依賴 theta、x_def 也依賴 theta）都要保留。"""
    torch.manual_seed(2)
    x = torch.rand(1, 3, 64, 64)
    m = _module()
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.normal_(0.0, 0.3)
    m.theta.requires_grad_(True)
    term = make_consistency_term(lambda: m, x, 1.0)
    term(m._rephase(x)).backward()
    assert m.theta.grad is not None
    assert float(m.theta.grad.abs().sum()) > 0


def test_權重是線性的():
    torch.manual_seed(3)
    x = torch.rand(1, 3, 64, 64)
    m = _module()
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.normal_(0.0, 0.3)
    xd = m._rephase(x)
    a = float(make_consistency_term(lambda: m, x, 1.0)(xd))
    b = float(make_consistency_term(lambda: m, x, 2.5)(xd))
    assert abs(b - 2.5 * a) < 1e-5


def test_形狀不合時報錯():
    with pytest.raises(ValueError):
        amplitude_deviation(torch.zeros(2, 2, dtype=torch.cfloat),
                            torch.zeros(3, 3, dtype=torch.cfloat))


def test_驅動在非相位參數化上拒絕而不是靜默略過():
    import ip2p_run

    class NoModule:
        pass

    args = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--radius", "2.5", "--consistency-weight", "0.5"])
    with pytest.raises(SystemExit):
        ip2p_run._with_consistency(NoModule(), torch.rand(1, 3, 64, 64),
                                   args, lambda z: z.sum())


def test_模組尚未建好時不當場拒絕():
    """`PhaseParam.module` 在 reset() 之前是 None，那時**不可以**判它不合格
    ——實際踩過，三個工作點全部被自己的守門擋下。"""
    import ip2p_run

    class LazyPhase:
        module = None

    args = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--radius", "2.5", "--consistency-weight", "0.5"])
    wrapped = ip2p_run._with_consistency(
        LazyPhase(), torch.rand(1, 3, 64, 64), args, lambda z: z.sum())
    assert wrapped is not None


def test_關閉時驅動回傳原本那個函式物件():
    import ip2p_run
    args = ip2p_run.build_parser().parse_args(["--out", "x", "--radius", "2.5"])
    fn = lambda z: z.sum()
    assert ip2p_run._with_consistency(None, None, args, fn) is fn


def test_退火排程的端點與線性():
    from src.defense.consistency_loss import weight_at
    assert weight_at(0, 100, 0.3, 0.5) == pytest.approx(0.3)
    assert weight_at(25, 100, 0.3, 0.5) == pytest.approx(0.15)
    assert weight_at(50, 100, 0.3, 0.5) == 0.0
    assert weight_at(99, 100, 0.3, 0.5) == 0.0


def test_不退火時權重恆定():
    from src.defense.consistency_loss import weight_at
    for i in (0, 50, 99):
        assert weight_at(i, 100, 0.3, 0.0) == 0.3


def test_退火需要_steps():
    with pytest.raises(ValueError):
        make_consistency_term(lambda: None, torch.rand(1, 3, 8, 8), 0.3,
                              steps=0, decay_frac=0.5)


def test_權重歸零之後整項為零且梯度為零():
    """歸零之後不該再算 analyze，也不該留下任何梯度。"""
    torch.manual_seed(5)
    x = torch.rand(1, 3, 64, 64)
    m = _module()
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.normal_(0.0, 0.3)
    m.theta.requires_grad_(True)
    term = make_consistency_term(lambda: m, x, 0.3, steps=100, decay_frac=0.5)
    term.advance(60)                       # 已過歸零點
    xd = m._rephase(x)
    v = term(xd)
    assert float(v) == 0.0
    v.backward()
    assert m.theta.grad is None or float(m.theta.grad.abs().sum()) == 0.0


def test_step_hook_不影響未給時的行為():
    from src.defense.param_pgd import run_param_pgd

    class Toy:
        name = "toy"; radius = 1.0
        def reset(self, x01, seed): self.p = torch.zeros_like(x01).requires_grad_(True)
        def params(self): return [self.p]
        def render(self, x01): return x01 + self.p
        def project(self):
            with torch.no_grad(): self.p.clamp_(-self.radius, self.radius)

    x = torch.full((1, 1, 4, 4), 0.5)
    loss = lambda y: ((y - 1.0) ** 2).mean()
    a = run_param_pgd(x, Toy(), loss, steps=5, step_size=0.1)
    b = run_param_pgd(x, Toy(), loss, steps=5, step_size=0.1, step_hook=None)
    assert torch.equal(a.x_def, b.x_def)


def test_step_hook_收到的是遞增的步數():
    from src.defense.param_pgd import run_param_pgd

    class Toy:
        name = "toy"; radius = 1.0
        def reset(self, x01, seed): self.p = torch.zeros_like(x01).requires_grad_(True)
        def params(self): return [self.p]
        def render(self, x01): return x01 + self.p
        def project(self):
            with torch.no_grad(): self.p.clamp_(-self.radius, self.radius)

    seen = []
    run_param_pgd(torch.full((1, 1, 4, 4), 0.5), Toy(),
                  lambda y: ((y - 1.0) ** 2).mean(),
                  steps=5, step_size=0.1, step_hook=seen.append)
    assert seen == [0, 1, 2, 3, 4]
