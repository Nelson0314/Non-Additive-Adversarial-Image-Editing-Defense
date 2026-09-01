"""AdvDrop 的「論文模式」——重現 Table 1／2 需要的那組設定。

2026-08-21 取得論文全文後發現：正文 §3.1 寫 `q_init = 1`、約束
`‖q − q_init‖∞ < ε`，§4.3 掃 `ε ∈ {20, 60, 100}`；而官方 `infod_sample.py`
的簽章預設是 `q_size = 10`、`factor_range = [5, 10]`、初始值 10。**可動區間
差一個數量級，方向也相反**（論文由 1 往上，程式碼由 10 往下）。

這一支釘住四件事：

1. 預設仍是程式碼模式，逐位元與 08-19 的實作相同——不能因為加了新模式就
   把既有批次的意義改掉；
2. 論文模式的初始值與夾取範圍確實是 `[q_init, q_init + eps]`；
3. 兩個欄位只給一個時拒絕——那時可動區間沒有定義，靜默取預設會量到第三種
   東西；
4. 論文寫死的三個數字（q_init、ε 掃描、步數）不被改動。
"""

import pytest
import torch

from src.baselines.advdrop import (
    PAPER_EPS_SWEEP, PAPER_Q_INIT, PAPER_Q_MIN, PAPER_Q_SIZE,
    PAPER_TARGETED_STEPS, PAPER_UNTARGETED_STEPS, AdvDropSpec, run_advdrop,
)


class _StubSD:
    """`encode_image` 取每 8×8 區塊的平均，形狀對得上 VAE 且可微。"""

    def encode_image(self, x01, use_ckpt: bool = False):
        return torch.nn.functional.avg_pool2d(x01, 8)


def _img(size=32):
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, size, size, generator=g, dtype=torch.float64)


def test_預設是程式碼模式():
    s = AdvDropSpec(name="x")
    assert s.paper_mode is False
    assert s.bounds() == (PAPER_Q_MIN, PAPER_Q_SIZE, PAPER_Q_SIZE)


def test_論文模式的區間由q_init與eps決定():
    s = AdvDropSpec(name="x", q_init=PAPER_Q_INIT, eps=100.0)
    assert s.paper_mode is True
    assert s.bounds() == (1.0, 101.0, 1.0)


@pytest.mark.parametrize("kw", [{"q_init": 1.0}, {"eps": 20.0}])
def test_只給一個時拒絕(kw):
    with pytest.raises(ValueError, match="同時"):
        AdvDropSpec(name="x", **kw)


def test_論文模式拒絕非正的eps():
    with pytest.raises(ValueError, match="eps"):
        AdvDropSpec(name="x", q_init=1.0, eps=0.0)


def test_論文模式的量化表確實留在區間內():
    x = _img()
    spec = AdvDropSpec(name="p", q_init=1.0, eps=20.0, steps=6)
    res = run_advdrop(_StubSD(), x, spec)
    # 由輸出反推不可靠（IDCT ＋ 夾取），故直接檢查前向沒有爆掉且有作用
    assert res.x_def.shape == x.shape
    assert torch.isfinite(res.x_def).all()
    assert not torch.allclose(res.x_def, x), "論文模式完全沒有作用"


def test_兩種模式給出不同的結果():
    """區間差一個數量級，結果必須不同——否則 bounds() 沒有真的接上去。"""
    x = _img()
    kw = dict(steps=6)
    a = run_advdrop(_StubSD(), x, AdvDropSpec(name="code", **kw)).x_def
    b = run_advdrop(_StubSD(), x,
                    AdvDropSpec(name="paper", q_init=1.0, eps=100.0, **kw)).x_def
    assert not torch.allclose(a, b, atol=1e-6)


def test_論文寫死的數字():
    """改動會讓重現的對象變成別的東西，故釘住。出處：arXiv:2108.09034
    §3.1（q_init=1）、§4.3（ε 掃描與步數）。"""
    assert PAPER_Q_INIT == 1.0
    assert PAPER_EPS_SWEEP == (20.0, 60.0, 100.0)
    assert (PAPER_UNTARGETED_STEPS, PAPER_TARGETED_STEPS) == (50, 500)


def test_真正的round讓最佳化幾乎推不動():
    """論文 §4.5 的消融：硬四捨五入報 5.00±0.98%，即攻擊幾乎沒發生。

    **梯度不是零**——`round(c/q)·q` 對 q 的導數是 `round(c/q)`，來自外層那個
    乘回去的 `·q`，只有 `round` 本身的導數是零。所以量化表仍會動，只是方向
    由一個逐段常數的量決定，與「增大 q 是否真的有幫助」無關，於是最佳化
    幾乎推不動損失。這一點與「把 alpha 設到 1e-20」不同：那時 phi_diff 仍是
    平滑的，梯度帶有真實方向，本專案實測成功率 1.000。

    這裡比的是損失的下降幅度，不是量化表有沒有動。
    """
    x = _img()
    kw = dict(q_init=1.0, eps=100.0, steps=12, step_size=4.0)

    def drop(spec):
        h = run_advdrop(_StubSD(), x, spec, log_every=1).history
        losses = [r["loss"] for r in h if "loss" in r]
        return losses[0] - losses[-1]

    soft = drop(AdvDropSpec(name="soft", **kw))
    hard = drop(AdvDropSpec(name="hard", hard_round=True,
                            modified_from_paper=True,
                            modification_note="真正的 round，重現 §4.5 的消融",
                            **kw))
    assert soft > hard, f"軟 {soft:.4f} 不大於硬 {hard:.4f}，對照不成立"


def test_hard_round沒標modified時拒絕():
    with pytest.raises(ValueError, match="hard_round"):
        AdvDropSpec(name="x", hard_round=True)
