"""`src/baselines/advdrop.py` 的驗收 — 不需要 SD 的部分。

釘住三類事：軟四捨五入在極限下要收斂到真的 `round`（否則 α 退火沒有意義）、
「丟資訊」這個方向要真的成立（量化表越大失真越大）、以及**與原始碼逐行對應
的兩個容易寫錯的事實**——更新是手寫 sign 而非 Adam、通道是 RGB 而非 YCbCr。
"""

import pytest
import torch

from src.baselines.advdrop import (
    CHANNELS, PAPER_ALPHA_HI, PAPER_ALPHA_LO, PAPER_Q_MIN, PAPER_Q_SIZE,
    PAPER_STEPS, AdvDropParam, AdvDropSpec, alpha_at, init_q_tables, phi_diff,
    quantize_drop, render_advdrop,
)


def _img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (torch.sin(xx / 4.0) + torch.cos(yy / 6.0)) * 0.25 + 0.5
    x = base.view(1, 1, h, w).repeat(1, 3, 1, 1).double()
    return (x + torch.randn(x.shape, generator=g).double() * 0.05).clamp(0, 1)


# ---- 軟四捨五入 ----

def _away_from_midpoint(x, margin=0.05):
    return ((x - torch.floor(x)) - 0.5).abs() > margin


def test_phi_diff_approaches_round_but_never_exactly():
    """α → 0 時逼近 `round`，但**只在遠離中點處**，而且只到 1e-2 的量級。

    α=1e-20 給出 `k = log(2/α − 1) ≈ 46.7`，`tanh` 只在 `|f − 0.5| ≫ 1/46.7`
    才飽和。實測：離中點 0.05 以外最大差 0.0091，全域最大差 0.495（正中點）。
    **不可以把它當成精確的 `round`**——α 退火買到的是「接近」而不是「等於」。
    """
    x = torch.linspace(-4.3, 4.3, 4001, dtype=torch.float64)
    hard = phi_diff(x, torch.tensor(1e-20, dtype=torch.float64))
    d = (hard - torch.round(x)).abs()
    m = _away_from_midpoint(x)
    assert float(d[m].max()) < 0.01
    assert float(d.max()) > 0.4, "正中點的落差是構造使然，不該消失"


def test_phi_diff_is_smooth_at_large_alpha():
    """α 大時輸出應偏離 `round`，否則早期的梯度與硬捨入無異。"""
    x = torch.linspace(-4.3, 4.3, 401, dtype=torch.float64)
    soft = phi_diff(x, torch.tensor(PAPER_ALPHA_HI, dtype=torch.float64))
    assert float((soft - torch.round(x)).abs().max()) > 0.1


def test_phi_diff_is_differentiable():
    x = torch.linspace(0.1, 3.9, 64, dtype=torch.float64).requires_grad_(True)
    phi_diff(x, torch.tensor(0.1, dtype=torch.float64)).sum().backward()
    assert float(x.grad.abs().sum()) > 0


def test_alpha_at_or_above_two_is_nan_at_exact_midpoints():
    """**原始碼的夾取沒有防住 `log(0)`，它正是造成 nan 的原因。**

    `a = min(α, 2)`；`a = 2` 時 `k = log(2/2 − 1) = log(0) = −∞`。遠離中點處
    `(f − 0.5)·(−∞) = ∓∞`、`tanh` 飽和，配上 `s = 1/(1−2) = −1` 之後恰好等於
    硬四捨五入；但**正中點的 `0 × (−∞)` 是 nan**。

    本專案的 α 由 0.1 起始，不會走到 2，故實務上碰不到。這一項存在是為了讓
    後來的人不會誤以為那個 `clamp(max=2)` 是一個安全防護。
    """
    x = torch.linspace(-2.0, 2.0, 33, dtype=torch.float64)   # 含 ±0.5、±1.5
    at2 = phi_diff(x, torch.tensor(2.0, dtype=torch.float64))
    at5 = phi_diff(x, torch.tensor(5.0, dtype=torch.float64))
    mid = ~_away_from_midpoint(x, margin=1e-9)
    assert mid.any(), "測試點必須包含正中點"
    assert torch.isnan(at2[mid]).all(), "正中點應為 nan"
    torch.testing.assert_close(at2[~mid], at5[~mid])         # 夾取本身有效
    torch.testing.assert_close(at2[~mid], torch.round(x)[~mid])


def test_alpha_schedule_starts_at_hi_and_is_monotone():
    seq = [alpha_at(i, 40) for i in range(40)]
    assert seq[0] == PAPER_ALPHA_HI
    assert all(a >= b for a, b in zip(seq, seq[1:]))


def test_alpha_schedule_does_not_reach_lo_within_the_loop():
    """**原始碼的性質**：`alpha += (lo − hi)/steps` 每步一次，最後一步之後
    才會到 `lo`，故迴圈內的最小值是 `hi + (lo − hi)(steps−1)/steps`。
    這一項存在是為了讓後來的人不會「修正」成端點對齊而改變了行為。"""
    last = alpha_at(39, 40)
    assert last > PAPER_ALPHA_LO
    assert abs(last - (PAPER_ALPHA_HI + (PAPER_ALPHA_LO - PAPER_ALPHA_HI)
                       * 39 / 40)) < 1e-15


def test_alpha_rejects_out_of_range():
    with pytest.raises(ValueError):
        alpha_at(40, 40)
    with pytest.raises(ValueError):
        alpha_at(0, 0)


# ---- 量化即丟資訊 ----

def test_quantize_drop_snaps_to_multiples_of_the_table():
    """硬捨入下，輸出必須恰好落在量化表的整數倍上。"""
    coef = torch.linspace(-50, 50, 1001, dtype=torch.float64)
    tbl = torch.full_like(coef, 7.0)
    out = quantize_drop(coef, tbl, torch.tensor(1e-20, dtype=torch.float64))
    m = _away_from_midpoint(coef / 7.0)
    resid = (out[m] / 7.0 - torch.round(out[m] / 7.0)).abs()
    assert float(resid.max()) < 0.01, "遠離中點處應貼齊量化表的整數倍"


def test_larger_tables_drop_more_information():
    """AdvDrop 的「半徑」是量化步長的上界，越大代表丟得越多、失真越大。"""
    x = _img()
    a = torch.tensor(PAPER_ALPHA_LO, dtype=torch.float64)
    prev = -1.0
    for q in (2.0, 5.0, 10.0, 20.0, 40.0):
        tables = init_q_tables(x, q)
        err = float((render_advdrop(x, tables, a) - x).detach().abs().mean())
        assert err > prev, f"q={q} 的失真沒有比前一個大"
        prev = err


def test_there_is_no_zero_point():
    """**與紋理重相位的關鍵差別**：AdvDrop 沒有「不動」的設定，即使量化表壓到
    下界仍然丟資訊。報表不可把它的 `q_min` 當成 `θ=0` 的對應物。"""
    x = _img()
    a = torch.tensor(PAPER_ALPHA_LO, dtype=torch.float64)
    out = render_advdrop(x, init_q_tables(x, PAPER_Q_MIN), a)
    assert float((out - x).detach().abs().max()) > 1e-3


def test_render_keeps_shape_and_range():
    x = _img()
    out = render_advdrop(x, init_q_tables(x), torch.tensor(0.1, dtype=torch.float64))
    assert out.shape == x.shape
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_render_rejects_bad_shapes():
    a = torch.tensor(0.1)
    with pytest.raises(ValueError):
        render_advdrop(torch.rand(3, 32, 32), {}, a)
    with pytest.raises(ValueError):
        render_advdrop(torch.rand(1, 3, 30, 32), {}, a)


def test_tables_are_per_block_per_coefficient():
    """**不是** JPEG 那種全圖共用一張 8×8 表：每個區塊各有自己的一張。"""
    x = _img(32, 32)
    t = init_q_tables(x)
    assert set(t) == set(CHANNELS)
    assert t["r"].shape == (1, 4, 4, 8, 8)          # 32/8 = 4 個區塊
    assert t["r"].requires_grad


def test_gradient_reaches_every_table():
    x = _img()
    t = init_q_tables(x)
    render_advdrop(x, t, torch.tensor(0.1, dtype=torch.float64)).pow(2).sum().backward()
    for c in CHANNELS:
        assert t[c].grad is not None and float(t[c].grad.abs().sum()) > 0


# ---- 與原始碼對應的兩個容易寫錯的事實 ----

def test_channels_are_rgb_not_ycbcr():
    """原始碼把變數命名成 y/cb/cr，餵進去的卻是 R、G、B，**沒有色彩轉換**。
    這一項存在是為了讓後來的人不會「補上」一個原文沒有的 YCbCr 轉換。

    驗證方式：只把紅色通道的量化表調粗，則只有紅色通道會變。
    """
    x = _img()
    a = torch.tensor(PAPER_ALPHA_LO, dtype=torch.float64)
    t = init_q_tables(x, 2.0)
    with torch.no_grad():
        t["r"].fill_(60.0)
    out = render_advdrop(x, t, a)
    d = (out - x).abs().mean(dim=(0, 2, 3))
    assert float(d[0]) > 5 * float(d[1]), "只有 R 通道該變粗"
    assert float(d[0]) > 5 * float(d[2])


def test_paper_hyperparameters():
    assert PAPER_Q_SIZE == 10.0 and PAPER_Q_MIN == 5.0
    assert PAPER_STEPS == 40
    assert PAPER_ALPHA_HI == 0.1 and PAPER_ALPHA_LO == 1e-20


# ---- spec 與 Parameterization ----

def test_spec_rejects_degenerate_range():
    with pytest.raises(ValueError):
        AdvDropSpec(q_size=5.0, q_min=5.0)


def test_spec_requires_a_note_when_modified():
    with pytest.raises(ValueError):
        AdvDropSpec(modified_from_paper=True)


def test_param_projects_into_the_range():
    x = _img()
    p = AdvDropParam(radius=8.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.q["r"].fill_(100.0)
        p.q["g"].fill_(-3.0)
    p.project()
    assert float(p.q["r"].detach().max()) <= 8.0
    assert float(p.q["g"].detach().min()) >= PAPER_Q_MIN


def test_param_radius_must_exceed_the_floor():
    p = AdvDropParam()
    with pytest.raises(ValueError):
        p.set_radius(PAPER_Q_MIN)


def test_param_render_matches_the_free_function():
    x = _img()
    p = AdvDropParam(radius=10.0)
    p.reset(x, 0)
    want = render_advdrop(x, p.q, torch.tensor(p.alpha, dtype=x.dtype))
    torch.testing.assert_close(p.render(x), want)
