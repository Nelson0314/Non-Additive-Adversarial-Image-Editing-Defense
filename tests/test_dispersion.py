"""色散變形的構造檢定。

這一支釘的不是接線而是**構造本身**：K=1 的相位斜坡真的是一個平移、K 越大
色散度越高、折疊比例真的分得出微分同胚與非微分同胚。這些是後面所有歸因的
前提，講錯了不會有症狀。

設計見 `docs/superpowers/specs/IMAGE_GUIDANCE_AND_DISPERSIVE_WARP.md` §2。
"""

from __future__ import annotations

import math

import pytest
import torch

from src.defense.dispersion import (
    R_CORNER, apply_theta, band_index, bandpass, displacement_theta,
    fold_fraction, make_operator, random_displacements, random_phase_theta,
)


BLOCK = 32
SIZE = 128
HOP = 16
R_MIN = 0.12


def _op(x: torch.Tensor):
    return make_operator(x, BLOCK, HOP, R_MIN)


def _img(seed: int, ch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, ch, SIZE, SIZE, generator=g, dtype=torch.float64)


def _n_windows(op) -> int:
    return int(op.n_blocks)


def _side(op) -> int:
    """視窗格點的邊長。場是在這個格點上生成的，`n_blocks = side * side`。"""
    return int(op.side)


# ---- 頻帶編號 ----

def test_band_index_rejects_empty_passband():
    with pytest.raises(ValueError, match="通帶是空的"):
        band_index(BLOCK, 4, 0.5, "cpu", r_max=0.4)


def test_band_index_rejects_zero_bands():
    with pytest.raises(ValueError, match="n_bands"):
        band_index(BLOCK, 0, R_MIN, "cpu")


def test_out_of_band_is_minus_one_and_matches_radial_gate():
    """通帶外一律 −1，且與 `radial_gate` 的支撐**逐格相同**。

    兩者若不一致，「色散變形」與「現行方法」就跑在不同的頻格集合上，
    等失真比較會把支撐差異讀成色散度的效果。
    """
    from src.residual.texture_rephase import radial_gate

    idx = band_index(BLOCK, 4, R_MIN, "cpu", r_max=R_CORNER)
    gate = radial_gate(BLOCK, R_MIN, "cpu", torch.float32, R_CORNER) > 0
    assert torch.equal(idx >= 0, gate)


def test_every_band_is_non_empty():
    """八度分帶下每一帶都要有格子，否則某些 K 實際上少於 K 帶。"""
    for k in (1, 2, 3, 4, 8):
        idx = band_index(BLOCK, k, R_MIN, "cpu")
        present = {int(v) for v in idx.unique() if int(v) >= 0}
        assert present == set(range(k)), (k, present)


def test_band_index_is_monotone_in_radius():
    idx = band_index(BLOCK, 4, R_MIN, "cpu")
    fy = torch.fft.fftfreq(BLOCK)[:, None] * 2.0
    fx = torch.fft.rfftfreq(BLOCK)[None, :] * 2.0
    r = torch.sqrt(fy ** 2 + fx ** 2)
    inside = idx >= 0
    for a in range(4):
        for b in range(a + 1, 4):
            ra = r[inside & (idx == a)]
            rb = r[inside & (idx == b)]
            assert float(ra.max()) <= float(rb.min()) + 1e-6


# ---- K = 1 真的是一個平移 ----

def test_k1_uniform_displacement_is_a_translation():
    """K=1 的相位斜坡真的把通帶內容**平移**該格數。

    這是平移定理的直接檢定，也是整條「色散度」軸的地基：K=1 必須就是古典
    位移場。判準取互相關的極大值落在正確的位移上——比對絕對誤差會被區塊內
    的循環繞回與重疊相加的邊界效應干擾，而那些與「是不是平移」無關。
    """
    x = _img(0)
    op = _op(x)
    bands = band_index(BLOCK, 1, R_MIN, "cpu", dtype=torch.float64)
    ref = bandpass(op, x, bands)

    shift = 3
    u = torch.zeros(_n_windows(op), 1, 2, dtype=torch.float64)
    u[:, 0, 0] = float(shift)          # 只在 x 方向
    moved = bandpass(op, apply_theta(
        op, x, displacement_theta(BLOCK, bands, u)), bands)

    c = slice(BLOCK, SIZE - BLOCK)      # 深內部，避開反射填補
    a = moved[..., c, c]
    scores = {s: float((a * torch.roll(ref, shifts=s, dims=-1)[..., c, c]).sum())
              for s in range(-6, 7)}
    assert max(scores, key=scores.get) == shift, scores


def test_residual_is_generated_only_from_in_band_coefficients():
    """殘差完全由通帶內的係數造出來：`x' − x = synth(analyze(x)·(e^{iθ}−1))`，
    而括號裡的因子在通帶外恰為零。

    **不可以改寫成「重新分析之後通帶外的係數不變」**——那是錯的，而且錯得
    沒有症狀：逐區塊改過相位之後的係數一般**不一致**，重疊相加會把它投影回
    一致集合，通帶外的係數因此跟著動。那個投影誤差就是 FND-049 的 `amp_dev`，
    是構造使然不是實作錯誤。
    """
    x = _img(1)
    op = _op(x)
    bands = band_index(BLOCK, 2, R_MIN, "cpu", dtype=torch.float64)
    u = random_displacements(_side(op), 2, 2.0, seed=11, device="cpu",
                             dtype=torch.float64)
    theta = displacement_theta(BLOCK, bands, u)
    assert float(theta[0][:, bands < 0].abs().max()) == 0.0

    out = op.synthesize(op.analyze(x) * torch.polar(
        torch.ones_like(theta), theta).unsqueeze(1))
    factor = torch.polar(torch.ones_like(theta), theta) - 1.0
    resid = op.synthesize(op.analyze(x) * factor.unsqueeze(1))
    assert float((out - x - resid).abs().max()) < 1e-9


def test_all_zero_theta_over_full_spectrum_is_identity():
    """整個頻譜都不動時逐位元恆等——`analyze`／`synthesize` 那一對的保證。"""
    x = _img(2, ch=3)
    op = _op(x)
    theta = torch.zeros(1, _n_windows(op), BLOCK, BLOCK // 2 + 1,
                        dtype=torch.float64)
    out = apply_theta(op, x, theta)
    assert float((out - x).abs().max()) < 1e-10


# ---- 相位超過 π 要繞回去而不是被夾平 ----

def test_theta_beyond_pi_wraps_instead_of_clamping():
    """`theta` 與 `theta + 2π` 必須給出同一張圖。

    `PhaseResidual._rephase` 會夾在 ±π，斜坡因此被削平；本模組刻意不走那條
    路徑。這一條就是那個差別的檢定。
    """
    x = _img(3)
    op = _op(x)
    theta = torch.rand(1, _n_windows(op), BLOCK, BLOCK // 2 + 1,
                       dtype=torch.float64) * 6.0 - 3.0
    a = apply_theta(op, x, theta)
    b = apply_theta(op, x, theta + 2.0 * math.pi)
    assert float((a - b).abs().max()) < 1e-10


# ---- 色散度：K 越大，逐頻格的相位越不一致 ----

def _phase_spread(theta: torch.Tensor, bands: torch.Tensor) -> float:
    """通帶內、逐視窗相位的標準差，當作「色散度」的粗略讀數。"""
    m = bands >= 0
    vals = theta[0][:, m]
    return float(vals.std(dim=1).mean())


def test_dispersion_increases_with_band_count():
    """同一個位移振幅下，K 越大相位越分散——這就是「色散度」這個旋鈕。"""
    op = _op(_img(4))
    n = _side(op)
    spreads = []
    for k in (1, 2, 4, 8):
        bands = band_index(BLOCK, k, R_MIN, "cpu")
        u = random_displacements(n, k, 1.0, seed=7, device="cpu")
        spreads.append(_phase_spread(displacement_theta(BLOCK, bands, u), bands))
    assert spreads == sorted(spreads), spreads


def test_random_phase_endpoint_has_the_documented_shape():
    op = _op(_img(5))
    bands = band_index(BLOCK, 4, R_MIN, "cpu")
    theta = random_phase_theta(_side(op), BLOCK, math.pi, 0, bands, "cpu")
    assert theta.shape == (1, _n_windows(op), BLOCK, BLOCK // 2 + 1)
    assert float(theta[0][:, bands < 0].abs().max()) == 0.0


def test_displacement_theta_rejects_wrong_shape():
    bands = band_index(BLOCK, 2, R_MIN, "cpu")
    with pytest.raises(ValueError, match=r"\(L, K, 2\)"):
        displacement_theta(BLOCK, bands, torch.zeros(4, 2))


# ---- 折疊比例 ----

def test_zero_field_has_no_folds():
    assert fold_fraction(torch.zeros(1, 2, 16, 16)) == 0.0


def test_smooth_small_field_has_no_folds():
    """平緩的場是微分同胚：位移的梯度遠小於 1，行列式恆為正。"""
    ys = torch.linspace(0, 1, 64)
    d = torch.stack(torch.meshgrid(ys, ys, indexing="ij"))[None] * 2.0
    assert fold_fraction(d) == 0.0


def test_high_frequency_field_folds():
    """逐像素振盪的場一定折疊——位移梯度超過 1 就翻面。"""
    g = torch.Generator().manual_seed(0)
    d = torch.randn(1, 2, 64, 64, generator=g) * 3.0
    assert fold_fraction(d) > 0.2


def test_fold_fraction_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(1,2,H,W\)"):
        fold_fraction(torch.zeros(1, 3, 8, 8))


# ---- 場的空間粗糙度：K 才是唯一的變因 ----

def test_coarse_grid_field_is_spatially_smoother_than_iid():
    """`grid > 0` 的場在視窗格點上比逐視窗獨立抽的平緩。

    這是「固定空間粗糙度、只讓 K 變」那個控制的地基：不成立的話，K=1 就同時
    是最粗的場，色散度的效果會與空間粗糙度混在一起而分不開。
    """
    from src.defense.dispersion import random_field

    side = 32
    rough = random_field(side, (1,), 1.0, 0, 0, "cpu").reshape(side, side)
    smooth = random_field(side, (1,), 1.0, 0, 8, "cpu").reshape(side, side)
    def tv(t):
        return float((t[1:] - t[:-1]).abs().mean() + (t[:, 1:] - t[:, :-1]).abs().mean())
    assert tv(smooth) < 0.4 * tv(rough), (tv(smooth), tv(rough))


def test_coarse_grid_field_stays_within_the_amplitude():
    """雙三次會過衝，故上採樣之後要夾回 ±amp，否則「強度」沒有定義。"""
    from src.defense.dispersion import random_field

    f = random_field(24, (3, 2), 0.7, 5, 6, "cpu")
    assert f.shape == (24 * 24, 3, 2)
    assert float(f.abs().max()) <= 0.7 + 1e-6


def test_random_field_rejects_negative_grid():
    from src.defense.dispersion import random_field

    with pytest.raises(ValueError, match="grid"):
        random_field(8, (1,), 1.0, 0, -1, "cpu")


# ---- 接進主線管線 ----

def test_dispersion_conditions_are_registered():
    """`ip2p_run.py` 要認得這些條件，否則它們只活在探針裡、不存防禦圖。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import ip2p_run
    from phase_ablation import build

    for name in ip2p_run.DISP_CONDS:
        assert name in ip2p_run.PHASE_CONDS, name
        param, lo, hi = build(name, 0, block=32, r_min=0.12, hop=8)
        assert param.params() == [], "隨機對照不應該有可學參數"
        assert 0 < lo < hi


def test_band_and_phase_families_get_different_search_ranges():
    """逐頻帶位移的半徑單位是**像素**、逐頻格相位是**弧度**，區間不可共用。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from phase_ablation import build

    _, _, hi_px = build("disp_k4", 0, block=32, r_min=0.12, hop=8)
    _, _, hi_rad = build("disp_kfull", 0, block=32, r_min=0.12, hop=8)
    assert hi_px == 16.0
    assert hi_rad == pytest.approx(math.pi)


def test_set_radius_resamples_the_field():
    """半徑是抽樣的尺度不是事後夾的界，改了要重抽（同 `WarpRandomParam`）。"""
    from src.defense.dispersion import DispersionParam

    x = _img(9, ch=3).to(torch.float32)
    p = DispersionParam(radius=1.0, n_bands=4, block=BLOCK, hop=HOP,
                        r_min=R_MIN)
    p.reset(x, 0)
    small = float((p.render(x) - x).pow(2).mean().sqrt())
    p.set_radius(4.0)
    big = float((p.render(x) - x).pow(2).mean().sqrt())
    assert big > small


def test_dispersion_param_rejects_zero_bands():
    from src.defense.dispersion import DispersionParam

    with pytest.raises(ValueError, match="n_bands"):
        DispersionParam(radius=1.0, n_bands=0)
