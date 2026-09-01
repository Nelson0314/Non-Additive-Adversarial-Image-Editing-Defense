"""`src/purify/freq_grid.py` 的驗收：不需要擴散檢查點的部分。

檢查點（`256x256_diffusion_uncond.pt`）只在遠端有，故 `gridpure_real` 與
`fdpure_real` 的端到端行為在本機測不到。這裡釘住的是**幾何與代數**——網格
覆蓋、低通遮罩的中心對齊、相位投影的週期性、半徑換算。這幾項寫錯都不會有
症狀：輸出仍是一張看起來合理的淨化圖，只是淨化的位置全錯。
"""

import math

import pytest
import torch

from src.purify import ops as purify_ops
from src.purify.freq_grid import (
    FDPURE_CIFAR_NYQUIST, FDPURE_DA_CIFAR, GRIDPURE_GRID, GRIDPURE_STRIDE,
    _project_phase, grid_specs, lowpass_mask, scale_radius,
)


# ---- GrIDPure 的網格幾何 ----

def test_grid_specs_match_paper_layout_on_512():
    """論文圖 9：512² 切成 3×3 = 9 個規則 256² 網格（stride 128），
    加上四角合併的第十格，共十格。"""
    specs = grid_specs(512)
    assert len(specs) == 10
    regular = [s for s in specs if s[0] == 0 and s[1] == 0]
    assert len(regular) == 9
    assert sorted({t for _, _, t, _ in regular}) == [0, 128, 256]


def test_every_pixel_covered_at_least_twice():
    """論文 §5.2 步驟 (1)：每一塊像素至少落在兩個網格裡。

    這是 GrIDPure 能夠平均合併而不留接縫的前提；覆蓋數為 1 的區域會直接
    暴露單格的擴散輸出。
    """
    for size in (512, 384, 320):
        cover = torch.zeros(1, 1, size, size)
        for (st, sl, t, l) in grid_specs(size):
            one = torch.zeros(1, 1, size, size)
            one[..., t:t + GRIDPURE_GRID, l:l + GRIDPURE_GRID] = 1.0
            if st or sl:
                one = torch.roll(one, shifts=(-st, -sl), dims=(-2, -1))
            cover += one
        assert float(cover.min()) >= 2.0, f"size={size} 有像素只被覆蓋一次"


def test_grid_specs_reject_too_small():
    with pytest.raises(ValueError):
        grid_specs(128)


def test_corner_grid_gathers_the_four_corners():
    """第十格必須恰好蒐集四個角落的 128×128，不多不少。"""
    size, g = 512, GRIDPURE_GRID
    st, sl, top, left = grid_specs(size)[-1]
    assert (st, sl) != (0, 0), "第十格必須是平移格"
    marker = torch.zeros(1, 1, size, size)
    h = g // 2
    for a in (slice(0, h), slice(size - h, size)):
        for b in (slice(0, h), slice(size - h, size)):
            marker[..., a, b] = 1.0
    taken = torch.roll(marker, shifts=(st, sl), dims=(-2, -1))[
        ..., top:top + g, left:left + g]
    assert float(taken.sum()) == float(marker.sum()) == 4 * h * h


def test_grid_and_stride_are_the_paper_values():
    assert GRIDPURE_GRID == 256 and GRIDPURE_STRIDE == 128


# ---- FD-Pure 的低通遮罩與相位投影 ----

def test_lowpass_mask_is_centred_on_dc_after_ifftshift():
    """`fft2` 把零頻放在角落，故遮罩必須在 (0,0) 為 1、在中心為 0。"""
    m = lowpass_mask(64, radius=5.0)[0, 0]
    assert float(m[0, 0]) == 1.0          # 直流
    assert float(m[32, 32]) == 0.0        # Nyquist 角落
    assert float(m[0, 1]) == 1.0 and float(m[1, 0]) == 1.0


def test_lowpass_mask_count_matches_disc_area():
    """遮罩為 1 的格數應接近半徑 r 的圓面積 πr²。"""
    r = 8.0
    m = lowpass_mask(128, radius=r)
    assert abs(float(m.sum()) - math.pi * r * r) / (math.pi * r * r) < 0.06


def test_lowpass_mask_zero_radius_is_empty():
    assert float(lowpass_mask(32, radius=0.0).sum()) == 0.0


def test_scale_radius_preserves_nyquist_fraction():
    """半徑換算是按 Nyquist 比例——本專案指定的換算，必須可驗證。"""
    got = scale_radius(FDPURE_DA_CIFAR, 256)
    assert abs(got - FDPURE_DA_CIFAR / FDPURE_CIFAR_NYQUIST * 128.0) < 1e-9
    assert abs(scale_radius(FDPURE_DA_CIFAR, 32) - FDPURE_DA_CIFAR) < 1e-9


def test_phase_projection_clamps_within_delta():
    ref = torch.rand(1, 1, 8, 8) * 2 * math.pi - math.pi
    est = torch.rand(1, 1, 8, 8) * 2 * math.pi - math.pi
    out = _project_phase(ref, est, 0.2)
    diff = torch.remainder(out - ref + math.pi, 2 * math.pi) - math.pi
    assert float(diff.abs().max()) <= 0.2 + 1e-6


def test_phase_projection_is_identity_inside_the_band():
    ref = torch.zeros(1, 1, 4, 4)
    est = torch.full((1, 1, 4, 4), 0.1)
    torch.testing.assert_close(_project_phase(ref, est, 0.2), est)


def test_phase_projection_wraps_across_pi():
    """相位是週期量：ref=+3.1、est=−3.1 的真實差是 0.083 而不是 6.2。"""
    ref = torch.full((1, 1, 2, 2), 3.1)
    est = torch.full((1, 1, 2, 2), -3.1)
    out = _project_phase(ref, est, 0.2)
    torch.testing.assert_close(out, est + 2 * math.pi, atol=1e-6, rtol=0)


# ---- 接進 ops.Purifier ----

def test_new_kinds_registered():
    assert "gridpure" in purify_ops.KINDS and "fdpure" in purify_ops.KINDS


def test_new_kinds_are_not_claimed_differentiable():
    """兩者都在 `torch.no_grad()` 裡跑擴散，不得宣稱可微。"""
    assert not purify_ops.Purifier("gridpure").differentiable
    assert not purify_ops.Purifier("fdpure").differentiable


def test_required_hyperparameters_are_not_silently_defaulted():
    """`t`／`gamma`／`iters`／`t_star` 論文正文未載，缺了必須直接失敗。

    給一個看起來合理的預設是本專案明令禁止的（`src/baselines/__init__.py`
    模組 docstring）：填錯不會有症狀，事後也補不回來。
    """
    from src.purify.freq_grid import fdpure_real, gridpure_real
    x = torch.rand(1, 3, 64, 64)
    with pytest.raises(TypeError):
        gridpure_real(x)
    with pytest.raises(TypeError):
        fdpure_real(x)
