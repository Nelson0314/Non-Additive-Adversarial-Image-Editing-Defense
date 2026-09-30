"""`src/baselines/blurguard.py` 的驗收 — 不需要 SD、不需要 SAM 的部分。

BlurGuard 的每一個零件寫錯都不會有症狀：輸出仍是一張合理的防禦圖，只是模糊
強度綁錯區域、或頻譜約束根本沒在約束任何東西。這裡釘住的是可驗證的代數與
幾何性質，以及**與原始碼逐行對應的參數值**。
"""

import math

import pytest
import torch

from src.baselines.blurguard import (
    PAPER_BLUR_WIDTH, PAPER_EPSILON_255, PAPER_EPS_SIGMA, PAPER_LR,
    PAPER_SIGMA_WEIGHTING, PAPER_STEPS, PAPER_WARMUP_ITERS, SPEC_PAPER,
    BlurGuardParam, BlurGuardSpec, check_partition, fft_power, filter_delta,
    gaussian_blur, radial_histogram, sam_masks, spectrum_deviation,
)


def _img(seed=0, h=32, w=32):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (torch.sin(xx / 4.0) + torch.cos(yy / 6.0)) * 0.25 + 0.5
    x = base.view(1, 1, h, w).repeat(1, 3, 1, 1).double()
    return (x + torch.randn(x.shape, generator=g).double() * 0.05).clamp(0, 1)


def _quad_masks(h=32, w=32, device=None):
    """四象限的互斥分割。**這不是 BlurGuard 的遮罩**（原文用 SAM），只是
    用來測試 `filter_delta` 的代數性質。"""
    out = {}
    for i, (ys, xs) in enumerate([(slice(0, h // 2), slice(0, w // 2)),
                                  (slice(0, h // 2), slice(w // 2, w)),
                                  (slice(h // 2, h), slice(0, w // 2)),
                                  (slice(h // 2, h), slice(w // 2, w))]):
        m = torch.zeros(1, 1, h, w, dtype=torch.float64, device=device)
        m[..., ys, xs] = 1.0
        out[f"mask{i + 1}"] = m
    return out


# ---- 高斯模糊 ----

def test_blur_kernel_is_normalised():
    """核必須總和為 1：常數影像模糊後應完全不變（反射填補使邊界也成立）。"""
    x = torch.full((1, 3, 32, 32), 0.4, dtype=torch.float64)
    out = gaussian_blur(x, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(out, x, atol=1e-12, rtol=0)


def test_blur_preserves_shape_and_width_is_odd():
    x = _img()
    assert gaussian_blur(x, torch.tensor(1.5, dtype=torch.float64)).shape == x.shape
    assert PAPER_BLUR_WIDTH % 2 == 1


def test_blur_reduces_high_frequency_energy():
    """模糊必須壓掉高頻——這是 BlurGuard 賴以重塑頻譜的機制本身。"""
    x = _img()
    hi = lambda t: float(fft_power(t * 2 - 1)[8:24, 8:24].sum())  # noqa: E731
    assert hi(gaussian_blur(x, torch.tensor(2.0, dtype=torch.float64))) < hi(x)


def test_blur_is_differentiable_wrt_sigma():
    """σ 是被 Adam 最佳化的變數，核必須落在計算圖上。"""
    s = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    gaussian_blur(_img(), s).pow(2).sum().backward()
    assert s.grad is not None and float(s.grad.abs()) > 0


def test_blur_rejects_wrong_rank():
    with pytest.raises(ValueError):
        gaussian_blur(torch.rand(3, 32, 32), torch.tensor(1.0))


# ---- 逐區域模糊 ----

def test_filter_delta_equals_per_region_blur():
    """每個區域的輸出必須等於「整張擾動用該區域的 σ 模糊後、取該區域」。"""
    x, masks = _img(1), _quad_masks()
    ls = torch.log(torch.tensor([0.8, 1.5, 2.5, 4.0], dtype=torch.float64))
    got = filter_delta(ls, x, masks)
    for i in range(4):
        m = masks[f"mask{i + 1}"]
        want = gaussian_blur(x, ls[i].exp()) * m
        torch.testing.assert_close(got * m, want, atol=1e-12, rtol=0)


def test_filter_delta_with_equal_sigmas_is_plain_blur():
    x, masks = _img(1), _quad_masks()
    ls = torch.log(torch.full((4,), 2.0, dtype=torch.float64))
    torch.testing.assert_close(filter_delta(ls, x, masks),
                               gaussian_blur(x, torch.tensor(2.0, dtype=torch.float64)),
                               atol=1e-12, rtol=0)


def test_filter_delta_rejects_length_mismatch():
    with pytest.raises(ValueError):
        filter_delta(torch.zeros(3), _img(), _quad_masks())


def test_check_partition_rejects_overlap_and_gaps():
    masks = _quad_masks()
    check_partition(masks)                       # 正常的分割不該拋
    masks["mask1"] = masks["mask1"] + masks["mask2"]      # 重疊
    with pytest.raises(ValueError):
        check_partition(masks)
    gap = _quad_masks()
    del gap["mask4"]
    with pytest.raises(ValueError):
        check_partition(gap)


# ---- 頻譜項 ----

def test_fft_power_is_real_nonnegative_and_dc_at_corner():
    """`fft_fps` 沒有 fftshift，零頻在 (0,0)，且該格是全圖最大。"""
    p = fft_power(_img() * 2 - 1)
    assert p.shape == (32, 32)
    assert float(p.min()) >= 0.0
    assert p.argmax().item() == 0


def test_radial_histogram_conserves_total_power():
    """分箱只是重新分組，總和必須守恆。"""
    p = fft_power(_img() * 2 - 1)
    torch.testing.assert_close(radial_histogram(p).sum(), p.sum(),
                               atol=1e-8, rtol=1e-10)


def test_radial_histogram_bins_match_source_definition():
    """分箱索引照原始碼：圓心 ((H−1)/2,(W−1)/2)、`round()`、`max+1` 個箱。"""
    h = w = 16
    p = torch.ones(h, w, dtype=torch.float64)
    hist = radial_histogram(p)
    cy = cx = (h - 1) / 2.0
    want = {}
    for y in range(h):
        for x in range(w):
            k = round(math.sqrt((x - cx) ** 2 + (y - cy) ** 2))
            want[k] = want.get(k, 0) + 1
    assert hist.numel() == max(want) + 1
    for k, v in want.items():
        assert abs(float(hist[k]) - v) < 1e-9


def test_radial_binning_is_inverted_dc_lands_in_the_last_bin():
    """**模組 docstring 記的觀察**：圓心取在陣列中心，而 `fft2` 的零頻在角落，
    所以分箱與頻率大小的對應是**反過來的**——直流落在最後一箱，Nyquist 落在
    最前面。這一項存在是為了讓該性質不會被後來的人誤讀成「徑向頻譜」。

    另外，邊長為偶數時圓心落在像素之間（16×16 的圓心是 7.5），最近的四個
    索引距離 0.707、四捨五入後進第 1 箱，**第 0 箱恆為空**。
    """
    h = w = 16
    dc = torch.zeros(h, w, dtype=torch.float64)
    dc[0, 0] = 1.0
    hist_dc = radial_histogram(dc)
    assert int(hist_dc.argmax()) == hist_dc.numel() - 1, "直流應落在最後一箱"

    nyq = torch.zeros(h, w, dtype=torch.float64)
    nyq[h // 2, w // 2] = 1.0
    hist_nyq = radial_histogram(nyq)
    assert int(hist_nyq.argmax()) == 1, "Nyquist 應落在第 1 箱"

    assert float(hist_dc[0]) == 0.0 and float(hist_nyq[0]) == 0.0, "第 0 箱恆為空"


def test_spectrum_deviation_is_zero_for_identical_images():
    x = _img() * 2 - 1
    assert float(spectrum_deviation(x, x)) == 0.0


def test_spectrum_deviation_grows_with_noise():
    x = _img() * 2 - 1
    g = torch.Generator().manual_seed(3)
    small = (x + torch.randn(x.shape, generator=g).double() * 0.01).clamp(-1, 1)
    big = (x + torch.randn(x.shape, generator=g).double() * 0.20).clamp(-1, 1)
    assert float(spectrum_deviation(x, small)) < float(spectrum_deviation(x, big))


# ---- 設定值必須與 repo 的 config 一致 ----

def test_paper_hyperparameters_match_the_repo_config():
    """來自 `configs/attack/base.yaml`，**不是**函式簽章的預設。
    `sigma_weighting` 在 repo 裡有 10 與 10000 兩個不同的預設，生效的是 10。
    """
    assert PAPER_EPSILON_255 == 16
    assert PAPER_STEPS == 150
    assert PAPER_LR == 0.06
    assert PAPER_EPS_SIGMA == 0.0
    assert PAPER_SIGMA_WEIGHTING == 10.0
    assert PAPER_WARMUP_ITERS == 50
    assert abs(SPEC_PAPER.eps_pixel01 - 16 / 255) < 1e-12


def test_spec_rejects_warmup_that_starves_the_perturbation():
    """暖身期不更新擾動；warmup >= steps 會產出一張完全沒被最佳化的圖，
    而且看起來仍然合理——必須直接失敗。"""
    with pytest.raises(ValueError):
        BlurGuardSpec(steps=40, warmup=50)


def test_spec_requires_a_note_when_modified():
    with pytest.raises(ValueError):
        BlurGuardSpec(modified_from_paper=True)


# ---- Parameterization ----

def test_param_render_at_zero_delta_is_identity():
    x, masks = _img(), _quad_masks()
    p = BlurGuardParam(masks)
    p.reset(x, 0)
    torch.testing.assert_close(p.render(x), x, atol=1e-12, rtol=0)


def test_param_projects_and_respects_radius():
    x, masks = _img(), _quad_masks()
    p = BlurGuardParam(masks, radius=0.01)
    p.reset(x, 0)
    with torch.no_grad():
        p.delta.add_(5.0)
    p.project()
    assert float(p.delta.detach().abs().max()) <= 0.01 + 1e-12
    assert float((p.render(x) - x).detach().abs().max()) <= 0.01 + 1e-9


def test_param_rejects_sigma_count_mismatch():
    with pytest.raises(ValueError):
        BlurGuardParam(_quad_masks(), sigmas=[1.0, 2.0])


def test_sam_masks_refuses_to_substitute_a_partition():
    """相依不齊時必須拋出，不得回傳任何替代分割。"""
    with pytest.raises(NotImplementedError):
        sam_masks(_img())
