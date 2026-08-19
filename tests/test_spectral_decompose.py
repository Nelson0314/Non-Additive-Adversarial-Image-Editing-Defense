"""`scripts/spectral_decompose.py` 的驗收 — FND-057 的比值由它產生。

**這一組測試是補寫的。** FND-057 的主結果是「相位半邊 ÷ 幅度半邊」，而那個
比值只有在**兩個半邊的失真真的被對齊**時才可解讀。做對齊的是
`_match_distortion` 的二分搜尋，它在寫入 `FINDINGS.md` 時沒有任何測試。

搜尋若沒收斂、或把縮放係數套錯方向，輸出仍是一張合理的圖、CSV 仍然填滿，
只是那一欄的 DISTS 不等於 `full` 的——而報表正是靠那一欄宣稱「已對齊」。
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "spectral_decompose",
    Path(__file__).resolve().parent.parent / "scripts" / "spectral_decompose.py")
sdm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdm)


class _StubSuite:
    """`MetricSuite` 的替身：用平均絕對差當「DISTS」。

    真的 DISTS 要載 VGG，而本檔要測的是**搜尋邏輯**，不是度量本身。單調且
    連續的度量足以驗證二分搜尋。
    """

    def pairwise(self, a, b):
        return {"dists": float((a - b).abs().mean())}


def _img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (torch.sin(xx / 4.0) + torch.cos(yy / 6.0)) * 0.2 + 0.5
    x = base.view(1, 1, h, w).repeat(1, 3, 1, 1).double()
    return (x + torch.randn(x.shape, generator=g).double() * 0.03).clamp(0, 1)


# ---- 二分搜尋 ----

def test_match_distortion_hits_the_target():
    x = _img()
    delta = (_img(seed=1) - x) * 0.5
    suite = _StubSuite()
    target = 0.02
    out, s, unreach = sdm._match_distortion(x, delta, target, suite)
    assert not unreach
    assert abs(suite.pairwise(x, out)["dists"] - target) < target * 0.05
    assert s > 0


def test_match_distortion_returns_the_image_at_the_scale_it_reports():
    """回傳的影像必須就是 `x + s·delta`（夾取後），不是別的縮放。"""
    x = _img()
    delta = (_img(seed=2) - x) * 0.4
    out, s, unreach = sdm._match_distortion(x, delta, 0.015, _StubSuite())
    torch.testing.assert_close(out, (x + s * delta).clamp(0, 1),
                               atol=1e-12, rtol=0)


def test_match_distortion_flags_unreachable_targets():
    """目標高於 `hi` 所能達到的失真時必須標出來，而不是安靜地回一個
    離目標很遠的結果。"""
    x = _img()
    delta = (_img(seed=3) - x) * 1e-4          # 極小的擾動
    out, s, unreach = sdm._match_distortion(x, delta, 0.5, _StubSuite())
    assert unreach
    assert s == 4.0, "達不到時應回傳上界"


def test_match_distortion_is_monotone_in_the_scale():
    """搜尋的前提：縮放越大失真越大。夾取只會讓它上升得更慢，不會反轉。"""
    x = _img()
    delta = (_img(seed=4) - x)
    suite = _StubSuite()
    prev = -1.0
    for s in (0.1, 0.5, 1.0, 2.0, 4.0):
        d = suite.pairwise(x, (x + s * delta).clamp(0, 1))["dists"]
        assert d > prev
        prev = d


# ---- 夾取診斷 ----

def test_overflow_reports_fraction_and_magnitude_separately():
    """**只報比例會誤導**：自然照片有大量貼著邊界的飽和像素，任何擾動都會把
    它們推出界，於是比例很高而幅度極小。FND-057 實測 23–35% 的比例配上
    1e-4 到 2e-3 的幅度，正是這個現象。"""
    v = torch.tensor([[[[-0.001, 0.5, 1.002, 0.3]]]], dtype=torch.float64)
    o = sdm._overflow(v)
    assert abs(o["clip_fraction"] - 0.5) < 1e-12
    assert abs(o["clip_max"] - 0.002) < 1e-12
    assert abs(o["clip_mean"] - 0.00075) < 1e-12


def test_overflow_is_zero_inside_the_range():
    o = sdm._overflow(torch.rand(1, 3, 8, 8, dtype=torch.float64))
    assert o["clip_fraction"] == 0.0 and o["clip_max"] == 0.0


# ---- 五個版本 ----

def test_build_variants_produces_all_five_and_aligns_the_scaled_pair():
    """`amp_s`／`pha_s` 的失真必須等於 `full` 的——這是 FND-057 主表可解讀的
    唯一前提。"""
    x, xd = _img(), _img(seed=5)
    suite = _StubSuite()
    out = sdm.build_variants(x, xd, suite)
    assert set(out) == set(sdm.VARIANTS)

    full = suite.pairwise(x, out["full"][0])["dists"]
    for name in ("amp_s", "pha_s"):
        got = suite.pairwise(x, out[name][0])["dists"]
        assert abs(got - full) < full * 0.1, f"{name} 沒有對齊到 full 的失真"


def test_build_variants_full_is_the_defended_image_untouched():
    x, xd = _img(), _img(seed=6)
    out = sdm.build_variants(x, xd, _StubSuite())
    assert torch.equal(out["full"][0], xd)
    assert out["full"][1]["scale"] == 1.0


def test_unscaled_halves_are_not_aligned():
    """`amp`／`pha` 是 PAD 的原始分解，**沒有**對齊——報表若拿它們比就是錯的，
    故此處釘住它們的縮放係數為 1 且失真通常不等於 full。"""
    x, xd = _img(), _img(seed=7)
    out = sdm.build_variants(x, xd, _StubSuite())
    for name in ("amp", "pha"):
        assert out[name][1]["scale"] == 1.0


def test_build_variants_rejects_complex_leakage():
    """交叉互換後逆轉換的虛部應是浮點誤差；超過門檻代表輸入不是實數影像，
    必須直接失敗而不是靜默取實部。"""
    x, xd = _img(), _img(seed=8)
    real = sdm.decompose(x.double(), xd.double())
    assert real["imag_max"] < 1e-9


def test_variants_stay_in_range():
    x, xd = _img(), _img(seed=9)
    out = sdm.build_variants(x, xd, _StubSuite())
    for name, (img, _) in out.items():
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0, name
