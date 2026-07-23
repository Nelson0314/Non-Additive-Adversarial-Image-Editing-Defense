"""淨化模組單元測試（CPU、不需 SD 模型）。

驗證：輕量三法之形狀/值域/決定性與強度單調性、AdverseCleaner 兩變體、
GrIDPure grid 機制（切分覆蓋、恆等淨化器不變性、合併平均）、purify() 分派。
"""

import pytest
import torch

from src.purify import purify
from src.purify.adverse_cleaner import adverse_clean
from src.purify.gridpure import corner_boxes, grid_boxes, grid_pure
from src.purify.lightweight import crop_resize, gaussian_blur, jpeg_compress


def _image(n=64, seed=0):
    torch.manual_seed(seed)
    return torch.rand(1, 3, n, n)


def _assert_valid(out, x):
    assert out.shape == x.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert torch.isfinite(out).all()


# --- 輕量淨化 ---

def test_jpeg_valid_and_monotonic():
    x = _image()
    hi, lo = jpeg_compress(x, 90), jpeg_compress(x, 30)
    _assert_valid(hi, x)
    assert torch.equal(jpeg_compress(x, 90), hi)  # 決定性
    # 品質越低失真越大
    assert (lo - x).abs().mean() > (hi - x).abs().mean()


def test_blur_smooths():
    x = _image()
    out = gaussian_blur(x, sigma=1.5)
    _assert_valid(out, x)
    tv = lambda t: (t[..., 1:, :] - t[..., :-1, :]).abs().mean()
    assert tv(out) < tv(x)


def test_crop_resize():
    x = _image()
    out = crop_resize(x, ratio=0.2)
    _assert_valid(out, x)
    assert not torch.equal(out, x)
    assert torch.equal(crop_resize(x, 0.2), out)


# --- AdverseCleaner ---

@pytest.mark.parametrize("variant", ["bf_only", "bf_gf"])
def test_adverse_clean(variant):
    x = _image()
    out = adverse_clean(x, variant=variant)
    _assert_valid(out, x)
    assert not torch.equal(out, x)


def test_adverse_clean_variants_differ():
    x = _image()
    assert not torch.equal(adverse_clean(x, "bf_only"), adverse_clean(x, "bf_gf"))


def test_adverse_clean_unknown_variant():
    with pytest.raises(ValueError):
        adverse_clean(_image(), variant="nope")


# --- GrIDPure ---

def test_grid_boxes_512():
    """官方設定：512×512 → 九個 256×256 grid + 四角落。"""
    boxes = grid_boxes(512, 512, 256, 128)
    assert len(boxes) == 9
    assert all(x1 - x0 == 256 and y1 - y0 == 256 for x0, y0, x1, y1 in boxes)
    corners = corner_boxes(512, 512, 128)
    assert len(corners) == 4
    # 覆蓋檢查：每像素至少落在一個 grid（角落區另有第十個 grid 覆蓋）
    cnt = torch.zeros(512, 512)
    for x0, y0, x1, y1 in boxes:
        cnt[y0:y1, x0:x1] += 1
    assert (cnt >= 1).all()
    # 內部區域與兩個以上 grid 重疊（GrIDPure 設計要求）
    assert cnt.max() >= 2


def test_grid_pure_identity():
    """恆等淨化器 + 合併平均 + γ 混合 → 輸出應等於輸入。"""
    x = _image(64)
    out = grid_pure(x, lambda p: p, grid_size=32, stride=16, gamma=0.1, iterations=2)
    assert torch.allclose(out, x, atol=1e-5)


def test_grid_pure_with_blur_purifier():
    x = _image(64)
    out = grid_pure(x, lambda p: gaussian_blur(p, 1.0),
                    grid_size=32, stride=16, gamma=0.1, iterations=2)
    _assert_valid(out, x)
    assert not torch.equal(out, x)


def test_grid_pure_too_small():
    with pytest.raises(ValueError):
        grid_pure(_image(64), lambda p: p, grid_size=256)


# --- 分派介面 ---

def test_purify_dispatch():
    x = _image()
    assert torch.equal(purify(x, "jpeg", 65), jpeg_compress(x, 65))
    assert torch.equal(purify(x, "blur", 1.0), gaussian_blur(x, 1.0))
    assert torch.equal(purify(x, "crop_resize", 0.2), crop_resize(x, 0.2))
    assert torch.equal(purify(x, "advclean_bf"), adverse_clean(x, "bf_only"))
    assert torch.equal(purify(x, "advclean_bfgf"), adverse_clean(x, "bf_gf"))
    cfg = {"gridpure": {"grid_size": 32, "stride": 16, "gamma": 0.1}}
    out = purify(x, "gridpure", {"pure_steps": 1, "iterations": 1},
                 config=cfg, purifier=lambda p: p)
    assert torch.allclose(out, x, atol=1e-5)


def test_purify_errors():
    x = _image()
    with pytest.raises(ValueError):
        purify(x, "nope")
    with pytest.raises(ValueError):
        purify(x, "gridpure", {})  # 缺 purifier
    with pytest.raises(ValueError):
        purify(x, "blur")  # 缺 sigma
