"""佈局讀數的不變量。只用合成的類別圖，不載任何權重。"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.layout import layout_row  # noqa: E402


def seg(fill, side=32, box=None, cls=4):
    s = torch.full((side, side), fill, dtype=torch.long)
    if box:
        a, b, c, d = box
        s[a:b, c:d] = cls
    return s


def test_完全相同時兩個_IoU_都是一():
    a = seg(0, box=(4, 20, 4, 20))
    r = layout_row(a, a.clone())
    assert r["iou_person"] == pytest.approx(1.0)
    assert r["iou_classes"] == pytest.approx(1.0)
    assert r["n_classes"] == 1


def test_完全不重疊時是零():
    a = seg(0, box=(0, 8, 0, 8))
    b = seg(0, box=(20, 28, 20, 28))
    r = layout_row(a, b)
    assert r["iou_person"] == pytest.approx(0.0)
    assert r["iou_classes"] == pytest.approx(0.0)


def test_部分重疊介於零與一之間():
    a = seg(0, box=(0, 16, 0, 16))       # 256 px
    b = seg(0, box=(8, 24, 0, 16))       # 256 px，重疊 128
    r = layout_row(a, b)
    assert r["iou_person"] == pytest.approx(128 / 384, abs=1e-4)


def test_兩邊都沒有人時回空字串而不是一或零():
    """「兩張圖都沒有人」與「佈局完全一致」是不同的事。"""
    empty = seg(0)
    r = layout_row(empty, empty.clone())
    assert r["iou_person"] == "" and r["iou_classes"] == ""
    assert r["n_classes"] == 0


def test_類別換掉時_person_仍高而_classes_塌掉():
    """衣服變成別的東西：人的輪廓沒變，但逐類 IoU 會抓到。

    這正是身分讀數與 LPIPS 都判不準的那一類失效。
    """
    a = seg(0, box=(4, 28, 4, 28), cls=4)      # Upper-clothes
    b = seg(0, box=(4, 28, 4, 28), cls=6)      # Pants
    r = layout_row(a, b)
    assert r["iou_person"] == pytest.approx(1.0)
    assert r["iou_classes"] == pytest.approx(0.0)
    assert r["n_classes"] == 2


def test_背景不進逐類平均():
    """背景佔面積最大，算進去會把所有比較都拉向 1。"""
    a = seg(0, box=(0, 4, 0, 4))
    b = seg(0, box=(0, 4, 0, 4))
    assert layout_row(a, b)["n_classes"] == 1     # 只有 cls=4，不含背景


def test_形狀不同時拋錯():
    with pytest.raises(ValueError, match="形狀"):
        layout_row(seg(0, side=16), seg(0, side=32))
