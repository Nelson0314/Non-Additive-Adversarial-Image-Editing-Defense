"""抗淨化表的兩件事：逐圖扣地板、以及標籤不可併錯。

`ours_nonadd` 與 `ours_add` 的 `condition` 欄都是 `phase_gain`，**只有檔名
分得開**；把標籤取成 `split("_")[0]` 會把兩個主線設定併成一格而不會有任何
症狀。地板逐圖不同（同一個模糊在不同影像上推開的量差好幾倍），故相減必須在
逐圖層級做。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from retention_table import net_gain, tag_of, usable_rows  # noqa: E402


def _row(file, image, cond, pur, effect, usable=True):
    return {"_file": file, "image": image, "condition": cond,
            "purifier": pur, "effect_mean": str(effect),
            "usable": str(usable)}


def test_標籤由檔名還原而不是取第一段():
    assert tag_of("ours_add_color.csv") == "ours_add"
    assert tag_of("ours_nonadd_scene.csv") == "ours_nonadd"
    assert tag_of("dct_y_e14_object.csv") == "dct_y_e14"


def test_分片名不認得時拋錯():
    with pytest.raises(ValueError, match="分片名"):
        tag_of("ours_add_unknownshard.csv")


def test_兩個主線設定不會被併成一格():
    rows = [
        _row("ours_add_color.csv", "a", "phase_gain", "blur1", 0.5),
        _row("ours_nonadd_color.csv", "a", "phase_gain", "blur1", 0.4),
        _row("floor_all.csv", "a", "none", "blur1", 0.2),
    ]
    table, dropped = net_gain(rows)
    assert dropped == []
    tags = sorted(r["condition"] for r in table)
    assert tags == ["ours_add|phase_gain", "ours_nonadd|phase_gain"]


def test_地板逐圖相減而不是先平均再相減():
    """兩張影像的地板差十倍。先平均再相減會得到另一個數字。"""
    rows = [
        _row("ours_add_color.csv", "a", "phase_gain", "blur1", 0.50),
        _row("ours_add_color.csv", "b", "phase_gain", "blur1", 0.30),
        _row("floor_all.csv", "a", "none", "blur1", 0.40),
        _row("floor_all.csv", "b", "none", "blur1", 0.04),
    ]
    table, _ = net_gain(rows)
    assert len(table) == 1
    # 逐圖：(0.50−0.40) 與 (0.30−0.04) → 平均 0.18
    assert table[0]["net_gain"] == pytest.approx(0.18)
    assert table[0]["wins_over_floor"] == 2


def test_缺地板的格子被排除而不是補零():
    rows = [
        _row("ours_add_color.csv", "a", "phase_gain", "blur1", 0.5),
        _row("ours_add_color.csv", "b", "phase_gain", "blur1", 0.5),
        _row("floor_all.csv", "a", "none", "blur1", 0.2),
    ]
    table, dropped = net_gain(rows)
    assert len(dropped) == 1 and "缺地板" in dropped[0]
    assert table[0]["n_images"] == 1


def test_usable為false的列被排除():
    rows = [
        _row("ours_add_color.csv", "a", "phase_gain", "blur1", 0.5),
        _row("ours_add_color.csv", "b", "phase_gain", "blur1", 0.5, usable=False),
    ]
    keep, skipped = usable_rows(rows)
    assert skipped == 1 and len(keep) == 1
