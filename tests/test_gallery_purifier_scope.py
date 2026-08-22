"""`defense_purify_gallery.py` 的淨化階可以關掉。

存在理由
────────────────────────────────────────────────────────────────────
這支比對頁原本只服務抗淨化那一段，故每個條件固定展開四個淨化算子。
把它拿來做**條件對條件**的比較時（防禦圖與防禦後編輯並列多個方法），那四階
既不是要看的東西，又讓頁面體積漲三倍——`main()` 自己在超過 16 MB 時會警告。

關掉之後每個條件只剩「防禦圖」與「防禦後編輯」兩階，其餘版面邏輯不變。
預設維持 `standard`，逐位元等於加這個旗標之前。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import defense_purify_gallery as gallery  # noqa: E402


def test_default_scope_is_the_four_standard_purifiers():
    ns = gallery.build_parser().parse_args(["--src", "a", "--out", "b"])
    assert ns.purifiers == "standard"
    assert [n for n, _ in gallery.purifier_set("standard")] == [
        "blur1", "jpeg75", "crop_resize0.1", "jpeg_then_resize"]


def test_none_scope_yields_no_purifiers():
    ns = gallery.build_parser().parse_args(
        ["--src", "a", "--out", "b", "--purifiers", "none"])
    assert ns.purifiers == "none"
    assert gallery.purifier_set("none") == []


def test_unknown_scope_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match="purifiers"):
        gallery.purifier_set("blur_only")
