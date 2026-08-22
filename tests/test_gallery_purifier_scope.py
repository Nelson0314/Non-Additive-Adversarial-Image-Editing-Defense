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


def test_page_copy_follows_the_scope():
    """關掉淨化階之後，標題與說明也要跟著換。

    原本的抬頭、標題與警告框整段都在講淨化（「同一張防禦圖走過四個淨化算子
    之後還剩下什麼」「本頁回答淨化把擾動抹掉了多少」）。條件對條件的比較裡
    那些話全部是錯的，而錯的說明比沒有說明更糟——讀的人會照它去解讀畫面。
    """
    cards = [{"image": "img", "box": [0, 0],
              "stages": {"原圖": {"full": "", "zoom": ""}},
              "conditions": {"a": {"防禦圖": {"full": "", "zoom": ""}}}}]
    purify = gallery.render(cards, [], [], scope="standard")
    compare = gallery.render(cards, [], [], scope="none")

    assert "<title>擾動存活檢視台</title>" in purify
    assert "淨化算子" in purify

    assert "<title>擾動存活檢視台</title>" not in compare
    assert "淨化" not in compare.split("</header>")[0]


def test_render_defaults_to_the_purification_copy():
    """預設值不變，逐位元等於加這個旗標之前。"""
    cards = [{"image": "img", "box": [0, 0],
              "stages": {"原圖": {"full": "", "zoom": ""}},
              "conditions": {}}]
    assert gallery.render(cards, [], []) == gallery.render(
        cards, [], [], scope="standard")


def test_title_can_be_overridden_per_page():
    """一份批次拆成多頁時，各頁必須有自己的名字。

    頁面體積超過 16 MB 就得拆（`main()` 會警告），而拆出來的每一頁若共用
    同一個 `<title>`，在瀏覽器分頁與 Artifact 清單裡就分不出誰是誰。
    """
    cards = [{"image": "img", "box": [0, 0],
              "stages": {"原圖": {"full": "", "zoom": ""}},
              "conditions": {}}]
    out = gallery.render(cards, [], [], scope="none", title="改色與天氣")
    assert "<title>改色與天氣</title>" in out
    assert "<h1>改色與天氣</h1>" in out
    # 抬頭與說明不受影響：換的是名字，不是這一頁在回答什麼。
    assert gallery.PAGE_COPY["none"]["eyebrow"] in out

    ns = gallery.build_parser().parse_args(["--src", "a", "--out", "b"])
    assert ns.title is None
