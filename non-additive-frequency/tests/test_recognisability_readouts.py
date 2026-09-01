"""量「原圖還認得出來嗎」的讀數，配對必須是原圖對防禦後編輯。

存在理由
────────────────────────────────────────────────────────────────────
擋下的判準由使用者裁定為**原圖還認得出來嗎**，而現行代理
`siglip_sim` 量的是 `image_similarity(編輯(原圖), 編輯(防禦圖))`——兩張
**編輯輸出**之間的距離。配對就錯了：它回答「防禦有沒有改變輸出」，不回答
「輸出還看不看得出原圖」。

`drift` 已經是 `LPIPS(原圖, 編輯(防禦圖))`，即該配對的像素版；語意版
（CLIP／SigLIP）從未量過，而語意正是「認得出來」這個問法的層級。

`*_to_orig` 相對於 `*_to_orig_base`（原圖對未防禦編輯）的差則扣掉「這張圖
本來就會被編輯推開多遠」，即空白地板的語意版。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import defense_outcome_metrics as dom  # noqa: E402


REQUIRED = ("clip_to_orig", "siglip_to_orig",
            "clip_to_orig_base", "siglip_to_orig_base",
            "siglip_to_orig_gain")


def test_recognisability_columns_are_written_literally():
    src = (ROOT / "scripts" / "defense_outcome_metrics.py").read_text(
        encoding="utf-8")
    for col in REQUIRED:
        assert f'"{col}":' in src, col


def test_the_new_pair_is_original_against_defended_edit():
    """必須有一次 `image_similarity(x, ed)`。

    比對字面字串而非執行整條管線：這裡不載 VLM 權重，而配對接錯的失效方式
    正是「那一行量的是別的兩張」——結果仍是合法的數字，不會有症狀。
    """
    src = (ROOT / "scripts" / "defense_outcome_metrics.py").read_text(
        encoding="utf-8")
    assert "image_similarity(x, ed)" in src
    assert "image_similarity(x, eo)" in src
    assert "image_similarity(eo, ed)" in src


def test_readout_list_includes_the_new_ones():
    """候選讀數的清單要含新欄位，否則 AUC 那一段不會評它們。"""
    assert "siglip_to_orig" in dom.READOUTS
    assert "siglip_to_orig_gain" in dom.READOUTS
