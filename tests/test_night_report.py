"""`scripts/night_report.py` 的彙總邏輯 — FND-061 的表由它算。

**這是從原始 CSV 到公布數字之間最後一個沒有測試的環節。** 主讀數是
「扣掉空白地板的淨增益」，而那個減法只存在於報告產生器裡：`phase_retention.py`
只寫出絕對位移量，地板是另一批 CSV，兩者相減發生在這裡。

減錯的後果是靜默的：表格照樣填滿、比值照樣有大有小，只是每一格都偏移了一個
常數。FND-061 的「紋理重相位 +0.0888 對 DCT-Shield base +0.0645」正是這個
減法的輸出。
"""

import importlib.util
import re
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "night_report",
    Path(__file__).resolve().parent.parent / "scripts" / "night_report.py")
nr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nr)

HEAD = ("image,condition,budget_target,purifier,effect_mean,effect_sd,"
        "effect_identity_mean,effect_identity_sd,retention,edit_strength,"
        "usable,seconds\n")


def _row(image, cond, pur, eff):
    return f"{image},{cond},human,{pur},{eff},0.01,0.5,0.02,1.0,0.7,True,10\n"


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEAD + "".join(rows), encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """把報告的資料根目錄指到暫存區，不碰真實的 `runs/`。"""
    monkeypatch.setattr(nr, "ROOT", tmp_path)
    monkeypatch.setattr(nr, "IMG", tmp_path / "img")
    return tmp_path


def _cells(html):
    """把一列 HTML 表格拆成純文字格。"""
    return [re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<td>(.*?)</td>", html, re.S)]


# ---- 淨增益的減法 ----

def test_net_gain_is_effect_minus_floor(root):
    """單一算子、單一影像：淨增益必須恰好是 `effect − floor`。"""
    _write(root / "runs/freqret/ret_0.csv",
           [_row("a", "phase", "identity", 0.40),
            _row("a", "phase", "jpeg75", 0.30)])
    _write(root / "runs/freqret/floor_0.csv",
           [_row("a", "none", "jpeg75", 0.12)])

    html = nr.retention_section()
    assert "+0.1800" in html, "淨增益應為 0.30 − 0.12 = 0.18"


def test_net_gain_averages_over_images_before_subtracting(root):
    """地板是**跨影像的平均**，與條件的平均相減。逐圖相減再平均在算術上
    相同，但本專案一律報「平均比平均」（FND-037／039），故此處釘住實作
    走的是哪一條。"""
    _write(root / "runs/freqret/ret_0.csv",
           [_row("a", "phase", "identity", 0.40),
            _row("b", "phase", "identity", 0.60),
            _row("a", "phase", "jpeg75", 0.20),
            _row("b", "phase", "jpeg75", 0.40)])
    _write(root / "runs/freqret/floor_0.csv",
           [_row("a", "none", "jpeg75", 0.10),
            _row("b", "none", "jpeg75", 0.20)])

    html = nr.retention_section()
    # 條件平均 0.30、地板平均 0.15 → 淨增益 0.15
    assert "+0.1500" in html


def test_identity_column_is_not_floor_subtracted(root):
    """`identity` 的地板由構造為 0（同 seed、同輸入、SDEdit 是確定性的），
    故該欄只報絕對值，不做減法。"""
    _write(root / "runs/freqret/ret_0.csv",
           [_row("a", "phase", "identity", 0.4537)])
    html = nr.retention_section()
    assert "0.4537" in html
    assert "−地板" not in html.split("0.4537")[0][-80:]


def test_conditions_are_ordered_by_net_gain(root):
    """表格依淨增益由高到低排序——報表的第一列就是結論。"""
    _write(root / "runs/freqret/ret_0.csv",
           [_row("a", "weak", "identity", 0.2), _row("a", "weak", "jpeg75", 0.15),
            _row("a", "strong", "identity", 0.5), _row("a", "strong", "jpeg75", 0.45)])
    _write(root / "runs/freqret/floor_0.csv", [_row("a", "none", "jpeg75", 0.10)])

    html = nr.retention_section()
    assert html.index("strong") < html.index("weak")


def test_all_four_source_globs_are_merged(root):
    """四批 CSV（第一趟、gridpure 補跑、DCT-Shield、預算對齊）必須都被讀進來。
    漏掉任何一批，那個條件會安靜地從表上消失。"""
    for name, cond in (("ret_0", "phase"), ("gret_0", "phase_g"),
                       ("dret_0", "dct_shield"), ("aret_0", "dct_shield")):
        _write(root / f"runs/freqret/{name}.csv",
               [_row("a", cond, "identity", 0.4), _row("a", cond, "jpeg75", 0.3)])
    _write(root / "runs/freqret/floor_0.csv", [_row("a", "none", "jpeg75", 0.1)])

    html = nr.retention_section()
    for want in ("phase", "phase_g", "DCT-Shield"):
        assert want in html, f"{want} 沒有出現在表上"


def test_aligned_batch_is_renamed_so_it_cannot_merge_with_the_native_one(root):
    """`aret_*.csv` 的 `condition` 欄與原生 ε=1 同名（都是 `dct_shield`）。
    合併前必須改名，否則兩個不同預算的結果會被平均在同一列上。"""
    _write(root / "runs/freqret/dret_0.csv",
           [_row("a", "dct_shield", "identity", 0.60),
            _row("a", "dct_shield", "jpeg75", 0.50)])
    _write(root / "runs/freqret/aret_0.csv",
           [_row("a", "dct_shield", "identity", 0.20),
            _row("a", "dct_shield", "jpeg75", 0.15)])
    _write(root / "runs/freqret/floor_0.csv", [_row("a", "none", "jpeg75", 0.10)])

    html = nr.retention_section()
    assert "ε=1" in html and "對齊 DISTS" in html, "兩批應各自成列"
    assert "+0.4000" in html and "+0.0500" in html


# ---- 缺資料時要說出來，不要給空表 ----

def test_missing_data_says_so_rather_than_rendering_an_empty_table(root):
    assert "尚未產出" in nr.retention_section()
    assert "尚未產出" in nr.dct_section()


def test_floorless_purifier_is_shown_without_a_net_gain(root):
    """某個算子沒有地板資料時，該格只報絕對值且不進淨增益的平均——
    拿 0 當地板會把那一格灌水。"""
    _write(root / "runs/freqret/ret_0.csv",
           [_row("a", "phase", "identity", 0.40),
            _row("a", "phase", "blur1", 0.35),
            _row("a", "phase", "jpeg75", 0.30)])
    _write(root / "runs/freqret/floor_0.csv", [_row("a", "none", "jpeg75", 0.10)])

    html = nr.retention_section()
    # 只有 jpeg75 有地板，故淨增益均值就是它自己的 0.20
    assert "<b>+0.2000</b>" in html


# ---- 小工具 ----

def test_fl_returns_none_on_blanks_so_means_skip_them():
    assert nr.fl({"x": ""}, "x") is None
    assert nr.fl({"x": "1.5"}, "x") == 1.5
    assert nr.mean([1.0, None, 3.0]) == 2.0
