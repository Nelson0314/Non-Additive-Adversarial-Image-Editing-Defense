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


def test_分批補跑時identity不會被重複計入():
    """補跑某個算子時 `--purifiers` 必須帶上 identity（它是分母），於是同一格
    會出現兩次。兩次的 seed 與輸入相同、數值也相同，重複計入只會讓 n_images
    看起來變兩倍。"""
    from retention_table import dedupe
    rows = [
        _row("ours_add_color.csv", "a", "phase_gain", "identity", 0.5),
        _row("ours_add_color.csv", "a", "phase_gain", "identity", 0.5),
        _row("ours_add_color.csv", "a", "phase_gain", "gridpure", 0.3),
    ]
    keep, dup = dedupe(rows)
    assert dup == 1 and len(keep) == 2


def test_不同標籤的同一格不算重複():
    from retention_table import dedupe
    rows = [
        _row("ours_add_color.csv", "a", "phase_gain", "identity", 0.5),
        _row("ours_nonadd_color.csv", "a", "phase_gain", "identity", 0.4),
    ]
    keep, dup = dedupe(rows)
    assert dup == 0 and len(keep) == 2


# ------------------------------------------------ 出表：兩個絕對值並列

def _h2h_src(tmp_path):
    """blur1（非幾何，地板 0.2）與 crop_resize0.1（幾何，地板 0）各一格。"""
    import csv as _csv
    src = tmp_path / "h2h"
    src.mkdir()
    fields = ["image", "condition", "purifier", "effect_mean", "reference"]
    rows = [
        ("a", "none", "blur1", "0.2", "orig"),
        ("a", "none", "crop_resize0.1", "0.0", "purified_orig"),
        ("a", "phase_gain", "blur1", "0.5", "orig"),
        ("a", "phase_gain", "crop_resize0.1", "0.36", "purified_orig"),
    ]
    with (src / "floor_all.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(fields)
        for r in rows[:2]:
            w.writerow(r)
    with (src / "ours_all.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(fields)
        for r in rows[2:]:
            w.writerow(r)
    return src


def _h2h_out(tmp_path, monkeypatch, capsys):
    import retention_table as rt
    src = _h2h_src(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["retention_table.py", "--src", str(src),
                         "--out", str(tmp_path / "net_gain.csv")])
    rt.main()
    return capsys.readouterr().out


def test_出表同時印出總增益與淨增益(tmp_path, monkeypatch, capsys):
    out = _h2h_out(tmp_path, monkeypatch, capsys)
    lines = out.splitlines()
    assert len([ln for ln in lines if ln.startswith("總增益")]) == 1
    assert len([ln for ln in lines if ln.startswith("淨增益")]) == 1
    rows = [ln for ln in lines if ln.startswith("ours|phase_gain")]
    assert len(rows) == 2
    gross = [float(x) for x in rows[0].split()[1:]]
    net = [float(x) for x in rows[1].split()[1:]]
    # 欄序為 sorted(purifiers)：blur1、crop_resize0.1。
    assert gross == pytest.approx([0.5, 0.36])
    assert net == pytest.approx([0.3, 0.36])
    assert net[1] == gross[1]                       # 幾何欄：地板 0，兩者相等
    assert gross[0] - net[0] == pytest.approx(0.2)  # 非幾何欄：恰差一個地板


def test_出表兩張表都印出地板與參照(tmp_path, monkeypatch, capsys):
    out = _h2h_out(tmp_path, monkeypatch, capsys)
    floors = [ln for ln in out.splitlines() if ln.startswith("空白地板")]
    refs = [ln for ln in out.splitlines() if ln.startswith("參照")]
    assert len(floors) == 2 and len(refs) == 2
    for ln in floors:
        assert [float(x) for x in ln.split()[1:]] == pytest.approx([0.2, 0.0])
    for ln in refs:
        assert ln.split()[1:] == ["orig", "purified_orig"]


def test_出表不含佔比讀數(tmp_path, monkeypatch, capsys):
    """主讀數是兩個絕對值，任何「佔可達範圍的比例」都不得回到表上。"""
    out = _h2h_out(tmp_path, monkeypatch, capsys)
    # 末尾那一行是寫出的檔案路徑（含 tmp_path），不屬於表的內容。
    body = "\n".join(ln for ln in out.splitlines()
                     if not ln.startswith("寫出"))
    assert "可達" not in body and "%" not in body
    assert "0.772" in body and "飽和" in body        # 只留飽和值的說明


def test_出表在幾何地板非零時拋錯(tmp_path, monkeypatch):
    import retention_table as rt
    src = _h2h_src(tmp_path)
    text = (src / "floor_all.csv").read_text(encoding="utf-8")
    (src / "floor_all.csv").write_text(
        text.replace("crop_resize0.1,0.0", "crop_resize0.1,0.5193"),
        encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["retention_table.py", "--src", str(src),
                         "--out", str(tmp_path / "net_gain.csv")])
    with pytest.raises(ValueError, match="舊參照"):
        rt.main()
