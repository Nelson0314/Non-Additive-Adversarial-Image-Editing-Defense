"""`phase_retention.cell_of`：由 results.csv 的一列還原防禦圖的檔名段。

三種批次的命名不同（預算對齊／人眼門檻／baseline），此處寫死一種就會在
`FileNotFoundError` 上停住——2026-08-14 的人眼門檻批次即撞到這一點。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phase_retention import cell_of  # noqa: E402


def test_預算對齊的列還原成_d_檔名段():
    c = cell_of({"image": "horse_03", "condition": "phase",
                 "budget_target": "0.075", "budget_mode": "dists"})
    assert c["tag"] == "phase__d0.075"
    assert c["budget"] == "0.075"


def test_預算為_0_04_時不留尾隨零():
    """`:g` 的作用：0.04 而非 0.040000。檔名是既有 runs/ 已經寫死的形狀。"""
    assert cell_of({"image": "x", "condition": "add",
                    "budget_target": "0.04"})["tag"] == "add__d0.04"


def test_人眼門檻的列還原成_human_檔名段():
    c = cell_of({"image": "man_02", "condition": "phase",
                 "budget_target": "human", "budget_mode": "human"})
    assert c["tag"] == "phase__human"
    assert c["budget"] == "human"


def test_只有_budget_mode_為_human_時也認得():
    """`budget_target` 若被改成別的字串，判斷仍應由 `budget_mode` 主導。"""
    assert cell_of({"image": "x", "condition": "add", "budget_target": "",
                    "budget_mode": "human"})["tag"] == "add__human"


def test_baseline_的列沒有_budget_欄位_檔名段就是條件名():
    """`apa_baseline` 存的是 {image}__{cond}__def.png，沒有 budget 段。"""
    c = cell_of({"image": "dog_03", "condition": "photoguard_c"})
    assert c["tag"] == "photoguard_c"
    assert c["budget"] == "native"


def test_影像與條件原樣帶過():
    c = cell_of({"image": "woman_02", "condition": "dia_r"})
    assert (c["image"], c["condition"]) == ("woman_02", "dia_r")
