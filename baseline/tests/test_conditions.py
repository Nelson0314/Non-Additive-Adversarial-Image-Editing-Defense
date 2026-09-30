"""主表十二個條件的規格釘樁。

**為什麼要有這一支。** 主線目錄的 `tests/test_baselines.py` 斷言
`set(AUDIT) == set(REGISTRY)`，而那個 `REGISTRY` 只收共用 `immunization_baseline/attacks/pgd.py`
骨幹的六個 spec（`photoguard_c`、`mist`、`dia_pt`、`dia_r`、`advpaint`、
`promptflare`）。它與主表十二列的交集只有四個：`photoguard_c`、`mist`、
`dia_pt`、`dia_r`。其餘八列走各自的模組（`dct_shield` 有自己的 `DCTShieldSpec`
與自己的 `REGISTRY`，`dayn`／`sifm`／`danp`／`diffvax` 各自獨立，
`color` 走本專案的顏色載體（`lab/code/color_defence.py`）），**改動它們的常數不會有任何測試失敗**。

這一支釘的是**已交付的讀數自己記下的設定**：`results/defense_<方法>.csv` 的
求解欄位、讀數 CSV 的規模與鍵、以及協定欄位。它不重跑求解，也不判定任何條件
的好壞——只確保重跑後寫出來的設定與已發布的主表是同一組。

期望值全部抄自 `results/` 的現況，出處欄見主線目錄的
`../../docs/reference/BASELINE_PROVENANCE.md` 與 `../../docs/reference/SOURCE_AUDIT.md` §10。
"""

from __future__ import annotations

import csv

import pytest

from immunization_baseline import layout

E = 1.0 / 255.0

#: 條件 → (eps, eps_pixel01, norm, steps, grad_reps, modified_from_paper)。
#: `eps_pixel01` 為 None 表示該欄在 CSV 中是空的——束縛不是像素域的 L∞／L2，
#: 逐列的單位見 `../../docs/reference/BASELINE_PROVENANCE.md`
#: §「`eps_pixel01` 欄的單位，逐列」。
SOLVER = {
    # 共用 PGD 骨幹的四列，與主線 `tests/test_baselines.py` 的 AUDIT 重疊。
    "photoguard_c":      (16.0,   8.0,     "l2",   200, 10, True),
    "photoguard_linf":   (32 * E, 16 * E,  "linf", 200, 10, True),
    "mist":              (32 * E, 16 * E,  "linf", 100,  1, False),
    "dia_pt":            (0.05,   0.025,   "linf",  20,  1, False),
    "dia_r":             (0.05,   0.025,   "linf",  20,  1, False),
    # 以下八列在主線的 REGISTRY 之外，只有這裡釘。
    "dayn":              (0.06,   0.03,    "linf", 100, 10, True),
    "sifm":              (0.03,   0.03,    "linf", 100,  1, True),
    "danp":              (0.03,   0.03,    "linf", 100, 10, True),
    "dct_shield":        (1.0,    None,    "dct_coeff_linf",  1000, 1, False),
    "dct_shield_y":      (1.0,    None,    "dct_coeff_linf",  1000, 1, False),
    "color":             (32.0,   None,    "delta_e00_cap",   None, None, None),
    "diffvax":           (None,   None,    "none", 1, None, False),
}

#: `../../docs/reference/SOURCE_AUDIT.md` §10 的值域對照表。三篇在 `[-1,1]` 上最佳化，
#: 換算方式逐篇不同：Mist 有乘 2、DIA 直接用。CSV 的 `eps_pixel01` 必須與此一致。
VALUE_RANGE_TABLE = {
    "photoguard_c": 8.0,      # L2 半徑，非 L∞，不可與下列並排比大小
    "mist": 16 * E,
    "dia_pt": 0.025,
    "dia_r": 0.025,
}

IMAGES = ("man_00", "man_01", "man_02", "man_03",
          "woman_00", "woman_01", "woman_02", "woman_03")
PURIFIERS = ("blur1", "blur2", "crop_resize0.1",
             "jpeg30", "jpeg50", "jpeg80", "rotate15")
GEOMETRIC = ("crop_resize0.1", "rotate15")
SCENARIOS = ("ip2p", "inpaint")
UNDEFENDED_ARMS = {"ip2p": "ip2p_si18", "inpaint": "inpaint_undefended"}
SIGLIP_THRESHOLD = "0.837"
PROMPT_INDICES = ("0", "1", "2", "3")


def read(name: str) -> list[dict]:
    with (layout.RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(text: str):
    return None if text == "" else float(text)


@pytest.mark.parametrize("condition", sorted(SOLVER))
def test_solver_settings_match_the_published_table(condition):
    rows = read(f"defense_{condition}.csv")
    eps, eps01, norm, steps, reps, modified = SOLVER[condition]
    for row in rows:
        assert row["condition"] == condition
        assert number(row["eps"]) == pytest.approx(eps) if eps is not None \
            else row["eps"] == ""
        if eps01 is None:
            assert row["eps_pixel01"] == "", (
                f"{condition} 的束縛不是像素域的 L∞／L2，該欄應為空")
        else:
            assert number(row["eps_pixel01"]) == pytest.approx(eps01)
        assert row["norm"] == norm
        if steps is not None:
            assert int(row["steps"]) == steps
        if reps is not None:
            assert int(row["grad_reps"]) == reps
        if modified is not None:
            assert row["modified_from_paper"] == str(modified)


@pytest.mark.parametrize("condition", sorted(SOLVER))
def test_each_condition_covers_the_eight_images_once(condition):
    rows = read(f"defense_{condition}.csv")
    assert len(rows) == len(IMAGES)
    assert sorted(r["image"] for r in rows) == sorted(IMAGES)


@pytest.mark.parametrize("condition,eps01", sorted(VALUE_RANGE_TABLE.items()))
def test_value_range_table_agrees_with_the_csv(condition, eps01):
    """`SOURCE_AUDIT.md` §10 的像素域 eps 與求解 CSV 記的是同一個數。"""
    rows = read(f"defense_{condition}.csv")
    assert number(rows[0]["eps_pixel01"]) == pytest.approx(eps01)


def test_modified_conditions_say_what_was_changed():
    """`modified_from_paper=True` 的列必須有 `modification_note`。"""
    for condition in sorted(SOLVER):
        for row in read(f"defense_{condition}.csv"):
            if row["modified_from_paper"] == "True":
                assert row["modification_note"].strip(), (
                    f"{condition} 標了 modified_from_paper 卻沒寫改了什麼")


def test_every_condition_names_its_source():
    for condition in sorted(SOLVER):
        for row in read(f"defense_{condition}.csv"):
            assert row["spec_source"].strip(), f"{condition} 沒有 spec_source"


def test_displacement_is_twelve_conditions_by_sixty_four_cells():
    rows = read("displacement.csv")
    assert len(rows) == 768 == len(SOLVER) * len(SCENARIOS) * len(IMAGES) * 4
    assert {r["condition"] for r in rows} == set(SOLVER)
    keys = {(r["condition"], r["scenario"], r["image"], r["prompt_index"])
            for r in rows}
    assert len(keys) == len(rows), "displacement.csv 有重複的格"
    assert {r["scenario"] for r in rows} == set(SCENARIOS)
    assert {r["prompt_index"] for r in rows} == set(PROMPT_INDICES)


def test_displacement_protocol_columns_are_constant():
    for row in read("displacement.csv"):
        assert row["undefended_arm"] == UNDEFENDED_ARMS[row["scenario"]]
        assert row["arm"] == f"{row['scenario']}_{row['condition']}"
        assert row["siglip_blocked_threshold"] == SIGLIP_THRESHOLD


def test_retention_is_seven_purifiers_on_top_of_displacement():
    rows = read("retention.csv")
    assert len(rows) == 5376 == 768 * len(PURIFIERS)
    assert {r["purifier"] for r in rows} == set(PURIFIERS)
    keys = {(r["condition"], r["purifier"], r["scenario"],
             r["image"], r["prompt_index"]) for r in rows}
    assert len(keys) == len(rows), "retention.csv 有重複的格"


def test_geometric_flag_matches_the_purifier_name():
    for row in read("retention.csv"):
        assert row["geometric"] == str(row["purifier"] in GEOMETRIC)


@pytest.mark.parametrize("name,rows", [
    ("additional_metrics/displacement.csv", 768),
    ("additional_metrics/retention.csv", 5376),
    ("additional_metrics/fidelity.csv", 96),
    ("additional_metrics/aesthetic.csv", 104),
])
def test_additional_metric_tables_keep_their_size(name, rows):
    assert len(read(name)) == rows


def test_fidelity_covers_twelve_conditions_times_eight_images():
    rows = read("additional_metrics/fidelity.csv")
    assert len(rows) == len(SOLVER) * len(IMAGES) == 96
    assert {r["condition"] for r in rows} == set(SOLVER)
