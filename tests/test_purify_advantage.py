"""主張階層之首的判定腳本 —— `legacy/scripts/purify_advantage.py`。

這裡驗的是**判準本身**，不是某一批的結論：判準若沒有分辨力，跑出來的
「成立」不代表任何事。`p6_purify_retention.py` 的模組 docstring 記載了
實測到的那次失效——寬鬆判準下七對全部 4/4 成立——本檔把嚴格判準與
那條教訓一起釘住。
"""

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "purify_advantage", ROOT / "legacy" / "scripts" / "purify_advantage.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


COLUMNS = ["condition", "image_id", "tau", "purify_kind", "purify_strength",
           "seed", "effect_abs", "retention", "retention_usable"]


def write_grid(tmp_path: Path, rows) -> Path:
    d = tmp_path / "batch"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    return d


def row(cond, op, strength, ret, tau=0.2, usable=True, seed=0, img="a"):
    return {"condition": cond, "image_id": img, "tau": tau,
            "purify_kind": op, "purify_strength": strength, "seed": seed,
            "effect_abs": 0.1, "retention": ret,
            "retention_usable": str(usable).lower()}


def test_identity不進保留率統計(tmp_path):
    """identity 是**分母**，它自己的 retention 恆為 1，收進來會稀釋每個平均。"""
    d = write_grid(tmp_path, [
        row("N2", "identity", 0.0, 1.0),
        row("N2", "jpeg", 75, 0.6),
    ])
    got = pa.collect(pa.load_rows(d), 0.2)
    assert list(got) == [("N2", "jpeg", 75.0)]


def test_不可用的列被排除(tmp_path):
    """`retention_usable` 為 false 表示分母在雜訊裡。

    先驗實驗曾因為沒有這道閘門而出現 −43、−98 的比值。
    """
    d = write_grid(tmp_path, [
        row("N2", "jpeg", 75, 0.6, usable=True),
        row("N2", "jpeg", 75, -43.0, usable=False),
    ])
    got = pa.collect(pa.load_rows(d), 0.2)
    assert got[("N2", "jpeg", 75.0)] == [0.6], "不可用的列被算進平均了"


def test_只取指定的tau(tmp_path):
    d = write_grid(tmp_path, [
        row("N2", "jpeg", 75, 0.6, tau=0.2),
        row("N2", "jpeg", 75, 0.9, tau=0.35),
    ])
    assert pa.collect(pa.load_rows(d), 0.2)[("N2", "jpeg", 75.0)] == [0.6]


def test_nan的保留率不進平均(tmp_path):
    """`_fill_retention` 在分母為零時寫 NaN。它不是一個「很小的值」。"""
    d = write_grid(tmp_path, [
        row("N2", "jpeg", 75, 0.6),
        row("N2", "jpeg", 75, float("nan")),
    ])
    assert pa.collect(pa.load_rows(d), 0.2)[("N2", "jpeg", 75.0)] == [0.6]


def test_同一算子的多個強度只算一個獨立算子():
    """相鄰強度高度相關，湊三個強度不構成三個獨立證據。"""
    means = {("N2", "jpeg", s): 0.9 for s in (30, 75)}
    means |= {("mist", "jpeg", s): 0.1 for s in (30, 75)}
    r = pa.compare(means, "N2", "mist")
    assert r["n_ops"] == 1, "同一算子的兩個強度被算成兩個算子"
    assert r["ops_majority"] == 1


def test_嚴格判準不讓單一強度扛下整個算子():
    """`p6_purify_retention` 實測記載的失效：寬鬆判準下七對全部 4/4 成立。

    六個強度裡贏一個就讓該算子計入，判準因此失去分辨力。嚴格判準要求
    多數強度佔優，此處以「六個強度贏一個」釘住兩者的差別。
    """
    means = {}
    for i, s in enumerate((0.5, 1.0, 2.0, 3.0, 4.0, 5.0)):
        means[("N2", "blur", s)] = 0.9 if i == 0 else 0.1
        means[("mist", "blur", s)] = 0.2
    r = pa.compare(means, "N2", "mist")
    assert r["ops_any"] == 1, "寬鬆判準應該計入（這正是它的問題）"
    assert r["ops_majority"] == 0, "嚴格判準不該讓一個強度扛下整個算子"


def test_對手缺該算子時不算勝場():
    """對照側沒有那一格時，比較沒有定義；算成勝場等於憑空得分。"""
    means = {("N2", "jpeg", 75.0): 0.9, ("N2", "diffpure", 150.0): 0.8,
             ("mist", "jpeg", 75.0): 0.1}
    r = pa.compare(means, "N2", "mist")
    assert r["n_ops"] == 1 and "diffpure" not in r["detail"]


def test_沒有可用資料時明確中止(tmp_path):
    """印一張空表等於讓「還沒有資料」看起來像「沒有優勢」。"""
    d = write_grid(tmp_path, [row("N2", "jpeg", 75, 0.6, usable=False)])
    with pytest.raises(SystemExit):
        pa.main([str(d), "--tau", "0.2"])


def test_判準的門檻與先驗腳本一致():
    """三個獨立算子。改動它等於改動主張的成立條件，必須是刻意的。"""
    assert pa.MIN_OPS == 3
