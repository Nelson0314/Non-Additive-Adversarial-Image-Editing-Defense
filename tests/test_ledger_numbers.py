"""把 `docs/LEDGER.md` 引用的關鍵數字釘回 `runs/` 的原始資料。

本專案的帳（LEDGER）與證據（`runs/`）是兩份東西，而帳是人寫的。
先前已經出現過帳與資料不符的情形——例如 3.19 說「綁住 site S 的是鈍化約束」，
而該批的 `history.json` 顯示三道 hinge 一次都沒有啟動（7.10）。
那一次不是數字抄錯，是**解釋**錯了，但同一個風險對數字也成立。

`runs/` 是唯一的證據來源且實驗無法重跑，故這些數字是常數而非會漂移的量：
測試失敗只有兩種可能，帳抄錯了、或有人動了 `runs/`。兩者都必須立刻知道。

只釘 2026-08-04 新增的那一批（1.15–1.26、5.10–5.12、1.22）。舊條目的資料
散在幾十個目錄裡，逐一釘住的維護成本高於效益。
"""

import csv
import json
import statistics as st
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _cells(path):
    rows = list(csv.DictReader(open(ROOT / path, encoding="utf-8")))
    for r in rows:
        for k in r:
            if k not in ("image", "attack"):
                r[k] = float(r[k])
    return rows


def _mean(rows, attack, key):
    d = [r for r in rows if r["attack"] == attack]
    return st.mean(r[key] for r in d), len(d)


# ---------------------------------------------------------------------------
# 1.15／1.17：三類判準逐攻擊（只留未防禦編輯明顯成功的 18 張）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attack,dsiglip,dniqe,n_sem", [
    # `semantic` 是唯一 Δsiglip 為負、也是唯一在語意軸上得分的
    ("semantic", -0.01239, -0.5149, 6),
    # 兩個 PhotoGuard 變體 的 Δsiglip 為正，即免疫後的編輯更服從 prompt
    ("pg_encoder", 0.00656, 5.7480, 1),
    ("pg_diffusion", 0.00908, 0.6558, 0),
])
def test_三類判準逐攻擊(attack, dsiglip, dniqe, n_sem):
    rows = _cells("runs/l3_criterion_axes_working/cells.csv")
    m, n = _mean(rows, attack, "dsiglip_mean")
    assert n == 18
    assert m == pytest.approx(dsiglip, abs=5e-5)
    assert _mean(rows, attack, "dniqe_mean")[0] == pytest.approx(dniqe, abs=5e-4)
    d = [r for r in rows if r["attack"] == attack]
    got = sum(1 for r in d
              if r["dsiglip_mean"] < 0
              and abs(r["dsiglip_mean"]) > r["dsiglip_sd"])
    assert got == n_sem


def test_語意軸與距離軸的排序相反():
    """1.15 的主張：Table 1 把 `semantic` 排最後，語意軸把它排第一。"""
    rows = _cells("runs/l3_criterion_axes_working/cells.csv")
    dist = {a: _mean(rows, a, "dist_mean")[0]
            for a in ("semantic", "pg_encoder", "pg_diffusion")}
    sem = {a: _mean(rows, a, "dsiglip_mean")[0] for a in dist}
    assert min(dist, key=dist.get) == "semantic"     # 距離軸最低 = 最差
    assert min(sem, key=sem.get) == "semantic"       # 語意軸最低 = 最好


# ---------------------------------------------------------------------------
# 1.20／1.23／1.25／1.26：閘門
# ---------------------------------------------------------------------------


def _gate():
    return {r["arm"]: r for r in
            csv.DictReader(open(ROOT / "runs/gate_suppress/summary.csv",
                                encoding="utf-8"))}


@pytest.mark.parametrize("arm,pert,dsig,dniqe,edit", [
    ("opt", 0.5352, -0.05674, 0.4355, 0.5299),
    ("rand", 0.5314, -0.04187, 2.7123, 0.5465),
])
def test_閘門逐條件(arm, pert, dsig, dniqe, edit):
    r = _gate()[arm]
    assert float(r["pert_lpips"]) == pytest.approx(pert, abs=5e-5)
    assert float(r["dsiglip_mean"]) == pytest.approx(dsig, abs=5e-5)
    assert float(r["dniqe_mean"]) == pytest.approx(dniqe, abs=5e-4)
    assert float(r["edit_lpips"]) == pytest.approx(edit, abs=5e-4)


def test_距離判準把隨機排在最佳化之上():
    """1.20：文獻主流的判準給出與方法設計意圖相反的排序。"""
    g = _gate()
    assert float(g["rand"]["edit_lpips"]) > float(g["opt"]["edit_lpips"])


def test_同LPIPS下隨機的劣化是最佳化的數倍():
    """1.25／1.26：匹配 LPIPS 不等於匹配可辨失真。"""
    g = _gate()
    ratio = float(g["rand"]["dniqe_mean"]) / float(g["opt"]["dniqe_mean"])
    assert ratio > 5.0


def test_配對分析的效果量與所需樣本數():
    """1.23：兩個條件共用同一組 ε，配對使所需 n 由約 48 降到約 7。"""
    rows = list(csv.DictReader(
        open(ROOT / "runs/gate_suppress/results.csv", encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["eval_seed"]] = (
            float(r["edit_siglip_b"]) - float(r["edit_siglip_a"]))
    seeds = sorted(set(by["opt"]) & set(by["rand"]))
    d = [by["opt"][s] - by["rand"][s] for s in seeds]
    assert len(d) == 5
    m, sd = st.mean(d), st.stdev(d)
    assert m == pytest.approx(-0.01487, abs=5e-5)
    assert sd == pytest.approx(0.01400, abs=5e-5)
    assert m / sd == pytest.approx(-1.06, abs=0.01)      # Cohen d
    assert sum(1 for v in d if v < 0) == 4               # 5 個種子 4 個同向
    n_need = (1.96 + 0.84) ** 2 * (sd / abs(m)) ** 2
    assert 6 <= n_need <= 8


# ---------------------------------------------------------------------------
# 5.10–5.12：起點梯度
# ---------------------------------------------------------------------------


def test_untargeted_在真實run的第0步梯度精確為零():
    """5.10：site PF 的 φ=0 加上 identity 淨化，兩條分支逐元素相同。"""
    for img in ("man_00", "man_01", "man_02"):
        h = json.load(open(ROOT / f"runs/ours_lo/{img}__PF__history.json"))
        assert h[0]["grad_norm"] == 0.0
        assert h[0]["edit_shift"] == 0.0
        assert h[1]["grad_norm"] > 0.1        # 第 1 步換成 blur 才有梯度


def test_site_S_的起步梯度小三個數量級():
    """5.12：S 的起步梯度來自 grid_sample 的數值底線，與參數化無關。"""
    pf = json.load(open(ROOT / "runs/ours_lo/man_00__PF__history.json"))
    s = json.load(open(ROOT / "runs/ours_lo/man_00__S__history.json"))
    assert 0 < s[0]["grad_norm"] < 1e-3
    assert pf[1]["grad_norm"] / s[0]["grad_norm"] > 500


def test_site_S_三格的三道hinge一次都沒啟動():
    """3.23／7.10：跑滿上限不是未收斂的證據，是準則沒有定義。"""
    for img in ("man_00", "man_01", "man_02"):
        h = json.load(open(ROOT / f"runs/ours_lo/{img}__S__history.json"))
        assert len(h) == 150
        for k in ("fid_pen_lpips", "fid_pen_acut", "fid_pen_chroma"):
            assert all(step.get(k, 0.0) == 0.0 for step in h), k


# ---------------------------------------------------------------------------
# 1.22：未防禦編輯的成功與否
# ---------------------------------------------------------------------------


def test_六張影像的未防禦編輯不算成功():
    rows = list(csv.DictReader(
        open(ROOT / "runs/l3_criterion_axes/edit_success.csv",
             encoding="utf-8")))
    assert len(rows) == 24
    assert sum(r["edit_worked_clearly"] == "True" for r in rows) == 18
    bad = {r["image"] for r in rows if r["edit_worked_clearly"] != "True"}
    assert bad == {"dog_03", "man_02", "cat_01", "dog_02", "woman_00", "bird_03"}
    # dog_03 的編輯效果是負的：編輯反而讓輸出更遠離 prompt
    d3 = next(r for r in rows if r["image"] == "dog_03")
    assert float(d3["edit_effect"]) < 0
