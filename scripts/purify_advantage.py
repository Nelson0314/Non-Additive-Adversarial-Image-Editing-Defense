"""主張階層之首的判定：**非加性抗淨化是否勝過加性**。

`DESIGN_2026-08-05` §1 的階層（使用者 2026-08-05 定案）：

| 層級 | 主張 |
|---|---|
| **主** | 非加性**抗淨化**勝過加性 |
| 次 | 抗編輯持平或小輸 |
| 三 | 保真受控，報全部指標不挑選 |

本腳本只回答主層級那一條。次層級由 `grid.csv` 的 `effect_*` 直接讀，
不在此處混進來——兩個主張各自的判準不同，混在一張表上會讓「哪一條成立」
變得含糊。

量什麼
──────────────────────────────────────────────────────────────────────
`retention = effect(該淨化算子) / effect(identity)`，由 `run_report` 的
`_fill_retention` 算好並寫進 `grid.csv`。它問的不是「誰的防禦比較強」
（那由 identity 那一格決定，屬次層級），而是**「誰掉得比較慢」**。

`retention_usable` 為 false 的列一律排除：那表示 identity 那一格的效果
本身就在雜訊裡（`effect < 3σ`），除以它得到的比值不可解讀。先驗實驗曾
因為沒有這道閘門而出現 −43、−98 這種比值。

判準（沿用 `p6_purify_retention.py`，並採用它記下的教訓）
──────────────────────────────────────────────────────────────────────
該腳本實測後發現寬鬆判準「該算子的**任一**強度佔優即計入」沒有分辨力
——六個強度中的一個就能扛下整個算子，七對全部 4/4 成立。故此處以
**多數強度佔優**為準（`ops_majority`），寬鬆版一併輸出但只作參考。

「獨立算子」是 jpeg / crop_resize / adverse_cleaner / cnn_denoise /
diffpure / blur / noise / quantize 各算一個；同一算子的不同強度**只算一個**
——相鄰強度高度相關，湊三個強度不構成三個獨立證據。

用法
──────────────────────────────────────────────────────────────────────
    python scripts/purify_advantage.py runs/v14_bird_03 [runs/... ...]
    python scripts/purify_advantage.py runs/v14_merged --tau 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

import sys                                                  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.experiment import grid                             # noqa: E402

# 由格點的登記表導出（2026-08-09）。before：三行寫死的名字，新增條件之後
# 抗淨化的比較會靜默漏掉它們。
NONADDITIVE = grid.NONADDITIVE
ADDITIVE = grid.BASELINES
# 隨機對照逐參數化各一個（位移場 R、apa Ra），故不再是單一個名字。
RANDOM_BY_SITE = {"warp": grid.RANDOM_CONTROL, "apa": grid.RANDOM_CONTROL_APA}
RANDOM = grid.RANDOM_CONTROL

# 主判定的量。`effect_abs` 是 `run_report` 用來算 retention 的分子，
# 定義見 `executors._fill_retention`；此處不另訂一個，兩處必須同源。
MIN_OPS = 3


def load_rows(batch_dir: Path) -> List[Dict[str, str]]:
    p = batch_dir / "grid.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 不存在。段 4 還沒跑：`run_stage.py report --batch <批次>`")
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v: str) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if x == x else None          # NaN → None


def collect(rows, tau: float
            ) -> Dict[Tuple[str, str, float], List[float]]:
    """{(條件, 算子, 強度): [各影像各種子的 retention]}，只收可用的列。"""
    out: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        if r.get("purify_kind") in (None, "", "identity"):
            continue
        if _f(r.get("tau")) != tau:
            continue
        if str(r.get("retention_usable")).lower() != "true":
            continue
        ret = _f(r.get("retention"))
        if ret is None:
            continue
        out[(r["condition"], r["purify_kind"],
             _f(r.get("purify_strength")) or 0.0)].append(ret)
    return out


def summarize(data) -> Dict[Tuple[str, str, float], float]:
    return {k: statistics.fmean(v) for k, v in data.items() if v}


def compare(means, lhs: str, rhs: str) -> Dict:
    """lhs 對 rhs 的逐算子勝負。回傳寬鬆與嚴格兩種計數。"""
    by_op: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    for (cond, op, s), m in means.items():
        if cond != lhs:
            continue
        other = means.get((rhs, op, s))
        if other is None:
            continue
        by_op[op].append((s, m > other))

    detail = {}
    for op, pts in sorted(by_op.items()):
        wins = sum(1 for _, w in pts if w)
        detail[op] = {"n_strengths": len(pts), "wins": wins,
                      "any": wins > 0, "majority": wins * 2 > len(pts)}
    return {
        "ops_any": sum(1 for d in detail.values() if d["any"]),
        "ops_majority": sum(1 for d in detail.values() if d["majority"]),
        "n_ops": len(detail),
        "detail": detail,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batches", nargs="+", type=Path)
    ap.add_argument("--tau", type=float, default=0.20,
                    help="主表所在的失真預算（grid.MAIN_TAU）")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows: List[Dict[str, str]] = []
    for b in args.batches:
        rows += load_rows(b)
    print(f"讀入 {len(rows)} 列，來自 {len(args.batches)} 個批次\n")

    data = collect(rows, args.tau)
    if not data:
        # 靜默印一張空表等於讓「還沒有資料」看起來像「沒有優勢」
        raise SystemExit(
            f"τ={args.tau} 上沒有任何 retention_usable 的列。可能是段 3 還沒"
            "跑完，或 identity 那一格的效果都在雜訊裡（見 _fill_retention）")
    means = summarize(data)

    conds = sorted({k[0] for k in means})
    ops = sorted({(k[1], k[2]) for k in means})
    print(f"τ = {args.tau}   條件 {conds}")
    print(f"{'淨化算子':<26}" + "".join(f"{c:>14}" for c in conds))
    for op, s in ops:
        label = f"{op}" + (f" {s:g}" if s else "")
        line = f"{label:<26}"
        for c in conds:
            m = means.get((c, op, s))
            line += f"{m:>14.4f}" if m is not None else f"{'—':>14}"
        print(line)

    print("\n逐對判定（左勝右的獨立算子數；判準：≥"
          f"{MIN_OPS} 個算子且多數強度佔優）")
    verdicts = {}
    for na in NONADDITIVE:
        for ad in ADDITIVE + (RANDOM,):
            if na not in conds or ad not in conds:
                continue
            r = compare(means, na, ad)
            ok = r["ops_majority"] >= MIN_OPS
            verdicts[f"{na}>{ad}"] = {**r, "passes": ok}
            print(f"  {na:>3} > {ad:<16} "
                  f"嚴格 {r['ops_majority']}/{r['n_ops']}   "
                  f"寬鬆 {r['ops_any']}/{r['n_ops']}   "
                  f"{'成立' if ok else '不成立'}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"tau": args.tau, "verdicts": verdicts,
                        "means": {f"{c}|{o}|{s}": m
                                  for (c, o, s), m in means.items()}},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n寫出 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
