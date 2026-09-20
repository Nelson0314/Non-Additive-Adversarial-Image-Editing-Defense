"""任一批編輯讀數對該批對照臂的逐格配對差。

為什麼要逐格相減
────────────────────────────────────────────────────────────────────
位移隨「哪一張圖、哪一句指令」大幅變動。兩個臂各取中位再相減，等於把那個
變異當成雜訊吞掉。以 `(scenario, image, prompt_index)` 為鍵逐格相減之後再取
中位，比的才是同一格上的差；改善格數數的也是逐格的正負。

對照臂必須是**同一批**解出來的。求解端跨卡不可重現（同設定同種子的全圖
ΔE00 量到 15.93／15.70／12.51），跨批拿別批的對照來減不成立。

**本腳本不判成立與否**，只把配對差中位、改善格數、格數總計擺出來。

用法
    python scripts/paired_difference.py \
        --displacement runs/amplitude_ladder_edits/displacement.csv \
        --control cap16 --out runs/amplitude_ladder_edits/paired.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: 逐格相減的鍵。`arm` 不進鍵：同一個 scenario 的臂名在各條件之間相同。
KEY = ("scenario", "image", "prompt_index")

#: 預設要配對的欄位。有哪一欄就算哪一欄，缺的欄不報錯——不同批的欄位不一樣。
COLUMNS = ("disp_lpips", "disp_lpips_matched_mean", "disp_dists",
           "disp_lpips_subject", "disp_lpips_background")


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--displacement", type=Path, required=True)
    parser.add_argument("--control", required=True,
                        help="同一批裡當對照的條件名")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", nargs="+", default=None)
    args = parser.parse_args()

    rows = read(args.displacement)
    if not rows:
        raise SystemExit(f"{args.displacement} 沒有資料列")
    columns = args.columns or [c for c in COLUMNS if c in rows[0]]
    if not columns:
        raise SystemExit(f"CSV 裡沒有任何可配對的欄位（找過 {COLUMNS}）")

    conditions = sorted({r["condition"] for r in rows})
    if args.control not in conditions:
        raise SystemExit(f"這一批裡沒有條件 {args.control!r}；有的是 {conditions}")

    table = {}
    for row in rows:
        table.setdefault(row["condition"], {})[tuple(row[k] for k in KEY)] = row
    control = table[args.control]

    out = []
    for condition in conditions:
        cells = table[condition]
        shared = sorted(set(cells) & set(control))
        missing = sorted(set(control) - set(cells))
        record = {"condition": condition, "control": args.control,
                  "cells": len(shared), "cells_in_control": len(control),
                  "cells_missing": len(missing)}
        for column in columns:
            diffs = [float(cells[k][column]) - float(control[k][column])
                     for k in shared]
            record[f"{column}_median"] = round(
                float(statistics.median(diffs)), 6) if diffs else ""
            record[f"{column}_improved"] = sum(1 for d in diffs if d > 0)
            record[f"{column}_self_median"] = round(
                float(statistics.median(
                    [float(cells[k][column]) for k in shared])), 6) if diffs else ""
        out.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    width = max(len(c) for c in conditions)
    head = "  ".join(f"{c}" for c in columns)
    print(f"對照臂 {args.control}｜逐格相減後取中位（改善格數／共用格數）  {head}")
    for record in out:
        cells = "  ".join(
            f"{record[f'{c}_median']:>9}({record[f'{c}_improved']:>2}/{record['cells']})"
            for c in columns)
        print(f"  {record['condition']:<{width}}  {cells}")
    print(f"[ALLDONE] {args.out}")


if __name__ == "__main__":
    main()
