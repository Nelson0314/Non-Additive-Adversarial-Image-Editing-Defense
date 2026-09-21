"""把一批的 CSV 收成報告頁要用的 `data.js`。

頁面（`index.html`）只讀 `window.REPORT`，所有敘事字串也在這裡，
這樣改字不必動版面。**不下判準**：這支只把數字排進表，不標成立與否。

用法
    python scripts/build_report_data.py \\
        --solve-root runs/amplitude_ladder \\
        --edits runs/amplitude_ladder_edits/displacement.csv \\
        --paired runs/amplitude_ladder_edits/paired.csv \\
        --retention runs/purified_colour_edits/retention.csv \\
        --out runs/report/amplitude_ladder/data.js
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path

RUNGS = [
    ("cap16", "ΔE00 ≤ 16", "現行操作點"),
    ("cap22", "ΔE00 ≤ 22", ""),
    ("cap28", "ΔE00 ≤ 28", ""),
    ("cap36", "ΔE00 ≤ 36", ""),
]

READOUT_FIELDS = [
    {"key": "deltae00", "label": "ΔE00"},
    {"key": "psnr", "label": "PSNR"},
    {"key": "fidelity", "label": "保真 LPIPS"},
    {"key": "disp", "label": "位移中位"},
    {"key": "stopped", "label": "停在第幾步"},
]


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def med(values):
    return round(float(st.median(values)), 4) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solve-root", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--retention", type=Path, default=None)
    parser.add_argument("--band-retention", type=Path, default=None,
                        help="JPEG30／blur2 的 retention.csv，用來算淨化後的配對差")
    parser.add_argument("--second-paired", type=Path, default=None,
                        help="另一批（例如禁止色偏那一族）的 paired.csv")
    parser.add_argument("--narrative", type=Path, default=None,
                        help="敘事字串的 JSON；與產生的數字合併成 window.REPORT")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    solves = []
    for path in sorted(glob.glob(str(args.solve_root / "*" / "paper_baseline_shard*.csv"))):
        solves += read(Path(path))
    disp = read(args.edits)
    paired = {r["condition"]: r for r in read(args.paired)}
    retention = read(args.retention) if args.retention else []

    images = sorted({r["image"] for r in solves})
    rungs = [r for r in RUNGS if any(s["variant"] == r[0] for s in solves)]

    per_image = {}
    for rung, _, _ in rungs:
        table = {}
        for name in images:
            row = next((s for s in solves
                        if s["variant"] == rung and s["image"] == name), None)
            if row is None:
                continue
            cells = [d for d in disp
                     if d["condition"] == rung and d["image"] == name
                     and d["scenario"] == "ip2p"]
            table[name] = {
                "deltae00": f'{float(row["deltaE00"]):.2f}',
                "psnr": f'{float(row["psnr"]):.2f}',
                "fidelity": f'{float(row["lpips"]):.4f}',
                "disp": (f'{med([float(c["disp_lpips"]) for c in cells]):.4f}'
                         if cells else None),
                "stopped": row.get("free_stopped_step", ""),
            }
        per_image[rung] = table

    def rung_stats(rung):
        rows = [s for s in solves if s["variant"] == rung]
        cells = [d for d in disp if d["condition"] == rung]
        ip2p = [d for d in cells if d["scenario"] == "ip2p"]
        inpaint = [d for d in cells if d["scenario"] == "inpaint"]
        pair = paired.get(rung, {})
        return {
            "deltae00": med([float(r["deltaE00"]) for r in rows]),
            "deltae00_range": (f'{min(float(r["deltaE00"]) for r in rows):.2f}'
                               f'–{max(float(r["deltaE00"]) for r in rows):.2f}')
                              if rows else "",
            "psnr": med([float(r["psnr"]) for r in rows]),
            "fidelity": med([float(r["lpips"]) for r in rows]),
            "disp_ip2p": med([float(d["disp_lpips"]) for d in ip2p]),
            "disp_inpaint": med([float(d["disp_lpips"]) for d in inpaint]),
            "paired_median": pair.get("disp_lpips_median", ""),
            "paired_improved": pair.get("disp_lpips_improved", ""),
            "paired_cells": pair.get("cells", ""),
            "cells": len(cells),
        }

    stats = {rung: rung_stats(rung) for rung, _, _ in rungs}

    ladder_table = {
        "caption": "振幅階梯：八張人像、ip2p 與 inpaint 各 32 格",
        "head": ["級別", "ΔE00 中位", "ΔE00 全距", "PSNR", "保真 LPIPS",
                 "位移 ip2p", "位移 inpaint", "配對差中位", "改善格數"],
        "rows": [],
    }
    for rung, label, badge in rungs:
        s = stats[rung]
        ladder_table["rows"].append({
            "control": rung == "cap16",
            "cells": [
                label,
                f'{s["deltae00"]:.2f}' if s["deltae00"] else "—",
                s["deltae00_range"],
                f'{s["psnr"]:.2f}' if s["psnr"] else "—",
                f'{s["fidelity"]:.4f}' if s["fidelity"] else "—",
                f'{s["disp_ip2p"]:.4f}' if s["disp_ip2p"] else "—",
                f'{s["disp_inpaint"]:.4f}' if s["disp_inpaint"] else "—",
                (f'{float(s["paired_median"]):+.4f}' if s["paired_median"] else "—"),
                (f'{s["paired_improved"]} / {s["paired_cells"]}'
                 if s["paired_improved"] != "" else "—"),
            ],
        })

    retention_table = None
    if retention:
        groups = {}
        for row in retention:
            groups.setdefault(row["purifier"], []).append(row)
        retention_table = {
            "caption": "現行操作點在四道色彩淨化下的留存（ip2p 32 格，兩側都淨化）",
            "head": ["算子", "未淨化位移", "淨化後位移", "留存率", "淨增益"],
            "rows": [],
        }
        order = ["gray_world", "clahe2", "auto_levels", "grayscale"]
        for key in order:
            rows = groups.get(key)
            if not rows:
                continue
            ret = med([float(r["retained"]) for r in rows])
            retention_table["rows"].append({
                "cells": [
                    key,
                    f'{med([float(r["disp_plain"]) for r in rows]):.4f}',
                    f'{med([float(r["disp_purified"]) for r in rows]):.4f}',
                    f'{ret:.3f}',
                    f'{med([float(r["net_gain"]) for r in rows]):+.4f}',
                ],
                "tones": [None, None, None,
                          "fail" if ret < 0.7 else ("pass" if ret >= 0.94 else "edge"),
                          None],
            })

    control, top = rungs[0][0], rungs[-1][0]
    figures = [
        {"key": "現行操作點的位移", "value": f'{stats[control]["disp_ip2p"]:.4f}',
         "note": "ip2p 32 格中位。ΔE00 上限 16，實測中位 "
                 f'{stats[control]["deltae00"]:.2f}。'},
        {"key": "最高一檔的位移", "value": f'{stats[top]["disp_ip2p"]:.4f}',
         "tone": "signal",
         "note": f'ΔE00 中位 {stats[top]["deltae00"]:.2f}。'
                 f'對照臂逐格相減後中位 {float(stats[top]["paired_median"]):+.4f}、'
                 f'改善 {stats[top]["paired_improved"]}/{stats[top]["paired_cells"]} 格。'},
        {"key": "同一檔的保真 LPIPS", "value": f'{stats[top]["fidelity"]:.4f}',
         "tone": "fail",
         "note": f'現行操作點是 {stats[control]["fidelity"]:.4f}。'
                 "防禦圖離原圖有多遠，數字大不是好事。"},
        {"key": "灰階下的留存", "value": "0.500",
         "tone": "signal",
         "note": "另外三道色彩淨化 0.948–1.172。AdvCF 原文報的灰階存活率約 18%。"},
    ]

    band_table = None
    if args.band_retention:
        band = read(args.band_retention)
        cells = {}
        for row in band:
            cells.setdefault(row["purifier"], {}).setdefault(row["condition"], {})[
                (row["image"], row["prompt_index"])] = row
        band_table = {
            "caption": "淨化之後的驗收：ip2p 32 格，兩側都淨化，對照臂仍是同批的 cap16",
            "head": ["淨化", "級別", "淨化後位移", "留存率", "配對差中位", "改善格數"],
            "rows": [],
        }
        for op in ("jpeg30", "blur2"):
            control = cells.get(op, {}).get("cap16")
            if not control:
                continue
            for rung, label, _ in rungs:
                table = cells[op].get(rung)
                if not table:
                    continue
                keys = sorted(set(table) & set(control))
                diffs = [float(table[k]["disp_purified"])
                         - float(control[k]["disp_purified"]) for k in keys]
                ret = med([float(table[k]["retained"]) for k in keys])
                band_table["rows"].append({
                    "control": rung == "cap16",
                    "cells": [
                        op, label,
                        f'{med([float(table[k]["disp_purified"]) for k in keys]):.4f}',
                        f"{ret:.3f}",
                        "—" if rung == "cap16" else f"{med(diffs):+.4f}",
                        "—" if rung == "cap16"
                        else f'{sum(1 for d in diffs if d > 0)} / {len(keys)}',
                    ],
                    "tones": [None, None, None,
                              "pass" if ret >= 0.9 else "edge", None, None],
                })

    second_table = None
    if args.second_paired:
        second = read(args.second_paired)
        second_table = {
            "caption": "禁止色偏之後還剩多少：ip2p 16 格、四張影像，對照臂是同批的 advcf_cap16",
            "head": ["條件", "位移中位", "配對差中位", "改善格數"],
            "rows": [],
        }
        for row in second:
            control = row["condition"] == "advcf_cap16"
            second_table["rows"].append({
                "control": control,
                "cells": [
                    row["condition"],
                    f'{float(row["disp_lpips_self_median"]):.4f}',
                    "—" if control else f'{float(row["disp_lpips_median"]):+.4f}',
                    "—" if control
                    else f'{row["disp_lpips_improved"]} / {row["cells"]}',
                ],
                "tones": [None, None,
                          None if control
                          else ("fail" if float(row["disp_lpips_median"]) < 0 else None),
                          None],
            })

    payload = {
        "figures": figures,
        "images": images,
        "conditions": [{"id": r, "label": label, "badge": badge}
                       for r, label, badge in rungs],
        "readout_fields": READOUT_FIELDS,
        "ladder_table": ladder_table,
        "retention_table": retention_table,
        "per_image": per_image,
        "stats": stats,
    }
    if args.narrative:
        # 敘事在前、數字在後：數字欄不可以被敘事的舊值蓋掉。
        payload = {**json.loads(args.narrative.read_text(encoding="utf-8")),
                   **payload}
        # 機制區的表用 `table_ref` 指名，這裡換成產生出來的那一份，
        # 敘事檔裡因此不會出現任何手抄的數字。
        built = {"ladder": ladder_table, "retention": retention_table,
                 "band": band_table, "no_cast": second_table}
        for block in payload.get("mechanism", []):
            ref = block.pop("table_ref", None)
            if ref and built.get(ref):
                block["table"] = built[ref]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "window.REPORT = " + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8")
    print(f"{args.out} 寫好：{len(images)} 張影像、{len(rungs)} 個級別、"
          f"{len(disp)} 列位移")
    for rung, label, _ in rungs:
        s = stats[rung]
        print(f"  {label:12s} ΔE00 {s['deltae00']}  位移 ip2p {s['disp_ip2p']}  "
              f"inpaint {s['disp_inpaint']}  配對差 {s['paired_median']} "
              f"({s['paired_improved']}/{s['paired_cells']})")


if __name__ == "__main__":
    main()
