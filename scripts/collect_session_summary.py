"""把幾個批次的防禦端與抗淨化讀數收成一份 JSON，給報告頁用。

**不跑 GPU。** 只讀已存的 `results.csv` 與 `purify/*_all.csv`。

抗淨化那一側**照實輸出總增益與空白地板兩個絕對值**，不算比例——比例讀數
（保留率、佔可達範圍）在分母塌陷或參照改變時不可解讀，讀者要看得到差額
從哪來。位移是 LPIPS，在兩張不相干的自然影像之間飽和於 0.772
（`runs/readout_ceiling/`），該值只作為飽和參考，不進任何算式。

**參照的協定要照實標。** 幾何類算子（`crop_resize` 等）的參照現行是「同一個
算子淨化過的原圖」，其空白地板由構造為 0；`runs/` 底下既有的抗淨化 CSV 是在
舊參照（未淨化的原圖）下量的，該欄的地板非 0。本支輸出 `protocol` 欄照實記，
不代為換算。

用法：python scripts/collect_session_summary.py --out session_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

LPIPS_CEILING = 0.772

# 批次 → （防禦端目錄, 抗淨化目錄）。抗淨化為 None 表示只有防禦端。
BATCHES = {
    "ip2p_ig_loss": "runs/ip2p_ig_loss",
    "ip2p_ig_lowdist": "runs/ip2p_ig_lowdist",
    "ip2p_blur_band": "runs/ip2p_blur_band",
    "ip2p_split_band": "runs/ip2p_split_band",
    "ip2p_sbsurv_long": "runs/ip2p_sbsurv_long",
    "ip2p_stadv": "runs/ip2p_stadv",
    "ip2p_stadv_radius": "runs/ip2p_stadv_radius",
}
FIELDS = ["fid_dists", "fid_lpips", "fid_psnr", "fid_ssim", "fid_vif_p",
          "fid_rms", "fid_linf", "edit_lpips"]
KEEP = ["radius", "loss", "spectral_floor", "floor_survival", "floor_r_max",
        "survival_weight", "r_max", "purify_aware", "flow_tau", "warp_grid",
        "update", "stop_reason", "stopped_at", "resumed", "steps"]


def num(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def defence(root: Path) -> dict:
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "results.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        if not rows:
            continue
        rec = {"n": len(rows)}
        for k in FIELDS:
            vals = [num(r, k) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                rec[k] = round(st.mean(vals), 4)
        # 設定欄逐列應相同，取第一列；不同則照實列出全部相異值。
        for k in KEEP:
            vs = sorted({r.get(k, "") for r in rows if r.get(k, "") != ""})
            if vs:
                rec[k] = vs[0] if len(vs) == 1 else "|".join(vs)
        out[d.name] = rec
    return out


def purify(root: Path) -> dict:
    """`{tag: {purifier: {總增益, 空白地板, 淨增益, protocol}}}`。"""
    gal = root / "purify"
    if not gal.is_dir():
        return {}
    eff = defaultdict(dict)
    ref = defaultdict(dict)
    for f in sorted(gal.glob("*_all.csv")):
        tag = f.name[: -len("_all.csv")]
        for r in csv.DictReader(f.open(encoding="utf-8")):
            v = num(r, "effect_mean")
            if v is None:
                continue
            eff[tag][(r["image"], r["purifier"])] = v
            ref[tag][r["purifier"]] = r.get("reference", "orig")
    floor = eff.pop("floor", {})
    if not floor:
        return {}
    out = {}
    for tag, cells in eff.items():
        per = {}
        purs = sorted({p for _, p in cells})
        for p in purs:
            tot = [v for (i, q), v in cells.items() if q == p]
            net = [v - floor[(i, p)] for (i, q), v in cells.items()
                   if q == p and (i, p) in floor]
            fl = [v for (i, q), v in floor.items() if q == p]
            per[p] = {
                "total": round(st.mean(tot), 4) if tot else None,
                "floor": round(st.mean(fl), 4) if fl else None,
                "net": round(st.mean(net), 4) if net else None,
                "protocol": ref[tag].get(p, "orig"),
            }
        out[tag] = per
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("session_summary.json"))
    args = ap.parse_args()

    data = {"lpips_ceiling": LPIPS_CEILING, "batches": {}}
    for name, path in BATCHES.items():
        root = Path(path)
        if not root.is_dir():
            print(f"[缺] {root}")
            continue
        data["batches"][name] = {
            "defence": defence(root),
            "purify": purify(root),
        }
        n_d = len(data["batches"][name]["defence"])
        n_p = len(data["batches"][name]["purify"])
        print(f"{name:<20s} 防禦 {n_d} 條件  抗淨化 {n_p} 條件")

    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1000:.0f} KB）")


if __name__ == "__main__":
    main()
