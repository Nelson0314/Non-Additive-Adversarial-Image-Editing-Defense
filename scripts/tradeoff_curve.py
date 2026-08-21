"""失真—防禦效果的取捨曲線，以及曲線上的兩個錨點（DEC-029）。

**這支不跑 GPU**，只讀既有批次的 `results.csv`。掃描本身由既有的兩支驅動
負責，各自都已經有強度旗標，不必另寫驅動：

    DCT-Shield    scripts/dct_shield_run.py --mode paper --eps <ε>
    紋理重相位     scripts/phase_ablation.py --human-threshold --phase-radius <θ>

為什麼要曲線而不是單點：兩個方法的強度參數不同單位（`ε` 是量化階、`θ` 是
相位半徑），論文的 baseline 又走 `L∞ = 16/255` 而本專案走原生預算，**沒有
任何一個軸天然對齊**。文獻上四種處理方式的查證見
`docs/reference/BASELINE_ALIGNMENT.md` §2；本專案採其中的掃描曲線
（IMPASTO, arXiv:2403.19254 §IV-A 的 Fig. 8 作法），並在曲線上標出兩個錨點。

兩個錨點
────────────────────────────────────────────────────────────────────
    等失真   在對方曲線上取失真相同之處，比防禦效果（FND-062 的作法）
    等效果   在對方曲線上取防禦效果相同之處，比失真（IMPASTO 的作法）

兩者都以**線性內插**求得，不再另跑二分搜尋——二分搜尋要重跑 GPU，內插不用，
而且內插會把所依據的兩個端點一起報出來，讀者看得到它有多可靠。

**落在掃描範圍之外的一律拒絕外插**，回傳 `out_of_range` 而不是硬算。FND-062
的單點對齊之所以難解讀，正是因為只有一個點，看不出附近的曲線長什麼樣。

用法：
    python scripts/tradeoff_curve.py --run runs/sweep0820 \
        --x fid_dists --ref phase --out runs/sweep0820/curve
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import write_csv  # noqa: E402

# 曲線的預設兩軸。橫軸是失真、縱軸是防禦效果。
DEFAULT_X = "fid_dists"
DEFAULT_Y = "edit_lpips"


def curve_points(rows: Sequence[dict], condition: str, x_key: str,
                 y_key: str) -> List[Tuple[float, float, float, int]]:
    """把某條件的全部列摺成曲線點 `(x, y, radius, n)`，依 x 遞增排序。

    同一個 `radius` 的多張影像取平均——曲線比的是條件不是影像。`radius` 是
    強度參數（DCT-Shield 的 `ε`、紋理重相位的 `θ`），兩支驅動都寫這個欄名。
    """
    by_r: Dict[float, List[Tuple[float, float]]] = {}
    for r in rows:
        if r["condition"] != condition:
            continue
        by_r.setdefault(round(float(r["radius"]), 6), []).append(
            (float(r[x_key]), float(r[y_key])))
    pts = [(statistics.fmean(x for x, _ in v), statistics.fmean(y for _, y in v),
            rad, len(v)) for rad, v in by_r.items()]
    return sorted(pts, key=lambda p: p[0])


def interp_at(points: Sequence[Tuple[float, float, float, int]], target: float,
              axis: int = 0) -> Optional[dict]:
    """在曲線上內插出 `axis` 等於 `target` 的那一點。

    `axis=0` 依 x 內插（等失真錨點），`axis=1` 依 y 內插（等效果錨點）。
    回傳含所依據的兩個端點；`target` 落在範圍外時回傳 `None`——**不外插**。

    依 y 內插要求曲線在 y 上單調。取捨曲線在構造上應該是單調遞增的（失真
    越大、防禦越強），但實測未必；不單調時取**第一個**跨過 target 的區間，
    並在回傳裡標 `crossings` 讓呼叫端看得到有幾個。
    """
    if len(points) < 2:
        return None
    vals = [p[axis] for p in points]
    other = 1 - axis
    lo, hi = min(vals), max(vals)
    if not (lo <= target <= hi):
        return None
    crossings = sum(1 for a, b in zip(vals, vals[1:])
                    if (a - target) * (b - target) <= 0 and a != b)
    for pa, pb in zip(points, points[1:]):
        a, b = pa[axis], pb[axis]
        if a == b:
            if a == target:
                return {"value": pa[other], "lo": pa, "hi": pb, "t": 0.0,
                        "crossings": crossings}
            continue
        if (a - target) * (b - target) <= 0:
            t = (target - a) / (b - a)
            return {"value": pa[other] + t * (pb[other] - pa[other]),
                    "lo": pa, "hi": pb, "t": t, "crossings": crossings}
    return None


def anchors(ref_pts, other_pts, ref_radius: Optional[float] = None) -> List[dict]:
    """兩個錨點。`ref_radius` 給定時以參照曲線上該強度的點為準，否則取中位點。"""
    if not ref_pts or not other_pts:
        return []
    if ref_radius is None:
        anchor = ref_pts[len(ref_pts) // 2]
    else:
        match = [p for p in ref_pts if abs(p[2] - ref_radius) < 1e-9]
        if not match:
            raise ValueError(
                f"參照曲線上沒有 radius={ref_radius} 的點；有的是 "
                f"{sorted(p[2] for p in ref_pts)}")
        anchor = match[0]
    out = []
    for name, axis, ref_val in (("等失真", 0, anchor[0]), ("等效果", 1, anchor[1])):
        hit = interp_at(other_pts, ref_val, axis=axis)
        out.append({
            "anchor": name, "ref_radius": anchor[2],
            "ref_x": round(anchor[0], 5), "ref_y": round(anchor[1], 5),
            "other_value": "" if hit is None else round(hit["value"], 5),
            "out_of_range": hit is None,
            "seg_lo_radius": "" if hit is None else hit["lo"][2],
            "seg_hi_radius": "" if hit is None else hit["hi"][2],
            "seg_lo": "" if hit is None else round(hit["lo"][axis], 5),
            "seg_hi": "" if hit is None else round(hit["hi"][axis], 5),
            "crossings": "" if hit is None else hit["crossings"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True,
                    help="一個或多個含 results.csv 的批次目錄")
    ap.add_argument("--x", default=DEFAULT_X, help="橫軸（失真）欄名")
    ap.add_argument("--y", default=DEFAULT_Y, help="縱軸（防禦效果）欄名")
    ap.add_argument("--ref", default="phase", help="參照條件，錨點由它決定")
    ap.add_argument("--ref-radius", type=float, default=None,
                    help="參照條件取哪一個強度；預設取中位點")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="只畫這些條件；預設是資料裡的全部")
    ap.add_argument("--out", type=Path, required=True, help="輸出前綴")
    args = ap.parse_args()

    rows: List[dict] = []
    for d in args.run:
        with (d / "results.csv").open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise SystemExit("沒有讀到任何列")
    missing = [k for k in (args.x, args.y, "radius", "condition")
               if k not in rows[0]]
    if missing:
        raise SystemExit(f"results.csv 缺欄位 {missing}；有的是 {sorted(rows[0])}")

    conds = args.conditions or sorted({r["condition"] for r in rows})
    if args.ref not in conds:
        raise SystemExit(f"參照條件 {args.ref} 不在 {conds} 裡")

    curves = {c: curve_points(rows, c, args.x, args.y) for c in conds}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pt_rows = [{"condition": c, "radius": rad, "n_images": n,
                "x_key": args.x, "x": round(x, 5),
                "y_key": args.y, "y": round(y, 5)}
               for c, pts in curves.items() for x, y, rad, n in pts]
    write_csv(Path(f"{args.out}_points.csv"), pt_rows)

    anc_rows = []
    for c in conds:
        if c == args.ref:
            continue
        for a in anchors(curves[args.ref], curves[c], args.ref_radius):
            anc_rows.append({"ref": args.ref, "other": c, **a})
    write_csv(Path(f"{args.out}_anchors.csv"), anc_rows)

    for c, pts in curves.items():
        print(f"\n=== {c}（{len(pts)} 點）===")
        for x, y, rad, n in pts:
            print(f"  r={rad:<8.4g} {args.x}={x:.4f} {args.y}={y:.4f} (n={n})")
    for a in anc_rows:
        state = "**外插，不可解讀**" if a["out_of_range"] else (
            f"由 r={a['seg_lo_radius']}–{a['seg_hi_radius']} 內插")
        print(f"\n{a['anchor']}：{a['ref']} r={a['ref_radius']} "
              f"→ {a['other']} = {a['other_value']}　{state}")
    print(f"\n表：{args.out}_points.csv、{args.out}_anchors.csv")


if __name__ == "__main__":
    main()
