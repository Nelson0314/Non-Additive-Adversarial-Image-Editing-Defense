"""等失真的頭對頭表。**不跑 GPU**，只讀既有批次的 `results.csv`。

為什麼不能直接用 `tradeoff_curve.py`
────────────────────────────────────────────────────────────────────
那一支以 `condition` 欄分曲線。本專案現在有一批比較是**同一個 condition、
不同旗標**（`--spectral-floor`、`--floor-gate`、`--quantile`），那些列的
`condition` 全是 `phase_gain`，直接餵進去會被摺成同一條曲線而沒有症狀。
本支把「哪些欄一起決定一條曲線」開放成 `--group-by`。

協定不變（DEC-029）：曲線點是逐強度的平均，錨點以**線性內插**求得，
**落在掃描範圍外一律拒絕外插**並標 `out_of_range`。內插所依據的兩個端點
一併輸出。

逐圖對齊
────────────────────────────────────────────────────────────────────
`--images` 給定時只保留這些影像。不同分片的完成張數可能不同，把張數不同的
平均並排讀會把「跑到哪張」讀成方法差異，故預設**取所有曲線共有的影像交集**，
並把 `n_images` 逐列寫出。

用法：
    python scripts/matched_distortion_table.py \
        --run runs/ip2p_axis_necessity/*/ --group-by condition spectral_floor \
        --anchor 0.1377 --out runs/ip2p_axis_necessity/matched.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradeoff_curve import interp_at  # noqa: E402

from src.utils.io import write_csv  # noqa: E402

BLOCKED_THRESHOLD = 0.8445          # docs/EVALUATION.md，CLIP 影像相似度
DEFAULT_X = "fid_dists"
DEFAULT_Y = "edit_lpips"
# 錨點上一併回報的量。擋下數是離散的且有 ±2/13 的重跑雜訊，故它與位移
# 一樣走內插，但讀的時候要記得那個雜訊。
CARRIED = ("fid_psnr", "fid_lpips", "fid_ssim", "edit_clip_sim", "blocked_rate")


def load_rows(runs: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for run in runs:
        path = run / "results.csv" if run.is_dir() else run
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("condition"):
                    continue
                r["_src"] = str(path.parent)
                rows.append(r)
    return rows


def group_key(row: dict, keys: Sequence[str]) -> str:
    return "|".join(f"{k}={row.get(k, '')}" for k in keys)


def build_curves(rows: Sequence[dict], keys: Sequence[str], x_key: str,
                 y_key: str, images: Sequence[str]
                 ) -> Dict[str, List[Tuple[float, float, float, int, dict]]]:
    """每組一條曲線，點是 `(x, y, radius, n, 其餘量的平均)`，依 x 遞增。"""
    buckets: Dict[Tuple[str, float], List[dict]] = {}
    keep = set(images)
    for r in rows:
        if r["image"] not in keep:
            continue
        buckets.setdefault(
            (group_key(r, keys), round(float(r["radius"]), 6)), []).append(r)
    curves: Dict[str, List[Tuple[float, float, float, int, dict]]] = {}
    for (g, rad), rs in buckets.items():
        if len(rs) != len(keep):
            raise SystemExit(
                f"{g} radius={rad} 只有 {len(rs)} 張，交集是 {len(keep)} 張——"
                "分片沒跑完就不要合併，張數不同的平均不可並排讀")
        extra = {k: statistics.fmean(float(r[k]) for r in rs)
                 for k in CARRIED if k != "blocked_rate"}
        extra["blocked_rate"] = statistics.fmean(
            1.0 if float(r["edit_clip_sim"]) < BLOCKED_THRESHOLD else 0.0
            for r in rs)
        curves.setdefault(g, []).append((
            statistics.fmean(float(r[x_key]) for r in rs),
            statistics.fmean(float(r[y_key]) for r in rs),
            rad, len(rs), extra))
    for g in curves:
        curves[g].sort(key=lambda p: p[0])
    return curves


def common_images(rows: Sequence[dict], keys: Sequence[str]) -> List[str]:
    """所有 (組, radius) 都跑到的影像。空集合直接拋錯，不靜默出空表。"""
    per: Dict[Tuple[str, float], set] = {}
    for r in rows:
        per.setdefault((group_key(r, keys), round(float(r["radius"]), 6)),
                       set()).add(r["image"])
    inter = set.intersection(*per.values()) if per else set()
    if not inter:
        raise SystemExit("各組沒有共同的影像，無法逐圖對齊")
    return sorted(inter)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--group-by", nargs="+", default=["condition"],
                    help="哪些欄一起決定一條曲線")
    ap.add_argument("--x", default=DEFAULT_X)
    ap.add_argument("--y", default=DEFAULT_Y)
    ap.add_argument("--anchor", type=float, nargs="+", required=True,
                    help="等失真錨點（--x 的值），可給多個")
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.run)
    images = args.images or common_images(rows, args.group_by)
    curves = build_curves(rows, args.group_by, args.x, args.y, images)

    pts_out = []
    for g, pts in sorted(curves.items()):
        for x, y, rad, n, extra in pts:
            pts_out.append({
                "group": g, "radius": rad, "n_images": n,
                args.x: round(x, 5), args.y: round(y, 5),
                **{k: round(v, 5) for k, v in extra.items()}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out.with_name(args.out.stem + "_points.csv"), pts_out)

    table = []
    for anchor in args.anchor:
        for g, pts in sorted(curves.items()):
            hit = interp_at([(x, y, r, n) for x, y, r, n, _ in pts], anchor,
                            axis=0)
            row = {"anchor_x": anchor, "group": g, "n_images": len(images),
                   "out_of_range": hit is None,
                   "x_min": round(pts[0][0], 5), "x_max": round(pts[-1][0], 5)}
            if hit is None:
                row[args.y] = ""
                for k in CARRIED:
                    row[k] = ""
            else:
                row[args.y] = round(hit["value"], 5)
                t = hit["t"]
                lo = next(p for p in pts if p[2] == hit["lo"][2])
                hi = next(p for p in pts if p[2] == hit["hi"][2])
                for k in CARRIED:
                    row[k] = round(lo[4][k] + t * (hi[4][k] - lo[4][k]), 5)
                row["seg_lo_radius"] = hit["lo"][2]
                row["seg_hi_radius"] = hit["hi"][2]
            table.append(row)
    write_csv(args.out, table)

    print(f"共同影像 {len(images)} 張\n")
    for g, pts in sorted(curves.items()):
        print(f"— {g}")
        for x, y, rad, n, extra in pts:
            print(f"    r={rad:<7g} {args.x}={x:.4f} {args.y}={y:.4f} "
                  f"psnr={extra['fid_psnr']:.2f} "
                  f"blocked={extra['blocked_rate'] * n:.0f}/{n}")
    print(f"\n{'錨點':>8s}  {'組':46s}{args.y:>9s}{'PSNR':>8s}{'擋下':>8s}")
    for r in table:
        if r["out_of_range"]:
            print(f"{r['anchor_x']:8.4f}  {r['group']:46s}"
                  f"{'範圍外':>11s}（{r['x_min']:.4f}–{r['x_max']:.4f}）")
        else:
            print(f"{r['anchor_x']:8.4f}  {r['group']:46s}{r[args.y]:9.4f}"
                  f"{r['fid_psnr']:8.2f}"
                  f"{r['blocked_rate'] * r['n_images']:6.1f}/{r['n_images']}")
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
