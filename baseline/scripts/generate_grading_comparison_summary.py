"""measure_grading_comparison 的彙總：color3d 在三組全域調色分布中的百分位，與各組的中位數與範圍。

百分位：逐張計算「該組在同一張影像上的值 ≤ color3d 值」的比例 ×100（並列各計一半），再對 8 張取平均。
中位數與範圍：該組全部（影像 × 成員）的值；color3d 為 8 張的值。

用法：python scripts/generate_grading_comparison_summary.py <grading_comparison CSV…> --output-csv <CSV>
"""
import argparse
import csv
import statistics
from pathlib import Path

GROUPS = ("rt", "random_raw", "random_matched")
SKIP = {"condition", "group", "image", "device", "alpha", "target_lpips"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()
    rows = [r for p in args.inputs for r in csv.DictReader(p.open(encoding="utf-8"))]
    metrics = [k for k in rows[0] if k not in SKIP]
    color = {r["image"]: r for r in rows if r["group"] == "color3d"}
    out = []
    for m in metrics:
        rec = {"metric": m, "color3d_median": round(statistics.median(float(r[m]) for r in color.values()), 5),
               "color3d_min": min(float(r[m]) for r in color.values()),
               "color3d_max": max(float(r[m]) for r in color.values())}
        for g in GROUPS:
            vals = [r for r in rows if r["group"] == g]
            per = []
            for im, c in color.items():
                v = [float(r[m]) for r in vals if r["image"] == im]
                cv = float(c[m])
                per.append(100 * (sum(x < cv for x in v) + 0.5 * sum(x == cv for x in v)) / len(v))
            allv = [float(r[m]) for r in vals]
            rec.update({f"{g}_percentile": round(statistics.mean(per), 1), f"{g}_median": round(statistics.median(allv), 5),
                        f"{g}_min": min(allv), f"{g}_max": max(allv), f"{g}_n": len(allv)})
        out.append(rec)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
