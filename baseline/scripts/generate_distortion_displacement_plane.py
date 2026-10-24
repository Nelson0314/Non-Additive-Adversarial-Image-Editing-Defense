"""失真—位移平面：每個（方法, 預算）點的防禦圖失真與 ip2p 編輯結果 LPIPS，及逐方法與合併的直線擬合。

資料（data/portraits 8 張、ip2p、四句、主種子；防禦圖失真讀交付的 PNG、CPU）：
    native         主表原生預算：results/additional_metrics/fidelity.csv、results/displacement.csv
    aligned        等失真臂：results/aligned/additional_metrics/fidelity.csv、results/aligned/displacement.csv
    aligned_1p255  等失真臂 mist、photoguard_linf 改為 2/255 前的解（PNG 上為 1 級）：
                   artifacts/aligned/superseded/ 的防禦圖、`git show ebb51e0^` 的位移表（呼叫端存成 --superseded-displacement）
    plane          倍率掃描：runtime/plane/<方法>/scale_<倍率>/{fidelity.csv,displacement.csv}
    color          color 專案：<color trial>/results/perceptual_quality/aligned_full_reference.csv（alignment=none）
                   與 <color trial>/results/<條件>/displacement.csv
PieAPP 以 pyiqa（CPU）對每張防禦圖與原圖計算；color 的 PieAPP 取自其表。

輸出：<輸出目錄>/points.csv（每點一列：方法、來源、倍率、八張平均的 LPIPS／DISTS／PieAPP 與編輯結果 LPIPS）、
fits.csv（橫軸 × 方法：點數、斜率、截距、r；另有全部點合併的一列 `all`）。

用法（遠端 baseline 專案根）：
    python scripts/generate_distortion_displacement_plane.py --superseded-displacement <CSV> \\
        --color-trial <color trial 目錄> --output-dir results/distortion_displacement
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics as st
from pathlib import Path

COLOR_CONDITIONS = ("color3d_orig", "color3d_adaptive_set", "color3d_orig_0d4")
SUPERSEDED = {"mist": ("mist_eps_scale_0p093", "0.093"),
              "photoguard_linf": ("photoguard_linf_eps_scale_0p0857", "0.0857")}


def read(path) -> list:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, col, **eq):
    v = [float(r[col]) for r in rows if all(r[k] == x for k, x in eq.items()) and r[col] != ""]
    if not v:
        raise SystemExit(f"沒有 {col} 的值：{eq}")
    return st.mean(v)


def fit(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    return {"n": n, "slope": round(slope, 4), "intercept": round(my - slope * mx, 4),
            "r": round(sxy / math.sqrt(sxx * syy), 4) if syy > 0 else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--superseded-displacement", type=Path, required=True)
    ap.add_argument("--color-trial", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    import pyiqa
    import torch

    from immunization_core.io import load_image_tensor

    pieapp = pyiqa.create_metric("pieapp", device="cpu")

    def pieapp_mean(defenses_dir: Path, cond: str) -> float:
        vals = []
        for d in sorted(defenses_dir.glob(f"*__{cond}__def.png")):
            o = d.with_name(f"{d.name.split('__')[0]}__orig.png")
            with torch.no_grad():
                vals.append(float(pieapp(load_image_tensor(d, "cpu", size=512),
                                         load_image_tensor(o, "cpu", size=512))))
        if len(vals) != 8:
            raise SystemExit(f"{defenses_dir} 的 {cond} 防禦圖不是 8 張")
        return st.mean(vals)

    points = []

    def add(method, source, scale, fid_rows, disp_rows, defenses_dir, **eq):
        points.append({
            "method": method, "source": source, "scale": scale,
            "defense_lpips": round(mean(fid_rows, "fid_lpips", condition=method, **eq), 4),
            "defense_dists": round(mean(fid_rows, "fid_dists", condition=method, **eq), 4),
            "defense_pieapp": round(pieapp_mean(defenses_dir, method), 4),
            "edit_lpips": round(mean(disp_rows, "disp_lpips_full", condition=method, scenario="ip2p"), 4),
        })
        print(points[-1], flush=True)

    for source, fid, disp, root in (
            ("native", "results/additional_metrics/fidelity.csv", "results/displacement.csv", Path("artifacts/defenses")),
            ("aligned", "results/aligned/additional_metrics/fidelity.csv", "results/aligned/displacement.csv",
             Path("artifacts/aligned/defenses"))):
        fr, dr = read(fid), read(disp)
        scales = {}
        if source == "aligned":
            scales = {c: read(f"results/aligned/defense_{c}.csv")[0]["eps_scale"]
                      for c in {r["condition"] for r in fr}}
        for c in sorted({r["condition"] for r in dr}):
            add(c, source, scales.get(c, "1"), fr, dr, root / c)
    sup_fid = read("runtime/semantic_identity/fidelity_png/aligned/fidelity.csv")
    sup_disp = read(args.superseded_displacement)
    for c, (d, scale) in SUPERSEDED.items():
        add(c, "aligned_1p255", scale, sup_fid, sup_disp, Path(f"artifacts/aligned/superseded/{d}/defenses/{c}"))
    for point_dir in sorted(glob.glob("runtime/plane/*/scale_*")):
        p = Path(point_dir)
        c = p.parent.name
        add(c, "plane", p.name.removeprefix("scale_").replace("p", "."),
            read(p / "fidelity.csv"), read(p / "displacement.csv"), p / "defenses" / c)
    ref = [r for r in read(args.color_trial / "results/perceptual_quality/aligned_full_reference.csv")
           if r["alignment"] == "none"]
    for c in COLOR_CONDITIONS:
        rows = [r for r in ref if r["condition"] == c]
        disp = read(args.color_trial / f"results/{c}/displacement.csv")
        points.append({"method": c, "source": "color", "scale": "",
                       "defense_lpips": round(st.mean(float(r["lpips"]) for r in rows), 4),
                       "defense_dists": round(st.mean(float(r["dists"]) for r in rows), 4),
                       "defense_pieapp": round(st.mean(float(r["pieapp"]) for r in rows), 4),
                       "edit_lpips": round(mean(disp, "disp_lpips_full", scenario="ip2p"), 4)})
        print(points[-1], flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(points)
    fits = []
    for axis in ("defense_lpips", "defense_dists", "defense_pieapp"):
        groups = {m: [p for p in points if p["method"] == m] for m in sorted({p["method"] for p in points})}
        groups["all"] = points
        for m, ps in groups.items():
            if len(ps) < 3:
                continue
            fits.append({"axis": axis, "method": m,
                         **fit([p[axis] for p in ps], [p["edit_lpips"] for p in ps])})
    with open(args.output_dir / "fits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fits[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(fits)
    print(f"[ALLDONE] {args.output_dir}（{len(points)} 點、{len(fits)} 條擬合）", flush=True)


if __name__ == "__main__":
    main()
