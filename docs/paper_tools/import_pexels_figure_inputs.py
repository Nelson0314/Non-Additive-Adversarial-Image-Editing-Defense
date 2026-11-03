"""把 baseline 的 Pexels（306 張）讀數轉成 generate_paper_figures.py 的輸入格式。

輸出（--out-dir）：
  points.csv               method, source, defense_lpips, edit_lpips：原生預算（interim/fig3_points.csv 的 native 列，各 306 張）
                           與等失真臂（summary/main_table.csv）。
  condition_summary.csv    condition, defense_lpips, edit_lpips_none, edit_lpips_purified_mean（七道平均）：9 個等失真臂。
  color_summary.csv        condition = color3d, fid_lpips（防禦圖 LPIPS）, lpips_none, lpips_seven。
  perturbation_survival.csv  condition, image, purifier, survival：9 臂由 followup 各影像組合併，color3d 取 color3d_all。

遠端 CPU 執行（basic-1）：
  ~/venvs/wacv/bin/python in/import_pexels_figure_inputs.py --out-dir in_pexels
"""
import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "image-immunization/baseline/runtime/pexels_portraits"
N_IMAGES = 306
ARMS = ("danp", "dayn", "dct_shield", "dct_shield_y", "dia_pt", "dia_r", "mist", "photoguard_encoder", "sifm")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = {r["condition"]: r for r in read(ROOT / "summary/main_table.csv")}
    for c in ARMS + ("color3d_pexels",):
        if int(table[c]["n_images"]) != N_IMAGES:
            raise ValueError(f"{c} 的影像數不是 {N_IMAGES}")

    native = [r for r in read(ROOT / "interim/fig3_points.csv") if r["arm"] == "native" and r["condition"] in ARMS]
    if {r["condition"] for r in native} != set(ARMS) or any(int(r["n_images"]) != N_IMAGES for r in native):
        raise ValueError("原生預算點不齊或影像數不足")
    points = [{"method": r["condition"], "source": "native", "defense_lpips": r["defense_lpips"], "edit_lpips": r["disp_none"]}
              for r in native]
    points += [{"method": c, "source": "aligned", "defense_lpips": table[c]["defense_lpips"], "edit_lpips": table[c]["disp_none"]}
               for c in ARMS]
    write(args.out_dir / "points.csv", ["method", "source", "defense_lpips", "edit_lpips"], points)

    write(args.out_dir / "condition_summary.csv", ["condition", "defense_lpips", "edit_lpips_none", "edit_lpips_purified_mean"],
          [{"condition": c, "defense_lpips": table[c]["defense_lpips"], "edit_lpips_none": table[c]["disp_none"],
            "edit_lpips_purified_mean": table[c]["disp_seven"]} for c in ARMS])

    ours = table["color3d_pexels"]
    write(args.out_dir / "color_summary.csv", ["condition", "fid_lpips", "lpips_none", "lpips_seven"],
          [{"condition": "color3d", "fid_lpips": ours["defense_lpips"], "lpips_none": ours["disp_none"],
            "lpips_seven": ours["disp_seven"]}])

    survival = {}
    for f in sorted(glob.glob(str(ROOT / "followup/sets/*/results/aligned/survival_seven.csv"))) + \
            [str(ROOT / "followup/results/aligned/survival_seven.csv")]:
        for r in read(f):
            survival[(r["condition"], r["image"], r["purifier"])] = r["survival"]
    for r in read(ROOT / "followup/results/color3d_all/survival_seven.csv"):
        survival[("color3d", r["image"], r["purifier"])] = r["survival"]
    count = Counter(k[0] for k in survival)
    for c in ARMS + ("color3d",):
        if count[c] != N_IMAGES * 7:
            raise ValueError(f"{c} 的存續比列數 {count[c]}，應為 {N_IMAGES * 7}")
    write(args.out_dir / "perturbation_survival.csv", ["condition", "image", "purifier", "survival"],
          [{"condition": k[0], "image": k[1], "purifier": k[2], "survival": v} for k, v in sorted(survival.items())])


if __name__ == "__main__":
    main()
