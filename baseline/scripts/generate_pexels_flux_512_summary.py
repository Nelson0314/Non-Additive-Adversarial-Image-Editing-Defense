"""Pexels 人像 FLUX 512 協定（資料版本 flux_512，50 張）的彙整：等失真 photoguard_encoder、dct_shield、dct_shield_y、mist 與
color3d_pexels，每條件一列；分母為同一資料版本的 undefended。

來源 runtime/pexels_portraits/flux_512/readouts/（scripts/evaluate_pexels_flux_512_readouts.sh）：displacement.csv、retention.csv、
semantic_identity.csv、arcface_pair.csv、clip_score.csv、edit_fidelity.csv、distribution/ 的 FID／KID；
防禦圖 LPIPS 取 ip2p 主表的 fidelity.csv（同一批防禦圖，只取這 50 張）：等失真臂為 runtime/pexels_portraits/aligned/<根>/results/，
color3d_pexels 為 runtime/pexels_portraits/followup/<根>/results/color3d/。
欄與平均、區間的算法同 scripts/generate_pexels_main_table_summary.py（none＝淨化前、seven＝七道、vae_trans）：
    七道欄：7 個逐道平均的平均，_std 為其樣本標準差，_ci95＝t(0.975, 6)·s_p／√7
    其餘欄：逐格平均，_std 為逐張值（4 句平均）的影像間標準差，_ci95＝1.96·s／√n（n＝50；ArcFace 為有臉影像數）
    fid_seven、kid_seven：7 組各自的值，同七道規則；fid_none、fid_vae_trans 無區間
任一條件的影像數不是 50 或格數不齊時失敗。
輸出 runtime/pexels_portraits/flux_512/readouts/summary.csv 與 per_purifier.csv。
用法（遠端 baseline 專案根）：python scripts/generate_pexels_flux_512_summary.py
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_pexels_main_table_summary import (SEVEN, faces, image_stat, purifier_means,  # noqa: E402
                                                 purifier_stat, put, read, require)

COLOR3D = "color3d_pexels"
CONDITIONS = ("photoguard_encoder", "dct_shield", "dct_shield_y", "mist", COLOR3D)
R = Path("runtime/pexels_portraits/flux_512/readouts")
CT = Path("../color/trials/lut3d_defense_and_naturalness")
N = 50


def kind_of(purifier):
    return "seven" if purifier in SEVEN else purifier


def main():
    images = set((CT / "pexels_subsets/flux.txt").read_text(encoding="utf-8").split())
    require("flux.txt", "影像數", len(images), N)
    metric = defaultdict(dict)   # (指標, 條件, none|seven|vae_trans) -> {(影像, 句, 道): 值}
    for row in read(R / "displacement.csv"):
        key = (row["image"], row["prompt_index"], "none")
        metric[("disp", row["condition"], "none")][key] = float(row["disp_lpips_full"])
    for row in read(R / "retention.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        kind = kind_of(row["purifier"])
        metric[("disp", row["condition"], kind)][key] = float(row["disp_purified"])
        metric[("retained", row["condition"], kind)][key] = float(row["retained"])
    for row in read(R / "semantic_identity.csv"):
        c, kind = row["condition"], kind_of(row["purifier"])
        if c == "undefended" or kind not in ("seven", "vae_trans"):
            continue
        key = (row["image"], row["prompt_index"], row["purifier"])
        for m, name in (("siglip", "siglip"), ("clip", "clip_i")):
            if row[f"{m}_pair"] != "":
                metric[(name, c, kind)][key] = float(row[f"{m}_pair"])
        if kind == "seven":
            metric[("arcface_original", c, kind)][key] = float(row["arcface_id"]) if row["arcface_id"] != "" else None
    for row in read(R / "arcface_pair.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        metric[("arcface_pair", row["condition"], kind_of(row["purifier"]))][key] = (
            float(row["arcface_pair"]) if row["arcface_pair"] != "" else None)
    for row in read(R / "clip_score.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        metric[("clip_s", row["condition"], kind_of(row["purifier"]))][key] = float(row["clip_s"])
    for row in read(R / "edit_fidelity.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        for m in ("ssim", "psnr"):
            metric[(f"disp_{m}", row["condition"], kind_of(row["purifier"]))][key] = float(row[f"disp_{m}"])
    distribution = defaultdict(dict)
    for path in glob.glob(str(R / "distribution" / "*.csv")):
        for row in read(path):
            if "fid" in row and row.get("end") == "displacement":
                distribution[row["condition"]][row["purifier"]] = row
    defense = defaultdict(dict)
    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    for s in sets:
        base = Path("runtime/pexels_portraits/aligned") / (s["root"] or "") / "results"
        for c in CONDITIONS:
            if c == COLOR3D:
                path = Path("runtime/pexels_portraits/followup") / (s["root"] or "") / "results/color3d/fidelity.csv"
            else:
                path = (base / c if c == "photoguard_encoder" else base) / "fidelity.csv"
            for row in read(path):
                if row["condition"] == c and row["image"] in images:
                    defense[c][(row["image"],)] = float(row["fid_lpips"])

    n_cells = N * 4
    expected = {"none": n_cells, "seven": n_cells * 7, "vae_trans": n_cells}
    columns = ([("disp", k) for k in ("none", "seven", "vae_trans")]
               + [(f"disp_{m}", k) for m in ("ssim", "psnr") for k in ("none", "seven", "vae_trans")]
               + [("retained", "seven"), ("retained", "vae_trans")]
               + [(m, k) for m in ("siglip", "clip_i") for k in ("seven", "vae_trans")]
               + [("clip_s", k) for k in ("none", "seven", "vae_trans")]
               + [("arcface_original", "seven"), ("arcface_pair", "seven"), ("arcface_pair", "vae_trans")])
    table, per_purifier = [], []
    for c in CONDITIONS:
        require(c, "防禦圖影像數", len(defense[c]), N)
        row = {"condition": c, "n_images": N}
        put(row, "defense_lpips", image_stat(defense[c]))
        by_purifier = {}
        for m, kind in columns:
            name = f"{m}_{kind}"
            cells = metric[(m, c, kind)]
            require(c, f"{name} 格數", len(cells), expected[kind])
            if m.startswith("arcface"):
                row[f"{name}_missing"] = len(cells) - len(faces(cells))
                cells = faces(cells)
            if kind == "seven":
                by_purifier[m] = purifier_means(cells)
                put(row, name, purifier_stat(by_purifier[m].values()))
            else:
                put(row, name, image_stat(cells))
        row["retained_seven_ratio"] = row["disp_seven"] / row["disp_none"]
        d = distribution[c]
        for key in ("none", "vae_trans", *SEVEN):
            if key not in d:
                raise SystemExit(f"{c} 缺分布指標組 {key}")
            require(c, f"分布指標 {key} 影像數", int(d[key]["n_images"]), N)
        for m in ("fid", "kid"):
            row[f"{m}_none"] = float(d["none"][m])
            put(row, f"{m}_seven", purifier_stat(float(d[p][m]) for p in SEVEN))
            row[f"{m}_vae_trans"] = float(d["vae_trans"][m])
            by_purifier[m] = {p: float(d[p][m]) for p in SEVEN}
        for p in SEVEN:
            per_purifier.append({"condition": c, "purifier": p, **{m: round(v[p], 5) for m, v in by_purifier.items()}})
        table.append({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
    for name, rows in (("summary.csv", table), ("per_purifier.csv", per_purifier)):
        with open(R / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"[ALLDONE] {R / 'summary.csv'}（{len(table)} 列）", flush=True)


if __name__ == "__main__":
    main()
