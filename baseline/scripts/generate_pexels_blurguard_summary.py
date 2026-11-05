"""BlurGuard 在 Pexels 100 張子集上的主表列（ip2p、四句、主種子；防禦圖經事後縮放到等失真錨點，見 STATUS.md）。

來源 runtime/pexels_portraits/blurguard/results/<影像組>/（runtime/pexels_portraits/blurguard/run_blurguard_set.sh 與
scripts/evaluate_pexels_blurguard_followup.sh）：displacement_、retention_、retention_vae_trans_blurguard.csv、semantic_identity{,_vae_trans}.csv、
arcface_pair.csv、clip_score.csv、survival.csv、edit_fidelity.csv、fidelity.csv、aesthetic.csv；FID／KID 取 results/all/distribution/
（scripts/evaluate_pexels_blurguard_summary.sh）。欄、平均與區間的算法同 scripts/generate_pexels_main_table_summary.py
（n＝100；ArcFace 為有臉影像數）。影像數不是 100 或格數不齊時失敗。
輸出 runtime/pexels_portraits/blurguard/results/all/{summary,per_purifier}.csv。
用法（遠端 baseline 專案根）：python scripts/generate_pexels_blurguard_summary.py
"""
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_pexels_main_table_summary import (SEVEN, faces, image_stat, purifier_means,  # noqa: E402
                                                 purifier_stat, put, read, require)

C = "blurguard"
B = Path("runtime/pexels_portraits/blurguard")
OUT = B / "results/all"


def kind_of(purifier):
    return "seven" if purifier in SEVEN else purifier


def main():
    images = set((B / "images.txt").read_text(encoding="utf-8").split())
    N = len(images)
    metric = defaultdict(dict)
    defense, aesthetic = {}, {}
    for R in sorted(p for p in (B / "results").iterdir() if p.is_dir() and p.name != "all"):
        if not (R / f"displacement_{C}.csv").is_file():
            continue
        for row in read(R / f"displacement_{C}.csv"):
            key = (row["image"], row["prompt_index"], "none")
            metric[("disp", "none")][key] = float(row["disp_lpips_full"])
            for m in ("ssim", "psnr"):
                metric[(f"disp_{m}", "none")][key] = float(row[f"disp_{m}"])
        for name in (f"retention_{C}.csv", f"retention_vae_trans_{C}.csv"):
            for row in read(R / name):
                kind = kind_of(row["purifier"])
                if kind not in ("seven", "vae_trans"):
                    continue
                key = (row["image"], row["prompt_index"], row["purifier"])
                metric[("disp", kind)][key] = float(row["disp_purified"])
                metric[("retained", kind)][key] = float(row["retained"])
        for name in ("semantic_identity.csv", "semantic_identity_vae_trans.csv"):
            for row in read(R / name):
                kind = kind_of(row["purifier"])
                if row["condition"] != C or kind not in ("seven", "vae_trans"):
                    continue
                key = (row["image"], row["prompt_index"], row["purifier"])
                for m, label in (("siglip", "siglip"), ("clip", "clip_i")):
                    metric[(label, kind)][key] = float(row[f"{m}_pair"])
                if kind == "seven":
                    metric[("arcface_original", kind)][key] = float(row["arcface_id"]) if row["arcface_id"] != "" else None
        for row in read(R / "arcface_pair.csv"):
            key = (row["image"], row["prompt_index"], row["purifier"])
            metric[("arcface_pair", kind_of(row["purifier"]))][key] = float(row["arcface_pair"]) if row["arcface_pair"] != "" else None
        for row in read(R / "clip_score.csv"):
            if row["condition"] == C:
                metric[("clip_s", kind_of(row["purifier"]))][(row["image"], row["prompt_index"], row["purifier"])] = float(row["clip_s"])
        for row in read(R / "edit_fidelity.csv"):
            kind = kind_of(row["purifier"])
            if kind in ("seven", "vae_trans"):
                for m in ("ssim", "psnr"):
                    metric[(f"disp_{m}", kind)][(row["image"], row["prompt_index"], row["purifier"])] = float(row[f"disp_{m}"])
        for row in read(R / "survival.csv"):
            metric[("survival", kind_of(row["purifier"]))][(row["image"], row["purifier"])] = float(row["survival"])
        for row in read(R / "fidelity.csv"):
            if row["condition"] == C:
                defense[(row["image"],)] = float(row["fid_lpips"])
        for row in read(R / "aesthetic.csv"):
            if row["condition"] == C:
                aesthetic[row["image"]] = row
    distribution = {}
    for path in glob.glob(str(OUT / "distribution" / "*distribution*.csv")):
        for row in read(path):
            distribution["distortion" if row["end"] == "distortion" else row["purifier"]] = row

    require(C, "防禦圖影像數", len(defense), N)
    require(C, "防禦圖美學影像數", len(aesthetic), N)
    if {k[0] for k in defense} != images:
        raise SystemExit("防禦圖影像與 images.txt 不同")
    cells = N * 4
    expected = {"none": cells, "seven": cells * 7, "vae_trans": cells}
    columns = ([("disp", k) for k in ("none", "seven", "vae_trans")]
               + [(f"disp_{m}", k) for m in ("ssim", "psnr") for k in ("none", "seven", "vae_trans")]
               + [("retained", "seven"), ("retained", "vae_trans")]
               + [(m, k) for m in ("siglip", "clip_i") for k in ("seven", "vae_trans")]
               + [("clip_s", k) for k in ("none", "seven", "vae_trans")]
               + [("arcface_original", "seven"), ("arcface_pair", "seven"), ("arcface_pair", "vae_trans")]
               + [("survival", "seven"), ("survival", "vae_trans")])
    row = {"condition": C, "n_images": N}
    put(row, "defense_lpips", image_stat(defense))
    for col in ("aes_laion", "aes_musiq_ava"):
        put(row, col, image_stat({(i,): float(r[col]) for i, r in aesthetic.items()}))
    by_purifier = {}
    for m, kind in columns:
        name = f"{m}_{kind}"
        data = metric[(m, kind)]
        want = (N * (7 if kind == "seven" else 1)) if m == "survival" else expected[kind]
        require(C, f"{name} 格數", len(data), want)
        if m.startswith("arcface"):
            row[f"{name}_missing"] = len(data) - len(faces(data))
            data = faces(data)
        if kind == "seven":
            by_purifier[m] = purifier_means(data)
            put(row, name, purifier_stat(by_purifier[m].values()))
        else:
            put(row, name, image_stat(data))
        if name == "disp_seven":
            worst = defaultdict(list)
            for (i, pi, _), v in data.items():
                worst[(i, pi)].append(v)
            put(row, "disp_worst", image_stat({k: min(v) for k, v in worst.items()}))
    row["retained_seven_ratio"] = row["disp_seven"] / row["disp_none"]
    for key in ("distortion", "none", "vae_trans", *SEVEN):
        if key not in distribution:
            raise SystemExit(f"缺分布指標組 {key}")
        require(C, f"分布指標 {key} 影像數", int(distribution[key]["n_images"]), N)
    for m in ("fid", "kid"):
        row[f"{m}_none"] = float(distribution["none"][m])
        put(row, f"{m}_seven", purifier_stat(float(distribution[p][m]) for p in SEVEN))
        row[f"{m}_vae_trans"] = float(distribution["vae_trans"][m])
        row[f"{m}_distortion"] = float(distribution["distortion"][m])
        by_purifier[m] = {p: float(distribution[p][m]) for p in SEVEN}
    per_purifier = [{"condition": C, "purifier": p, **{m: round(v[p], 5) for m, v in by_purifier.items()}} for p in SEVEN]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", [{k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}]),
                       ("per_purifier.csv", per_purifier)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"[ALLDONE] {OUT / 'summary.csv'}（n＝{N}）", flush=True)


if __name__ == "__main__":
    main()
