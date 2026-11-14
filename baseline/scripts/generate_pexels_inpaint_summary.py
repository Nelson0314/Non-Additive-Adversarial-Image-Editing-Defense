"""Pexels 人像 SD inpainting 協定的彙整：等失真 photoguard_encoder、dia_pt、dct_shield、sifm、BlurGuard 與 ImmuLUT
（color3d_scale2p5_pexels），每條件一列；分母為同一協定的 undefended。

來源 runtime/pexels_portraits/inpaint/readouts/（scripts/evaluate_pexels_inpaint_readouts.sh）：displacement.csv、retention.csv、
semantic_identity.csv、arcface_pair.csv、clip_score.csv、edit_fidelity.csv、distribution/ 的 FID／KID；防禦圖 LPIPS 取 ip2p 主表的
fidelity.csv（同一批防禦圖），來源規則同 scripts/generate_pexels_flux_512_summary.py。
淨化設定逐一成欄（none＝淨化前、jpeg50、blur2、vae_trans），不做跨淨化平均：
    各欄為逐格平均，_std 為逐張值（4 句平均）的影像間標準差，_ci95＝1.96·s／√n（ArcFace 的 n 為有臉影像數）；
    FID／KID 為各設定一個值，無區間。
任一條件的影像數不等於清單張數或格數不齊時失敗。輸出 <readouts>/summary.csv。
用法（遠端 baseline 專案根）：python scripts/generate_pexels_inpaint_summary.py --images-file <檔> [--conditions ...] [--readouts-dir <目錄>]
"""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_pexels_main_table_summary import faces, image_stat, put, read, require  # noqa: E402

CONDITIONS = ("photoguard_encoder", "dia_pt", "dct_shield", "sifm", "blurguard", "color3d_scale2p5_pexels")
SETTINGS = ("none", "jpeg50", "blur2", "vae_trans")
PURIFIED = SETTINGS[1:]
R = Path("runtime/pexels_portraits/inpaint/readouts")


def defense_lpips(images, conditions):
    """防禦圖 LPIPS（ip2p 主表的 fidelity.csv）：{條件: {(影像,): 值}}。"""
    defense = defaultdict(dict)
    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    for s in sets:
        base = Path("runtime/pexels_portraits/aligned") / (s["root"] or "") / "results"
        for c in conditions:
            if c == "blurguard":
                continue
            if c.startswith("color3d_"):
                group = "color3d" if c == "color3d_pexels" else c.removesuffix("_pexels")
                path = Path("runtime/pexels_portraits/followup") / (s["root"] or "") / "results" / group / "fidelity.csv"
            else:
                path = (base / c if c == "photoguard_encoder" else base) / "fidelity.csv"
            for row in read(path):
                if row["condition"] == c and row["image"] in images:
                    defense[c][(row["image"],)] = float(row["fid_lpips"])
    if "blurguard" in conditions:
        for path in sorted(glob.glob("runtime/pexels_portraits/blurguard*/results/*/fidelity.csv")):
            if "/results/all/" in path.replace("\\", "/"):
                continue
            for row in read(path):
                if row["condition"] == "blurguard" and row["image"] in images:
                    defense["blurguard"][(row["image"],)] = float(row["fid_lpips"])
    return defense


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-file", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--readouts-dir", type=Path, default=R)
    args = ap.parse_args()
    images = set(args.images_file.read_text(encoding="utf-8").split())
    conditions, R_ = tuple(args.conditions), args.readouts_dir
    N = len(images)
    if N == 0:
        raise SystemExit("影像清單為空")

    metric = defaultdict(dict)   # (指標, 條件, 設定) -> {(影像, 句, 設定): 值}
    for row in read(R_ / "displacement.csv"):
        metric[("disp", row["condition"], "none")][(row["image"], row["prompt_index"], "none")] = float(row["disp_lpips_full"])
    for row in read(R_ / "retention.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        metric[("disp", row["condition"], row["purifier"])][key] = float(row["disp_purified"])
        metric[("retained", row["condition"], row["purifier"])][key] = float(row["retained"])
    for row in read(R_ / "semantic_identity.csv"):
        c, p = row["condition"], row["purifier"]
        if c == "undefended" or p not in PURIFIED:
            continue
        key = (row["image"], row["prompt_index"], p)
        for m, name in (("siglip", "siglip"), ("clip", "clip_i")):
            if row[f"{m}_pair"] != "":
                metric[(name, c, p)][key] = float(row[f"{m}_pair"])
    for row in read(R_ / "arcface_pair.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        metric[("arcface_pair", row["condition"], row["purifier"])][key] = (
            float(row["arcface_pair"]) if row["arcface_pair"] != "" else None)
    for row in read(R_ / "clip_score.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        metric[("clip_s", row["condition"], row["purifier"])][key] = float(row["clip_s"])
    for row in read(R_ / "edit_fidelity.csv"):
        key = (row["image"], row["prompt_index"], row["purifier"])
        for m in ("ssim", "psnr"):
            metric[(f"disp_{m}", row["condition"], row["purifier"])][key] = float(row[f"disp_{m}"])
    distribution = defaultdict(dict)
    for path in glob.glob(str(R_ / "distribution" / "*.csv")):
        for row in read(path):
            if "fid" in row and row.get("end") == "displacement":
                distribution[row["condition"]][row["purifier"]] = row
    defense = defense_lpips(images, conditions)

    columns = ([(m, k) for m in ("disp", "disp_ssim", "disp_psnr", "clip_s") for k in SETTINGS]
               + [(m, k) for m in ("retained", "siglip", "clip_i") for k in PURIFIED]
               + [("arcface_pair", k) for k in SETTINGS])
    table = []
    for c in conditions:
        require(c, "防禦圖影像數", len(defense[c]), N)
        row = {"condition": c, "n_images": N}
        put(row, "defense_lpips", image_stat(defense[c]))
        for m, k in columns:
            name, cells = f"{m}_{k}", metric[(m, c, k)]
            require(c, f"{name} 格數", len(cells), N * 4)
            if m.startswith("arcface"):
                row[f"{name}_missing"] = len(cells) - len(faces(cells))
                cells = faces(cells)
            put(row, name, image_stat(cells))
        for k in SETTINGS:
            if k not in distribution[c]:
                raise SystemExit(f"{c} 缺分布指標組 {k}")
            require(c, f"分布指標 {k} 影像數", int(distribution[c][k]["n_images"]), N)
            for m in ("fid", "kid"):
                row[f"{m}_{k}"] = float(distribution[c][k][m])
        table.append({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
    with open(R_ / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    print(f"[ALLDONE] {R_ / 'summary.csv'}（{len(table)} 列）", flush=True)


if __name__ == "__main__":
    main()
