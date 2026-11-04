"""Pexels 人像 FLUX 512 協定（資料版本 flux_512，50 張）的逐格表，供 baseline 的共用指標程式讀取。

條件：未防禦（分母，同一資料版本）、等失真 photoguard_encoder、dct_shield、dct_shield_y、mist，以及 color3d_pexels；
每條件 {淨化前、七道、VAE-Trans} × 4 句。來源 `runtime/pexels_portraits/flux_512/chunks/<影像>/edits_<條件>__<設定>.csv`
（`scripts/evaluate_pexels_flux_512.sh`），影像清單為 color trial `pexels_subsets/flux.txt`。

輸出 `runtime/pexels_portraits/flux_512/readouts/`：
- `edits_<條件>.csv`（含 `edits_undefended.csv`）：欄 arm、image、prompt_index、prompt、png、purifier（none、七道、vae_trans），
  每條件 50 × 4 × 9 ＝ 1,800 格；供 measure_semantic_identity、measure_clip_score
- `plain/edits_<條件>.csv`：只含淨化前的格，供 measure_flux_displacement（`--edits-csv-dir`）
- `pairs.csv`：欄 condition、end、purifier、a_png（防禦後編輯）、b_png（同一淨化的未防禦編輯），供 measure_distribution_metrics 與編輯端 SSIM／PSNR
任一條件格數不齊、格重複、影像缺檔或指令與未防禦不同時失敗。
用法（遠端 baseline 專案根，CPU）：python scripts/generate_pexels_flux_512_cell_tables.py
"""
import csv
from pathlib import Path

UNDEFENDED = "undefended"
CONDITIONS = ("photoguard_encoder", "dct_shield", "dct_shield_y", "mist", "color3d_pexels")
SETTINGS = ("none", "blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15", "vae_trans")
COLS = ("arm", "image", "prompt_index", "prompt", "png", "purifier")
CT = Path("../color/trials/lut3d_defense_and_naturalness")
CHUNKS = Path("runtime/pexels_portraits/flux_512/chunks")
OUT = Path("runtime/pexels_portraits/flux_512/readouts")


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path, data, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(data)


def collect(condition, images):
    data = []
    for img in images:
        for setting in SETTINGS:
            for r in rows(CHUNKS / img / f"edits_{condition}__{setting}.csv"):
                if r["arm"] != condition or r["image"] != img:
                    raise SystemExit(f"{CHUNKS / img}/edits_{condition}__{setting}.csv 含 {r['arm']} {r['image']} 的列")
                data.append({"arm": condition, "image": img, "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                             "png": r["png"], "purifier": setting})
    keys = {(r["image"], r["prompt_index"], r["purifier"]) for r in data}
    want = len(images) * 4 * len(SETTINGS)
    if len(data) != len(keys) or len(keys) != want:
        raise SystemExit(f"{condition}：{len(data)} 列、{len(keys)} 個不重複格，應為 {want}")
    for r in data:
        if not Path(r["png"]).is_file():
            raise SystemExit(f"缺影像 {r['png']}")
    data.sort(key=lambda r: (r["purifier"], r["image"], int(r["prompt_index"])))
    return data


def main():
    images = (CT / "pexels_subsets/flux.txt").read_text(encoding="utf-8").split()
    cells = {}
    for c in (UNDEFENDED, *CONDITIONS):
        data = collect(c, images)
        write(OUT / f"edits_{c}.csv", data, COLS)
        write(OUT / "plain" / f"edits_{c}.csv", [r for r in data if r["purifier"] == "none"], COLS)
        cells[c] = {(r["image"], r["prompt_index"], r["purifier"]): r for r in data}
    for c in CONDITIONS:
        for k, r in cells[c].items():
            if r["prompt"] != cells[UNDEFENDED][k]["prompt"]:
                raise SystemExit(f"{c} {k} 的指令與未防禦不同：{r['prompt']} vs {cells[UNDEFENDED][k]['prompt']}")
    pairs = [{"condition": c, "end": "displacement", "purifier": k[2], "a_png": r["png"], "b_png": cells[UNDEFENDED][k]["png"]}
             for c in CONDITIONS for k, r in sorted(cells[c].items())]
    write(OUT / "pairs.csv", pairs, ("condition", "end", "purifier", "a_png", "b_png"))
    print(f"[TABLES] images={len(images)} conditions={len(CONDITIONS)} pairs={len(pairs)} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
