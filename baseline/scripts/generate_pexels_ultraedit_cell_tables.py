"""Pexels 人像 UltraEdit 50 張（color trial `pexels_subsets/flux.txt`）的逐格表與讀數用目錄，供 baseline 的共用指標程式讀取。

條件：等失真 photoguard_encoder、dct_shield、dct_shield_y、mist（ip2p 上求得的防禦圖直接送 UltraEdit，`scripts/evaluate_pexels_transfer.sh ultraedit`）；
未防禦分母取 color trial 的 UltraEdit 輸出（`results/cross_model_pexels/sd3/transfer50/edits_undefended.csv`，
`artifacts_pexels/cross_model/sd3/edits/{undefended_edits,purified_edits/undefended}/`，只讀）。
來源：`runtime/pexels_portraits/transfer/ultraedit/chunks/<影像>/{plain,vae_trans,seven}/edits_<條件>.csv`。

輸出 `runtime/pexels_portraits/ultraedit_readouts/`：
- `edits_<條件>.csv`、`edits_undefended.csv`：欄 arm、image、prompt_index、prompt、png、purifier（none、七道、vae_trans），每條件 50 × 4 × 9 ＝ 1,800 格
- `pairs.csv`：欄 condition、end、purifier、a_png（防禦後編輯）、b_png（同一淨化的未防禦編輯），供 measure_distribution_metrics 與編輯端 SSIM／PSNR
- `edits/defended_edits/<條件>/`：`preflight.csv`（50 張 × 4 句）與指向產物 `ultraedit_<條件>/` 的符號連結
- `edits/purified_edits/`：指向四條件與未防禦淨化後編輯的符號連結（只讀）
選項：--images-file（預設 flux.txt 的 50 張）、--conditions（預設上列四條件；color3d_<變體>_pexels 取 baseline 遷移產物）、--output-dir；
給 --images-file 時未防禦分母改讀 color trial results/cross_model_pexels/sd3/chunks/<影像組>/edits_undefended.csv 的聯集。
用法（遠端 baseline 專案根，CPU）：python scripts/generate_pexels_ultraedit_cell_tables.py [--images-file <檔>] [--conditions ...] [--output-dir <目錄>]
"""
import argparse
import csv
import glob
import os
from pathlib import Path

CONDITIONS = ("photoguard_encoder", "dct_shield", "dct_shield_y", "mist")
OPS = ("blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15", "vae_trans")
COLS = ("arm", "image", "prompt_index", "prompt", "png", "purifier")
CT = Path("../color/trials/lut3d_defense_and_naturalness")
CHUNKS = Path("runtime/pexels_portraits/transfer/ultraedit/chunks")
EDITS = Path("artifacts/pexels_portraits/transfer/ultraedit/edits")
COLOR_EDITS = CT / "artifacts_pexels/cross_model/sd3/edits"
OUT = Path("runtime/pexels_portraits/ultraedit_readouts")


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path, data, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(data)


def link(target, name):
    """name → target 的符號連結；已存在時須指向同一處。"""
    target = Path(os.path.relpath(Path(target).resolve(), Path(name).parent.resolve()))
    if name.is_symlink():
        if Path(os.readlink(name)) != target:
            raise SystemExit(f"{name} 已指向 {os.readlink(name)}，不是 {target}")
        return
    if name.exists():
        raise SystemExit(f"{name} 已存在且不是符號連結")
    name.symlink_to(target, target_is_directory=True)


def check(arm, data, n_images):
    keys = {(r["image"], r["prompt_index"], r["purifier"]) for r in data}
    want = n_images * 4 * (1 + len(OPS))
    if len(data) != len(keys) or len(keys) != want:
        raise SystemExit(f"{arm}：{len(data)} 列、{len(keys)} 個不重複格，應為 {want}")
    for r in data:
        if not Path(r["png"]).is_file():
            raise SystemExit(f"缺影像 {r['png']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-file", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.output_dir
    conditions = tuple(args.conditions)
    images = (args.images_file or CT / "pexels_subsets/flux.txt").read_text(encoding="utf-8").split()
    out.mkdir(parents=True, exist_ok=True)
    cells = {}
    if args.images_file is None:
        source = rows(CT / "results/cross_model_pexels/sd3/transfer50/edits_undefended.csv")
    else:
        source = [r for p in sorted(glob.glob(str(CT / "results/cross_model_pexels/sd3/chunks/*/edits_undefended.csv"))) for r in rows(p)]
    undefended = [{k: r[k] for k in COLS} | {"arm": "undefended"} for r in source if r["image"] in set(images)]
    check("undefended", undefended, len(images))
    write(out / "edits_undefended.csv", undefended, COLS)
    cells["undefended"] = {(r["image"], r["prompt_index"], r["purifier"]): r for r in undefended}
    for c in conditions:
        data = []
        for img in images:
            for part in ("plain", "vae_trans", "seven"):
                for r in rows(CHUNKS / img / part / f"edits_{c}.csv"):
                    data.append({"arm": c, "image": r["image"], "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                                 "png": r["png"], "purifier": r["purifier"]})
        check(c, data, len(images))
        data.sort(key=lambda r: (r["purifier"], r["image"], int(r["prompt_index"])))
        write(out / f"edits_{c}.csv", data, COLS)
        cells[c] = {(r["image"], r["prompt_index"], r["purifier"]): r for r in data}
        for k, r in cells[c].items():
            if r["prompt"] != cells["undefended"][k]["prompt"]:
                raise SystemExit(f"{c} {k} 的指令與未防禦不同：{r['prompt']} vs {cells['undefended'][k]['prompt']}")
        d = out / "edits/defended_edits" / c
        d.mkdir(parents=True, exist_ok=True)
        link(EDITS / "defended_edits" / c / f"ultraedit_{c}", d / f"ultraedit_{c}")
        pre = [{"scenario": "ip2p", "arm": f"ultraedit_{c}", "image": r["image"], "prompt_index": r["prompt_index"], "prompt": r["prompt"]}
               for r in sorted(data, key=lambda r: (r["image"], int(r["prompt_index"]))) if r["purifier"] == "none"]
        write(d / "preflight.csv", pre, ("scenario", "arm", "image", "prompt_index", "prompt"))
    p = out / "edits/purified_edits"
    p.mkdir(parents=True, exist_ok=True)
    for c in conditions:
        link(EDITS / "purified_edits" / c, p / c)
    link(COLOR_EDITS / "purified_edits/undefended", p / "undefended")
    pairs = [{"condition": c, "end": "displacement", "purifier": k[2], "a_png": r["png"], "b_png": cells["undefended"][k]["png"]}
             for c in conditions for k, r in sorted(cells[c].items())]
    write(out / "pairs.csv", pairs, ("condition", "end", "purifier", "a_png", "b_png"))
    print(f"[TABLES] images={len(images)} conditions={len(conditions)} pairs={len(pairs)} → {out}", flush=True)


if __name__ == "__main__":
    main()
