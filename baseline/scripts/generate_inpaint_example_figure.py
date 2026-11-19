"""Pexels SD inpainting 的質性圖：版面同 Fig. 4（scripts/generate_purification_example_figure.py 的 draw()），取 inpainting 既有產物，不重跑編輯。

欄：Undefended、PhotoGuard、DIA-PT、DCT-Shield、SIFM（等失真臂）、BlurGuard、ImmuLUT；列：防禦圖、不淨化的 inpainting 編輯、
淨化後（--purification：jpeg50、blur2 或 vae_trans）的 inpainting 編輯。協定見 scripts/run_inpaint_edits.py
（runwayml/stable-diffusion-inpainting、主體遮罩的補集、主種子 20260812、prompts.yaml 的 edits.inpaint）。
防禦圖沿用 ip2p 流程（與 inpainting 共用 VAE），路徑規則同 run_inpaint_edits.build_manifest；編輯在 artifacts/pexels_portraits/inpaint/。
所用檔案寫入 <輸出根>/<影像>_p<k>_<淨化>.csv；輸出 <輸出根>/<影像>_p<k>_<淨化>.{pdf,png}（預設輸出根 artifacts/pexels_portraits/inpaint_example/）。
用法（遠端 baseline 專案根，CPU）：$PY scripts/generate_inpaint_example_figure.py --image man_08 --prompt-index 2 --purification blur2
"""
import argparse
import csv
import json
from pathlib import Path

import yaml

from generate_purification_example_figure import WIDTH, draw
from run_inpaint_edits import build_manifest

ARMS = ("photoguard_encoder", "dia_pt", "dct_shield", "sifm", "blurguard")
DATA = Path("data/pexels_portraits")
EDITS = Path("artifacts/pexels_portraits/inpaint")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt-index", type=int, required=True)
    ap.add_argument("--purification", default="blur2", choices=("jpeg50", "blur2", "vae_trans"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/pexels_portraits/inpaint_example"))
    ap.add_argument("--print-width", type=float, default=0.62 * WIDTH, help="論文置入寬度（英吋）")
    ap.add_argument("--min-print-pt", type=float, default=7.0, help="印刷字級下限（pt）")
    a = ap.parse_args()
    img, k, purifier = a.image, a.prompt_index, a.purification
    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    first = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
    units = {p.stem: set(p.read_text(encoding="utf-8").split())
             for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))}
    cls, rows = build_manifest(img, sets, first, units)
    protected = {c: path for c, op, path in rows if op == "none"}
    files = {}
    for c in ("undefended", *ARMS, "color3d_scale2p5_pexels"):
        triple = (protected[c], EDITS / "edits" / c / f"{img}__p{k}.png", EDITS / "purified_edits" / c / purifier / f"{img}__p{k}.png")
        missing = [x.as_posix() for x in triple if not x.is_file()]
        if missing:
            raise SystemExit(f"缺檔：{missing}")
        files["immulut" if c == "color3d_scale2p5_pexels" else c] = triple
    spec = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))
    prompt = spec["edits"]["inpaint"][k].format(content=spec[cls]["content"])
    name = f"{img}_p{k}_{purifier}"
    a.output_root.mkdir(parents=True, exist_ok=True)
    with open(a.output_root / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["condition", "row", "source_png"])
        w.writerows((c, row, p.as_posix()) for c, t in files.items() for row, p in zip(("immunized", "none", purifier), t))
    draw(files, prompt, purifier, a.min_print_pt * WIDTH / a.print_width, a.output_root / name, arms=ARMS)


if __name__ == "__main__":
    main()
