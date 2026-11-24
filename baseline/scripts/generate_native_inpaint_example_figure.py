"""SD inpainting 質性圖的原生預算版：版面同 scripts/generate_inpaint_example_figure.py（Fig. 4 版面），
加性五法改用原生預算的防禦圖與編輯（scripts/run_native_inpaint_edits.py 的產物），Undefended 與 ImmuLUT 沿用等失真流程的既有產物。

原生預算產物根（--native-root）：
    man_26  artifacts/pexels_portraits/inpaint_native_man_26/
    其餘    artifacts/pexels_portraits/inpaint_native/<影像>/
防禦圖取 <根>/defenses/<條件>/<影像>__<條件>__def.png，不存在時取 ip2p 流程的原生預算防禦圖 artifacts/pexels_portraits/<影像組根>/defenses/<條件>/；
編輯取 <根>/edits/<條件>/{none,<淨化>}/<影像>__p<k>.png。
所用檔案寫入 <輸出根>/<影像>_p<k>_<淨化>_native.csv；輸出 <輸出根>/<影像>_p<k>_<淨化>_native.{pdf,png}。
用法（遠端 baseline 專案根，CPU）：
    $PY scripts/generate_native_inpaint_example_figure.py --image woman_14 --prompt-index 0 --purification jpeg50 \\
        --native-root artifacts/pexels_portraits/inpaint_native/woman_14
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
    ap.add_argument("--purification", required=True, choices=("jpeg50", "blur2", "vae_trans"))
    ap.add_argument("--native-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/pexels_portraits/inpaint_example_native"))
    ap.add_argument("--print-width", type=float, default=0.62 * WIDTH, help="論文置入寬度（英吋）")
    ap.add_argument("--min-print-pt", type=float, default=7.0, help="印刷字級下限（pt）")
    a = ap.parse_args()
    img, k, purifier = a.image, a.prompt_index, a.purification
    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    root = next(s["root"] for s in sets if img in s["images"]) or ""
    first = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
    units = {p.stem: set(p.read_text(encoding="utf-8").split())
             for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))}
    cls, rows = build_manifest(img, sets, first, units)
    protected = {c: path for c, op, path in rows if op == "none"}
    files = {}
    for c in ("undefended", *ARMS, "color3d_scale2p5_pexels"):
        if c in ARMS:
            d = a.native_root / "defenses" / c / f"{img}__{c}__def.png"
            if not d.is_file():
                d = Path("artifacts/pexels_portraits") / root / "defenses" / c / f"{img}__{c}__def.png"
            e = a.native_root / "edits" / c
            triple = (d, e / "none" / f"{img}__p{k}.png", e / purifier / f"{img}__p{k}.png")
        else:
            triple = (protected[c], EDITS / "edits" / c / f"{img}__p{k}.png",
                      EDITS / "purified_edits" / c / purifier / f"{img}__p{k}.png")
        missing = [x.as_posix() for x in triple if not x.is_file()]
        if missing:
            raise SystemExit(f"缺檔：{missing}")
        files["immulut" if c == "color3d_scale2p5_pexels" else c] = triple
    spec = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))
    prompt = spec["edits"]["inpaint"][k].format(content=spec[cls]["content"])
    name = f"{img}_p{k}_{purifier}_native"
    a.output_root.mkdir(parents=True, exist_ok=True)
    with open(a.output_root / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["condition", "row", "source_png"])
        w.writerows((c, row, p.as_posix()) for c, t in files.items() for row, p in zip(("immunized", "none", purifier), t))
    draw(files, prompt, purifier, a.min_print_pt * WIDTH / a.print_width, a.output_root / name, arms=ARMS)


if __name__ == "__main__":
    main()
