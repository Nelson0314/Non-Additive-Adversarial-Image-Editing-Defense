"""論文 Fig. 5（purification_example）版面的 Pexels 主表質性圖：一張影像、一句 ip2p 指令，取主表 306 張的既有產物，不重跑編輯。

欄：原圖、PhotoGuard、Mist、DIA-PT、DCT-Shield（等失真臂）、BlurGuard（306 張版）、ImmuLUT（color3d_scale2p5_pexels）；
列：防禦圖、不淨化的編輯、淨化後的編輯（--purification，blur2 或 jpeg<q>）。圖內 "No defense" 改為 "Undefended"、
列標 "Protection" 改為 "Immunized"（與論文正文用詞一致）。編輯協定為主表 ip2p（text 7.5、image 1.8、50 步、
主種子 20260812）。版面、字型與尺寸由 color trial tools/generate_teaser_paper_figure.py 的 figure() 產生（只呼叫、不修改）；
本腳本把主表產物複製成該函式讀取的 run-dir 版面：
    <輸出根>/<影像>/inputs/<條件>/none/<影像>.png（undefended 為原圖）、edits/<條件>/{none,<淨化>}/<影像>__p<k>.png、data/man/<影像>.png、
    edits.csv、prompts.json
產物路徑規則同 scripts/run_inpaint_edits.py（等失真臂 aligned/<根>/、BlurGuard 兩批、ImmuLUT 於 color trial）；缺檔即失敗。
輸出 <輸出根>/<影像>_p<k>[_<淨化>].{pdf,png}（預設輸出根 artifacts/pexels_portraits/purification_example/）。
用法（遠端 baseline 專案根，CPU）：$PY scripts/generate_purification_example_figure.py --image woman_111 --prompt-index 1 [--purification blur2]
"""
import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import yaml

from immunization_baseline.figure_text import relabel

# 圖內用詞與論文正文一致（undefended、immunized image）；figure() 的原字串由 immunization_baseline.figure_text.relabel 替換。
LABELS = {"No defense": "Undefended", "Protection": "Immunized"}
ARMS = ("photoguard_encoder", "mist", "dia_pt", "dct_shield")
SEED, TEXT_GUIDANCE, IMAGE_GUIDANCE = 20260812, 7.5, 1.8
DATA = Path("data/pexels_portraits")
COLOR_TRIAL = Path("../color/trials/lut3d_defense_and_naturalness")
FIGURE_TOOL = COLOR_TRIAL / "tools/generate_teaser_paper_figure.py"


def sources(img, k, purifier):
    """{條件: (防禦圖, 不淨化編輯, 淨化後編輯)}；條件 undefended 的防禦圖為原圖。"""
    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    root = next(s for s in sets if img in s["images"])["root"] or ""
    na = Path("artifacts/pexels_portraits") / root
    aa = Path("artifacts/pexels_portraits/aligned") / root
    edit = f"{img}__p{k}.png"
    out = {"undefended": (DATA / img.split("_")[0] / f"{img}.png", na / "undefended_edits/ip2p_si18" / edit,
                          na / "purified_edits/undefended" / purifier / f"ip2p_undefended_{purifier}" / edit)}
    for c in ARMS:
        out[c] = (aa / "defenses" / c / f"{img}__{c}__def.png", aa / "defended_edits" / c / f"ip2p_{c}" / edit,
                  aa / "purified_edits" / c / purifier / f"ip2p_{c}_{purifier}" / edit)
    first = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
    if img in first:
        b = na
    else:
        unit = next(p.stem for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))
                    if img in p.read_text(encoding="utf-8").split())
        b = Path("runtime/pexels_portraits/blurguard_rest/artifacts") / unit
    out["blurguard"] = (b / "defenses/blurguard" / f"{img}__blurguard__def.png",
                        b / "defended_edits/blurguard/ip2p_blurguard" / edit,
                        b / "purified_edits/blurguard" / purifier / f"ip2p_blurguard_{purifier}" / edit)
    c = "color3d_scale2p5_pexels"
    e = COLOR_TRIAL / "artifacts_pexels/edits_by_image" / c / img
    out["immulut"] = (COLOR_TRIAL / "artifacts_pexels/defenses" / c / f"{img}__{c}__def.png",
                      e / "defended" / f"ip2p_{c}" / edit, e / "purified" / purifier / f"ip2p_{c}_{purifier}" / edit)
    missing = [p.as_posix() for files in out.values() for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"缺 {len(missing)} 個檔：{missing}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt-index", type=int, required=True)
    ap.add_argument("--purification", default="blur2")
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/pexels_portraits/purification_example"))
    a = ap.parse_args()
    img, k, purifier = a.image, a.prompt_index, a.purification
    prompts = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))["edits"]["ip2p"]
    run = a.output_root / img
    if run.exists():
        shutil.rmtree(run)
    rows = []
    for c, (protected, plain, purified) in sources(img, k, purifier).items():
        # figure() 也讀 undefended 的 inputs（原圖），故原圖同時放 data/man/ 與 inputs/undefended/none/。
        dsts = [run / "inputs" / c / "none" / f"{img}.png"] + ([run / "data/man" / f"{img}.png"] if c == "undefended" else [])
        for dst in dsts:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(protected, dst)
        for setting, src in (("none", plain), (purifier, purified)):
            d = run / "edits" / c / setting
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, d / f"{img}__p{k}.png")
            rows.append({"condition": c, "purifier": setting, "image": img, "prompt_index": k, "source_png": src.as_posix(),
                         "text_guidance": TEXT_GUIDANCE, "image_guidance": IMAGE_GUIDANCE, "seed": SEED})
    with open(run / "edits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (run / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("teaser_paper_figure", FIGURE_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tool.OUT = a.output_root
    name = f"{img}_p{k}" + ("" if purifier == "blur2" else f"_{purifier}")
    with relabel(LABELS):
        tool.figure(run, k, "full", name, purifier)


if __name__ == "__main__":
    main()
