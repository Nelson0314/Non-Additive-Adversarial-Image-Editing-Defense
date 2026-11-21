"""SD inpainting 的 ISR 評審素材：306 張 Pexels 人像 × 4 句 inpaint 指令 ＝ 1,224 張拼圖，取既有編輯結果，不重跑編輯。

每張拼圖對應一張影像 × 一句指令（標題為影像 ID 與實際送入編輯器的指令全文）。列為 7 個條件（Undefended、等失真 PhotoGuard、
DIA-PT、DCT-Shield、SIFM、BlurGuard、ImmuLUT），列序逐張以 numpy.random.default_rng(20260812) 依拼圖序號依序打亂，列標只寫 R1–R7，
條件名只記在 key.csv。欄為 C1 Input（送入編輯器的影像；Undefended 列為原圖）與 4 種淨化 none、jpeg50、blur2、vae_trans。
格號 C<欄>-R<列> 印於每格左上，版面與 ip2p 的 ISR 拼圖相同（每格 256²）。
輸入與編輯的路徑規則同 scripts/run_inpaint_edits.py（防禦圖與淨化後輸入沿用 ip2p 流程；編輯在 artifacts/pexels_portraits/inpaint/）。
輸出 artifacts/pexels_portraits/inpaint_isr_sheets/：
    <影像>__p<k>.jpg   拼圖
    cells.csv          每個編輯格一列（sheet、cell、image、prompt_id、prompt_text、purification、edit_png、original_png、
                       condition_input_png、undefended_same_purification_png）；路徑含條件名，不交給評審
    key.csv            sheet、row、condition（解盲用）
    images.txt         影像清單
缺任何檔即失敗。用法（遠端 baseline 專案根，CPU）：$PY scripts/generate_inpaint_isr_sheets.py [--images-file <檔>]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inpaint_edits import build_manifest  # noqa: E402

SEED = 20260812
DATA = Path("data/pexels_portraits")
EDITS = Path("artifacts/pexels_portraits/inpaint")
OUT = Path("artifacts/pexels_portraits/inpaint_isr_sheets")
CONDS = [("undefended", "Undefended"), ("photoguard_encoder", "PhotoGuard"), ("dia_pt", "DIA-PT"), ("dct_shield", "DCT-Shield"),
         ("sifm", "SIFM"), ("blurguard", "BlurGuard"), ("color3d_scale2p5_pexels", "ImmuLUT")]
COLS = [("input", "Input"), ("none", "No purification"), ("jpeg50", "JPEG 50"), ("blur2", "Gaussian blur σ2"), ("vae_trans", "VAE-Trans")]
CELL, LEFT, HEAD, TITLE, PAD = 256, 60, 34, 46, 4


def tile(path):
    if not Path(path).is_file():
        raise SystemExit(f"缺 {path}")
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-file", type=Path, default=Path("runtime/pexels_portraits/vibe_distilled/images_306.txt"))
    a = ap.parse_args()
    images = a.images_file.read_text(encoding="utf-8").split()
    spec = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))
    templates = spec["edits"]["inpaint"]
    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    first = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
    units = {p.stem: set(p.read_text(encoding="utf-8").split())
             for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))}
    font = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    f_title, f_head, f_id = (ImageFont.truetype(font, s) for s in (17, 16, 14))
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    W, H = LEFT + len(COLS) * (CELL + PAD), TITLE + HEAD + len(CONDS) * (CELL + PAD)
    cells, keys = [], []
    for img in images:
        cls, rows = build_manifest(img, sets, first, units)
        inputs = {c: p for c, op, p in rows if op == "none"}
        for k, template in enumerate(templates):
            prompt = template.format(content=spec[cls]["content"])
            name = f"{img}__p{k}.jpg"
            order = rng.permutation(len(CONDS))
            sheet = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(sheet)
            d.text((8, 12), f'{img} — p{k}: "{prompt}"', fill="black", font=f_title)
            for j, (_, label) in enumerate(COLS):
                d.text((LEFT + j * (CELL + PAD) + CELL // 2, TITLE + HEAD // 2), label, fill="black", font=f_head, anchor="mm")
            edit = lambda c, op: EDITS / ("edits" / Path(c) if op == "none" else Path("purified_edits") / c / op) / f"{img}__p{k}.png"
            for i, ci in enumerate(order):
                c = CONDS[ci][0]
                y = TITLE + HEAD + i * (CELL + PAD)
                d.text((8, y + CELL // 2), f"R{i + 1}", fill="black", font=f_head, anchor="lm")
                keys.append({"sheet": name, "row": f"R{i + 1}", "condition": CONDS[ci][1]})
                for j, (col, _) in enumerate(COLS):
                    path = inputs[c] if col == "input" else edit(c, col)
                    x = LEFT + j * (CELL + PAD)
                    sheet.paste(tile(path), (x, y))
                    cid = f"C{j + 1}-R{i + 1}"
                    box = d.textbbox((x + 4, y + 4), cid, font=f_id)
                    d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill="white")
                    d.text((x + 4, y + 4), cid, fill="black", font=f_id)
                    if col != "input":
                        cells.append({"sheet": name, "cell": cid, "image": img, "prompt_id": f"p{k}", "prompt_text": prompt,
                                      "purification": col, "edit_png": Path(path).as_posix(),
                                      "original_png": inputs["undefended"].as_posix(),
                                      "condition_input_png": Path(inputs[c]).as_posix(),
                                      "undefended_same_purification_png": edit("undefended", col).as_posix()})
            sheet.save(OUT / name, quality=90)
        print(f"[SHEET] {img}", flush=True)
    for fname, data in (("cells.csv", cells), ("key.csv", keys)):
        with open(OUT / fname, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (OUT / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    print(f"[INPAINT ISR SHEETS] {len(images) * len(templates)} 張，cells {len(cells)} 列，key {len(keys)} 列，{W}×{H}", flush=True)


if __name__ == "__main__":
    main()
