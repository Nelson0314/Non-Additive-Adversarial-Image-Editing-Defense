"""VIBE-DistilledCFG 與 FLUX.2 [klein] 4B 的 ISR 評審素材（盲化拼圖），取既有編輯結果，不重跑編輯。

版面與 scripts/generate_inpaint_isr_sheets.py 相同：每張拼圖對應一張影像 × 一句指令（標題為影像 ID 與實際送入編輯器的指令全文），
列為 6 個條件（Undefended、等失真 PhotoGuard、DCT-Shield、Mist、BlurGuard、ImmuLUT），列序逐張以
numpy.random.default_rng(20260812) 依拼圖序號依序打亂，列標只寫 R1–R6，條件名只記在 key.csv；欄為 C1 Input（送入編輯器的影像；
Undefended 列為原圖）與 4 種淨化 none、jpeg50、blur2、vae_trans；格號 C<欄>-R<列> 印於每格左上（每格 256²）。
輸入路徑取 runtime/pexels_portraits/<流程>/chunks/<影像>/manifest.csv（purifier=none），指令取同目錄 edits_undefended__none.csv；
編輯在 artifacts/pexels_portraits/<編輯器目錄>/{edits,purified_edits}/。
    --editor vibe_distilled  runtime/pexels_portraits/vibe_distilled/、artifacts/pexels_portraits/vibe_distilled/
    --editor klein           runtime/pexels_portraits/klein/、artifacts/pexels_portraits/klein/
輸出 artifacts/pexels_portraits/<編輯器目錄>/isr_blind_sheets/：拼圖 <影像>__p<k>.jpg、cells.csv（含條件路徑，不交給評審）、
key.csv（sheet、row、condition）、images.txt。缺任何檔即失敗。
用法（遠端 baseline 專案根，CPU）：
    $PY scripts/generate_editor_isr_sheets.py --editor klein --images-file runtime/pexels_portraits/vibe_distilled/images_isr50.txt
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

SEED = 20260812
CONDS = [("undefended", "Undefended"), ("photoguard_encoder", "PhotoGuard"), ("dct_shield", "DCT-Shield"), ("mist", "Mist"),
         ("blurguard", "BlurGuard"), ("color3d_scale2p5_pexels", "ImmuLUT")]
COLS = [("input", "Input"), ("none", "No purification"), ("jpeg50", "JPEG 50"), ("blur2", "Gaussian blur σ2"), ("vae_trans", "VAE-Trans")]
CELL, LEFT, HEAD, TITLE, PAD = 256, 60, 34, 46, 4


def tile(path):
    if not Path(path).is_file():
        raise SystemExit(f"缺 {path}")
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", required=True, choices=("vibe_distilled", "klein"))
    ap.add_argument("--images-file", type=Path, required=True)
    a = ap.parse_args()
    run = Path("runtime/pexels_portraits") / a.editor
    art = Path("artifacts/pexels_portraits") / a.editor
    out = art / "isr_blind_sheets"
    images = a.images_file.read_text(encoding="utf-8").split()
    font = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    f_title, f_head, f_id = (ImageFont.truetype(font, s) for s in (17, 16, 14))
    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    W, H = LEFT + len(COLS) * (CELL + PAD), TITLE + HEAD + len(CONDS) * (CELL + PAD)
    cells, keys = [], []
    for img in images:
        with open(run / "chunks" / img / "manifest.csv", encoding="utf-8", newline="") as f:
            inputs = {r["arm"]: r["input_png"] for r in csv.DictReader(f) if r["purifier"] == "none"}
        with open(run / "chunks" / img / "edits_undefended__none.csv", encoding="utf-8", newline="") as f:
            prompts = {int(r["prompt_index"]): r["prompt"] for r in csv.DictReader(f)}
        for k in sorted(prompts):
            prompt = prompts[k]
            name = f"{img}__p{k}.jpg"
            order = rng.permutation(len(CONDS))
            sheet = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(sheet)
            d.text((8, 12), f'{img} — p{k}: "{prompt}"', fill="black", font=f_title)
            for j, (_, label) in enumerate(COLS):
                d.text((LEFT + j * (CELL + PAD) + CELL // 2, TITLE + HEAD // 2), label, fill="black", font=f_head, anchor="mm")
            edit = lambda c, op: art / ("edits" / Path(c) if op == "none" else Path("purified_edits") / c / op) / f"{img}__p{k}.png"
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
                                      "original_png": Path(inputs["undefended"]).as_posix(),
                                      "condition_input_png": Path(inputs[c]).as_posix(),
                                      "undefended_same_purification_png": edit("undefended", col).as_posix()})
            sheet.save(out / name, quality=90)
        print(f"[SHEET] {img}", flush=True)
    for fname, data in (("cells.csv", cells), ("key.csv", keys)):
        with open(out / fname, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (out / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    print(f"[{a.editor} ISR SHEETS] {len(keys) // len(CONDS)} 張，cells {len(cells)} 列，key {len(keys)} 列，{W}×{H}", flush=True)


if __name__ == "__main__":
    main()
