"""teaser 案例挑選用總覽圖：讀 scripts/run_teaser_edits.py 的 --run-dir，每個（淨化 × 指令）一張圖，
列＝條件，欄＝影像；淨化含 none 時另產生一張輸入總覽（列＝條件，欄＝影像，格內為不淨化的防禦圖）。
每格左上角印格號 C<欄>-R<列>，欄頭為影像名稱，標題含淨化與指令全文。
輸出：<--run-dir>/overview/inputs.jpg、<--run-dir>/overview/<淨化>__p<k>.jpg（JPEG 品質 90）。
用法（遠端 baseline 專案根，CPU）：python scripts/generate_teaser_overview.py --run-dir artifacts/teaser/<名稱> [--cell 256]
"""
import argparse
import json
from pathlib import Path

LABELS = {"undefended": "Undefended", "photoguard_encoder": "PhotoGuard", "mist": "Mist", "dia_pt": "DIA-PT",
          "dct_shield": "DCT-Shield", "blurguard": "BlurGuard", "immulut": "ImmuLUT"}
PURIFICATION_LABELS = {"none": "No purification", "jpeg50": "JPEG 50", "blur2": "Gaussian blur σ2", "vae_trans": "VAE-Trans"}


def sheet(title, rows, cols, cell_path, out, cell):
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    fp = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    f_title, f_head, f_id = (ImageFont.truetype(fp, s) for s in (17, 14, 13))
    left, head, top, pad = 140, 30, 44, 4
    img = Image.new("RGB", (left + len(cols) * (cell + pad), top + head + len(rows) * (cell + pad)), "white")
    d = ImageDraw.Draw(img)
    d.text((8, 12), title, fill="black", font=f_title)
    for j, c in enumerate(cols):
        d.text((left + j * (cell + pad) + cell // 2, top + head // 2), c, fill="black", font=f_head, anchor="mm")
    for i, r in enumerate(rows):
        y = top + head + i * (cell + pad)
        d.text((8, y + cell // 2), LABELS[r], fill="black", font=f_head, anchor="lm")
        for j, c in enumerate(cols):
            p = cell_path(r, c)
            if not p.is_file():
                raise SystemExit(f"缺 {p}")
            x = left + j * (cell + pad)
            img.paste(Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y))
            cid = f"C{j + 1}-R{i + 1}"
            box = d.textbbox((x + 4, y + 4), cid, font=f_id)
            d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill="white")
            d.text((x + 4, y + 4), cid, fill="black", font=f_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)
    print(f"[OVERVIEW] {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cell", type=int, default=256, help="每格邊長（像素）")
    args = ap.parse_args()
    R = args.run_dir
    if not (R / "prompts.json").is_file() or not (R / "edits").is_dir():
        raise SystemExit(f"{R} 不是 run_teaser_edits.py 的輸出目錄（缺 prompts.json 或 edits/）")
    prompts = json.loads((R / "prompts.json").read_text(encoding="utf-8"))
    conds = [c for c in LABELS if (R / "edits" / c).is_dir()]
    purs = [p for p in PURIFICATION_LABELS if any((R / "edits" / c / p).is_dir() for c in conds)]
    images = sorted(p.stem for p in (R / "data" / "man").glob("*.png"))
    if "none" in purs:
        sheet("Inputs (defended images, no purification)", conds, images,
              lambda c, im: R / "inputs" / c / "none" / f"{im}.png", R / "overview" / "inputs.jpg", args.cell)
    else:
        print("[NOTE] 輸入總覽取 inputs/<條件>/none/；本次未執行 --purification none，不產生 inputs.jpg", flush=True)
    for p in purs:
        for k, prompt in enumerate(prompts):
            sheet(f'{PURIFICATION_LABELS[p]} — p{k}: "{prompt}"', conds, images,
                  lambda c, im: R / "edits" / c / p / f"{im}__p{k}.png", R / "overview" / f"{p}__p{k}.jpg", args.cell)


if __name__ == "__main__":
    main()
