"""消融降強度擴充（crosseditor100 全部 100 張）的 ISR 評審拼圖：100 張 × ip2p 4 句 ＝ 400 張，每張拼圖 5 列。

列：Undefended、Full（ImmuLUT）、B2_lpips0p85（色相旋轉＋飽和度縮放，防禦圖 LPIPS 目標 0.85 倍 Full）、
B3_lpips0p55、B3_lpips0p40（3×3 仿射，目標 0.55、0.40 倍 Full）。列序逐張以 numpy.random.default_rng(20260812) 依拼圖序號依序打亂
（permutation），列標只寫 R1–R5，條件名只記在 key.csv。欄為 C1 Input（各條件送入編輯器的影像；Undefended 列為原圖）與淨化
none、jpeg50、blur2、vae_trans 的 ip2p 編輯。版面同 generate_ablation_strength_sheets.py（每格 256²）。
路徑：Undefended 與 Full 取 baseline artifacts/pexels_portraits/ip2p_isr100_sheets/cells.csv（與消融 ISR 同一批編輯）；
降強度列取 trial artifacts_pexels 的 defense_shards、edits_by_image、purified_edits；VAE-Trans 編輯取 vae_trans_crosseditor100/<條件>/，
該處沒有的影像（30 張樣本）取既有的 vae_trans/<條件>/。
某條件在某張影像無可行檢查點時，該列各格為灰底並印 N/A，cells.csv 不列該列的格，key.csv 的 status 記 no_feasible_checkpoint。
其餘任何檔案缺少即失敗。
輸出 trial artifacts_pexels/ablation_strength_crosseditor100_sheets/：<影像>__p<k>.jpg、cells.csv、key.csv（sheet、row、condition、status）、
images.txt。用法（遠端，CPU）：python color/scripts/generate_ablation_strength_crosseditor100_sheets.py
"""
import csv
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

SEED = 20260812
COLOR = Path(__file__).resolve().parents[1]
TRIAL = COLOR / "trials/lut3d_defense_and_naturalness"
AP = TRIAL / "artifacts_pexels"
BASE = Path.home() / "image-immunization/baseline"
ISR_CELLS = BASE / "artifacts/pexels_portraits/ip2p_isr100_sheets/cells.csv"
IMAGES = COLOR / "results/ablation_strength_crosseditor100/images.txt"
OUT = AP / "ablation_strength_crosseditor100_sheets"
REUSED = {"undefended": "Undefended", "color3d_scale2p5_pexels": "Full"}
CONDS = [("undefended", "Undefended"), ("color3d_scale2p5_pexels", "Full"),
         ("color_baseline_huesat_lpips0p85_pexels", "B2_lpips0p85"), ("color_baseline_affine_lpips0p55_pexels", "B3_lpips0p55"),
         ("color_baseline_affine_lpips0p40_pexels", "B3_lpips0p40")]
PURS = ["none", "jpeg50", "blur2", "vae_trans"]
COLS = [("input", "Input"), ("none", "No purification"), ("jpeg50", "JPEG 50"), ("blur2", "Gaussian blur σ2"), ("vae_trans", "VAE-Trans")]
CELL, LEFT, HEAD, TITLE, PAD = 256, 60, 34, 46, 4


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else BASE / p


def tile(path):
    if not Path(path).is_file():
        raise SystemExit(f"缺 {path}")
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


def vae_edit(c, f):
    """VAE-Trans 編輯：先找 vae_trans_crosseditor100/<條件>/，沒有再找既有 vae_trans/<條件>/。"""
    rel = Path("edits") / c / "vae_trans" / f"ip2p_{c}_vae_trans" / f
    new = AP / "vae_trans_crosseditor100" / c / rel
    return new if new.is_file() else AP / "vae_trans" / c / rel


def strength_paths(c, img, k):
    """回傳 (防禦圖, {淨化: 編輯})；無防禦圖時回傳 None（求解無可行檢查點，以 shard results.csv 的 outcome 確認）。"""
    shard = AP / "defense_shards" / c / img
    defense = shard / f"{img}__{c}__def.png"
    if not defense.is_file():
        rows = list(csv.DictReader(open(shard / "results.csv", encoding="utf-8")))
        if [r.get("outcome") for r in rows] != ["failed_no_feasible_checkpoint"]:
            raise SystemExit(f"{c} {img} 沒有防禦圖，results.csv 也未記為 failed_no_feasible_checkpoint")
        return None
    f = f"{img}__p{k}.png"
    return defense, {"none": AP / "edits_by_image" / c / img / "defended" / f"ip2p_{c}" / f,
                     "jpeg50": AP / "purified_edits" / c / "jpeg50" / f"ip2p_{c}_jpeg50" / f,
                     "blur2": AP / "purified_edits" / c / "blur2" / f"ip2p_{c}_blur2" / f,
                     "vae_trans": vae_edit(c, f)}


def main():
    images = IMAGES.read_text(encoding="utf-8").split()
    src, prompts, originals = {}, {}, {}
    for r in csv.DictReader(open(ISR_CELLS, encoding="utf-8")):
        if r["condition"] not in REUSED or r["image"] not in images:
            continue
        k = int(r["prompt_id"][1:])
        _, ed = src.setdefault((r["condition"], r["image"], k), (resolve(r["condition_input_png"]), {}))
        ed[r["purification"]] = resolve(r["edit_png"])
        prompts[(r["image"], k)] = r["prompt_text"]
        originals[r["image"]] = resolve(r["original_png"])
    font = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    f_title, f_head, f_id = (ImageFont.truetype(font, s) for s in (17, 16, 14))
    gray = Image.new("RGB", (CELL, CELL), (128, 128, 128))
    ImageDraw.Draw(gray).text((CELL // 2, CELL // 2), "N/A", fill="white", font=f_head, anchor="mm")
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    W, H = LEFT + len(COLS) * (CELL + PAD), TITLE + HEAD + len(CONDS) * (CELL + PAD)
    cells, keys = [], []
    for img in images:
        for k in range(4):
            paths = {}
            for c, _ in CONDS:
                if c in REUSED:
                    if (c, img, k) not in src or [p for p in PURS if p not in src[(c, img, k)][1]]:
                        raise SystemExit(f"ISR cells.csv 缺 {c} {img} p{k}")
                    paths[c] = src[(c, img, k)]
                else:
                    paths[c] = strength_paths(c, img, k)
            prompt = prompts[(img, k)]
            name = f"{img}__p{k}.jpg"
            order = rng.permutation(len(CONDS))
            sheet = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(sheet)
            d.text((8, 12), f'{img} — p{k}: "{prompt}"', fill="black", font=f_title)
            for j, (_, label) in enumerate(COLS):
                d.text((LEFT + j * (CELL + PAD) + CELL // 2, TITLE + HEAD // 2), label, fill="black", font=f_head, anchor="mm")
            for i, ci in enumerate(order):
                c, label = CONDS[ci]
                y = TITLE + HEAD + i * (CELL + PAD)
                d.text((8, y + CELL // 2), f"R{i + 1}", fill="black", font=f_head, anchor="lm")
                keys.append({"sheet": name, "row": f"R{i + 1}", "condition": label,
                             "status": "ok" if paths[c] else "no_feasible_checkpoint"})
                for j, (col, _) in enumerate(COLS):
                    x = LEFT + j * (CELL + PAD)
                    cid = f"C{j + 1}-R{i + 1}"
                    if paths[c] is None:
                        sheet.paste(gray, (x, y))
                    else:
                        inp, ed = paths[c]
                        path = inp if col == "input" else ed[col]
                        sheet.paste(tile(path), (x, y))
                        if col != "input":
                            cells.append({"sheet": name, "cell": cid, "image": img, "prompt_id": f"p{k}", "prompt_text": prompt,
                                          "condition": label, "purification": col, "edit_png": Path(path).as_posix(),
                                          "original_png": originals[img].as_posix(), "condition_input_png": Path(inp).as_posix(),
                                          "undefended_same_purification_png": Path(paths["undefended"][1][col]).as_posix()})
                    box = d.textbbox((x + 4, y + 4), cid, font=f_id)
                    d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill="white")
                    d.text((x + 4, y + 4), cid, fill="black", font=f_id)
            sheet.save(OUT / name, quality=90)
        print(f"[SHEET] {img}", flush=True)
    for fname, data in (("cells.csv", cells), ("key.csv", keys)):
        with open(OUT / fname, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (OUT / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    print(f"[ABLATION STRENGTH CROSSEDITOR100 SHEETS] {len(images) * 4} 張，cells {len(cells)} 列，key {len(keys)} 列，{W}×{H}", flush=True)


if __name__ == "__main__":
    main()
