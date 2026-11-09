"""Candidate qualitative cases for the edit-result LPIPS versus ISR analysis (ip2p ISR pilot).

Two groups per method (CAPGuard and BlurGuard by default), eligible cells only (undefended edit succeeds):
  high_lpips_edit_succeeded  defense failed (edit judged successful), highest RGB edit-result LPIPS
  blocked                    defense succeeded, highest grayscale edit-result LPIPS
Each row: original | undefended edit (same purification) | defended edit, annotated with method,
purification, instruction, RGB and grayscale LPIPS and the labels. Writes <out>.jpg and <out>.csv.

Usage: python3 generate_isr_lpips_case_sheet.py lpips_gray.csv judgments_human.csv cells.csv <baseline root> <out> [k]
"""
import csv
import os
import sys

from PIL import Image, ImageDraw, ImageFont

LPIPS_CSV, JUDGMENTS, CELLS, ROOT, OUT = sys.argv[1:6]
K = int(sys.argv[6]) if len(sys.argv) > 6 else 6
METHODS = ["CAPGuard", "BlurGuard"]
T = 256

ok = lambda r: r["accessory"].strip() == "1" and r["mismatch"].strip() == "0" and r["degraded"].strip() == "0"
lab, und = {}, {}
for r in csv.DictReader(open(JUDGMENTS)):
    s = r["sheet"].strip().replace(".jpg", "")
    lab[(s, r["cell"].strip())] = r
    if r["condition"] == "Undefended":
        und[(s, r["purification"])] = ok(r)
paths = {(r["sheet"].replace(".jpg", ""), r["cell"]): r for r in csv.DictReader(open(CELLS))}

rows = []
for r in csv.DictReader(open(LPIPS_CSV)):
    k = (r["sheet"], r["cell"])
    j = lab[k]
    if j["condition"] not in METHODS or not und[(r["sheet"], r["purification"])]:
        continue
    rows.append(dict(sheet=r["sheet"], cell=r["cell"], method=j["condition"], purification=r["purification"],
                     rgb=float(r["lpips_rgb"]), gray=float(r["lpips_gray"]), blocked=0 if ok(j) else 1,
                     labels=f"acc{j['accessory']} mis{j['mismatch']} deg{j['degraded']}", note=j["note"],
                     prompt=paths[k]["prompt_text"], edit=paths[k]["edit_png"],
                     ref=paths[k]["undefended_same_purification_png"], orig=paths[k]["original_png"]))

picked = []
for m in METHODS:
    a = sorted([r for r in rows if r["method"] == m and not r["blocked"]], key=lambda r: -r["rgb"])[:K]
    b = sorted([r for r in rows if r["method"] == m and r["blocked"]], key=lambda r: -r["gray"])[:K]
    picked += [dict(r, group="high_lpips_edit_succeeded") for r in a] + [dict(r, group="blocked") for r in b]

font = ImageFont.load_default()
sheet = Image.new("RGB", (3 * T + 330, len(picked) * T), "white")
for i, r in enumerate(picked):
    for x, p in enumerate([r["orig"], r["ref"], r["edit"]]):
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        sheet.paste(Image.open(full).convert("RGB").resize((T, T), Image.BICUBIC), (x * T, i * T))
    d = ImageDraw.Draw(sheet)
    txt = [f"#{i + 1} {r['method']} | {r['group']}", f"{r['sheet']} {r['cell']}", f"purification: {r['purification']}",
           r["prompt"], f"LPIPS RGB {r['rgb']:.3f}", f"LPIPS gray {r['gray']:.3f}", f"gray/RGB {r['gray'] / r['rgb']:.2f}",
           r["labels"] + (" " + r["note"][:40] if r["note"] else "")]
    for n, t in enumerate(txt):
        d.text((3 * T + 10, i * T + 12 + 26 * n), t, fill="black", font=font)
sheet.save(OUT + ".jpg", quality=88)
with open(OUT + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group", "method", "sheet", "cell", "purification", "prompt", "rgb", "gray",
                                      "blocked", "labels", "note", "orig", "ref", "edit"])
    w.writeheader()
    for r in picked:
        w.writerow({k: r[k] for k in w.fieldnames})
print("saved", OUT + ".jpg", len(picked), "rows")
