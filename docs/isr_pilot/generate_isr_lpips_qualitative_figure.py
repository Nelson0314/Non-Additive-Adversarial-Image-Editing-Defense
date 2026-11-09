"""Qualitative figure: does CAPGuard's edit-result LPIPS mostly reflect color?

Selects CAPGuard cells of the ip2p ISR pilot (eligible cells only, labels after the human audit):
  edit succeeded, high LPIPS  defended edit judged successful; highest RGB edit-result LPIPS among cells
                              whose grayscale/RGB ratio is below 0.5; one cell per image
  edit blocked                accessory judged absent; highest grayscale edit-result LPIPS; one cell per image
Columns: original | undefended edit | CAPGuard edit | the two edits in grayscale.
Each row is annotated with instruction, purification, RGB and grayscale edit-result LPIPS.

Usage: python3 generate_isr_lpips_qualitative_figure.py lpips_gray.csv judgments_human.csv cells.csv <baseline root> <out.png> [k] [sheet:purification,...]
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LPIPS_CSV, JUDGMENTS, CELLS, ROOT, OUT = sys.argv[1:6]
K = int(sys.argv[6]) if len(sys.argv) > 6 else 3
EXCLUDE = set(sys.argv[7].split(",")) if len(sys.argv) > 7 else set()  # "<sheet>:<purification>" cells to skip
PUR = {"none": "no purification", "jpeg50": "JPEG 50", "blur2": "blur σ2", "vae_trans": "VAE-Trans"}

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
    if j["condition"] != "CAPGuard" or not und[(r["sheet"], r["purification"])] or f"{r['sheet']}:{r['purification']}" in EXCLUDE:
        continue
    rgb, gray = float(r["lpips_rgb"]), float(r["lpips_gray"])
    rows.append(dict(image=r["image"], pur=r["purification"], rgb=rgb, gray=gray, success=ok(j),
                     accessory=j["accessory"].strip(), degraded=j["degraded"].strip(), audited=j["note"].startswith("human:"), p=paths[k]))


def pick(cands, key):
    out, seen = [], set()
    for r in sorted(cands, key=key):
        if r["image"] not in seen:
            out.append(r)
            seen.add(r["image"])
        if len(out) == K:
            break
    return out


succ = pick([r for r in rows if r["audited"] and r["success"] and r["gray"] / r["rgb"] < 0.5], key=lambda r: -r["rgb"])
blk = pick([r for r in rows if r["audited"] and r["accessory"] == "0" and r["degraded"] == "0" and r["gray"] / r["rgb"] >= 0.55], key=lambda r: -r["gray"])


def load(rel, gray=False):
    im = Image.open(rel if os.path.isabs(rel) else os.path.join(ROOT, rel)).convert("RGB").resize((256, 256), Image.BICUBIC)
    if gray:
        im = im.convert("L").convert("RGB")
    return np.asarray(im)


cols = ["Original", "Undefended", "CAPGuard", "Undefended (gray)", "CAPGuard (gray)"]
n = len(succ) + len(blk)
# grid rows: header (a), succ rows, header (b), blk rows
ratios = [0.18] + [1] * len(succ) + [0.18] + [1] * len(blk)
fig = plt.figure(figsize=(12.5, 2.75 * n + 1.2), facecolor="white")
gs = fig.add_gridspec(len(ratios), 5, height_ratios=ratios, hspace=0.35, wspace=0.05)
row_of = [1 + i for i in range(len(succ))] + [2 + len(succ) + i for i in range(len(blk))]
axes = np.array([[fig.add_subplot(gs[row_of[i], c]) for c in range(5)] for i in range(n)])
for hdr, text in [(0, "(a) Edit succeeded, high LPIPS"), (1 + len(succ), "(b) Edit blocked")]:
    ax = fig.add_subplot(gs[hdr, :])
    ax.axis("off")
    ax.text(0, 0.0, text, fontsize=11, weight="bold", ha="left", va="bottom", transform=ax.transAxes)
for i, (group, r) in enumerate([("succ", r) for r in succ] + [("blk", r) for r in blk]):
    imgs = [load(r["p"]["original_png"]), load(r["p"]["undefended_same_purification_png"]), load(r["p"]["edit_png"]),
            load(r["p"]["undefended_same_purification_png"], True), load(r["p"]["edit_png"], True)]
    for c, im in enumerate(imgs):
        ax = axes[i, c]
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i in (0, len(succ)):
            ax.set_title(cols[c], fontsize=10)
    axes[i, 2].set_xlabel(f"LPIPS {r['rgb']:.2f}", fontsize=9)
    axes[i, 4].set_xlabel(f"LPIPS {r['gray']:.2f}", fontsize=9)
    instr = r["p"]["prompt_text"].replace("Let the person wear ", "")
    axes[i, 0].set_ylabel(f"{instr}\n{PUR[r['pur']]}", fontsize=9)
plt.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
with open(os.path.splitext(OUT)[0] + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "image", "prompt", "purification", "lpips_rgb", "lpips_gray", "human_audited", "edit_png"])
    for g, rs in [("edit_succeeded_high_lpips", succ), ("edit_blocked", blk)]:
        for r in rs:
            w.writerow([g, r["image"], r["p"]["prompt_text"], r["pur"], f"{r['rgb']:.4f}", f"{r['gray']:.4f}",
                        int(r["audited"]), r["p"]["edit_png"]])
print("saved", OUT)
