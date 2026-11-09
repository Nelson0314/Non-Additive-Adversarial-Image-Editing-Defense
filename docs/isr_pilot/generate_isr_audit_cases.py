"""Select and render the human-audit cases for the ip2p crosseditor100 ISR judgments.

Two sets:
  discordant  every eligible cell (Undefended judged successful) where CAPGuard and BlurGuard
              receive different success labels, in all four purifications; both cells are audited.
  random      a stratified random sample of all judged cells: PER_STRATUM cells for each
              condition x purification (7 x 4 strata, Undefended included), seed 20260812.

Each case is rendered blind as one panel: original | Undefended edit under the same purification |
audited edit, with the prompt text; the condition name and the judge labels are not shown.
Outputs in --out-dir: panels/<case_id>.jpg, audit_cases.csv (case_id, set, prompt) and
audit_key.csv (case_id, sheet, cell, condition, purification, judge labels, paths).

Usage: python3 generate_isr_audit_cases.py --judgments judgments.csv \
         --cells-csv <repo>/baseline/artifacts/pexels_portraits/ip2p_isr100_sheets/cells.csv \
         --repo-root <repo>/baseline --out-dir <dir>
"""
import argparse
import collections
import csv
import os
import random

from PIL import Image, ImageDraw, ImageFont

SEED = 20260812
PER_STRATUM = 5
TILE = 384

ap = argparse.ArgumentParser()
ap.add_argument("--judgments", required=True)
ap.add_argument("--cells-csv", required=True)
ap.add_argument("--repo-root", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()


def success(r):
    return r["accessory"].strip() == "1" and r["mismatch"].strip() == "0" and r["degraded"].strip() == "0"


judg = {(r["sheet"].strip(), r["cell"].strip()): r for r in csv.DictReader(open(args.judgments))}
cells = {(r["sheet"].removesuffix(".jpg"), r["cell"]): r for r in csv.DictReader(open(args.cells_csv))}
missing = [k for k in judg if k not in cells]
if missing:
    raise SystemExit(f"{len(missing)} judged cells absent from cells.csv, e.g. {missing[:3]}")

by_slot = {}  # (sheet, purification, condition) -> key
for k, r in judg.items():
    by_slot[(k[0], r["purification"], r["condition"])] = k

selected = collections.OrderedDict()  # key -> set name
for (sheet, pur, cond), k in sorted(by_slot.items()):
    if cond != "CAPGuard":
        continue
    und = judg[by_slot[(sheet, pur, "Undefended")]]
    blur_key = by_slot[(sheet, pur, "BlurGuard")]
    if success(und) and success(judg[k]) != success(judg[blur_key]):
        selected[k] = "discordant"
        selected[blur_key] = "discordant"

rng = random.Random(SEED)
strata = collections.defaultdict(list)
for k, r in sorted(judg.items()):
    strata[(r["condition"], r["purification"])].append(k)
for stratum in sorted(strata):
    pool = [k for k in strata[stratum] if k not in selected]
    for k in rng.sample(pool, PER_STRATUM):
        selected[k] = "random"

keys = list(selected)
rng.shuffle(keys)
os.makedirs(os.path.join(args.out_dir, "panels"), exist_ok=True)
font = ImageFont.load_default()


def load(rel):
    path = os.path.join(args.repo_root, rel)
    if not os.path.exists(path):
        raise SystemExit(f"missing image {path}")
    return Image.open(path).convert("RGB").resize((TILE, TILE), Image.BICUBIC)


with open(os.path.join(args.out_dir, "audit_cases.csv"), "w", newline="") as fc, \
        open(os.path.join(args.out_dir, "audit_key.csv"), "w", newline="") as fk:
    wc = csv.writer(fc)
    wk = csv.writer(fk)
    wc.writerow(["case_id", "set", "prompt"])
    wk.writerow(["case_id", "set", "sheet", "cell", "condition", "purification", "accessory", "mismatch",
                 "degraded", "note", "edit_png", "undefended_same_purification_png", "original_png"])
    for i, k in enumerate(keys, 1):
        cid = f"case_{i:03d}"
        j, c = judg[k], cells[k]
        panel = Image.new("RGB", (3 * TILE, TILE + 28), "white")
        for x, rel in enumerate([c["original_png"], c["undefended_same_purification_png"], c["edit_png"]]):
            panel.paste(load(rel), (x * TILE, 28))
        ImageDraw.Draw(panel).text((6, 8), f"{cid}   {c['prompt_text']}   [original | reference edit | audited edit]",
                                   fill="black", font=font)
        panel.save(os.path.join(args.out_dir, "panels", f"{cid}.jpg"), quality=88)
        wc.writerow([cid, selected[k], c["prompt_text"]])
        wk.writerow([cid, selected[k], k[0], k[1], j["condition"], j["purification"], j["accessory"],
                     j["mismatch"], j["degraded"], j["note"], c["edit_png"],
                     c["undefended_same_purification_png"], c["original_png"]])

counts = collections.Counter(selected.values())
print(f"cases {len(keys)}: discordant {counts['discordant']} (= {counts['discordant'] // 2} CAPGuard/BlurGuard pairs), "
      f"random {counts['random']}; out {args.out_dir}")
