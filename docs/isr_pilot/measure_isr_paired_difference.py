"""Paired ISR difference between CAPGuard and each baseline on the ip2p crosseditor100 judgments.

Eligible cells follow measure_isr_summary.py: same image, prompt and purification where the
Undefended edit is judged successful. For each baseline X the difference ISR(CAPGuard) - ISR(X)
is computed on the identical eligible cells; the 95% CI comes from an image-level paired
bootstrap (2000 resamples, seed 20260812). Discordant counts are also reported.

Usage: python3 measure_isr_paired_difference.py [judgments.csv]
"""
import collections
import csv
import os
import random
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "judgments.csv")
BASELINES = ["BlurGuard", "Mist", "SIFM", "PhotoGuard", "DCT-Shield"]
PURIFICATIONS = ["none", "jpeg50", "blur2", "vae_trans"]
PURIFIED = ["jpeg50", "blur2", "vae_trans"]
RESAMPLES = 2000
SEED = 20260812


def success(row):
    return row["accessory"].strip() == "1" and row["mismatch"].strip() == "0" and row["degraded"].strip() == "0"


rows = {}
for r in csv.DictReader(open(PATH)):
    rows[(r["sheet"].strip(), r["cell"].strip())] = r
sheet_counts = collections.Counter(s for s, _ in rows)
if len(rows) != 11200 or any(n != 28 for n in sheet_counts.values()):
    raise SystemExit(f"unexpected judgments shape: {len(rows)} cells, {len(sheet_counts)} sheets")

fail = {}  # (condition, image, prompt, purification) -> 1 if defense succeeded (edit failed)
undefended_ok = {}
for (sheet, _), r in rows.items():
    image, prompt = sheet.rsplit("__", 1)
    key = (image, prompt, r["purification"])
    if r["condition"] == "Undefended":
        undefended_ok[key] = success(r)
    else:
        fail[(r["condition"],) + key] = 0 if success(r) else 1
eligible = [k for k, v in undefended_ok.items() if v]


def paired(baseline, purifications):
    keys = [k for k in eligible if k[2] in purifications]
    by_image = collections.defaultdict(lambda: [0, 0, 0])  # cap fails, baseline fails, n
    b = c = 0
    for k in keys:
        cap, oth = fail[("CAPGuard",) + k], fail[(baseline,) + k]
        acc = by_image[k[0]]
        acc[0] += cap
        acc[1] += oth
        acc[2] += 1
        b += cap and not oth
        c += oth and not cap
    images = sorted(by_image)
    n = sum(v[2] for v in by_image.values())
    point = 100 * (sum(v[0] for v in by_image.values()) - sum(v[1] for v in by_image.values())) / n
    rng = random.Random(SEED)
    boots = []
    for _ in range(RESAMPLES):
        sample = [by_image[rng.choice(images)] for _ in images]
        m = sum(v[2] for v in sample)
        boots.append(100 * (sum(v[0] for v in sample) - sum(v[1] for v in sample)) / m)
    boots.sort()
    lo, hi = boots[int(0.025 * RESAMPLES)], boots[int(0.975 * RESAMPLES) - 1]
    share_positive = sum(x > 0 for x in boots) / RESAMPLES
    return point, lo, hi, share_positive, b, c, n, len(images)


print(f"judgments: {PATH}; eligible cells: {len(eligible)}; resamples {RESAMPLES}, seed {SEED}")
print("ISR(CAPGuard) - ISR(X), percentage points [95% paired bootstrap CI]; "
      "P(diff>0) over resamples; CAP-only defense successes / X-only defense successes; cells; images")
for baseline in BASELINES:
    print(f"\nvs {baseline}")
    for label, purs in [(p, [p]) for p in PURIFICATIONS] + [("purified (jpeg50+blur2+vae_trans)", PURIFIED)]:
        d, lo, hi, pos, b, c, n, ni = paired(baseline, purs)
        print(f"  {label:34s} {d:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]  P>0 {pos:.3f}  {b}/{c}  n={n}  images={ni}")
