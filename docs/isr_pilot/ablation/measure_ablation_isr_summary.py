"""Conditional ISR and input naturalness for the ablation conditions Full (ImmuLUT), B2 and B3.

Inputs: judgments.csv (sheet,row,cell,purification,accessory,color_cast,mismatch,degraded,
naturalness,note) in this directory and the blinding key (sheet,row,condition) passed as the
first argument. Protocol: crosseditor100 (100 portraits) x 4 InstructPix2Pix instructions,
purifications none / JPEG 50 / blur sigma 2 / VAE-Trans, main seed.

An edit succeeds when accessory=1, mismatch=0 and degraded=0. A defended cell counts toward
the denominator only when the Undefended edit of the same sheet and purification succeeds;
it is an immunization success when the defended edit does not succeed. Successful defended
edits are further split by color_cast (edit carried out with a remaining color cast).
CIs are image-level bootstrap (1000 resamples, seed 20260812); paired differences use the
same resamples.
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PURIFS = ["none", "jpeg50", "blur2", "vae_trans"]
PROMPTS = {"p0": "sunglasses", "p1": "police suit", "p2": "helmet", "p3": "bowtie"}
CONDS = ["Full", "B2", "B3"]


def main(key_path):
    key = {}
    for r in csv.DictReader(open(key_path)):
        key[(r["sheet"].rsplit(".", 1)[0], r["row"])] = r["condition"]
    cell = {}
    nat = defaultdict(list)
    for r in csv.DictReader(open(HERE / "judgments.csv")):
        cond = key[(r["sheet"], r["row"])]
        if r["cell"] == "C1":
            nat[cond].append(int(r["naturalness"]))
            continue
        good = r["accessory"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0"
        cell[(r["sheet"], r["purification"], cond)] = (good, r["color_cast"] == "1", r["accessory"] == "0")
    sheets = sorted({s for s, _, _ in cell})
    assert len(sheets) == 400, len(sheets)
    assert len(cell) == 400 * 4 * 4, len(cell)

    by_image = defaultdict(lambda: defaultdict(list))
    elig = defaultdict(lambda: [0, 0])
    for s in sheets:
        img, p = s.rsplit("__", 1)
        for pu in PURIFS:
            u = cell[(s, pu, "Undefended")][0]
            for g in (pu, p):
                elig[g][1] += 1
                elig[g][0] += u
            if not u:
                continue
            for c in CONDS:
                v = cell[(s, pu, c)]
                for g in (pu, p, "all") + (("purified",) if pu != "none" else ()):
                    by_image[img][(c, g)].append(v)
    images = sorted(by_image)
    rng = random.Random(20260812)
    boots = [[rng.choice(images) for _ in images] for _ in range(1000)]

    def counts(imgs, c, g, which):
        k = n = 0
        for img in imgs:
            for good, cast, noacc in by_image[img].get((c, g), []):
                n += 1
                k += {"isr": not good, "cast": good and cast, "noacc": noacc}[which]
        return k, n

    def stat(c, g, which="isr"):
        k, n = counts(images, c, g, which)
        vals = sorted(100 * kk / nn if nn else 0 for kk, nn in (counts(b, c, g, which) for b in boots))
        return 100 * k / n if n else float("nan"), vals[25], vals[974], k, n

    def paired(c1, c2, g):
        def d(imgs):
            k1, n1 = counts(imgs, c1, g, "isr")
            k2, n2 = counts(imgs, c2, g, "isr")
            return 100 * (k1 / n1 - k2 / n2)
        vals = sorted(d(b) for b in boots)
        return d(images), vals[25], vals[974]

    print("Undefended success (eligible cells):")
    for g in PURIFS + list(PROMPTS):
        a, b = elig[g]
        print(f"  {PROMPTS.get(g, g):13s} {a}/{b} ({100 * a / b:.1f}%)")
    cols = PURIFS + ["purified", "all"]
    for which, title in (("isr", "Conditional ISR % [95% CI] (blocked/eligible)"),
                         ("noacc", "Accessory absent % of eligible cells"),
                         ("cast", "Edit carried out with a remaining color cast % of eligible cells")):
        print(f"\n{title}")
        print("cond  " + " | ".join(f"{c:>22s}" for c in cols))
        for c in CONDS:
            out = []
            for g in cols:
                m, lo, hi, k, n = stat(c, g, which)
                out.append(f"{m:5.1f} [{lo:4.1f},{hi:4.1f}] ({k}/{n})")
            print(f"{c:5s} " + " | ".join(out))
    print("\nConditional ISR % by instruction (all purifications pooled)")
    for c in CONDS:
        print(f"{c:5s} " + "  ".join(f"{PROMPTS[p]}: {stat(c, p)[0]:5.1f}" for p in PROMPTS))
    print("\nPaired difference in conditional ISR, percentage points [95% CI]")
    for c2 in ("B2", "B3"):
        print(f"Full - {c2:3s} " + "  ".join(
            "{}: {:+.1f} [{:+.1f},{:+.1f}]".format(g, *paired("Full", c2, g)) for g in cols))
    print("\nInput naturalness (1-5), mean over rated rows")
    for c in ["Undefended"] + CONDS:
        v = nat[c]
        print(f"  {c:10s} {sum(v) / len(v):.2f} (n={len(v)})")


if __name__ == "__main__":
    main(sys.argv[1])
