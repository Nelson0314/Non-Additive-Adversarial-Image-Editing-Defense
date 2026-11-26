"""Conditional ISR for the transfer editors VIBE-DistilledCFG and FLUX.2 [klein].

Usage: measure_transfer_isr_summary.py <editor_dir> <key.csv>, where <editor_dir> holds judgments.csv
(sheet,row,cell,purification,accessory,color_cast,mismatch,degraded,note). Protocol: 50 Pexels portraits
x 4 instructions, purifications none / JPEG 50 / blur sigma 2 / VAE-Trans, main seed.

An edit succeeds when accessory=1, mismatch=0 and degraded=0. A defended cell counts toward the
denominator only when the Undefended edit of the same sheet and purification succeeds; it is an
immunization success when the defended edit does not succeed. CIs are image-level bootstrap
(1000 resamples, seed 20260812).
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

PURIFS = ["none", "jpeg50", "blur2", "vae_trans"]
PROMPTS = {"p0": "sunglasses", "p1": "police suit", "p2": "helmet", "p3": "bowtie"}
# With --exclude-mismatch, a defended cell labelled mismatch=1 is removed from the denominator.
EXCLUDE_MISMATCH = "--exclude-mismatch" in sys.argv


def main(editor_dir, key_path):
    key = {}
    for r in csv.DictReader(open(key_path)):
        key[(r["sheet"].rsplit(".", 1)[0], r["row"])] = r["condition"]
    ok = {}
    for r in csv.DictReader(open(Path(editor_dir) / "judgments.csv")):
        cond = key[(r["sheet"], r["row"])]
        good = r["accessory"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0"
        if EXCLUDE_MISMATCH and cond != "Undefended" and r["mismatch"] == "1":
            good = None
        ok[(r["sheet"], r["purification"], cond)] = good
    sheets = sorted({s for s, _, _ in ok})
    conds = sorted({c for _, _, c in ok} - {"Undefended"})
    assert len(sheets) == 200, len(sheets)
    assert len(ok) == 200 * 4 * 6, len(ok)

    by_image = defaultdict(lambda: defaultdict(list))
    elig = defaultdict(lambda: [0, 0])
    for s in sheets:
        img, p = s.rsplit("__", 1)
        for pu in PURIFS:
            u = ok[(s, pu, "Undefended")]
            for g in (pu, p):
                elig[g][1] += 1
                elig[g][0] += u
            if not u:
                continue
            for c in conds:
                if ok[(s, pu, c)] is None:
                    continue
                for g in (pu, p, "all") + (("purified",) if pu != "none" else ()):
                    by_image[img][(c, g)].append(ok[(s, pu, c)])
    images = sorted(by_image)
    rng = random.Random(20260812)
    boots = [[rng.choice(images) for _ in images] for _ in range(1000)]

    def counts(imgs, c, g):
        k = n = 0
        for img in imgs:
            v = by_image[img].get((c, g), [])
            k += sum(1 for x in v if not x)
            n += len(v)
        return k, n

    def stat(c, g):
        k, n = counts(images, c, g)
        vals = sorted(100 * kk / nn if nn else 0 for kk, nn in (counts(b, c, g) for b in boots))
        return 100 * k / n if n else float("nan"), vals[25], vals[974], k, n

    print("Undefended success (eligible cells):")
    for g in PURIFS + list(PROMPTS):
        a, b = elig[g]
        print(f"  {PROMPTS.get(g, g):13s} {a}/{b} ({100 * a / b:.1f}%)")
    cols = PURIFS + ["purified", "all"]
    print("\nConditional ISR % [95% CI] (blocked/eligible)")
    print("method      " + " | ".join(f"{c:>22s}" for c in cols))
    for c in conds:
        out = []
        for g in cols:
            m, lo, hi, k, n = stat(c, g)
            out.append(f"{m:5.1f} [{lo:4.1f},{hi:4.1f}] ({k}/{n})")
        print(f"{c:11s} " + " | ".join(out))
    print("\nConditional ISR % by instruction (all purifications pooled)")
    for c in conds:
        print(f"{c:11s} " + "  ".join(f"{PROMPTS[p]}: {stat(c, p)[0]:5.1f}" for p in PROMPTS))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
