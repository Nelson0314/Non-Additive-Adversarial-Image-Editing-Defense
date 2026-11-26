"""Conditional ISR for the SD Inpainting edits (306 Pexels portraits x 4 prompts).

Inputs: judgments.csv (sheet,cell,purification,content,mismatch,degraded,note) in this
directory and the blinding key (sheet,row,condition) passed as the first argument.
A cell counts toward the denominator only when the Undefended edit of the same sheet and
purification succeeds (content=1, mismatch=0, degraded=0); a defended cell is an
immunization success when any of the three labels fails. CIs are image-level bootstrap
(1000 resamples, seed 20260812).
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PURIFS = ["none", "jpeg50", "blur2", "vae_trans"]
PROMPTS = {"p0": "hospital", "p1": "gym", "p2": "snowy street", "p3": "dog"}


def main(key_path, review=None):
    key = {}
    for r in csv.DictReader(open(key_path)):
        key[(r["sheet"].rsplit(".", 1)[0], r["row"])] = r["condition"]
    ok = {}
    for r in csv.DictReader(open(HERE / "judgments.csv")):
        row = r["cell"].split("-")[1]
        cond = key[(r["sheet"], row)]
        good = r["content"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0"
        ok[(r["sheet"], r["purification"], cond)] = good
    sheets = sorted({s for s, _, _ in ok})
    conds = sorted({c for _, _, c in ok} - {"Undefended"})
    assert len(sheets) == 1224, len(sheets)
    removed = set()
    if review is not None:
        # Same protocol as the InstructPix2Pix ISR: reviewed cells take the human label, and slots whose
        # undefended reference the reviewer marks as failed are removed from the denominator.
        answers_path, review_key_path = review
        rkey = {r["item"]: r for r in csv.DictReader(open(review_key_path))}
        for a in csv.DictReader(open(answers_path)):
            k = rkey[a["item"]]
            if a["undefended_success"] != "1":
                removed.add((k["sheet"], k["purification"]))
                continue
            human_ok = a["content"] == "1" and a["mismatch"] == "0" and a["degraded"] == "0"
            ok[(k["sheet"], k["purification"], k["condition"])] = human_ok
        print(f"Human review applied: {len(rkey) - len(removed)} cells relabelled, {len(removed)} slots removed\n")

    def isr(cells):
        n = len(cells)
        k = sum(1 for c in cells if not c)
        return k, n

    by_image = defaultdict(lambda: defaultdict(list))  # image -> (cond, purif) -> successes
    elig = defaultdict(lambda: [0, 0])
    for s in sheets:
        img, p = s.rsplit("__", 1)
        for pu in PURIFS:
            u = ok[(s, pu, "Undefended")] and (s, pu) not in removed
            elig[pu][1] += 1
            elig[pu][0] += u
            elig[p][1] += 1
            elig[p][0] += u
            if not u:
                continue
            for c in conds:
                by_image[img][(c, pu)].append(ok[(s, pu, c)])
                by_image[img][(c, p)].append(ok[(s, pu, c)])
                by_image[img][(c, "all")].append(ok[(s, pu, c)])
                if pu != "none":
                    by_image[img][(c, "purified")].append(ok[(s, pu, c)])
    images = sorted(by_image)
    rng = random.Random(20260812)
    boots = [[rng.choice(images) for _ in images] for _ in range(1000)]

    def stat(c, key):
        k = n = 0
        for img in images:
            v = by_image[img].get((c, key), [])
            k += sum(1 for x in v if not x)
            n += len(v)
        vals = []
        for b in boots:
            kk = nn = 0
            for img in b:
                v = by_image[img].get((c, key), [])
                kk += sum(1 for x in v if not x)
                nn += len(v)
            vals.append(100 * kk / nn if nn else 0)
        vals.sort()
        return 100 * k / n if n else float("nan"), vals[25], vals[974], k, n

    print("Undefended success (eligible cells):")
    for g in PURIFS + list(PROMPTS):
        a, b = elig[g]
        print(f"  {PROMPTS.get(g, g):13s} {a}/{b} ({100 * a / b:.1f}%)")
    cols = PURIFS + ["purified", "all"]
    print("\nConditional ISR % [95% CI] (fails/eligible)")
    print("method      " + " | ".join(f"{c:>22s}" for c in cols))
    for c in conds:
        cells = []
        for col in cols:
            m, lo, hi, k, n = stat(c, col)
            cells.append(f"{m:5.1f} [{lo:4.1f},{hi:4.1f}] ({k}/{n})")
        print(f"{c:11s} " + " | ".join(cells))
    print("\nISR % by prompt (all purifications pooled)")
    for c in conds:
        print(f"{c:11s} " + "  ".join(f"{PROMPTS[p]}: {stat(c, p)[0]:5.1f}" for p in PROMPTS))


if __name__ == "__main__":
    # Optional: <answers.csv> <review key.csv> applies the human review (human_review/README.md).
    main(sys.argv[1], tuple(sys.argv[2:4]) if len(sys.argv) >= 4 else None)
