"""Per-instruction conditional ISR of CAPGuard and BlurGuard before and after the human audit.

Same layout as isr_by_prompt.png, restricted to BlurGuard, CAPGuard and their paired difference.
A value that changes after the audit is shown struck through, with the audited value in red below it.
Every audited value carries an image-level bootstrap 95% CI (2000 resamples, seed 20260812); the
difference row uses the paired bootstrap over images.

Usage: python3 generate_isr_audit_comparison_figure.py judgments.csv audit/judgments_human.csv out.png
"""
import collections
import csv
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JUDGE, HUMAN, OUT = sys.argv[1:4]
PURS = ["none", "jpeg50", "blur2", "vae_trans"]
COLS = ["No purification", "JPEG 50", "Gaussian blur σ2", "VAE-Trans", "Purified (3 combined)"]
PROMPTS = [("p0", "Sunglasses"), ("p1", "Police suit"), ("p2", "Helmet"), ("p3", "Bowtie"), (None, "All four instructions")]
RESAMPLES, SEED = 2000, 20260812
RED = "#c00000"


def load(path):
    ok = lambda r: r["accessory"].strip() == "1" and r["mismatch"].strip() == "0" and r["degraded"].strip() == "0"
    und, fail = {}, {}
    for r in csv.DictReader(open(path)):
        image, prompt = r["sheet"].strip().replace(".jpg", "").rsplit("__", 1)
        k = (image, prompt, r["purification"])
        if r["condition"] == "Undefended":
            und[k] = ok(r)
        elif r["condition"] in ("CAPGuard", "BlurGuard"):
            fail[(r["condition"],) + k] = 0 if ok(r) else 1
    return und, fail


def cells(und, prompt, purs):
    return [k for k, v in und.items() if v and k[2] in purs and (prompt is None or k[1] == prompt)]


def boot(by_image, stat):
    images = sorted(by_image)
    rng = random.Random(SEED)
    vals = sorted(stat([by_image[rng.choice(images)] for _ in images]) for _ in range(RESAMPLES))
    return vals[int(0.025 * RESAMPLES)], vals[int(0.975 * RESAMPLES) - 1]


def rate(fail, keys, cond):
    f = sum(fail[(cond,) + k] for k in keys)
    return f, len(keys)


def ci_rate(fail, keys, cond):
    by = collections.defaultdict(lambda: [0, 0])
    for k in keys:
        by[k[0]][0] += fail[(cond,) + k]
        by[k[0]][1] += 1
    return boot(by, lambda s: 100 * sum(a for a, _ in s) / sum(b for _, b in s))


def ci_diff(fail, keys):
    by = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys:
        by[k[0]][0] += fail[("CAPGuard",) + k]
        by[k[0]][1] += fail[("BlurGuard",) + k]
        by[k[0]][2] += 1
    return boot(by, lambda s: 100 * (sum(v[0] for v in s) - sum(v[1] for v in s)) / sum(v[2] for v in s))


judge, human = load(JUDGE), load(HUMAN)
fig, axes = plt.subplots(len(PROMPTS), 1, figsize=(13, 17), facecolor="white")
for ax, (prompt, name) in zip(axes, PROMPTS):
    ax.axis("off")
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4)
    widths = [1.6] + [1.0] * 5
    xs = [0]
    for w in widths:
        xs.append(xs[-1] + w)
    rows = ["BlurGuard", "CAPGuard", "CAPGuard − BlurGuard"]
    heights = [0.55, 1.15, 1.15, 1.15]
    ys = [4]
    for h in heights:
        ys.append(ys[-1] - h)
    for i in range(len(heights)):
        for j in range(len(widths)):
            fc = "#e8e8e8" if i == 0 and j > 0 else ("#fff2cc" if i == 2 else ("#f3f3f3" if i == 3 else "white"))
            if i == 0 and j == 0:
                continue
            ax.add_patch(plt.Rectangle((xs[j], ys[i + 1]), widths[j], heights[i], facecolor=fc, edgecolor="black", lw=0.8))
    for j, c in enumerate(COLS):
        ax.text((xs[j + 1] + xs[j + 2]) / 2, (ys[0] + ys[1]) / 2, c, ha="center", va="center", fontsize=10, weight="bold")
    for i, rname in enumerate(rows):
        ax.text(xs[0] + 0.08, (ys[i + 1] + ys[i + 2]) / 2, rname, ha="left", va="center", fontsize=10)

    for j, purs in enumerate([[p] for p in PURS] + [PURS[1:]]):
        kj, kh = cells(judge[0], prompt, purs), cells(human[0], prompt, purs)
        cx = (xs[j + 1] + xs[j + 2]) / 2
        for i, cond in enumerate(["BlurGuard", "CAPGuard", None]):
            cy = (ys[i + 1] + ys[i + 2]) / 2
            if cond:
                fo, no = rate(judge[1], kj, cond)
                fn, nn = rate(human[1], kh, cond)
                old = f"{100 * fo / no:.1f}% ({fo}/{no})"
                new = f"{100 * fn / nn:.1f}% ({fn}/{nn})"
                lo, hi = ci_rate(human[1], kh, cond)
            else:
                old = f"{100 * (rate(judge[1], kj, 'CAPGuard')[0] - rate(judge[1], kj, 'BlurGuard')[0]) / len(kj):+.1f} pp"
                new = f"{100 * (rate(human[1], kh, 'CAPGuard')[0] - rate(human[1], kh, 'BlurGuard')[0]) / len(kh):+.1f} pp"
                lo, hi = ci_diff(human[1], kh)
            ci = f"[{lo:.1f}, {hi:.1f}]" if cond else f"[{lo:+.1f}, {hi:+.1f}]"
            if old == new:
                ax.text(cx, cy + 0.16, new, ha="center", va="center", fontsize=10)
                ax.text(cx, cy - 0.22, ci, ha="center", va="center", fontsize=9, color="#444444")
            else:
                t = ax.text(cx, cy + 0.3, old, ha="center", va="center", fontsize=9.5, color="#555555")
                fig.canvas.draw()
                bb = t.get_window_extent().transformed(ax.transData.inverted())
                ax.plot([bb.x0, bb.x1], [cy + 0.3, cy + 0.3], color=RED, lw=1.2)
                ax.text(cx, cy - 0.02, new, ha="center", va="center", fontsize=10, color=RED, weight="bold")
                ax.text(cx, cy - 0.33, ci, ha="center", va="center", fontsize=9, color=RED)

    def und_counts(u):
        return [sum(1 for k, v in u.items() if v and k[2] == q and (prompt is None or k[1] == prompt)) for q in PURS]

    total = 100 if prompt else 400
    uo, un = und_counts(judge[0]), und_counts(human[0])
    counts = ", ".join(f"{b}/{total}" for b in un)
    changed = uo != un
    ax.text(0, 4.15, f"{name}  —  undefended edit success: {counts} (none / JPEG / blur / VAE-Trans)", fontsize=11,
            ha="left", va="bottom", color=RED if changed else "black")
    if changed:
        ax.text(6.6, 4.15, "before audit: " + ", ".join(f"{a}/{total}" for a in uo), fontsize=9, ha="right",
                va="bottom", color="#555555")

fig.suptitle("Conditional ISR on InstructPix2Pix, CAPGuard vs BlurGuard, after the human audit of 606 cells\n"
             "(100 Pexels portraits, seed 20260812; red = value changed by the audit; brackets = image-level bootstrap 95% CI, "
             "paired for the difference)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
plt.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
