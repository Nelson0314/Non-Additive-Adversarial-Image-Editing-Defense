"""Relation between edit-result LPIPS (RGB and grayscale) and defense success on the ip2p ISR pilot.

Inputs: lpips_gray.csv (measure_isr_lpips_grayscale.py) and a judgments CSV (judgments.csv or
audit/judgments_human.csv). Only eligible cells are used: the undefended edit under the same image,
instruction and purification is judged successful. Defense success = the defended edit is not a success
(accessory missing, mismatch or degraded).

Outputs: <out>.png (four panels) and <out>.txt (per method and purification: cell counts, defense success
rate, mean LPIPS in RGB and grayscale for blocked and not-blocked cells, gray/RGB ratio, AUC of LPIPS for
separating blocked from not-blocked cells).

Usage: python3 generate_isr_lpips_correlation_figure.py lpips_gray.csv judgments_human.csv <out prefix>
"""
import collections
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LPIPS_CSV, JUDGMENTS, OUT = sys.argv[1:4]
METHODS = ["CAPGuard", "BlurGuard", "Mist", "SIFM", "PhotoGuard", "DCT-Shield"]
ADDITIVE = ["Mist", "SIFM", "PhotoGuard", "DCT-Shield"]
PURS = ["none", "jpeg50", "blur2", "vae_trans"]
# Okabe-Ito order (colorblind-safe): CAPGuard, BlurGuard, additive pool
COLOR = {"CAPGuard": "#D55E00", "BlurGuard": "#0072B2", "additive": "#7f7f7f"}
GROUPS = [("CAPGuard", ["CAPGuard"]), ("BlurGuard", ["BlurGuard"]), ("additive", ADDITIVE)]
LABEL = {"CAPGuard": "CAPGuard", "BlurGuard": "BlurGuard", "additive": "Additive (4 methods)"}

ok = lambda r: r["accessory"].strip() == "1" and r["mismatch"].strip() == "0" and r["degraded"].strip() == "0"
labels, und = {}, {}
for r in csv.DictReader(open(JUDGMENTS)):
    s = r["sheet"].strip().replace(".jpg", "")
    labels[(s, r["cell"].strip())] = r
    if r["condition"] == "Undefended":
        und[(s, r["purification"])] = ok(r)

cells = []  # (method, purification, rgb, gray, blocked)
for r in csv.DictReader(open(LPIPS_CSV)):
    j = labels[(r["sheet"], r["cell"])]
    if j["purification"] != r["purification"]:
        raise SystemExit(f"purification mismatch at {r['sheet']} {r['cell']}")
    if not und[(r["sheet"], r["purification"])]:
        continue
    cells.append((j["condition"], r["purification"], float(r["lpips_rgb"]), float(r["lpips_gray"]), 0 if ok(j) else 1))
if collections.Counter(c[0] for c in cells).keys() != set(METHODS):
    raise SystemExit(f"unexpected methods {collections.Counter(c[0] for c in cells)}")


def auc(pos, neg):
    """P(LPIPS of a blocked cell > LPIPS of a not-blocked cell), ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


lines = ["edit-result LPIPS (piq.LPIPS, defended edit vs undefended edit under the same purification) and defense success",
         f"judgments: {JUDGMENTS}; eligible cells only; AUC = P(LPIPS of blocked cell > LPIPS of not-blocked cell)", ""]
hdr = f"{'method':11s} {'purif':9s} {'n':>5s} {'ISR%':>6s} | {'RGB blk':>8s} {'RGB ok':>8s} {'AUC':>5s} | {'gray blk':>8s} {'gray ok':>8s} {'AUC':>5s} | {'gray/RGB':>8s}"
lines.append(hdr)
for m in METHODS:
    for p in PURS + ["purified", "all"]:
        sel = [c for c in cells if c[0] == m and (p == "all" or (p == "purified" and c[1] != "none") or c[1] == p)]
        rb = np.array([c[2] for c in sel if c[4]]); ro = np.array([c[2] for c in sel if not c[4]])
        gb = np.array([c[3] for c in sel if c[4]]); go = np.array([c[3] for c in sel if not c[4]])
        rgb = np.array([c[2] for c in sel]); gr = np.array([c[3] for c in sel])
        f = lambda a: f"{a.mean():8.3f}" if len(a) else f"{'-':>8s}"
        lines.append(f"{m:11s} {p:9s} {len(sel):5d} {100 * len(rb) / len(sel):6.1f} | {f(rb)} {f(ro)} {auc(rb, ro):5.2f} | "
                     f"{f(gb)} {f(go)} {auc(gb, go):5.2f} | {gr.mean() / rgb.mean():8.3f}")
    lines.append("")
open(OUT + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor="white")
all_rgb = np.array([c[2] for c in cells]); all_gray = np.array([c[3] for c in cells])
for ax, idx, name, pooled in [(axes[0, 0], 2, "RGB", all_rgb), (axes[0, 1], 3, "grayscale", all_gray)]:
    edges = np.quantile(pooled, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    for g, members in GROUPS:
        sel = [c for c in cells if c[0] in members]
        xs, ys, ns = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            inb = [c for c in sel if a <= c[idx] < b]
            if len(inb) >= 15:
                xs.append(np.mean([c[idx] for c in inb])); ys.append(100 * np.mean([c[4] for c in inb])); ns.append(len(inb))
        ax.plot(xs, ys, "-o", color=COLOR[g], lw=2, ms=6, label=LABEL[g])
    ax.set_xlabel(f"edit LPIPS ({name})")
    ax.set_ylabel("defense success rate (%)")
    ax.set_title(f"({'a' if idx == 2 else 'b'}) {name.capitalize() if name != 'RGB' else name}", loc="left", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1, 0]
for g, members in [("additive", ADDITIVE), ("BlurGuard", ["BlurGuard"]), ("CAPGuard", ["CAPGuard"])]:
    sel = [c for c in cells if c[0] in members]
    for blocked, marker in [(0, "o"), (1, "x")]:
        pts = [c for c in sel if c[4] == blocked]
        ax.scatter([c[2] for c in pts], [c[3] for c in pts], s=9 if marker == "o" else 16, marker=marker,
                   color=COLOR[g], alpha=0.35 if marker == "o" else 0.8, lw=0.8,
                   label=f"{g if g != 'additive' else 'Additive'}, {'blocked' if blocked else 'not blocked'}")
lim = max(all_rgb.max(), all_gray.max()) * 1.03
ax.plot([0, lim], [0, lim], color="#333333", lw=0.8, ls="--")
ax.text(lim * 0.97, lim * 0.93, "gray = RGB", ha="right", fontsize=8, color="#333333")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("edit LPIPS (RGB)"); ax.set_ylabel("edit LPIPS (grayscale)")
ax.set_title("(c) Grayscale vs RGB, per cell", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1, markerscale=1.5)
ax.grid(color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 1]
x = np.arange(len(METHODS)); wdt = 0.36
for k, (idx, name, hatch) in enumerate([(2, "RGB", None), (3, "grayscale", "//")]):
    vals = []
    for m in METHODS:
        sel = [c for c in cells if c[0] == m]
        vals.append(auc(np.array([c[idx] for c in sel if c[4]]), np.array([c[idx] for c in sel if not c[4]])))
    bars = ax.bar(x + (k - 0.5) * wdt, vals, wdt, color=[COLOR.get(m, COLOR["additive"]) for m in METHODS],
                  alpha=1.0 if k == 0 else 0.55, hatch=hatch, edgecolor="white", label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.5, color="#333333", lw=0.8, ls="--")
ax.text(len(METHODS) - 0.5, 0.51, "chance", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(METHODS)
ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUC")
ax.set_title("(d) AUC (solid RGB, hatched gray)", loc="left", fontsize=11)
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)

n_by = collections.Counter(c[0] for c in cells)
fig.suptitle("Edit LPIPS vs defense success (ip2p, 100 portraits x 4 instructions x 4 purifications)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT + ".png", dpi=150, facecolor="white")
print("saved", OUT + ".png")
