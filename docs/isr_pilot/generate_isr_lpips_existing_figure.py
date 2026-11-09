"""Edit-result LPIPS versus defense success on the ip2p ISR pilot, using existing per-cell LPIPS only.

Existing per-cell edit-result LPIPS (`disp_lpips`, piq.LPIPS, defended edit vs undefended edit) is
collected from every displacement CSV under the given roots and matched to the ISR cells through the
defended edit path (`defended_png` == cells.csv `edit_png`). Only the no-purification column has paths
in the existing CSVs, so the figure covers purification = none. Cells are eligible when the undefended
edit is judged successful; defense success = the defended edit is not a success.

Usage: python3 generate_isr_lpips_existing_figure.py --cells-csv cells.csv --judgments judgments_human.csv \
         --repo <repo root> --out <prefix>
"""
import argparse
import collections
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cells-csv", required=True)
ap.add_argument("--judgments", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

METHODS = ["CAPGuard", "BlurGuard", "Mist", "SIFM", "PhotoGuard", "DCT-Shield"]
ADDITIVE = ["Mist", "SIFM", "PhotoGuard", "DCT-Shield"]
COLOR = {"CAPGuard": "#D55E00", "BlurGuard": "#0072B2", "Mist": "#009E73", "SIFM": "#CC79A7",
         "PhotoGuard": "#E69F00", "DCT-Shield": "#56B4E9", "additive": "#7f7f7f"}

ok = lambda r: r["accessory"].strip() == "1" and r["mismatch"].strip() == "0" and r["degraded"].strip() == "0"
labels, und = {}, {}
for r in csv.DictReader(open(args.judgments)):
    s = r["sheet"].strip().replace(".jpg", "")
    labels[(s, r["cell"].strip())] = r
    if r["condition"] == "Undefended":
        und[(s, r["purification"])] = ok(r)

want = {}  # edit_png -> (sheet, cell)
want_cap = {}  # (image, prompt index) -> edit path key, CAPGuard rows are stored under per-batch copies
for r in csv.DictReader(open(args.cells_csv)):
    if r["purification"] == "none" and r["condition"] != "undefended":
        want[os.path.realpath(os.path.join(args.repo, "baseline", r["edit_png"]))] = (r["sheet"].replace(".jpg", ""), r["cell"])
        if r["condition"] == "color3d_scale2p5_pexels":
            want_cap[(r["image"], r["prompt_id"][1:])] = os.path.realpath(os.path.join(args.repo, "baseline", r["edit_png"]))

found = collections.defaultdict(dict)  # edit_png -> {source: value}
sources = []
for root in ["baseline/results", "baseline/runtime", "color/trials", "color/results"]:
    sources += glob.glob(os.path.join(args.repo, root, "**", "displacement*.csv"), recursive=True)
for src in sorted(sources):
    with open(src) as f:
        head = f.readline()
        if "defended_png" not in head or "disp_lpips" not in head:
            continue
        f.seek(0)
        for r in csv.DictReader(f):
            raw = r["defended_png"]
            cands = [raw] if os.path.isabs(raw) else [os.path.join(args.repo, d, raw) for d in ("baseline", "color", "color/trials/lut3d_defense_and_naturalness")]
            p = next((q for q in map(os.path.realpath, cands) if q in want), None)
            if p is None and r.get("condition") == "color3d_scale2p5_pexels" and r.get("scenario") == "ip2p":
                p = want_cap.get((r["image"], r["prompt_index"]))
            if p is not None:
                found[p][os.path.relpath(src, args.repo)] = float(r["disp_lpips"])

cells, conflicts = [], 0
for p, (sheet, cell) in want.items():
    if p not in found:
        continue
    vals = list(found[p].values())
    if max(vals) - min(vals) > 1e-4:
        conflicts += 1
    j = labels[(sheet, cell)]
    if not und[(sheet, "none")]:
        continue
    cells.append((j["condition"], vals[0], 0 if ok(j) else 1))

cov = collections.Counter(labels[want[p]]["condition"] for p in found)
tot = collections.Counter(labels[v]["condition"] for v in want.values())
lines = [f"coverage of no-purification ISR cells by existing disp_lpips (matched on defended_png):"]
lines += [f"  {m:11s} {cov[m]}/{tot[m]}" for m in METHODS]
lines.append(f"cells with differing values across sources: {conflicts}")
srcs = collections.Counter(s for p in found for s in found[p])
lines += ["sources:"] + [f"  {n:5d}  {s}" for s, n in srcs.most_common()]


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


lines += ["", "purification = none, eligible cells; AUC = P(LPIPS of blocked > LPIPS of not blocked)",
          f"{'method':11s} {'n':>4s} {'ISR%':>6s} {'LPIPS all':>9s} {'blocked':>8s} {'not blk':>8s} {'AUC':>5s}"]
stats = {}
for m in METHODS:
    sel = [c for c in cells if c[0] == m]
    if not sel:
        lines.append(f"{m:11s} no matched cells")
        continue
    b = np.array([c[1] for c in sel if c[2]]); o = np.array([c[1] for c in sel if not c[2]])
    a = auc(b, o)
    stats[m] = a
    lines.append(f"{m:11s} {len(sel):4d} {100 * len(b) / len(sel):6.1f} {np.mean([c[1] for c in sel]):9.3f} "
                 f"{b.mean() if len(b) else float('nan'):8.3f} {o.mean():8.3f} {a:5.2f}")
open(args.out + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="white", gridspec_kw={"width_ratios": [1.15, 1.35, 0.9]})
ax = axes[0]
allv = np.array([c[1] for c in cells])
edges = np.quantile(allv, np.linspace(0, 1, 9)); edges[-1] += 1e-9
for g, members in [("CAPGuard", ["CAPGuard"]), ("BlurGuard", ["BlurGuard"]), ("additive", ADDITIVE)]:
    sel = [c for c in cells if c[0] in members]
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        inb = [c for c in sel if a <= c[1] < b]
        if len(inb) >= 15:
            xs.append(np.mean([c[1] for c in inb])); ys.append(100 * np.mean([c[2] for c in inb]))
    ax.plot(xs, ys, "-o", color=COLOR[g], lw=2, ms=6,
            label={"additive": "Additive (Mist, SIFM, PhotoGuard, DCT-Shield)"}.get(g, g))
ax.set_ylim(0, 100); ax.set_xlabel("edit-result LPIPS (bin mean; 8 equal-count bins over all methods)")
ax.set_ylabel("defense success rate (%)")
ax.set_title("(a) Defense success vs edit-result LPIPS", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.grid(color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
rng = np.random.default_rng(20260812)
for i, m in enumerate(METHODS):
    for blocked, dx, marker in [(0, -0.18, "o"), (1, 0.18, "x")]:
        v = np.array([c[1] for c in cells if c[0] == m and c[2] == blocked])
        if len(v) == 0:
            continue
        ax.scatter(i + dx + rng.uniform(-0.09, 0.09, len(v)), v, s=10 if marker == "o" else 18, marker=marker,
                   color=COLOR[m], alpha=0.35 if marker == "o" else 0.85, lw=0.8)
        ax.plot([i + dx - 0.13, i + dx + 0.13], [np.median(v)] * 2, color="#222222", lw=2)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS)
ax.set_ylabel("edit-result LPIPS")
ax.set_title("(b) Per cell: left o = not blocked, right x = blocked; bar = median", loc="left", fontsize=11)
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
ms = [m for m in METHODS if m in stats]
bars = ax.bar(range(len(ms)), [stats[m] for m in ms], color=[COLOR[m] for m in ms], width=0.6)
for b_, m in zip(bars, ms):
    ax.text(b_.get_x() + b_.get_width() / 2, stats[m] + 0.01, f"{stats[m]:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, color="#333333", lw=0.8, ls="--"); ax.text(len(ms) - 0.6, 0.51, "chance", ha="right", fontsize=8)
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=30, ha="right")
ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUC (LPIPS separates blocked from not blocked)")
ax.set_title("(c) Separability by method", loc="left", fontsize=11)
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)

n_by = collections.Counter(c[0] for c in cells)
fig.suptitle("Edit-result LPIPS and defense success, InstructPix2Pix, no purification, 100 Pexels portraits x 4 instructions, "
             "seed 20260812\nexisting per-cell LPIPS; labels after the human audit; eligible cells per method: "
             + ", ".join(f"{m} {n_by[m]}" for m in METHODS), fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(args.out + ".png", dpi=150, facecolor="white")
print("saved", args.out + ".png")
