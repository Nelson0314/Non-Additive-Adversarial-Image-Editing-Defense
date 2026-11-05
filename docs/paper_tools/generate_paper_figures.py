"""Draw the distortion--displacement figure of the paper (run on the remote host).

    python generate_paper_figures.py --points points.csv --aligned condition_summary.csv \
        --color-summary summary.csv --out-dir figs

distortion_displacement.pdf has two panels that share the least-squares line fitted to the
pre-purification points of the additive methods (baseline/results/distortion_displacement/
points.csv; rows of earlier color conditions are dropped, the final color3d comes from the color
paper table).
(a) Edit LPIPS before purification against defense LPIPS for every budget of every additive method; --ours-extra adds a second
    operating point of CAPGuard.
(b) The eleven matched-distortion arms and color3d: filled marker before purification, open marker
    after purification (mean over the seven standard purifications), joined by an arrow.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Additive methods reported in the paper; PhotoGuard appears only as its encoder attack.
ADDITIVE = {"danp", "dayn", "dct_shield", "dct_shield_y", "dia_pt", "dia_r", "mist",
            "photoguard_encoder", "sifm"}
OURS = "color3d"
OURS_LABEL = "CAPGuard"
GRAY = "#6b7785"
LIGHT = "#b9c1ca"
C_ADDITIVE = "#A94438"
C_ADDITIVE_LIGHT = "#D9A39C"
C_OURS = "#1F6F7C"
HIGHLIGHT = "dct_shield_y"  # additive method drawn separately (native budget; inputs merged by the caller)
HIGHLIGHT_LABEL = "DCT-Shield-Y (native)"
SOURCE_MARKER = {"native": "s", "aligned": "o", "aligned_1p255": "o", "plane": "^"}


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx, sxy / math.sqrt(sxx * syy)


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.color": "#e6e9ed", "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
    })


def draw_line(ax, slope, intercept, lo, hi):
    ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color=GRAY, lw=0.8,
            ls=(0, (4, 2)), zorder=1)


SURVIVAL_ORDER = [("jpeg80", "JPEG\n80"), ("jpeg50", "JPEG\n50"), ("jpeg30", "JPEG\n30"), ("blur1", "Blur\n1"),
                  ("blur2", "Blur\n2"), ("crop_resize0.1", "Crop"), ("rotate15", "Rot."), ("vae_trans", "VAE-\nTrans")]
SWEEPS = [
    ("JPEG quality", [("jpeg95", 95), ("jpeg90", 90), ("jpeg70", 70), ("jpeg50", 50), ("jpeg30", 30), ("jpeg10", 10)], True),
    (r"Blur $\sigma$", [("blur0.5", 0.5), ("blur1", 1), ("blur2", 2), ("blur3", 3)], False),
    ("Resize scale", [("resample_roundtrip384", 0.75), ("resample_roundtrip256", 0.5), ("resample_roundtrip128", 0.25)], True),
]
SWEEP_STYLE = {"dia_pt": ("DIA-PT", "o"), "mist": ("Mist", "s"), "photoguard_encoder": ("PhotoGuard", "^"),
               HIGHLIGHT: (HIGHLIGHT_LABEL, "D")}
SEED_FLOOR = 0.109  # edit LPIPS between two seeds on the same undefended image (8 portraits x 4 instructions)


def mean_by(rows, key_fields, value_field):
    acc = {}
    for row in rows:
        key = tuple(row[f] for f in key_fields)
        acc.setdefault(key, []).append(float(row[value_field]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def purification_figure(survival_rows, sweep_rows, out):
    survival = mean_by(survival_rows, ("condition", "purifier"), "survival")
    sweep = mean_by(sweep_rows, ("condition", "purifier"), "disp_purified")
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [1.55, 1, 0.85, 0.75]})

    ax = axes[0]
    order = [(p, label) for p, label in SURVIVAL_ORDER if (OURS, p) in survival]  # 缺讀數的淨化（如未算的 VAE-Trans）不畫
    # purifications have no order of strength: grouped bars, one group per purification
    others = [m for m in sorted(ADDITIVE - {"diffvax"}) if m != HIGHLIGHT]
    methods = others + [HIGHLIGHT, OURS]
    colors = [C_ADDITIVE_LIGHT] * len(others) + [C_ADDITIVE, C_OURS]
    width = 0.8 / len(methods)
    top_bar = 0.0
    for j, (method, color) in enumerate(zip(methods, colors)):
        ys = [survival.get((method, p)) for p, _ in order]
        if None in ys:
            raise SystemExit(f"survival readings missing for {method}")
        top_bar = max(top_bar, max(ys))
        ax.bar([i - 0.4 + (j + 0.5) * width for i in range(len(order))], ys, width=width, color=color,
               edgecolor="white", linewidth=0.2, zorder=2)
    ax.axhline(1.0, color=GRAY, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for _, label in order], fontsize=5.6)
    ax.set_ylim(0, top_bar + 0.3)
    ax.set_ylabel("Perturbation survival")
    ax.set_title("(a) Perturbation survival", loc="left")
    ax.legend(handles=[Patch(color=C_ADDITIVE_LIGHT, label=f"other additive ({len(others)})"),
                       Patch(color=C_ADDITIVE, label=HIGHLIGHT_LABEL), Patch(color=C_OURS, label=OURS_LABEL)],
              loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.8, fontsize=5.6, ncol=3)

    # the highlighted method keeps the color it has in panel (a); the others take lighter shades
    shades = ["#7A2E25", "#DDA59E", "#C7756B", C_ADDITIVE]  # DIA-PT, Mist, PhotoGuard, HIGHLIGHT (SWEEP_STYLE order)
    # panels b-d share one y range (c and d carry no tick labels)
    top = max(0.48, round(max(sweep[(m, p)] for m in list(SWEEP_STYLE) + [OURS]
                                for _, settings, _ in SWEEPS for p, _ in settings) + 0.05, 1))
    # unpurified edit LPIPS of the same cells (disp_plain), drawn as the leftmost "None" point of each panel
    plain = mean_by(sweep_rows, ("condition",), "disp_plain")
    top = max(top, round(max(plain[(m,)] for m in list(SWEEP_STYLE) + [OURS]) + 0.05, 1))
    for ax, (title, settings, _), letter in zip(axes[1:], SWEEPS, "bcd"):
        sx = list(range(len(settings) + 1))  # categorical positions, weakest to strongest after "None"
        for (method, (name, marker)), shade in zip(SWEEP_STYLE.items(), shades):
            ys = [plain[(method,)]] + [sweep[(method, p)] for p, _ in settings]
            ax.plot(sx, ys, color=shade, lw=0.9, marker=marker, ms=2.4, label=name, zorder=2)
        ys = [plain[(OURS,)]] + [sweep[(OURS, p)] for p, _ in settings]
        ax.plot(sx, ys, color=C_OURS, lw=1.3, marker="*", ms=5, label=OURS_LABEL, zorder=4)
        ax.axhline(SEED_FLOOR, color=GRAY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.set_xticks(sx)
        ax.set_xticklabels(["None"] + [f"{v:g}" for _, v in settings], fontsize=5.6)
        ax.set_xlim(-0.4, len(settings) + 0.4)
        ax.set_ylim(0, top)
        ax.set_xlabel(title)
        ax.set_title(f"({letter}) Edit LPIPS", loc="left")
        if letter == "b":
            ax.set_ylabel("Edit LPIPS")
            ax.text(sx[-1], SEED_FLOOR - 0.035, "seed-to-seed", color=GRAY, fontsize=5.5, ha="right")
        else:
            ax.tick_params(labelleft=False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.66, 1.0), ncol=5, frameon=False,
               fontsize=6, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    print(f"purification: ours survival 7-mean="
          f"{sum(survival[(OURS, p)] for p, _ in SURVIVAL_ORDER[:7]) / 7:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--color-summary", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True, help="baseline/results/aligned/perturbation_survival.csv")
    parser.add_argument("--sweep", type=Path, required=True, help="baseline/results/sweeps/ip2p/purifier_strength/retention.csv")
    parser.add_argument("--ours-extra", nargs=3, metavar=("DEFENSE_LPIPS", "EDIT_LPIPS", "LABEL"),
                        help="an additional operating point of our defense before purification, drawn in panel (a)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    purification_figure(read(args.survival), read(args.sweep), args.out_dir / "purification.pdf")

    points = [r for r in read(args.points) if r["method"] in ADDITIVE]
    xs = [float(r["defense_lpips"]) for r in points]
    ys = [float(r["edit_lpips"]) for r in points]
    slope, intercept, r = fit(xs, ys)
    ours = next(row for row in read(args.color_summary) if row["condition"] == OURS)
    ox, oy0, oy1 = float(ours["fid_lpips"]), float(ours["lpips_none"]), float(ours["lpips_seven"])
    arms = [row for row in read(args.aligned) if row["condition"] in ADDITIVE]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.05), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) every budget before purification
    draw_line(ax_a, slope, intercept, 0.0, 0.7)
    for marker in ("s", "o", "^"):
        sel = [row for row in points if SOURCE_MARKER.get(row["source"]) == marker]
        ax_a.scatter([float(row["defense_lpips"]) for row in sel], [float(row["edit_lpips"]) for row in sel],
                     s=14, marker=marker, color=C_ADDITIVE, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=2)
    ax_a.scatter(ox, oy0, s=70, marker="*", color=C_OURS, edgecolors="white", linewidths=0.4, zorder=4)
    ax_a.annotate(OURS_LABEL, (ox, oy0), xytext=(-14, 9), textcoords="offset points", color=C_OURS,
                  fontsize=7)
    if args.ours_extra:
        ex, ey = float(args.ours_extra[0]), float(args.ours_extra[1])
        ax_a.scatter(ex, ey, s=70, marker="*", facecolors="white", edgecolors=C_OURS, linewidths=0.9, zorder=4)
        ax_a.annotate(args.ours_extra[2], (ex, ey), xytext=(6, -3), textcoords="offset points", color=C_OURS,
                      fontsize=6.5, va="top")
    ax_a.set_xlim(0, 0.7)
    ax_a.set_ylim(0, 0.7)
    ax_a.set_xlabel("Defense LPIPS")
    ax_a.set_ylabel("Edit LPIPS")
    present = {SOURCE_MARKER.get(row["source"]) for row in points}
    ax_a.set_title("(a) Before purification, all budgets" if "^" in present else "(a) Before purification", loc="left")
    ax_a.text(0.69, 0.04, f"dashed: least-squares fit, $r$ = {r:.3f}", color=GRAY, fontsize=6.5, ha="right")
    handles = [Line2D([], [], ls="none", marker=m, color=C_ADDITIVE, ms=3.5, label=label)
               for m, label in (("s", "native budget"), ("o", "matched distortion"), ("^", "budget sweep")) if m in present]
    ax_a.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2)

    # (b) matched distortion, before -> after purification
    arm_x = [float(row["defense_lpips"]) for row in arms]
    lo, hi = min(arm_x + [ox]) - 0.02, max(arm_x + [ox]) + 0.02
    draw_line(ax_b, slope, intercept, lo, hi)
    for row in arms:
        x = float(row["defense_lpips"])
        y0, y1 = float(row["edit_lpips_none"]), float(row["edit_lpips_purified_mean"])
        ax_b.annotate("", xy=(x, y1 + 0.008), xytext=(x, y0 - 0.008),
                      arrowprops=dict(arrowstyle="-|>", color=C_ADDITIVE_LIGHT, lw=0.9, mutation_scale=6), zorder=2)
        ax_b.scatter(x, y0, s=16, color=C_ADDITIVE, edgecolors="white", linewidths=0.3, zorder=3)
        ax_b.scatter(x, y1, s=16, facecolors="white", edgecolors=C_ADDITIVE, linewidths=0.9, zorder=3)
    ax_b.scatter(ox, oy1, s=90, marker="*", facecolors="white", edgecolors=C_OURS, linewidths=0.9, zorder=4)
    ax_b.scatter(ox, oy0, s=14, color=C_OURS, edgecolors="white", linewidths=0.3, zorder=5)
    ax_b.annotate(f"{OURS_LABEL}\n{oy0:.3f} $\\rightarrow$ {oy1:.3f}", (ox, oy1), xytext=(-4, 12), textcoords="offset points",
                  color=C_OURS, fontsize=6.5, ha="center")
    after = [float(row["edit_lpips_purified_mean"]) for row in arms]
    ax_b.text(lo + 0.006, min(after) - 0.03, f"{len(arms)} additive methods", color=C_ADDITIVE, fontsize=6.5,
              ha="left", va="center")
    ax_b.set_xlim(lo, hi + 0.03)
    ax_b.set_ylim(0.0, 0.48)
    ax_b.set_xlabel("Defense LPIPS")
    ax_b.set_ylabel("Edit LPIPS")
    ax_b.set_title("(b) Matched distortion, before and after purification", loc="left")
    ax_b.legend(handles=[
        Line2D([], [], ls="none", marker="o", color=C_ADDITIVE, ms=3.5, label="before purification"),
        Line2D([], [], ls="none", marker="o", mfc="white", mec=C_ADDITIVE, mew=0.9, ms=3.5,
               label="after (mean of 7 purifications)"),
        Line2D([], [], color=GRAY, lw=0.8, ls=(0, (4, 2)), label="fit before purification"),
    ], loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    out = args.out_dir / "distortion_displacement.pdf"
    fig.savefig(out)
    print(f"fit n={len(points)} slope={slope:.4f} intercept={intercept:.4f} r={r:.4f}; "
          f"ours=({ox:.3f}, {oy0:.3f} -> {oy1:.3f}); arms={len(arms)}")


if __name__ == "__main__":
    main()
