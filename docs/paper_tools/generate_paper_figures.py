"""Draw the two distortion--displacement figures of the paper (run on the remote host).

    python generate_paper_figures.py --points points.csv --aligned condition_summary.csv \
        --color-summary summary.csv --color-arms arms_vae_siglip.csv --out-dir figs

plane.pdf: edit LPIPS before purification against defense LPIPS. Additive methods come from
baseline/results/distortion_displacement/points.csv (native, matched-distortion and budget-sweep
points); rows of earlier color conditions in that file are dropped and the final color3d is taken
from the color paper table. The pooled least-squares line is fitted on the additive points only.

purified_plane.pdf: edit LPIPS averaged over the seven purifications (filled) and after VAE-Trans
(open) against defense LPIPS, for the eleven matched-distortion arms and color3d.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {
    "danp": "DANP", "dayn": "DAYN", "dct_shield": "DCT-Shield", "dct_shield_y": "DCT-Shield-Y",
    "dia_pt": "DIA-PT", "dia_r": "DIA-R", "diffvax": "DiffVax", "mist": "Mist",
    "photoguard_c": r"PhotoGuard-$\ell_2$", "photoguard_encoder": "PhotoGuard-Enc",
    "photoguard_linf": r"PhotoGuard-$\ell_\infty$", "sifm": "SIFM",
}
EARLIER_COLOR = {"color", "color3d", "color3d_0d4", "color3d_adaptive_set"}
OURS = "color3d"


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
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def colors(methods):
    cmap = plt.get_cmap("tab20")
    return {m: cmap(i % 20) for i, m in enumerate(sorted(methods))}


def plane(points, ours, out):
    rows = [r for r in points if r["method"] not in EARLIER_COLOR]
    xs = [float(r["defense_lpips"]) for r in rows]
    ys = [float(r["edit_lpips"]) for r in rows]
    slope, intercept, r = fit(xs, ys)
    col = colors({r["method"] for r in rows})
    marker = {"native": "s", "aligned": "o", "aligned_1p255": "o", "plane": "^"}
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    seen = set()
    for row in rows:
        m = row["method"]
        ax.scatter(float(row["defense_lpips"]), float(row["edit_lpips"]), s=16,
                   marker=marker.get(row["source"], "o"), color=col[m], edgecolors="none",
                   label=NAMES.get(m, m) if m not in seen else None)
        seen.add(m)
    grid = [0.0, 0.7]
    ax.plot(grid, [slope * g + intercept for g in grid], color="0.5", lw=0.8, ls="--", zorder=0)
    ax.scatter(*ours, s=60, marker="*", color="crimson", edgecolors="k", linewidths=0.4,
               label="Ours", zorder=5)
    ax.set_xlabel("Defense LPIPS")
    ax.set_ylabel("Edit LPIPS (no purification)")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.legend(ncol=2, frameon=False, loc="upper left", handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    residual = ours[1] - (slope * ours[0] + intercept)
    print(f"plane: n={len(rows)} slope={slope:.4f} intercept={intercept:.4f} r={r:.4f} "
          f"ours_residual={residual:+.4f}")


def purified_plane(aligned, color_summary, color_arms, out):
    vae = {r["arm"]: float(r["lpips_vae_trans"]) for r in color_arms}
    arms = [r for r in aligned if r["condition"] in NAMES]
    col = colors({r["condition"] for r in arms})
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for row in arms:
        m = row["condition"]
        x = float(row["defense_lpips"])
        ax.scatter(x, float(row["edit_lpips_purified_mean"]), s=18, color=col[m],
                   edgecolors="none", label=NAMES[m])
        if m in vae:
            ax.scatter(x, vae[m], s=18, facecolors="none", edgecolors=col[m], linewidths=0.8)
    ours = next(r for r in color_summary if r["condition"] == OURS)
    x = float(ours["fid_lpips"])
    ax.scatter(x, float(ours["lpips_seven"]), s=60, marker="*", color="crimson",
               edgecolors="k", linewidths=0.4, label="Ours", zorder=5)
    ax.scatter(x, vae[OURS], s=60, marker="*", facecolors="none", edgecolors="crimson",
               linewidths=0.8, zorder=5)
    ax.scatter([], [], s=18, color="0.4", label="7-purification mean")
    ax.scatter([], [], s=18, facecolors="none", edgecolors="0.4", label="VAE-Trans")
    ax.set_xlabel("Defense LPIPS")
    ax.set_ylabel("Edit LPIPS after purification")
    ax.set_xlim(0.28, 0.39)
    ax.set_ylim(0, 0.5)
    ax.legend(ncol=2, frameon=False, loc="upper right", handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    print(f"purified_plane: arms={len(arms)} ours=({x:.3f}, {float(ours['lpips_seven']):.3f}, "
          f"{vae[OURS]:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--color-summary", type=Path, required=True)
    parser.add_argument("--color-arms", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    color_summary = read(args.color_summary)
    ours = next(r for r in color_summary if r["condition"] == OURS)
    plane(read(args.points), (float(ours["fid_lpips"]), float(ours["lpips_none"])),
          args.out_dir / "plane.pdf")
    purified_plane(read(args.aligned), color_summary, read(args.color_arms),
                   args.out_dir / "purified_plane.pdf")


if __name__ == "__main__":
    main()
