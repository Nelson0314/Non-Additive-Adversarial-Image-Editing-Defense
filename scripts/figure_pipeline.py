"""論文用的 pipeline 圖：同一張影像走完整條流程，每個環節都看得見。

一列是一組設定（注入位置 × 秩），一行是流程中的一個環節：

    x  ->  x_base = G(x; phi=0)  ->  x_def = G(x; phi)  ->  Delta = x_def - x
       ->  E(x)  ->  E(x_def)  ->  E(P(x_def))

前四行是防禦端在做什麼，後三行是攻擊端拿到圖之後的結果。把 x_base 單獨
列出來是必要的：沒有它，讀者分不清 x_def 的失真有多少來自防禦、多少來自
該注入位置本身的重建誤差——latent 注入的重建誤差比防禦本身大 20 倍以上，
只看 x_def 會把兩者混為一談。

圖內文字一律英文：論文本身是英文，且 matplotlib 預設字型不含 CJK 字符。

用法：
    python scripts/figure_pipeline.py --run runs/e2 --image car_00 \
        --arms P:16 L:16 --out runs/e2/fig_pipeline_car_00.png
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# (檔名, 欄標題, 說明)。檔名為 None 者由 --purify 決定。
STAGES = [
    ("orig.png",           "x",              "original"),
    ("baseline_phi0.png",  "G(x; 0)",        "site floor, no defense"),
    ("defended.png",       "x_def = G(x; f)", "released image"),
    ("residual.png",       "x_def - x",      "residual, amplified"),
    ("edit_orig.png",      "E(x)",           "attacker edits original"),
    ("edit_def_blur_0.0.png", "E(x_def)",    "attacker edits defended"),
    (None,                 "E(P(x_def))",    "after purification"),
]


def load_rows(run):
    with open(os.path.join(run, "results.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cell_metrics(rows, image, site, rank):
    """回傳 (該格的訓練期指標, {(purify,strength): 該淨化下的評測指標})。"""
    sel = [r for r in rows
           if r["image"] == image and r["site"] == site and r["rank"] == str(rank)]
    if not sel:
        raise SystemExit(f"results.csv 沒有 {image} / site {site} / r{rank}")
    by_purify = {(r["purify"], r["strength"]): r for r in sel}
    return sel[0], by_purify


def psnr_from_png(a_path, b_path):
    """由留存的 PNG 直接算 PSNR。

    這是 8-bit 量化後的值，與訓練期以 float32 記錄的數字會有小幅出入，故
    標註時明說來源，不與 results.csv 的欄位混用。用途只是讓讀者在圖上看得
    到「這一格離原圖多遠」，不作為報告數字。
    """
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return None
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def annotate(row, by_purify, stage_file, purify_tag, cell):
    """該環節底下要標的數字。沒有可標的就回傳空字串。"""
    f = lambda k: float(row[k])
    if stage_file == "defended.png":
        return (f"PSNR {f('final_psnr_total'):.1f} dB\n"
                f"LPIPS {f('final_lpips'):.4f}\n"
                f"SSIM {f('final_ssim'):.4f}")
    if stage_file == "baseline_phi0.png":
        # x_base 與 x 的距離就是該注入位置在「完全沒有防禦」時的失真下限。
        # 少了這個數字，x_def 的失真會被整筆算到防禦頭上。
        p = psnr_from_png(os.path.join(cell, "baseline_phi0.png"),
                          os.path.join(cell, "orig.png"))
        if p is None:
            return "reconstruction floor\nof this site"
        if p == float("inf"):
            return "identical to x\n(no floor)"
        return f"PSNR {p:.1f} dB to x\n(site's floor,\nfrom saved PNG)"
    if stage_file == "residual.png":
        return (f"display gain x{f('residual_gain'):.3g}\n"
                f"eff. rank {f('eff_rank_mean'):.0f}\n"
                f"99% energy rank {f('energy_rank_99_mean'):.1f}")
    if stage_file == "edit_def_blur_0.0.png":
        r = by_purify.get(("blur", "0.0"))
        return f"net shift {float(r['net_lpips']):.4f}" if r else ""
    if stage_file is None:
        r = by_purify.get(purify_tag)
        if not r:
            return ""
        return (f"net shift {float(r['net_lpips']):.4f}\n"
                f"attacker's own\ncost {float(r['ctrl_lpips']):.4f}")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/e2")
    ap.add_argument("--image", default="car_00")
    ap.add_argument("--arms", nargs="+", default=["P:16", "L:16"],
                    help="site:rank，每個佔一列")
    ap.add_argument("--purify", default="blur:3.0",
                    help="最後一行用哪個淨化設定，需有對應的 edit_def_*.png")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.run)
    pfam, pstr = args.purify.split(":")
    purify_file = f"edit_def_{pfam}_{pstr}.png"
    purify_tag = (pfam, pstr)

    arms = [a.split(":") for a in args.arms]
    nrow, ncol = len(arms), len(STAGES)

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.62 * nrow),
                             squeeze=False)
    for ri, (site, rank) in enumerate(arms):
        cell = os.path.join(args.run, f"{args.image}__{site}__r{rank}")
        row, by_purify = cell_metrics(rows, args.image, site, int(rank))

        for ci, (fname, header, sub) in enumerate(STAGES):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("#999")

            path = os.path.join(cell, purify_file if fname is None else fname)
            if os.path.exists(path):
                ax.imshow(np.asarray(Image.open(path).convert("RGB")))
            else:
                # 缺圖要看得出來，不能靜默留白被誤讀成全黑影像
                ax.imshow(np.full((8, 8, 3), 0.93))
                ax.text(0.5, 0.5, "not saved", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#b00")

            if ri == 0:
                ax.set_title(header, fontsize=10, pad=9)
                ax.text(0.5, 1.015, sub, ha="center", va="bottom", fontsize=6.5,
                        color="#666", transform=ax.transAxes)
            if ci == 0:
                label = {"P": "pixel", "L": "latent",
                         "LA": "latent-anchored"}.get(site, site)
                ax.set_ylabel(f"{label}\nr = {rank}", fontsize=10)

            txt = annotate(row, by_purify, fname, purify_tag, cell)
            if txt:
                ax.text(0.5, -0.035, txt, ha="center", va="top", fontsize=6.6,
                        family="monospace", color="#222",
                        transform=ax.transAxes, linespacing=1.35)

    fig.suptitle(
        f"One image through the full pipeline  —  {args.image}, "
        f"purification = {pfam} {pstr}",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.subplots_adjust(hspace=0.42, wspace=0.06)

    out = args.out or os.path.join(args.run, f"fig_pipeline_{args.image}.png")
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
