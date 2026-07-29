"""攻擊者代價前緣 —— 重新分析既有結果，不需 GPU。

先前的呈現方式是「在某個淨化強度下防禦還剩多少」，隱含假設淨化對攻擊者
免費。實際上淨化會同時傷害攻擊者自己的產出，而該量已經記在 results.csv
的 `ctrl_lpips` 欄（未防禦影像經同一淨化後編輯，與未淨化編輯結果的距離）。

本腳本把橫軸換成 ctrl_lpips、縱軸為 net_lpips，得到一條前緣：攻擊者每多
付出一分自身失真，能削掉多少防禦。防禦在某點「失效」與攻擊者在該點「產
出已毀」是兩件事，只有並列才能判讀。

    net_lpips  = d(E(P(x_def)), E(x)) - d(E(P(x)), E(x))
    ctrl_lpips = d(E(P(x)),     E(x))

用法：python scripts/frontier_attacker_cost.py --runs runs/e2 --out runs/e2
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths):
    rows = []
    for p in paths:
        f = os.path.join(p, "results.csv")
        if not os.path.exists(f):
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_run"] = os.path.basename(p.rstrip("/\\"))
                rows.append(r)
    return rows


def aggregate(rows):
    """對影像取平均，鍵為 (run, site, rank, purify, strength)。"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["_run"], r["site"], int(r["rank"]),
               r["purify"], float(r["strength"]))
        for col in ("net_lpips", "ctrl_lpips", "edit_lpips",
                    "defimg_psnr", "defimg_lpips"):
            v = r.get(col, "")
            if v not in ("", "nan"):
                acc[key][col].append(float(v))
    out = {}
    for key, cols in acc.items():
        out[key] = {c: sum(v) / len(v) for c, v in cols.items() if v}
        out[key]["n"] = len(cols.get("net_lpips", []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/e2"])
    ap.add_argument("--out", default="runs/e2")
    ap.add_argument("--drop", nargs="*", default=["identity"],
                    help="不畫的 purify family（identity 與 blur 0.0 重複）")
    args = ap.parse_args()

    agg = aggregate(load(args.runs))
    if not agg:
        raise SystemExit("找不到任何 results.csv")

    arms = sorted({k[:3] for k in agg})
    fams = sorted({k[3] for k in agg} - set(args.drop))

    # ---- 前緣圖：每個 arm 一條線，點為淨化設定，依 ctrl_lpips 排序 ----
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.0),
                             squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for run, site, rank in arms:
            pts = sorted(
                ((v["ctrl_lpips"], v["net_lpips"])
                 for k, v in agg.items()
                 if k[:3] == (run, site, rank) and k[3] == fam
                 and "ctrl_lpips" in v),
                key=lambda t: t[0])
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            label = f"{site} r={rank}" if len(args.runs) == 1 else \
                    f"{run}/{site} r={rank}"
            ax.plot(xs, ys, "o-", ms=4, lw=1.4, label=label)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"purify = {fam}")
        # 圖內一律用英文：matplotlib 預設字型無 CJK 字符，中文會變成方框
        ax.set_xlabel("attacker's own distortion  ctrl_lpips")
        ax.set_ylabel("surviving defense  net_lpips")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Attacker-cost frontier: purification also damages the attacker's own output")
    fig.tight_layout()
    png = os.path.join(args.out, "frontier_attacker_cost.png")
    fig.savefig(png, dpi=140)

    # ---- 表：每個 arm 在「攻擊者代價超過門檻」時尚存的防禦 ----
    lines = ["# 攻擊者代價前緣", "",
             "`ctrl_lpips` 為未防禦影像走同一淨化後，編輯結果相對未淨化編輯的",
             "偏移，即攻擊者為淨化付出的自身代價。",
             "",
             "攻擊者可自由選擇淨化算子，故在給定的自身代價預算 B 下，其最佳",
             "應對是**挑選代價不超過 B 且把防禦壓得最低的那一個設定**：",
             "",
             "    best(B) = min { net_lpips(p) : ctrl_lpips(p) <= B }",
             "",
             "下表即為此值。取「首個超過門檻的設定」會高估防禦，因為那等於",
             "假設攻擊者隨機選算子。", ""]
    budgets = [0.0, 0.02, 0.05, 0.10, 0.20]
    lines.append("| arm | " + " | ".join(
        f"預算 {b:.2f}" for b in budgets) + " |")
    lines.append("|---|" + "---|" * len(budgets))
    for run, site, rank in arms:
        pts = sorted(
            ((v["ctrl_lpips"], v["net_lpips"], k[3], k[4])
             for k, v in agg.items()
             if k[:3] == (run, site, rank) and k[3] not in args.drop
             and "ctrl_lpips" in v),
            key=lambda t: t[0])
        if not pts:
            continue
        cells = []
        for b in budgets:
            afford = [p for p in pts if p[0] <= b + 1e-12]
            if not afford:  # 連最便宜的淨化都超出預算，攻擊者只能不淨化
                afford = [min(pts, key=lambda p: p[0])]
            best = min(afford, key=lambda p: p[1])
            cells.append(f"{best[1]:.4f} ({best[2]} {best[3]:g})")
        name = f"{site} r={rank}" if len(args.runs) == 1 else \
               f"{run} {site} r={rank}"
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    md = os.path.join(args.out, "frontier_attacker_cost.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n寫出 {png}\n寫出 {md}")


if __name__ == "__main__":
    main()
