"""E2/E3 結果分析 — spec §7.3、§7.4。

產出三類圖：

1. **防禦—保真前緣**（E2）：橫軸為 `x_def` 對 `x` 的失真、縱軸為編輯偏移，
   每個指標一張。低秩約束是否改善此前緣，由 P 與 L 各自的曲線位置回答。
2. **淨化強度掃描**（E3）：橫軸為淨化強度、縱軸為編輯偏移，P 與 L 疊圖。
   spec §7.4 的因果判斷直接讀這張圖。
3. **秩對照表**：注入秩 vs 實測像素秩（effective 與 energy 兩種），驗證
   site P 的 clamp 效應與 site L 湧現秩的實際數值。

**不對結果方向做任何假設**：三種可能結果（P 亦耐、P 不耐而 L 耐、兩者皆耐）
都是可發表的發現，繪圖與統計一律中性處理。

執行：python scripts/analyze.py --run runs/e2
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.suite import HIGHER_IS_BETTER

SITE_STYLE = {"P": ("tab:blue", "o", "-"), "L": ("tab:red", "s", "--")}
NL = chr(10)


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def mean(vals):
    vals = [v for v in vals if v == v]  # 濾掉 NaN（例如尺寸不足的 NIQE）
    return sum(vals) / len(vals) if vals else float("nan")


def plot_frontier(summary, out_dir: Path):
    """防禦—保真前緣。橫軸失真、縱軸偏移，每點為一個 (site, rank) 的跨圖平均。

    誤差長條為跨影像的標準差。n 通常很小（現有資料集僅 6 張，spec §9.1），
    故長條只作離散程度的提示，不足以支撐顯著性宣稱。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # 橫軸取相對**原圖**的絕對保真度：讀者關心的是 x_def 像不像 x。
    # final_psnr / final_linf 是相對 x_base 的量，屬於損失函數的內部
    # 尺度，不適合當作前緣的保真軸。
    fid_keys = [("final_psnr_total", "PSNR(x_def, x) dB", True),
                ("final_ssim", "SSIM(x_def, x)", True),
                ("final_lpips", "LPIPS(x_def, x)", False)]

    for ax, (key, label, higher_better) in zip(axes, fid_keys):
        for site in sorted({r["site"] for r in summary}):
            color, marker, ls = SITE_STYLE.get(site, ("gray", "x", ":"))
            ranks = sorted({int(r["rank"]) for r in summary if r["site"] == site})
            xs, ys, xe, ye = [], [], [], []
            for rk in ranks:
                sel = [r for r in summary if r["site"] == site and int(r["rank"]) == rk]
                fx = [fnum(r, key) for r in sel]
                fy = [fnum(r, "final_shift") for r in sel]
                xs.append(mean(fx)); ys.append(mean(fy))
                xe.append(_std(fx)); ye.append(_std(fy))
            ax.errorbar(
                xs, ys, xerr=xe, yerr=ye, color=color, marker=marker, ls=ls,
                capsize=3, lw=1.4, ms=6, label=f"site {site}",
            )
            for rk, x, y in zip(ranks, xs, ys):
                ax.annotate(f"r={rk}", (x, y), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel(label + ("  (higher = better fidelity)" if higher_better
                               else "  (lower = better fidelity)"))
        ax.set_ylabel("edit shift  LPIPS(edit(x), edit(x_def))")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("E2 defense-fidelity frontier (labels = injected rank r)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "frontier.png", dpi=140)
    plt.close(fig)


def _std(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def plot_purify(results, out_dir: Path):
    """E3 淨化強度掃描。每種淨化一張子圖，P 與 L 疊在一起直接對比。

    只取 noise_split == "heldout" 的列。訓練用種子的那一列量的是訓練集
    表現，混進曲線會把過擬合誤記為防禦效果。
    """
    results = [r for r in results if r.get("noise_split", "heldout") == "heldout"]
    kinds = sorted({r["purify"] for r in results})
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 4.0), squeeze=False)

    for ax, kind in zip(axes[0], kinds):
        sub = [r for r in results if r["purify"] == kind]
        for site in sorted({r["site"] for r in sub}):
            for rk in sorted({int(r["rank"]) for r in sub if r["site"] == site}):
                sel = [r for r in sub if r["site"] == site and int(r["rank"]) == rk]
                by_s = defaultdict(list)
                for r in sel:
                    # net = 防禦造成的偏移 − 淨化自己造成的偏移
                    by_s[fnum(r, "strength")].append(fnum(r, "net_lpips"))
                xs = sorted(by_s)
                ys = [mean(by_s[s]) for s in xs]
                color, marker, ls = SITE_STYLE.get(site, ("gray", "x", ":"))
                ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=1.3, ms=5,
                        alpha=0.4 + 0.6 * (rk / max(1, max(by_s and [rk] or [1]))),
                        label=f"{site} r={rk}")
        ax.set_title(f"purify: {kind}")
        ax.set_xlabel("strength")
        ax.set_ylabel("net edit shift  (defended - undefended control)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(
        "E3 purification sweep - net of undefended control "
        "(higher = defense survives better)", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "purify_sweep.png", dpi=140)
    plt.close(fig)


def overfit_table(results, out_dir: Path):
    """訓練種子 vs 未見種子的偏移差，即對特定噪聲的過擬合幅度。"""
    tr = [r for r in results if r.get("noise_split") == "train"]
    ho = [
        r for r in results
        if r.get("noise_split") == "heldout"
        and r["purify"] == "blur" and fnum(r, "strength") == 0.0
    ]
    if not tr or not ho:
        return
    lines = [
        "| site | r | 訓練種子偏移 | 未見種子偏移 | 差 | 比值 |",
        "|---|---|---|---|---|---|",
    ]
    for site in sorted({r["site"] for r in tr}):
        for rk in sorted({int(r["rank"]) for r in tr if r["site"] == site}):
            a = mean([fnum(r, "edit_lpips") for r in tr
                      if r["site"] == site and int(r["rank"]) == rk])
            b = mean([fnum(r, "edit_lpips") for r in ho
                      if r["site"] == site and int(r["rank"]) == rk])
            ratio = a / b if b and b == b and b > 0 else float("nan")
            lines.append(
                f"| {site} | {rk} | {a:.4f} | {b:.4f} | {a - b:+.4f} "
                f"| {'—' if ratio != ratio else f'{ratio:.2f}x'} |"
            )
    (out_dir / "overfit_table.md").write_text(NL.join(lines), encoding="utf-8")
    print(NL + "[過擬合幅度：訓練種子 vs 未見種子，identity/無淨化]")
    print(NL.join(lines))


def rank_table(summary, out_dir: Path):
    """注入秩 vs 實測秩。site P 的 clamp 效應在此表可直接讀出。"""
    lines = [
        "| site | 注入 r | eff_rank(x_def−x) | energy99(x_def−x) | eff_rank(Δ) | energy99(Δ) | clamp 比例 |",
        "|---|---|---|---|---|---|---|",
    ]
    for site in sorted({r["site"] for r in summary}):
        for rk in sorted({int(r["rank"]) for r in summary if r["site"] == site}):
            sel = [r for r in summary if r["site"] == site and int(r["rank"]) == rk]
            def m(k):
                v = mean([fnum(r, k) for r in sel])
                return "—" if v != v else f"{v:.1f}"
            cf = mean([fnum(r, "clamped_fraction") for r in sel])
            lines.append(
                f"| {site} | {rk} | {m('eff_rank_mean')} | {m('energy_rank_99_mean')} "
                f"| {m('raw_eff_rank_mean')} | {m('raw_energy_rank_99_mean')} "
                f"| {'—' if cf != cf else f'{cf * 100:.2f}%'} |"
            )
    (out_dir / "rank_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/e2")
    args = ap.parse_args()
    run = Path(args.run)

    summary = read_csv(run / "summary.csv")
    results = read_csv(run / "results.csv")
    if not summary:
        print(f"[analyze] {run / 'summary.csv'} 不存在或為空")
        return 1

    print(f"[analyze] summary {len(summary)} 列、results {len(results)} 列")
    plot_frontier(summary, run)
    if results:
        plot_purify(results, run)
        overfit_table(results, run)
    rank_table(summary, run)
    print(f"[analyze] 圖表寫入 {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
