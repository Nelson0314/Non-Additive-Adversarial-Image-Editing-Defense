"""把逐頻帶的相位保留與能量存活畫成曲線。**只讀 CSV，不重算。**

配 `scripts/phase_drift_figure.py` 用：那一支給單張影像的樣子，這一支給十張
平均的定量形狀。兩張圖回答同一個問題的兩半——**相位在哪一帶開始跑掉，
以及那一帶還剩多少幅度可以讓相位作用**。

用法：
    python scripts/phase_drift_curves.py \
        --summary runs/phase_drift_diagnosis --out <輸出目錄>
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                          "PMingLiU", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PURIFIERS: Sequence[Tuple[str, str]] = (
    ("blur1", "模糊 σ1"), ("blur2", "模糊 σ2"),
    ("jpeg75", "JPEG 75"), ("jpeg30", "JPEG 30"),
    ("crop_resize0.1", "裁切 10%"),
)
CONDITIONS: Sequence[Tuple[str, str]] = (
    ("ours_ph_q", "本方法 純相位＋量化 r0.9"),
    ("dct_aj85", "DCT-Shield 抗JPEG q0.85"),
)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("runs/phase_drift_diagnosis"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    with (args.summary / "phase_retention_by_band.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bands = sorted({(float(r["band_lo"]), float(r["band_hi"])) for r in rows})
    centres = [(lo + hi) / 2 for lo, hi in bands]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, (cond, label) in zip(axes, CONDITIONS):
        for key, name in PURIFIERS:
            ys = []
            for lo, _ in bands:
                s = [float(r["rho"]) for r in rows
                     if r["condition"] == cond and r["purifier"] == key
                     and float(r["band_lo"]) == lo]
                ys.append(st.mean(s))
            ax.plot(centres, ys, marker="o", ms=4, label=name)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("歸一化頻率半徑（1 = Nyquist）")
        ax.set_ylabel("相位保留 ρ（1 = 原封不動）")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("裝上去的相位偏移，在每一個頻帶上還剩多少（十張平均）", fontsize=12)
    args.out.mkdir(parents=True, exist_ok=True)
    p1 = args.out / "phase_retention_curves.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    print(f"→ {p1}")

    # 第二張：相位保留 vs 殘差能量存活，說明「保留」與「有用」不是同一件事。
    with (args.summary / "summary.csv").open(encoding="utf-8") as fh:
        summ = list(csv.DictReader(fh))
    fig2, ax = plt.subplots(figsize=(6.6, 5.0))
    marks = {"ours_ph_q": "o", "ours_ph_n": "s", "ours_pg_q20": "^", "dct_aj85": "D"}
    for cond, mk in marks.items():
        xs, ys, names = [], [], []
        for key, name in PURIFIERS:
            r = next((r for r in summ
                      if r["condition"] == cond and r["purifier"] == key), None)
            if r is None:
                continue
            xs.append(float(r["rho_mean"]))
            ys.append(float(r["energy_ratio"]))
            names.append(name)
        ax.scatter(xs, ys, marker=mk, s=48, label=cond, alpha=0.85)
        if cond == "ours_ph_q":
            for x, y, n in zip(xs, ys, names):
                ax.annotate(n, (x, y), fontsize=8,
                            textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("相位保留 ρ")
    ax.set_ylabel("殘差能量存活率")
    ax.set_yscale("log")
    # 對數軸的預設標籤走 mathtext，而 mathtext 的字型沒有 U+2212（真正的減號），
    # 於是 `10^-1` 會被換成一個假符號而**不會報錯**。改成純文字標籤。
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("相位還在，不代表它還推得動任何東西", fontsize=11)
    p2 = args.out / "phase_vs_energy.png"
    fig2.savefig(p2, dpi=140, bbox_inches="tight")
    print(f"→ {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
