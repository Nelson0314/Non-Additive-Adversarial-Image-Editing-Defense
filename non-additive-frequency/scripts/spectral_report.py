"""彙總 `spectral_decompose.py` 的分片結果，產出 PAD 第 3 節在本專案讀數下的表。

讀 `runs/spectral/dec_*.csv`，輸出：

1. 逐條件 × 版本的平均（位移量、失真、幅度偏移、夾取量）
2. 「相位那一半 vs 幅度那一半」的逐圖勝場與平均比（**只用 DISTS 對齊過的
   `amp_s`／`pha_s`**——未對齊的 `amp`／`pha` 失真高於 `full`，比不得）
3. 對照 PAD 表 1 的方向：該篇在分類上量到相位半邊破壞力更大且雜訊更小

用法：
    python scripts/spectral_report.py --run runs/spectral --out reports/spectral
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from collections import defaultdict
from pathlib import Path

VARIANTS = ("full", "amp", "pha", "amp_s", "pha_s")
NUM = ("effect_mean", "fid_lpips", "fid_dists", "fid_psnr", "fid_ssim",
       "fid_linf", "fid_rms", "amp_dev", "clip_fraction", "clip_mean",
       "clip_max", "scale")


def load(run: Path) -> list:
    rows = []
    for f in sorted(glob.glob(str(run / "dec_*.csv"))):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in NUM:
                    if r.get(k) not in (None, ""):
                        r[k] = float(r[k])
                rows.append(r)
    return rows


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), float)]
    return statistics.fmean(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/spectral"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load(args.run)
    if not rows:
        raise SystemExit(f"{args.run} 沒有 dec_*.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["variant"])].append(r)
    conds = sorted({r["condition"] for r in rows})
    images = sorted({r["image"] for r in rows})
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    w(f"# 幅度／相位分解（PAD 第 3 節）· {len(images)} 張影像 × {len(conds)} 條件")
    w()
    w("## 1. 逐條件 × 版本的平均")
    w()
    w("| 條件 | 版本 | 位移量↑ | LPIPS | DISTS | PSNR | 幅度偏移 | 夾取均量 | 縮放 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        for v in VARIANTS:
            g = by.get((c, v))
            if not g:
                continue
            w(f"| {c} | {v} | {mean(g,'effect_mean'):.4f} | "
              f"{mean(g,'fid_lpips'):.4f} | {mean(g,'fid_dists'):.4f} | "
              f"{mean(g,'fid_psnr'):.2f} | {mean(g,'amp_dev'):.5f} | "
              f"{mean(g,'clip_mean'):.2e} | {mean(g,'scale'):.3f} |")

    w()
    w("## 2. 對齊失真之後：相位半邊 vs 幅度半邊")
    w()
    w("**只用 `amp_s`／`pha_s`。** 未對齊的 `amp`／`pha` 失真高於 `full`"
      "（交叉互換會放大擾動，PAD 表 1 自己記過同一現象），比值不可解讀。")
    w()
    w("| 條件 | amp_s 位移 | pha_s 位移 | pha/amp | 逐圖 pha 勝 | full 位移 |")
    w("|---|---|---|---|---|---|")
    for c in conds:
        a = {r["image"]: r["effect_mean"] for r in by.get((c, "amp_s"), [])}
        p = {r["image"]: r["effect_mean"] for r in by.get((c, "pha_s"), [])}
        f = {r["image"]: r["effect_mean"] for r in by.get((c, "full"), [])}
        common = sorted(set(a) & set(p))
        if not common:
            continue
        ma = statistics.fmean(a[i] for i in common)
        mp = statistics.fmean(p[i] for i in common)
        wins = sum(1 for i in common if p[i] > a[i])
        mf = statistics.fmean(f[i] for i in common) if f else float("nan")
        w(f"| {c} | {ma:.4f} | {mp:.4f} | **{mp/ma:.3f}** | "
          f"{wins}/{len(common)} | {mf:.4f} |")

    w()
    w("## 3. 半邊 vs 完整擾動（同 DISTS）")
    w()
    w("比值 > 1 代表「只留一半的頻譜資訊、失真相同」比完整擾動更能推開編輯。")
    w()
    w("| 條件 | pha_s/full | amp_s/full | pha_s 勝 full | amp_s 勝 full |")
    w("|---|---|---|---|---|")
    for c in conds:
        f = {r["image"]: r["effect_mean"] for r in by.get((c, "full"), [])}
        a = {r["image"]: r["effect_mean"] for r in by.get((c, "amp_s"), [])}
        p = {r["image"]: r["effect_mean"] for r in by.get((c, "pha_s"), [])}
        common = sorted(set(f) & set(a) & set(p))
        if not common:
            continue
        mf = statistics.fmean(f[i] for i in common)
        w(f"| {c} | {statistics.fmean(p[i] for i in common)/mf:.3f} | "
          f"{statistics.fmean(a[i] for i in common)/mf:.3f} | "
          f"{sum(1 for i in common if p[i]>f[i])}/{len(common)} | "
          f"{sum(1 for i in common if a[i]>f[i])}/{len(common)} |")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "spectral.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
        print(f"\n寫入 {args.out / 'spectral.md'}")


if __name__ == "__main__":
    main()
