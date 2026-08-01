"""P1 的判定 —— 在等 LPIPS 下，哪些指標把模糊判得比雜訊貴？

讀 `runs/p1_iso_lpips_probe/probe.csv`，不重跑探針。

為什麼需要先做「失真化」轉換。各指標的完美值不同（GMSD 為 0、SSIM 為 1、
PSNR 為 +∞、銳利度比為 1），直接拿原始值算相對差距會系統性低估「1 為完美」
那一類：SSIM 0.9980 與 0.9970 的相對差只有 0.1%，但離完美的缺口是 0.0020
與 0.0030，差 1.5 倍。故一律先映射成「0 為完美、越大越糟」的失真量：

| 指標 | 失真化 |
|---|---|
| gmsd, nlpd, lpips, dists, stlpips | v |
| ssim, ms_ssim, haarpsi, vif_p | 1 − v |
| psnr | 10^(−v/10)，即還原成 MSE |
| acutance_ratio | \\|1 − v\\| |
| niqe | v − niqe(原圖) |
| musiq | musiq(原圖) − v |

無參考的兩項以「相對原圖退步多少」為失真量：它們量的是影像本身的自然度，
絕對值含有原圖自身的品質，不減掉就不是在量失真。

判定量為無尺度的相對差距

    gap = (d_模糊 − d_雜訊) / max(|d_模糊|, |d_雜訊|) ∈ [−1, 1]

gap > 0 代表該指標對模糊收的費比 LPIPS 多，因為 LPIPS 在此被固定為相等，
其自身的 gap 恆為 0。三種判定的意義不同：

- `gap > +0.10`：對模糊額外收費 → 有資格加入約束。
- `|gap| < 0.10`：與 LPIPS 共享盲區 → 加進去擋不住模糊。
- `gap < −0.10`：對雜訊收費更多 → 反效果。加進去會讓 site P（加性）
  更難通過、site S（模糊）更容易通過，正好與目的相反。
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "p1_iso_lpips_probe"

TARGETS = [0.02, 0.05, 0.10]
THRESH = 0.10

# 順序即報告的閱讀順序：先是被固定住的 LPIPS，再依 gap 由大到小人工排版。
KEYS = ["lpips", "acutance_ratio", "stlpips", "gmsd", "dists", "haarpsi",
        "ms_ssim", "ssim", "psnr", "vif_p", "nlpd", "niqe", "musiq"]

UNIT_TOP = {"ssim", "ms_ssim", "haarpsi", "vif_p"}   # 1 為完美
DIRECT = {"gmsd", "nlpd", "lpips", "dists", "stlpips"}  # 0 為完美


def distortion(key: str, row: dict, side: str) -> float:
    """把指標值映射成「0 為完美、越大越糟」的失真量。"""
    v = float(row[f"{side}_{key}"])
    if key in DIRECT:
        return v
    if key in UNIT_TOP:
        return 1.0 - v
    if key == "psnr":
        return 10.0 ** (-v / 10.0)
    if key == "acutance_ratio":
        return abs(1.0 - v)
    if key == "niqe":
        return v - float(row[f"{side}_niqe_orig"])
    if key == "musiq":
        return float(row[f"{side}_musiq_orig"]) - v
    raise KeyError(f"未定義失真化轉換的指標：{key}。新增指標時必須一併定義，"
                   "否則它會被靜默當成 0 為完美而給出錯誤的判定")


def main() -> None:
    rows = list(csv.DictReader((OUT / "probe.csv").open(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError("probe.csv 是空的，請先跑 scripts/p1_iso_lpips_probe.py")

    summary, lines = {}, []
    for tgt in TARGETS:
        sub = [r for r in rows if abs(float(r["target_lpips"]) - tgt) < 1e-9]
        print(f"\n=== 目標 LPIPS = {tgt:.2f}（n={len(sub)}）===")
        print(f"{'指標':>16s} {'模糊失真':>10s} {'雜訊失真':>10s} {'gap':>8s}  判定")
        print("-" * 62)
        for k in KEYS:
            db = np.array([distortion(k, r, "blur") for r in sub])
            dn = np.array([distortion(k, r, "noise") for r in sub])
            # 逐圖算 gap 再平均，而非先平均失真量：後者會被絕對量級大的影像主導
            per = (db - dn) / np.maximum(np.abs(db), np.abs(dn)).clip(1e-12)
            gap = float(per.mean())
            verdict = ("對模糊額外收費" if gap > THRESH else
                       "反效果：對雜訊收費更多" if gap < -THRESH else
                       "與 LPIPS 同盲區")
            summary[f"{k}@{tgt}"] = {
                "d_blur": float(db.mean()), "d_noise": float(dn.mean()),
                "gap": gap, "gap_std": float(per.std()), "verdict": verdict,
            }
            line = (f"{k:>16s} {db.mean():>10.5f} {dn.mean():>10.5f} "
                    f"{gap:>+8.3f}  {verdict}")
            print(line)
            lines.append(f"tau={tgt:.2f} " + line)

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print()
    print("失真量已統一為「0 為完美、越大越糟」。gap 為逐圖計算後平均。")
    print(f"gap > +{THRESH}：對模糊額外收費，有資格加入約束。")
    print(f"|gap| < {THRESH}：與 LPIPS 共享盲區。")
    print(f"gap < -{THRESH}：反效果，會讓模糊更容易通過而加性更難通過。")


if __name__ == "__main__":
    main()
