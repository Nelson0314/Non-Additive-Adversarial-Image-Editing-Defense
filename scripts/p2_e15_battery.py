"""P2 —— 用候選指標組重判 E15 的 site S vs site P。

P1 在**合成**的等 LPIPS 配對上篩出「會對模糊收費」的指標。P2 檢查那些指標
在**真實**資料上是否同樣有效：E15 主網格宣稱「匹配 LPIPS 後 site S 領先
site P 約 1.15 倍」，而該匹配只對齊了 LPIPS。

要回答的是兩個問題：

1. **哪些指標會揭穿那個匹配？** 若某指標在 S 與 P 之間差距顯著，它就能在
   訓練時擋住 site S 靠模糊換來的那部分防禦效果。
2. **`acutance` 的抵銷漏洞在真實資料上有多大？** `acutance` 是全域梯度能量
   比，一處模糊、他處加噪可以互相抵銷；GMSD 取的是逐點梯度相似度的空間
   標準差，理論上堵住該路徑。兩者在此並排量測。

不需 GPU：`runs/e15_{S,P}_tau*/` 的 `orig.png` 與 `defended.png` 已入庫。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.battery import HIGHER_IS_BETTER, MetricBattery

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "p2_e15_battery"

# **只有 τ=0.05 的逐圖影像入了版控。** 其餘兩個 τ 的 PNG（共 446 MB）留在
# TWCC 的 NFS 上（見 docs/NEXT_SESSION.md §6），本機沒有。此處明列可用的 τ
# 而非讓迴圈掃過三個再跳過缺檔者：靜默跳過會產生一整排 NaN，看起來像
# 「指標算不出來」，實際上是資料不在。
TAUS = ["0.05"]
SITES = ["S", "P"]


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bat = MetricBattery()
    rows = []

    for tau in TAUS:
        for site in SITES:
            run = RUNS / f"e15_{site}_tau{tau}"
            if not run.is_dir():
                raise FileNotFoundError(f"缺少 run 目錄：{run}")
            pairs = [(d / "orig.png", d / "defended.png") for d in sorted(run.iterdir())
                     if d.is_dir()]
            pairs = [(o, x) for o, x in pairs if o.exists() and x.exists()]
            if not pairs:
                raise FileNotFoundError(
                    f"{run} 沒有任何 orig.png / defended.png 配對。該 τ 的逐圖影像"
                    "可能未入庫（僅 τ=0.05 有），需先由 TWCC 的 NFS 取回；"
                    "此處不跳過，因為跳過會讓結果變成一整排看似算不出來的 NaN"
                )
            for o, x in pairs:
                d = o.parent
                m = bat.evaluate(load(o), load(x))
                rows.append({"tau": tau, "site": site,
                             "image": d.name.split("__")[0], **m})
                print(f"tau={tau} site={site} {d.name.split('__')[0]:>10s}  "
                      f"lpips={m['lpips']:.4f} gmsd={m['gmsd']:.4f} "
                      f"nlpd={m['nlpd']:.4f} acut={m['acutance_ratio']:.3f}")

    with (OUT / "battery.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report(rows)


def report(rows: list[dict]) -> None:
    """逐 τ 逐指標列出 S 與 P 的均值±標準差，並標示哪些指標揭穿了匹配。

    「揭穿」的操作型定義：在該 τ 下，S 的均值比 P 差，且差距大於兩者標準差
    之和（即兩組在 n=6 下不重疊）。這是刻意保守的判準——樣本只有 6 張，
    用重疊與否而非 t 檢定，避免對小樣本套用需要更多假設的統計量。
    """
    keys = [k for k in rows[0] if k not in ("tau", "site", "image")
            and not k.endswith("_orig")]
    lines = []
    for tau in TAUS:
        print()
        print(f"=== τ = {tau} ===")
        print(f"{'指標':>16s} {'site S':>18s} {'site P':>18s}  判定")
        print("-" * 68)
        for k in keys:
            s = np.array([r[k] for r in rows if r["tau"] == tau and r["site"] == "S"])
            p = np.array([r[k] for r in rows if r["tau"] == tau and r["site"] == "P"])
            if k == "acutance_ratio":
                worse_s = abs(s.mean() - 1) > abs(p.mean() - 1)
            elif HIGHER_IS_BETTER.get(k, False):
                worse_s = s.mean() < p.mean()
            else:
                worse_s = s.mean() > p.mean()
            separated = abs(s.mean() - p.mean()) > (s.std() + p.std())
            verdict = ("揭穿：S 明顯較差" if worse_s and separated else
                       "S 較差但重疊" if worse_s else
                       "P 較差")
            line = (f"{k:>16s} {s.mean():>10.4f}±{s.std():<7.4f} "
                    f"{p.mean():>10.4f}±{p.std():<7.4f}  {verdict}")
            print(line)
            lines.append(f"tau={tau} " + line)
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("「揭穿」= S 的均值較差且兩組在 n=6 下不重疊（差距 > 標準差之和）。")


if __name__ == "__main__":
    main()
