"""主讀數的飽和值：LPIPS 在「兩張不相干的影像」之間停在哪。**純 CPU。**

為什麼需要它
────────────────────────────────────────────────────────────────────
報表上的「倍數」欄是 `淨增益 ÷ 空白地板`。分母是地板，但**分子有天花板**：
主讀數是

    淨增益 = LPIPS(編輯(原圖), 編輯(算子(防禦圖))) − LPIPS(編輯(原圖), 編輯(算子(原圖)))
                    ↑ 有經驗上界 L_sat                        ↑ 地板

LPIPS 不會無限大——兩張語意不相干的自然影像之間會飽和。所以**每一欄真正的
可達範圍是 `L_sat − 地板`**。裁切 15% 的地板已經 0.5676，若 `L_sat ≈ 0.75`，
那一欄總共只剩 0.18 可拿，而我們拿了 0.0674——那是可達範圍的 37%，不是
「地板的 12%」。

**這不是說裁切其實不差**，是說那個數字裡有一大部分由分母的選擇造成，機制
診斷不該建立在它上面。

量兩個上界
────────────────────────────────────────────────────────────────────
1. `L_sat`（經驗飽和值）：十張**編輯結果**兩兩配對（45 對）的 LPIPS 分佈。
   這些都是真實的、語意各異的自然影像，是「完全擋下」時分子能到的地方。
   報中位數與各分位數——**不取最大值**，最大值是離群點不是飽和值。
2. `L_noise`（絕對上界）：編輯結果對純高斯雜訊圖。它比 L_sat 高，但那不是
   任何防禦達得到的狀態（把圖變成雜訊不叫「重畫成無關場景」）。並列是為了
   看 L_sat 離真正的天花板多遠。

**兩個都要報。** 只報其一會讓「可達範圍」的定義變成一個沒說清楚的選擇。

用法：
    python scripts/readout_ceiling_probe.py \\
        --defense runs/ip2p_ig_converge/ig_d25 --condition phase_gain \\
        --out runs/readout_ceiling/ig_d25.csv
"""

from __future__ import annotations

import argparse
import itertools
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defense", type=Path, required=True,
                    help="含 <影像>__<cond>__edit_orig.png 的目錄")
    ap.add_argument("--condition", default="phase_gain")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    import torch
    from src.metrics.suite import MetricSuite
    from src.utils.io import load_image_tensor, write_csv

    suite = MetricSuite()
    dev = suite.device
    paths = sorted(args.defense.glob(f"*__{args.condition}__edit_orig.png"))
    if len(paths) < 2:
        raise SystemExit(f"{args.defense} 裡的 edit_orig 少於兩張，配不出對")
    names = [p.name[: -len(f"__{args.condition}__edit_orig.png")] for p in paths]
    imgs = [load_image_tensor(p, dev, size=RESOLUTION) for p in paths]
    print(f"{len(imgs)} 張編輯結果，{len(imgs) * (len(imgs) - 1) // 2} 對")

    rows = []
    for (i, j) in itertools.combinations(range(len(imgs)), 2):
        v = suite.pairwise(imgs[i], imgs[j])
        rows.append({"kind": "pair", "a": names[i], "b": names[j],
                     "lpips": round(float(v["lpips"]), 5),
                     "dists": round(float(v["dists"]), 5)})

    # 絕對上界：對純高斯雜訊。**不是任何防禦達得到的狀態**，只用來看 L_sat
    # 離真正的天花板多遠。固定種子，跨批次可重現。
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.rand(1, 3, RESOLUTION, RESOLUTION, generator=g).to(dev)
    for k, x in zip(names, imgs):
        v = suite.pairwise(x, noise)
        rows.append({"kind": "noise", "a": k, "b": "gaussian_noise",
                     "lpips": round(float(v["lpips"]), 5),
                     "dists": round(float(v["dists"]), 5)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    pair = sorted(r["lpips"] for r in rows if r["kind"] == "pair")
    noi = [r["lpips"] for r in rows if r["kind"] == "noise"]
    q = lambda p: pair[min(len(pair) - 1, int(p * len(pair)))]
    print(f"\n{args.out}：{len(rows)} 列")
    print("L_sat（兩兩不相干的編輯結果）")
    print(f"  最小 {pair[0]:.4f}   10% {q(0.10):.4f}   中位 {q(0.50):.4f}"
          f"   90% {q(0.90):.4f}   最大 {pair[-1]:.4f}")
    print(f"L_noise（對純雜訊）平均 {st.fmean(noi):.4f}"
          f"（最小 {min(noi):.4f}）")
    print("\n**取哪一個當 L_sat 是裁定事項**：中位數是「典型的不相干」，"
          "10% 分位是保守下界。兩者都比 L_noise 低。")


if __name__ == "__main__":
    main()
