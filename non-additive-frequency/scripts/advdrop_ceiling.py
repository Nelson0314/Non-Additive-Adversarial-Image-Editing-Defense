"""AdvDrop 的失真天花板：把量化表推到允許的最大值，它最多能造出多少失真。

`runs/ip2p_advdrop_band` 量到 AdvDrop 在這個威脅模型上構不到失真帶（500 步、
eps 220 只到 DISTS 0.0378，工作點是 0.1286–0.1447），先前歸因成「步數不夠」。
那只是表面。真正的限制在**機制**：

    AdvDrop 只能**拿走**已經在圖裡的東西，不能**加**東西進去。

原文（arXiv:2108.09034 §3.1 式 1）：`x' = D_I(Q_diff(D(x), q))`，
`s.t. ‖q − q_init‖∞ < ε`，`q_init = 1`，且「We **increase** the value of
quantization table q gradually to **drop the information**」。量化只會把 DCT
係數往較粗的格點靠，所以殘差的上限由**影像自己的高頻內容**決定：一張平坦的
圖沒有東西可丟。加性攻擊沒有這條限制——它可以無限往上加。

這支腳本把 `q` 直接設成允許的最大值 `q_init + ε`（**每一格、每一個區塊都推到
底**，也就是最佳化在該 ε 下能達到的最極端的解），量它的失真。那個數字就是
AdvDrop 在該 ε 下的**天花板**，與跑幾步無關。

**不需要最佳化、不需要 GPU、不需要擴散模型**——只有 DCT、量化、IDCT 與指標。

用法：

    python scripts/advdrop_ceiling.py --out runs/ip2p_advdrop_band/ceiling.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.baselines.advdrop import (  # noqa: E402
    PAPER_ALPHA_LO, PAPER_Q_INIT, init_q_tables, render_advdrop,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 論文 §4.3 Table 1 掃的三個 eps，加上本專案在 band 掃描用過的兩個更大的值。
EPS_LIST = (20.0, 60.0, 100.0, 150.0, 220.0, 500.0, 1000.0)
# 本方法工作點的失真帶（`b_ph_rpi` 與 `b_pg_r20`）。
BAND_LO, BAND_HI = 0.1286, 0.1447


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    # 軟四捨五入的最硬設定，與最佳化跑到最後一步時相同。
    alpha = torch.tensor(PAPER_ALPHA_LO, device=device, dtype=torch.float32)

    rows = []
    for name in names:
        x = load_image_tensor(dataset[name]["path"], device, size=RESOLUTION)
        for eps in EPS_LIST:
            q = init_q_tables(x, PAPER_Q_INIT + eps)
            with torch.no_grad():
                x_adv = render_advdrop(x, q, alpha, "rgb", False)
                m = suite.pairwise(x, x_adv)
            rows.append({
                "image": name, "eps": eps,
                "q": PAPER_Q_INIT + eps,
                "dists": round(m["dists"], 6), "lpips": round(m["lpips"], 6),
                "psnr": round(m["psnr"], 4), "ssim": round(m["ssim"], 6),
                "rms": round(m["rms"], 6), "linf": round(m["linf"], 6),
            })
        write_csv(args.out, rows)
        print(f"{name} 完成", flush=True)

    print()
    print("把量化表整片推到 q = 1 + eps（該 eps 下最極端的解）的失真：")
    print(f"{'eps':>7s} {'q':>8s} {'DISTS':>8s} {'PSNR':>7s} {'RMS':>8s} {'L∞':>7s}")
    for eps in EPS_LIST:
        sel = [r for r in rows if r["eps"] == eps]
        f = lambda k: statistics.fmean(r[k] for r in sel)  # noqa: E731
        print(f"{eps:7.0f} {PAPER_Q_INIT + eps:8.0f} {f('dists'):8.5f} "
              f"{f('psnr'):7.2f} {f('rms'):8.5f} {f('linf'):7.4f}")

    top = [r for r in rows if r["eps"] == EPS_LIST[-1]]
    ceil = statistics.fmean(r["dists"] for r in top)
    print()
    print(f"本方法的失真帶是 DISTS {BAND_LO}–{BAND_HI}。")
    print(f"eps = {EPS_LIST[-1]:.0f}（q = {PAPER_Q_INIT + EPS_LIST[-1]:.0f}，"
          f"遠超論文 §4.3 掃的 20/60/100）的天花板是 {ceil:.5f}。")
    if ceil < BAND_LO:
        print("-> **即使把量化表推到底，AdvDrop 也進不了失真帶**，"
              "限制在機制不在步數。")
    else:
        print("-> 天花板進得了帶，構不到是最佳化的問題不是機制的問題。")
    print(f"表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
