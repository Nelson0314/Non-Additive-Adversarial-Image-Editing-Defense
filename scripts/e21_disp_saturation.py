"""E21/E22 的位移飽和度診斷。

e21_report 的貼頂偵測不夠。它比對 `disp_max_px` 是否等於 1.5√2 ≈ 2.121，
那只抓得到「兩個分量同時飽和」的角落情形。實際的夾取是逐分量的：一個
分量在 ±1.5、另一個為 0 的像素其量值只有 1.5，該偵測抓不到，但它同樣已經
被上界綁死。

正確的量是分量層級的飽和比例：`|f| ≥ max_disp − ε` 的分量佔全部分量的
比例。此值大代表位移場被上界壓住，而不是被保真約束壓住——兩者是完全不同的
實驗結論。

位移場沒有被單獨存檔，但 `defended.png` 與 `orig.png` 都在，且位移場是
`WarpResidual` 唯一的參數；此處改由 `results.csv` 已記錄的 `disp_mean_px`、
`disp_max_px`、`disp_p99_px` 三個統計量推斷，並明確標示這是推斷而非直接量測。

推斷規則：`disp_p99_px` 已接近上界 `max_disp`（單分量）或 1.5√2（雙分量）
即代表大量像素貼頂。三個統計量一併列出，讓判讀者看得到依據。
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# (run 目錄前綴, max_disp)。E21 用預設 1.5，E22 明確放寬到 6.0。
GROUPS = [
    ("e21_Sbic", 1.5, "bicubic, max_disp 1.5"),
    ("e21_Sbil", 1.5, "bilinear, max_disp 1.5"),
    ("e22_Sbic_d6", 6.0, "bicubic, max_disp 6.0"),
]
TAUS = ["0.02", "0.05", "0.10"]


def stats(run: Path) -> dict:
    rows = list(csv.DictReader((run / "results.csv").open(encoding="utf-8")))
    seen = {}
    for r in rows:
        if r["image"] not in seen:
            seen[r["image"]] = r
    out = {k: [] for k in ("mean", "max", "p99", "lpips")}
    for r in seen.values():
        out["mean"].append(float(r["disp_mean_px"]))
        out["max"].append(float(r["disp_max_px"]))
        out["p99"].append(float(r["disp_p99_px"]))
        out["lpips"].append(float(r["final_lpips"]))
    return out


def main() -> None:
    print(f"{'run':>24s} {'τ':>5s} {'平均':>7s} {'p99':>7s} {'最大':>7s} "
          f"{'上界':>7s} {'p99/上界':>9s} {'LPIPS':>7s} {'預算':>6s}")
    print("-" * 92)
    for prefix, cap, label in GROUPS:
        mag_cap = cap * np.sqrt(2.0)     # 兩分量同時飽和時的量值
        for tau in TAUS:
            run = RUNS / f"{prefix}_tau{tau}"
            if not run.is_dir():
                continue
            s = stats(run)
            p99 = float(np.mean(s["p99"]))
            print(f"{prefix:>24s} {tau:>5s} {np.mean(s['mean']):>7.3f} "
                  f"{p99:>7.3f} {np.mean(s['max']):>7.3f} {mag_cap:>7.3f} "
                  f"{p99 / cap:>9.2f} {np.mean(s['lpips']):>7.4f} {tau:>6s}")
    print()
    print("「p99/上界」以單分量上界為分母：>0.7 即代表位移場的高分位數已接近")
    print("夾取邊界，該臂是被位移上界綁住而非被保真約束綁住。")
    print("「LPIPS vs 預算」若明顯低於 τ，同樣代表保真預算沒用完，綁定的是別的東西。")


if __name__ == "__main__":
    main()
