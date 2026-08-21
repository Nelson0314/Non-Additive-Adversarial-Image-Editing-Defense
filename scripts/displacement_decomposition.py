"""把「位移」拆成**鈍化**與**內容改變**兩部分。**不跑 GPU。**

問題
────────────────────────────────────────────────────────────────────
本專案的主讀數是位移 `edit_lpips = LPIPS(編輯(原圖), 編輯(防禦圖))`。它量的是
兩張編輯結果差多少，而**防禦成功的定義是攻擊方拿不到他要的東西**，兩者不是
同一件事。人眼在比對頁上直接看到這個落差：位移 0.137 的那一格，盆栽仍是被
指令要求的粉紅色、人臉完整、場景完整——攻擊方拿到了他要的東西，而報表把它
算成 0.137 的防禦。

判別實驗
────────────────────────────────────────────────────────────────────
若位移主要來自「編輯結果變鈍」而不是「內容變了」，那麼**把未防禦的編輯模糊
到與防禦後編輯相同的銳利度**，應該就能重現同樣大的 LPIPS。

    1. 量 acutance(編輯(原圖) -> 編輯(防禦圖))，得銳利度保留率 a
    2. 二分搜尋高斯 sigma，使 acutance(編輯(原圖) -> blur_sigma(編輯(原圖))) = a
    3. 比 LPIPS(編輯(原圖), blur_sigma(編輯(原圖))) 與實際位移

    blur_explained = 該模糊對照的 LPIPS ÷ 實際位移

`blur_explained` 接近 1 表示**整個位移都能由單純的模糊複製出來**，那不是防禦；
接近 0 表示位移確實來自內容改變。

模糊是**單參數、無語意**的操作，做不出任何內容改變，故它是「位移的下界解釋」
而不是一個競爭方法。這一條不依賴任何感知指標的校準。

用法：
    python scripts/displacement_decomposition.py --src <擺著 PNG 的目錄> \
        --out runs/displacement_decomposition/decomposition.csv
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from defense_purify_gallery import RESOLUTION, discover  # noqa: E402
from src.metrics.acutance import acutance  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.purify.ops import gaussian_blur  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

SIGMA_LO, SIGMA_HI = 1e-3, 12.0
BISECT_ROUNDS = 40


def sigma_matching_acutance(x: torch.Tensor, target: float) -> float:
    """找出使 `acutance(x, blur_sigma(x))` 等於 `target` 的 sigma。

    銳利度保留率隨 sigma 單調遞減（高斯是低通，sigma 越大通帶越窄），故二分
    搜尋有意義。目標落在可達範圍外時回傳邊界值並由呼叫端標記，**不外插**。
    """
    lo, hi = SIGMA_LO, SIGMA_HI
    if acutance(x, gaussian_blur(x, hi))["acutance_ratio"] > target:
        return float("nan")          # 連最大模糊都還比目標銳：目標不可達
    if acutance(x, gaussian_blur(x, lo))["acutance_ratio"] < target:
        return 0.0                   # 目標比原圖還銳：不需要模糊
    for _ in range(BISECT_ROUNDS):
        mid = 0.5 * (lo + hi)
        if acutance(x, gaussian_blur(x, mid))["acutance_ratio"] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    suite = MetricSuite(device=torch.device("cpu"))
    found = discover(args.src)
    rows: List[Dict] = []
    for image, by_cond in sorted(found.items()):
        for cond, kinds in sorted(by_cond.items()):
            if not {"edit_orig", "edit_def"} <= set(kinds):
                continue
            e_orig = load_image_tensor(kinds["edit_orig"], torch.device("cpu"),
                                       size=RESOLUTION)
            e_def = load_image_tensor(kinds["edit_def"], torch.device("cpu"),
                                      size=RESOLUTION)
            displacement = float(suite.pairwise(e_orig, e_def)["lpips"])
            a = acutance(e_orig, e_def)["acutance_ratio"]
            sigma = sigma_matching_acutance(e_orig, a)
            if sigma != sigma:                       # NaN：目標不可達
                blur_lpips, explained, reach = float("nan"), float("nan"), False
            else:
                blurred = gaussian_blur(e_orig, sigma) if sigma > 0 else e_orig
                blur_lpips = float(suite.pairwise(e_orig, blurred)["lpips"])
                explained = blur_lpips / displacement if displacement > 0 else float("nan")
                reach = True
            rows.append({
                "image": image, "condition": cond,
                "displacement": round(displacement, 5),
                "acutance_ratio": round(a, 4),
                "matched_sigma": round(sigma, 4) if reach else "",
                "blur_control_lpips": round(blur_lpips, 5) if reach else "",
                "blur_explained": round(explained, 4) if reach else "",
                "sigma_reachable": reach,
            })
            print(f"  {image[:34]:34s} {cond:16s} 位移 {displacement:.4f}  "
                  f"銳利度 {a:.3f}  sigma {sigma:.3f}  "
                  f"模糊可解釋 {explained:.1%}" if reach else
                  f"  {image[:34]:34s} {cond:16s} 位移 {displacement:.4f}  "
                  f"銳利度 {a:.3f}  目標不可達", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    print("\n條件層彙總（模糊可解釋的比例越高，位移越不能當防禦讀）")
    by_cond: Dict[str, List[float]] = {}
    for r in rows:
        if r["sigma_reachable"]:
            by_cond.setdefault(r["condition"], []).append(float(r["blur_explained"]))
    for cond, vals in sorted(by_cond.items(), key=lambda kv: -st.fmean(kv[1])):
        print(f"  {cond:18s} {st.fmean(vals):6.1%}  (n={len(vals)})")


if __name__ == "__main__":
    main()
