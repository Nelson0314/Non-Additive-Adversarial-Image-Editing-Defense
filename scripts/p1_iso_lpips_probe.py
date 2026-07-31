"""P1 —— iso-LPIPS 的「模糊 vs 雜訊」探針。

**這是 MAD competition（Wang & Simoncelli 2008）的簡化版。** 該方法的作法是
合成一對刺激，把一個指標的值固定住、讓另一個指標盡量分開，用以暴露前者的
失效模式。此處把「固定住的指標」取為 LPIPS——因為它正是目前的綁定約束——
而把兩個刺激取為兩種**方向相反**的失真：

- **模糊**（高斯低通）：抽掉高頻。site S 與低 λ 的 latent_opt 買的就是這個。
- **雜訊**（加性高斯）：加上高頻。site P 這類加性方法的失真屬於這一側。

兩者都以二分搜尋校準到**相同的 LPIPS**。若某個候選指標在這個等 LPIPS 的
配對上給出相近的值，它就與 LPIPS 共享同一個盲區，加進約束也擋不住模糊；
若它明顯把模糊判得比較貴，它才有資格成為新的約束項。

目標 LPIPS 取 {0.02, 0.05, 0.10}，與 E15 主網格的三個 τ 相同，使結論可以
直接套回那組資料。

輸出：`runs/p1_iso_lpips_probe/`（CSV 與逐圖 PNG，供比對頁使用）。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.battery import MetricBattery

ROOT = Path(__file__).resolve().parent.parent
SRC_RUN = ROOT / "runs" / "e15_S_tau0.05"
OUT = ROOT / "runs" / "p1_iso_lpips_probe"

TARGET_LPIPS = [0.02, 0.05, 0.10]

# 二分搜尋的容差與上限。上限取得寬，讓 σ=8 這種極端模糊也在搜尋範圍內；
# 真正的落點由 LPIPS 決定，不由上限決定。
TOL = 1e-4
MAX_ITERS = 40
SIGMA_HI = 8.0
NOISE_HI = 0.5

# 雜訊的隨機種子固定，使整份結果可重現。逐圖不同種子會讓「同一張圖的
# 兩個 τ」用到不同的雜訊場，量到的差異會混入取樣變異。
SEED = 20260731


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def save(x: torch.Tensor, p: Path) -> None:
    a = (x[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).save(p)


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """高斯模糊。核大小隨 σ 調整至 ±3σ，避免核被截斷而使實際 σ 小於名目值。"""
    if sigma <= 0:
        return x.clone()
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def noisy(x: torch.Tensor, amp: float, gen: torch.Generator) -> torch.Tensor:
    """加性高斯雜訊後夾回 [0,1]。

    夾回是必要的：不夾就不是一張可顯示的影像，量到的指標也不對應人眼看到的
    東西。夾回會讓實際的雜訊強度略低於 amp，但校準是對 LPIPS 做的，這個
    偏差已被吸收。
    """
    n = torch.randn(x.shape, generator=gen, dtype=x.dtype)
    return (x + amp * n).clamp(0, 1)


def calibrate(make, lpips_fn, target: float, hi: float) -> tuple[float, torch.Tensor, float]:
    """二分搜尋單調參數，使 LPIPS(orig, make(t)) 命中 target。

    回傳 (參數, 影像, 實際 LPIPS)。搜尋前先檢查上限是否夠——若 LPIPS 在
    上限處仍低於 target，二分會靜默收斂到上限並回報一個看似命中的結果，
    故此處直接拋出而非讓錯誤往下游傳。
    """
    top = lpips_fn(make(hi))
    if top < target:
        raise ValueError(
            f"搜尋上限 {hi} 處的 LPIPS 僅 {top:.4f}，低於目標 {target}；"
            "二分會收斂到上限並回報假的命中，需調高上限"
        )
    lo, hi_ = 0.0, hi
    for _ in range(MAX_ITERS):
        mid = 0.5 * (lo + hi_)
        img = make(mid)
        val = lpips_fn(img)
        if abs(val - target) < TOL:
            return mid, img, val
        if val < target:
            lo = mid
        else:
            hi_ = mid
    img = make(0.5 * (lo + hi_))
    return 0.5 * (lo + hi_), img, lpips_fn(img)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bat = MetricBattery()

    def lpips_of(ref):
        return lambda y: float(bat._lpips(ref.clamp(0, 1), y.clamp(0, 1)))

    origs = sorted(SRC_RUN.glob("*/orig.png"))
    if not origs:
        raise FileNotFoundError(f"找不到原圖：{SRC_RUN}")

    rows = []
    for op in origs:
        name = op.parent.name.split("__")[0]
        x = load(op)
        lp = lpips_of(x)
        save(x, OUT / f"{name}__orig.png")

        for tgt in TARGET_LPIPS:
            sigma, x_blur, lp_b = calibrate(lambda s: blur(x, s), lp, tgt, SIGMA_HI)
            # 每次呼叫都重建 generator，使雜訊場只隨 amp 縮放而不隨搜尋步數
            # 改變；否則二分搜尋的目標函數不是單調的，收斂點沒有意義。
            amp, x_noise, lp_n = calibrate(
                lambda a: noisy(x, a, torch.Generator().manual_seed(SEED)),
                lp, tgt, NOISE_HI,
            )

            tag = f"{tgt:.2f}".replace(".", "p")
            save(x_blur, OUT / f"{name}__lpips{tag}__blur.png")
            save(x_noise, OUT / f"{name}__lpips{tag}__noise.png")

            mb = bat.evaluate(x, x_blur)
            mn = bat.evaluate(x, x_noise)
            rows.append({
                "image": name, "target_lpips": tgt,
                "sigma": sigma, "noise_amp": amp,
                "lpips_blur": lp_b, "lpips_noise": lp_n,
                **{f"blur_{k}": v for k, v in mb.items()},
                **{f"noise_{k}": v for k, v in mn.items()},
            })
            print(f"{name}  τ={tgt:.2f}  σ={sigma:.3f} (LPIPS {lp_b:.4f})  "
                  f"amp={amp:.4f} (LPIPS {lp_n:.4f})")

    with (OUT / "probe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 判定另置於 p1_summary：它只讀 CSV，不重跑探針，故修正判定方式時
    # 不必再花一次二分搜尋的成本。
    import p1_summary

    p1_summary.main()


if __name__ == "__main__":
    main()
