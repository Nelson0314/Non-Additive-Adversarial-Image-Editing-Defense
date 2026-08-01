"""E26 —— site C 的容量檢查：它到得了運作點嗎？

為什麼要在下 GPU 之前做這件事。site C 的色度變換是乘性的，其可達
失真受影像自身的飽和度限制：U = V = 0 的像素無論 ΔM 為何都不動。若在
`max_dev` 的上界下最大只能達到 LPIPS 0.01，而主網格的運作點是 τ ∈
{0.02, 0.05, 0.10}，那麼本位置根本進不了比較的場地，任何 GPU 時數都是浪費。

tiny-SD 的煙霧測試量到 site C 在 step 0 的梯度為 2.2e-08，而 site P 在同一
組態下是 1.9e-04，差四個數量級。那個數字受隨機權重影響不可解讀，但它是
做本檢查的動機。

量什麼。對真實的 512² 影像掃 `max_dev`，每個值取最壞情況的 ΔM
（全部元素頂到上界，正負隨機以免退化成單純的飽和度縮放），量：

- `lpips`：與 τ ∈ {0.02, 0.05, 0.10} 直接可比
- `local_acutance_dev`：與門檻 0.04 直接可比（預期結構上為 0）
- `linf`：與 site P 的 τ_linf = 0.06 可比
- `chroma_energy`：影像自身的色度能量，解釋跨影像的差異

判準（事先宣告）：site C 可用，要求存在某個 `max_dev` 使得 LPIPS 落在
[0.02, 0.10] 之內且鈍化偏差仍低於 0.04。若 LPIPS 在 `max_dev` 到 1.0 之前
都達不到 0.02，本位置在現行協定下不可用，應在下 GPU 前放棄。

不需要 SD、不需要 GPU。輸出 `runs/p8_site_c_capacity/capacity.csv`。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.local_acutance import local_acutance_dev  # noqa: E402
from src.residual.site_color import ColorResidual, rgb_to_yuv  # noqa: E402

OUT = ROOT / "runs" / "p8_site_c_capacity"
SIZE = 512
SEED = 20260728
MAX_DEVS = [0.05, 0.1, 0.15, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
TAUS = (0.02, 0.05, 0.10)


def load(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    import piq

    OUT.mkdir(parents=True, exist_ok=True)
    lpips = piq.LPIPS()
    files = sorted((ROOT / "data/dayn_testset").rglob("*.png"))
    rows = []

    for path in files:
        x = load(path)
        yuv = rgb_to_yuv(x)
        # 色度能量：本位置的可用容量直接由它決定
        chroma = float(yuv[:, 1:].pow(2).sum(dim=1).sqrt().mean())
        print(f"\n[{path.stem}] 平均色度量值 {chroma:.4f}")
        print(f"{'max_dev':>9s}{'lpips':>9s}{'acut_dev':>10s}{'linf':>8s}")

        for md in MAX_DEVS:
            mod = ColorResidual(size=SIZE, grid_size=32, init_std=0.0,
                                max_dev=md)
            # 最壞情況：全部元素頂到上界。正負號隨機，否則 ΔM 退化成
            # 「整體縮放色度」，那只是飽和度調整，不代表本參數化的能力。
            g = torch.Generator().manual_seed(SEED)
            sign = torch.randint(0, 2, mod.delta.shape, generator=g) * 2 - 1
            mod.delta.data = sign.float() * md * 10.0   # ×10 確保夾到上界

            with torch.no_grad():
                out = mod.pixel_residual(x)
                row = {
                    "image": path.stem, "max_dev": md, "chroma_energy": chroma,
                    "lpips": float(lpips(x.clamp(0, 1), out.clamp(0, 1))),
                    "acut_dev": float(local_acutance_dev(x, out)),
                    "linf": float((out - x).abs().max()),
                    "cdev_mean": mod.color_stats()["cdev_mean"],
                }
            rows.append(row)
            print(f"{md:>9.2f}{row['lpips']:>9.4f}{row['acut_dev']:>10.4f}"
                  f"{row['linf']:>8.4f}")

    with (OUT / "capacity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== 判準 ===")
    reach = {}
    for md in MAX_DEVS:
        sub = [r for r in rows if r["max_dev"] == md]
        lp = float(np.mean([r["lpips"] for r in sub]))
        ac = float(np.mean([r["acut_dev"] for r in sub]))
        reach[md] = (lp, ac)
        print(f"  max_dev={md:<5.2f} 平均 LPIPS {lp:.4f}  鈍化 {ac:.4f}")

    usable = [md for md, (lp, ac) in reach.items()
              if TAUS[0] <= lp <= TAUS[-1] and ac < 0.04]
    if usable:
        print(f"\n可用：max_dev ∈ {usable} 能把 LPIPS 帶進 "
              f"[{TAUS[0]}, {TAUS[-1]}] 且鈍化低於 0.04。site C 進得了運作點。")
    else:
        top = max(reach.values())[0]
        print(f"\n不可用：掃到 max_dev={MAX_DEVS[-1]} 時 LPIPS 最高只有 "
              f"{top:.4f}，未達最低的 τ={TAUS[0]}。色度變換的可達失真受影像"
              f"自身飽和度限制，本位置在現行協定下進不了比較的場地。"
              f"下 GPU 之前應放棄或改參數化。")
    print(f"\n寫入 {OUT}")


if __name__ == "__main__":
    main()
