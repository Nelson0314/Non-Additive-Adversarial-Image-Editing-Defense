"""色彩重映射的半徑→失真對照。**不跑最佳化、不需要擴散模型、不需要 GPU。**

存在理由：`ColorCurveParam` 與 `ColorGridParam` 的半徑單位不同，且兩者都與
既有的相位半徑、`ShadingParam` 的 log 增益半徑不可換算。上 GPU 之前必須先
知道每個半徑落在哪一段失真上，否則整批掃描可能全部落在帶外——
`runs/shading_field_cost/` 是同一個位置的先例（該批的步驟 0 也是先在本機
量失真再決定要不要送）。

量的是**同半徑隨機場**的失真。隨機是上界：`runs/ip2p_shading` 實測同一個
半徑下最佳化的失真比隨機低（0.0165 對 0.0354），因為最佳化不會把參數推到
盒子的邊界、隨機的會。所以本表讀作「這個半徑最多付多少」。

四個軸一起報。**`deltaE00` 是為這一族新加的**：LPIPS 與 DISTS 都是結構／
紋理度量，對全域色偏的懲罰偏輕，只用它們對齊失真會系統性偏袒色彩方法。

用法：

    python scripts/color_field_cost.py --out runs/color_field_cost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.defense.color_param import (  # noqa: E402
    ColorCurveRandomParam, ColorGridRandomParam,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

# 六張，六類任務各一張。**前五張逐字取自 `runs/ip2p_shading` 的十三張**，
# 於是色彩族的數字能與明暗場的逐圖相減；第六張（`task_obj_add_13726`）是
# 補上「加物件」這一類，該類在明暗場那一批用的影像不在版控的資料集裡。
IMAGES = (
    "task_attr_mod_color_11699",
    "task_attr_mod_color_6205",
    "task_env_weather_112463",
    "task_obj_remove_380621",
    "task_obj_swap_joint_mask_276754",
    "task_obj_add_13726",
)

# 半徑的候選。曲線族是斜率剖面的動態範圍 `1 + r`、網格族是仿射係數對單位
# 矩陣的 L∞ 偏移，**兩者不可互相換算**，故各自給一組。
CURVE_RADII = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)
GRID_RADII = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)

# 每個 (影像, 半徑) 抽幾個隨機實現。單一實現的失真在隨機場上抖得很兇。
SEEDS = (0, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, default=Path("runs/color_field_cost"))
    ap.add_argument("--images", nargs="+", default=list(IMAGES))
    ap.add_argument("--grid", type=int, default=8,
                    help="ColorGridParam 的空間網格邊長 G")
    ap.add_argument("--luma-bins", type=int, default=8,
                    help="ColorGridParam 的亮度格數 D")
    ap.add_argument("--pieces", type=int, default=64,
                    help="ColorCurveParam 的分段數 K")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.images:
        path = args.data / name / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"找不到影像 {path}")
        x = load_image_tensor(path, device, size=512)
        for family, radii in (("color_curve_rand", CURVE_RADII),
                              ("color_grid_rand", GRID_RADII)):
            for r in radii:
                for seed in SEEDS:
                    if family == "color_curve_rand":
                        p = ColorCurveRandomParam(radius=r, pieces=args.pieces)
                        n_params = 3 * args.pieces
                    else:
                        p = ColorGridRandomParam(radius=r, grid=args.grid,
                                                 luma_bins=args.luma_bins)
                        n_params = 12 * args.luma_bins * args.grid ** 2
                    p.reset(x, seed)
                    y = p.render(x)
                    m = suite.pairwise(x, y)
                    rows.append({
                        "image": name, "condition": family, "radius": r,
                        "seed": seed, "n_params": n_params,
                        "grid": args.grid, "luma_bins": args.luma_bins,
                        "pieces": args.pieces,
                        "dists": round(m["dists"], 5),
                        "lpips": round(m["lpips"], 5),
                        "psnr": round(m["psnr"], 3),
                        "ssim": round(m["ssim"], 5),
                        "deltaE00": round(m["deltaE00"], 4),
                        "rms": round(m["rms"], 5),
                        "linf": round(m["linf"], 5),
                    })
                    print(f"{name:38s} {family:17s} r={r:<5g} seed={seed} "
                          f"DISTS={m['dists']:.4f} LPIPS={m['lpips']:.4f} "
                          f"dE00={m['deltaE00']:.2f} PSNR={m['psnr']:.2f}",
                          flush=True)

    write_csv(args.out / "results.csv", rows)
    print(f"\n寫出 {len(rows)} 列到 {args.out / 'results.csv'}")

    # 逐 (條件, 半徑) 的中位數，供選掃描點。
    import statistics
    print(f"\n{'condition':18s} {'radius':>7s} {'DISTS':>8s} {'LPIPS':>8s} "
          f"{'dE00':>7s} {'PSNR':>7s}")
    for cond in ("color_curve_rand", "color_grid_rand"):
        for r in sorted({x["radius"] for x in rows if x["condition"] == cond}):
            sel = [x for x in rows if x["condition"] == cond and x["radius"] == r]
            print(f"{cond:18s} {r:7g} "
                  f"{statistics.median(s['dists'] for s in sel):8.4f} "
                  f"{statistics.median(s['lpips'] for s in sel):8.4f} "
                  f"{statistics.median(s['deltaE00'] for s in sel):7.2f} "
                  f"{statistics.median(s['psnr'] for s in sel):7.2f}")


if __name__ == "__main__":
    main()
