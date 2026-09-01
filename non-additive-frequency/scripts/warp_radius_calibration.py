"""位移場三元對照的半徑校準與殘餘幾何量測。**不跑 GPU，不載編輯模型。**

要回答兩件在派工之前就必須知道的事：

1. **每個條件要掃哪些半徑。** 三格的「半徑 → 失真」關係差很多——
   `warp_roundtrip` 的幾何幾乎抵消，同一個半徑上的失真遠低於單次 warp。
   等失真比較要求三條曲線都跨過錨點（`matched_distortion_table.py` 拒絕
   外插），半徑猜錯就是整批白跑，而卡是多人共用的。
2. **`warp_roundtrip` 的殘餘幾何有多大。** 那一格宣稱「只剩內插 artifact」，
   但 `−f` 只是 `f` 的一階逆，殘餘量的量級是 `radius² / 粗網格間距`。
   不把它量出來寫進報表，那句宣稱就沒有證據。

最佳化的 `warp` 這一格不在本支裡：它要跑 IP2P 的編碼器。本支量的是那兩個
不最佳化的條件，而 sign PGD 會把 L∞ 球用滿，故 `warp` 的失真與 `warp_rand`
同量級——用後者的曲線挑半徑就夠了。

用法：
    python scripts/warp_radius_calibration.py --out runs/ip2p_warp \
        --images-file runs/ip2p_fair_comparison/images13.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.defense.param_pgd import (  # noqa: E402
    WarpRandomParam, WarpRoundTripParam,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
CONDS = {"warp_rand": WarpRandomParam, "warp_roundtrip": WarpRoundTripParam}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/ip2p_warp"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images-file", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"))
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[2, 4, 8, 12, 16, 24, 32, 48])
    ap.add_argument("--conditions", nargs="+", default=sorted(CONDS),
                    help="兩個條件的『半徑 → 失真』差一個數量級，通常要分兩次"
                         "跑不同的半徑清單")
    ap.add_argument("--suffix", default="",
                    help="附加在輸出檔名後，讓兩次不同半徑清單的結果不互相覆寫")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = [n for n in args.images_file.read_text(encoding="utf-8").split()
             if n]
    suite = MetricSuite(device=torch.device(args.device))
    imgs = []
    for n in names:
        hits = sorted((args.data / n).glob("*.png")) + \
            sorted((args.data / n).glob("*.jpg"))
        if not hits:
            raise SystemExit(f"{args.data / n} 底下沒有影像")
        imgs.append((n, load_image_tensor(hits[0], torch.device(args.device),
                                          size=RESOLUTION)))
    print(f"{len(imgs)} 張、{len(args.radii)} 個半徑、"
          f"{len(args.conditions)} 個條件",
          flush=True)

    rows = []
    for cond in args.conditions:
        cls = CONDS[cond]
        for rad in args.radii:
            per = {"dists": [], "psnr": [], "lpips": [], "disp": []}
            for name, x in imgs:
                p = cls(radius=rad, grid=args.grid)
                p.reset(x, args.seed)
                with torch.no_grad():
                    y = p.render(x)
                m = suite.pairwise(x, y)
                eff = p.effective_displacement(x)
                per["dists"].append(float(m["dists"]))
                per["psnr"].append(float(m["psnr"]))
                per["lpips"].append(float(m["lpips"]))
                per["disp"].append(
                    float(eff.pow(2).sum(1).sqrt().mean()))
            rows.append({
                "condition": cond, "radius": rad, "warp_grid": args.grid,
                "n_images": len(imgs), "seed": args.seed,
                "fid_dists": round(statistics.fmean(per["dists"]), 5),
                "fid_psnr": round(statistics.fmean(per["psnr"]), 3),
                "fid_lpips": round(statistics.fmean(per["lpips"]), 5),
                # 每個像素實際被搬了多遠（像素）。`warp_roundtrip` 這一欄就是
                # 殘餘幾何——「只剩內插 artifact」這句話的證據就是它。
                "effective_disp_px": round(statistics.fmean(per["disp"]), 4),
            })
            print(rows[-1], flush=True)
            write_csv(args.out / f"radius_calibration{args.suffix}.csv", rows)

    print(f"\n表：{args.out / 'radius_calibration.csv'}")


if __name__ == "__main__":
    main()
