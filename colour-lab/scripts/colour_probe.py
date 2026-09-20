"""四自由度色彩濾鏡族的可行域取樣探針：只撒點，不最佳化。

這一支回答的問題
────────────────────────────────────────────────────────────────────
`src/defense/tone_desat_param.py` 的族只有四個參數。在**不做任何最佳化**的
前提下，把可行域撒滿取樣點，看這個族推得動的失真與（由
`scripts/edit_displacement.py` 接手的）編輯位移落在哪個量級。

撒點不是求解：這裡沒有目標函數、沒有梯度、沒有受害模型，所以也沒有
「指令外洩」的問題。輸出是每個點 × 每張圖的防禦 PNG 與一列讀數。

取樣規則
────────────────────────────────────────────────────────────────────
以可行域內點 `θ_c = (−0.15, 0, 0, 0.01)` 為中心，沿若干方向直線外推。每一個
方向的邊界距離 `r_max` 有封閉解（四道限制都是 θ 的仿射函數，見
`tone_desat_param.max_radius`），取樣半徑寫成它的比例：

| 半徑 | 名稱後綴 | 位置 |
|---|---|---|
| `0.5 · r_max` | `_mid` | 內部 |
| `0.9 · r_max` | `_near` | 貼近邊界 |

方向有兩類：

* **座標軸**，七個帶號方向。每一個軸方向對應曲線上一個看得懂的自由度：
  `tone_floor` = −t0（三個 Bernstein 係數一起往 0.75 走，壓得最暗）、
  `tone_ceiling` = +t0（往 1 走，趨近恆等）、`shadow_slope` = ±t1
  （暗端與亮端的斜率反向擺動）、`midtone_slope` = ±t2（中段斜率）、
  `desaturate` = +s。
  **`−s` 不取樣**：`θ_c` 的 `s = 0.01` 離下界 0 只有 0.01，整個方向走完與
  中心點的差只有 `s` 的 0.009，取樣點會與 `centre` 重覆。
* **隨機方向**，在各座標按其可行寬度 `(0.25, 0.20, 0.10, 0.10)` 縮放後的
  空間裡抽高斯向量再正規化——不縮放的話隨機方向幾乎全部落在 `t` 的三個軸
  張成的子空間裡，`s` 抽不到。種子寫進點的名字，抽樣可重現。

點數上限是 16，配額這樣分：中心點一個；七個座標軸與兩個隨機方向**一律**取
貼邊界的 `_near`；`_mid` 只給兩個隨機方向，以及邊界距離 `r_max` 在 θ 空間裡
最長的四個座標軸（`tone_ceiling` 0.150、`tone_floor` 0.100、`shadow_slope`
0.100、`highlight_slope` 0.100）。其餘三個軸（`desaturate` 0.090、
`midtone_slope` 0.075、`midtone_dip` 0.050）的內部點離中心不到 0.05，與
`centre` 的重覆度高，故只留邊界端。排序由程式算，不是寫死的名單。

合計 16 個點，完整的 θ 四欄逐點寫進 CSV。

每個點量什麼
────────────────────────────────────────────────────────────────────
一律在**量化到 8-bit 之後**量，交付的是 PNG，量化前的數字不是使用者拿到的
東西。

| 欄 | 來源 |
|---|---|
| `deltaE00` | `src/defense/color_amplitude.py::delta_e00`（skimage CIEDE2000） |
| `psnr` / `lpips` | `src/metrics/suite.py::MetricSuite.pairwise` |
| `low_freq_share` | `src/metrics/perturbation_band.py::low_frequency_share` |
| `blur_retention` | 同上檔的 `blur_retention`，σ = 2 |

**判準不由本腳本下。** 它只把數與圖擺出來。

用法（CPU，不佔卡）
    CUDA_VISIBLE_DEVICES= python scripts/colour_probe.py \\
        --data data/portraits --out runs/colour_probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.defense.assets import save_png  # noqa: E402
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.tone_desat_param import (  # noqa: E402
    THETA_CENTRE, apply_filter, feasible, max_radius, theta_to_coeffs,
    violations)
from src.metrics.perturbation_band import (  # noqa: E402
    BLUR_SIGMA, LOW_BAND, blur_retention, low_frequency_share)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

#: 各座標的可行寬度，抽隨機方向時用來把四個軸拉到同一個尺度。
COORD_WIDTH = (0.25, 0.20, 0.10, 0.10)

#: 座標軸方向。鍵是點名的前綴，值是 θ 空間裡的帶號單位向量。
AXIS_DIRECTIONS = {
    "tone_floor": (-1.0, 0.0, 0.0, 0.0),
    "tone_ceiling": (1.0, 0.0, 0.0, 0.0),
    "shadow_slope": (0.0, 1.0, 0.0, 0.0),
    "highlight_slope": (0.0, -1.0, 0.0, 0.0),
    "midtone_slope": (0.0, 0.0, 1.0, 0.0),
    "midtone_dip": (0.0, 0.0, -1.0, 0.0),
    "desaturate": (0.0, 0.0, 0.0, 1.0),
}

#: 取樣半徑（`r_max` 的比例）與它在點名裡的後綴。
RADII = {"mid": 0.5, "near": 0.9}

#: 隨機方向的種子。種子進點名，抽樣因此可重現也認得出來。
RANDOM_SEEDS = (11, 29)

#: 幾個座標軸方向可以多拿一個內部取樣點。見模組 docstring 的配額說明。
AXES_WITH_INTERIOR_POINT = 4


def random_direction(seed: int):
    """在按 `COORD_WIDTH` 縮放的空間裡抽一個單位方向。"""
    gen = np.random.default_rng(int(seed))
    g = gen.standard_normal(4) * np.asarray(COORD_WIDTH)
    return tuple(g / np.linalg.norm(g))


def build_points():
    """回傳 `[(名稱, 方向, 半徑比例, θ)]`。方向為 None 的那一筆是中心點。

    `_mid` 的配額依 `r_max` 由長到短發給座標軸，名單由這裡算出來，不寫死。
    """
    centre = tuple(float(v) for v in THETA_CENTRE)
    if not feasible(centre):
        raise SystemExit(f"中心點不可行：{violations(centre)}")
    out = [("centre", None, 0.0, centre)]

    reach = {n: max_radius(centre, d) for n, d in AXIS_DIRECTIONS.items()}
    ranked = sorted(reach, key=lambda n: (-reach[n], n))
    with_interior = set(ranked[:AXES_WITH_INTERIOR_POINT])

    plan = []
    for name, direction in AXIS_DIRECTIONS.items():
        keys = ("mid", "near") if name in with_interior else ("near",)
        plan.extend((name, direction, k) for k in keys)
    for seed in RANDOM_SEEDS:
        direction = random_direction(seed)
        plan.extend((f"random_seed{seed}", direction, k)
                    for k in ("mid", "near"))

    for name, direction, key in plan:
        span = max_radius(centre, direction)
        fraction = RADII[key]
        theta = tuple(c + fraction * span * d
                      for c, d in zip(centre, direction))
        if not feasible(theta, tol=1e-6):
            raise SystemExit(
                f"{name}_{key} 落在可行域外：{violations(theta)}。"
                "取樣規則或邊界距離的封閉解有問題，不要靜默夾回去")
        out.append((f"{name}_{key}", direction, fraction, theta))
    return out


def load_items(data: Path):
    """八張原圖，順序與 `data/portraits_manifest.json` 無關、只取檔名排序。"""
    items = []
    for cls in sorted(p for p in data.iterdir()
                      if p.is_dir() and p.name != "masks"):
        for png in sorted(cls.glob("*.png")):
            items.append((png.stem, png))
    if not items:
        raise SystemExit(f"{data} 底下沒有影像")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/portraits"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    help="預設 cpu。這一支不需要卡，別佔別人的")
    args = ap.parse_args()

    points = build_points()
    items = load_items(args.data)
    device = torch.device(args.device)
    suite = MetricSuite(device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(points)} 個取樣點 × {len(items)} 張影像，device={device}",
          flush=True)

    rows = []
    for name, direction, fraction, theta in points:
        d0, d1, d2 = theta_to_coeffs(theta)
        point_dir = args.out / "points" / name
        point_dir.mkdir(parents=True, exist_ok=True)
        for image, path in items:
            x = load_image_tensor(path, device, size=RESOLUTION).double()
            with torch.no_grad():
                y = apply_filter(x, theta)
            # 交付的是 PNG，所有讀數因此在量化後的影像上量。
            y8 = (y.clamp(0, 1) * 255).round() / 255
            save_png(y8.float(), point_dir / f"{image}__{name}__def.png")
            delta = (y8 - x)[0].cpu().numpy()
            with torch.no_grad():
                pair = suite.pairwise(x.float(), y8.float())
            rows.append({
                "point": name, "image": image,
                "direction": "centre" if direction is None
                             else "|".join(f"{v:+.6f}" for v in direction),
                "radius_fraction": fraction,
                "t0": round(float(theta[0]), 8),
                "t1": round(float(theta[1]), 8),
                "t2": round(float(theta[2]), 8),
                "s": round(float(theta[3]), 8),
                "d0": round(d0, 8), "d1": round(d1, 8), "d2": round(d2, 8),
                "deltaE00": round(float(delta_e00(x.float(), y8.float())), 4),
                "psnr": round(float(pair["psnr"]), 4),
                "lpips": round(float(pair["lpips"]), 6),
                "low_freq_share": round(low_frequency_share(delta), 6),
                "blur_retention": round(blur_retention(delta), 6),
                "low_freq_band_cycles_per_pixel": LOW_BAND,
                "blur_sigma": BLUR_SIGMA,
                "linf": round(float((y8 - x).abs().max()), 6),
                "defended_png": (point_dir
                                 / f"{image}__{name}__def.png").as_posix(),
            })
            write_csv(args.out / "colour_probe.csv", rows)
        last = rows[-1]
        print(f"[DONE] {name:24s} θ=({theta[0]:+.4f},{theta[1]:+.4f},"
              f"{theta[2]:+.4f},{theta[3]:.4f}) d=({d0:.3f},{d1:.3f},{d2:.3f})"
              f"  最後一張 ΔE00 {last['deltaE00']:.2f}"
              f" PSNR {last['psnr']:.2f}", flush=True)
    print(f"[ALLDONE] {args.out / 'colour_probe.csv'}（{len(rows)} 列）",
          flush=True)


if __name__ == "__main__":
    main()
