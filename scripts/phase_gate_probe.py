"""site F 的前置量測：紋理閘的有效面積，與 theta_max 對可見失真的校準。

在跑任何最佳化之前先回答規格 §6 的風險三——真實照片的紋理區佔比是否足以
支撐足夠的自由度。同時量隨機相位（即 RPN 本身）在各個 theta_max 下的失真,
用來決定 A 臂兩個預算點各自要從哪個 theta_max 起步。

只用 CPU，不載入 Stable Diffusion。
"""

import argparse
import math
from pathlib import Path

import torch

from src.metrics.suite import MetricSuite
from src.residual.site_phase import PhaseResidual
from src.utils.io import load_image_tensor, write_csv

THETA_GRID = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, math.pi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--images", nargs="+",
                    default=["horse_00", "man_00", "bird_03",
                             "cat_02", "dog_03", "woman_03"])
    ap.add_argument("--out", default="runs/phase_gate_probe")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--r-min", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite = MetricSuite(device=device)

    rows = []
    for name in args.images:
        path = next(Path(args.data).rglob(f"{name}.png"), None)
        if path is None:
            raise FileNotFoundError(f"{name}.png 不在 {args.data} 下")
        x = load_image_tensor(path, device, size=args.size)

        probe = PhaseResidual(size=args.size, block=args.block, r_min=args.r_min)
        probe.prepare_gates(x)
        active = probe.active_fraction()

        for theta_max in THETA_GRID:
            m = PhaseResidual(
                size=args.size, block=args.block, r_min=args.r_min,
                theta_max=theta_max, init_std=theta_max, seed=args.seed,
            )
            m.prepare_gates(x)
            with torch.no_grad():
                x_def = m.pixel_residual(x).clamp(0.0, 1.0)
            pw = suite.pairwise(x_def, x)
            row = {
                "image": name,
                "active_fraction": round(active, 4),
                "theta_max": theta_max,
                "amp_dev": round(m.amplitude_deviation(x), 6),
                "linf_pixel": round(float((x_def - x).abs().max()), 5),
            }
            row.update({k: round(v, 5) for k, v in pw.items()})
            rows.append(row)
            print(f"{name:10s} theta={theta_max:5.3f} "
                  f"g={active:.3f} amp_dev={row['amp_dev']:.4f} "
                  f"lpips={pw.get('lpips', float('nan')):.4f} "
                  f"dists={pw.get('dists', float('nan')):.4f} "
                  f"psnr={pw.get('psnr', float('nan')):.2f}", flush=True)

    write_csv(out / "gate_probe.csv", rows)
    print(f"\n寫入 {out / 'gate_probe.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
