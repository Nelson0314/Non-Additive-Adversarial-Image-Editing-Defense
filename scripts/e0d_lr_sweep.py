"""E0d — 學習率校準 — spec §7.1 延伸。

**動機**：pilot 實測 `lr = 0.05` 下總損失由 15.32 單調上升至 20.23，即優化
方向正確但步長過大而發散。原因是 Adam 的每座標更新量約為 `lr`，與梯度大小
無關；而 `U` 的初始尺度為 `init_std = 0.02`，`lr = 0.05` 等於每步把參數改動
2.5 倍於其自身尺度。

本腳本掃描 lr，判準是**總損失是否單調下降**，而非最終偏移大小——偏移大但
損失發散的設定不可用，因為它代表結果由隨機遊走決定而非優化。

回報三個量：
- `loss_final / loss_min`：發散程度，接近 1 表示收斂於最低點
- `monotone_frac`：相鄰步中損失下降的比例
- `shift_final`：達成的編輯偏移

執行：source env.sh && python scripts/e0d_lr_sweep.py --site P --out runs/e0d
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.defense.objective import LossConfig
from src.defense.optimize import OptimConfig, optimize
from src.models.sd import SDWrapper
from src.purify.ops import default_train_set
from src.residual.site_latent import LatentResidual
from src.residual.site_pixel import PixelResidual
from src.utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--data", default="data/dayn_testset")
    ap.add_argument("--out", default="runs/e0d")
    ap.add_argument("--sites", default="P,L")
    ap.add_argument("--lrs", default="0.05,0.02,0.008,0.003,0.001")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--k_inv", type=int, default=20)
    ap.add_argument("--t_max", type=int, default=500)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    sd = SDWrapper(args.model)
    from PIL import Image
    import torchvision.transforms as T

    p = sorted(Path(args.data).rglob("*.png"))[0]
    img = Image.open(p).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    x01 = T.ToTensor()(img).unsqueeze(0).to(device)
    print(f"[E0d] 影像 {p.stem}，rank={args.rank}，steps={args.steps}")

    rows = []
    for site in [s.strip() for s in args.sites.split(",")]:
        for lr in [float(v) for v in args.lrs.split(",")]:
            cfg = OptimConfig(
                steps=args.steps, lr=lr, k_inv=args.k_inv, t_max=args.t_max,
                n_edit=args.n_edit, n_eot=1, seed=args.seed,
                prompt_edit="a wrecked car after an accident", log_every=10**9,
            )
            if site == "P":
                mod = PixelResidual(
                    size=args.size, channels=3, max_rank=args.rank,
                    const_rank=args.rank, seed=args.seed,
                )
            else:
                lat = sd.latent_shape(args.size, args.size)
                mod = LatentResidual(
                    steps=cfg.k_inv, channels=lat[1], size=lat[-1],
                    max_rank=args.rank, const_rank=args.rank, seed=args.seed,
                )
            mod = mod.to(device)

            res = optimize(sd, mod, x01, cfg, LossConfig(), default_train_set())
            losses = [h["loss"] for h in res.history]
            drops = sum(1 for a, b in zip(losses, losses[1:]) if b < a)
            row = {
                "site": site, "lr": lr,
                "loss_first": round(losses[0], 4),
                "loss_final": round(losses[-1], 4),
                "loss_min": round(min(losses), 4),
                "ratio_final_min": round(losses[-1] / max(min(losses), 1e-9), 3),
                "monotone_frac": round(drops / max(1, len(losses) - 1), 3),
                "shift_final": round(res.history[-1]["edit_shift"], 4),
                "psnr_final": round(res.history[-1]["fid_psnr"], 2),
                "linf_final": round(res.history[-1]["fid_linf"], 4),
                "seconds": round(res.seconds, 1),
            }
            rows.append(row)
            print(
                f"[E0d] site={site} lr={lr:<7g} loss {row['loss_first']:>9.3f}"
                f" → {row['loss_final']:>9.3f} (min {row['loss_min']:>9.3f})"
                f"  單調下降 {row['monotone_frac']:.0%}"
                f"  shift={row['shift_final']:.4f}  psnr={row['psnr_final']:.2f}",
                flush=True,
            )
            del mod, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(out / "lr_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n[E0d] 建議（判準：單調下降比例最高且 ratio_final_min ≈ 1）")
    for site in sorted({r["site"] for r in rows}):
        sel = [r for r in rows if r["site"] == site]
        best = max(sel, key=lambda r: (r["monotone_frac"], -r["ratio_final_min"]))
        print(
            f"  site {site}: lr={best['lr']}  單調 {best['monotone_frac']:.0%}  "
            f"ratio={best['ratio_final_min']}  shift={best['shift_final']}"
        )
    print(f"[E0d] 寫入 {out / 'lr_sweep.csv'}")


if __name__ == "__main__":
    main()
