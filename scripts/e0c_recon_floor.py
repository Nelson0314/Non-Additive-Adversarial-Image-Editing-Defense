"""E0c — site L 的保真地板 — spec §7.1 延伸、§7.2 T1 修訂紀錄的量化。

**動機**：site P 在 φ=0 時 `x_def = x` 逐元素相等，保真項為 0。site L 在
φ=0 時 `x_def` 仍是「VAE 編碼 → DDIM inversion(k_inv) → 去噪(k_inv) →
VAE 解碼」來回一趟的重建，本身即帶誤差，保真項有一個 **φ 無法消除的地板**。

這個地板直接影響 E2 的公平性：若不量出來，P 與 L 的防禦—保真前緣比較會把
「重建誤差」誤記為「防禦造成的失真」，兩個 site 的起跑點不同而圖上看不出來。

本腳本量測該地板隨 k_inv 的變化，並與 VAE 單獨來回（k_inv=0）分離，以區分
誤差來自 VAE 還是來自 DDIM inversion 的不精確。

**判準**：若地板的 PSNR 低於 LossConfig.psnr_floor（預設 30 dB），則該地板
會持續觸發保真項的 PSNR hinge，此時 psnr_floor 必須依實測值下修，否則 site L
的優化會被一個它無法改善的常數項主導。

執行：source env.sh && python scripts/e0c_recon_floor.py --out runs/e0c
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.metrics.suite import MetricSuite
from src.models.sd import SDWrapper
from src.utils.artifacts import save_image
from src.utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--data", default="data/dayn_testset")
    ap.add_argument("--out", default="runs/e0c")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--k_list", default="0,2,5,10,20,30,50")
    ap.add_argument("--prompt_def", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    ks = [int(v) for v in args.k_list.split(",")]

    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)

    from PIL import Image
    import torchvision.transforms as T

    paths = sorted(Path(args.data).rglob("*.png"))
    print(f"[E0c] {len(paths)} 張影像，k_inv ∈ {ks}")

    rows = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        x = T.ToTensor()(img).unsqueeze(0).to(device)
        emb = sd.encode_text(args.prompt_def).detach()

        for k in ks:
            with torch.no_grad():
                z0 = sd.encode_image(x)
                if k == 0:
                    # k=0 只走 VAE 來回，用以分離 VAE 誤差與 inversion 誤差
                    xr = sd.decode_latent(z0)
                else:
                    ts = sd.timesteps(k)
                    z_inv = sd.ddim_inversion(z0, emb, ts, k)
                    z_back, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
                    xr = sd.decode_latent(z_back)

            m = suite.pairwise(x, xr)
            m["niqe_recon"] = suite.niqe(xr)
            rows.append({"image": p.stem, "k_inv": k, **m})
            print(
                f"[E0c] {p.stem:<12s} k={k:>2d}  psnr={m['psnr']:>6.2f}  "
                f"lpips={m['lpips']:.4f}  ssim={m['ssim']:.4f}  linf={m['linf']:.4f}",
                flush=True,
            )
            if p.stem == paths[0].stem:
                save_image(xr, out / f"recon_k{k:02d}.png")

        if p.stem == paths[0].stem:
            save_image(x, out / "orig.png")

    with open(out / "recon_floor.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n[E0c] 各 k_inv 的平均（n = %d 張）" % len(paths))
    print(f"{'k_inv':>6} {'PSNR':>8} {'LPIPS':>8} {'SSIM':>8} {'Linf':>8}")
    for k in ks:
        sel = [r for r in rows if r["k_inv"] == k]
        n = len(sel)
        print(
            f"{k:>6} {sum(r['psnr'] for r in sel) / n:>8.2f} "
            f"{sum(r['lpips'] for r in sel) / n:>8.4f} "
            f"{sum(r['ssim'] for r in sel) / n:>8.4f} "
            f"{sum(r['linf'] for r in sel) / n:>8.4f}"
        )
    print(f"\n[E0c] 寫入 {out / 'recon_floor.csv'}")


if __name__ == "__main__":
    main()
