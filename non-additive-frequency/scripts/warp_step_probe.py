"""位移場上 sign PGD 的步長診斷：`--steps 1000` 那一批為什麼一動也沒動。

症狀：`runs/ip2p_warp/opt_r*` 第一輪跑出 `dists=0.0001`／`effect=0.0005`
——1000 步跑完，防禦圖與原圖幾乎逐位相同。**那不是「最佳化買不到東西」，
是最佳化沒有動過**，照那個數字寫結論會得到一個假的否證。

要分辨的兩件事：

1. **梯度是不是 0。** 是的話問題在前向（圖被切斷、精度、閘）。
2. **梯度不是 0，但 `sign` 每一步翻面。** 那是步長問題：雙線性重取樣對座標
   的梯度在**整數位移**處是單邊的，`c = 0` 恰好落在那個折點上，步長遠小於
   一個像素時 `c` 會在 0 與 ±α 之間形成週期 2 的振盪並**精確回到起點**。
   `run_param_pgd` 的步長是 `radius / (steps × saturate_at)`，`--steps 1000`
   配 `radius 3` 只有 0.012 px。

本支對同一張圖掃幾個步長，逐步印出損失與 `|c|`，讓上面兩者可以分開。
**要 GPU**（用 IP2P 自己的 VAE），但只跑一張圖、幾十步。

用法：
    CUDA_VISIBLE_DEVICES=6 python scripts/warp_step_probe.py \
        --image task_attr_mod_color_11699 --radius 3 --steps 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.defense.param_pgd import WarpParam  # noqa: E402
from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/ip2p_warp"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--image", default="task_attr_mod_color_11699")
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.012, 0.05, 0.125, 0.25, 0.5])
    ap.add_argument("--loss", choices=("latent_norm", "encoder_target"),
                    default="latent_norm",
                    help="要診斷的目標。`latent_norm` 是 DCT-Shield §4.2 的"
                         "`‖E(x′)‖₂`（本批原本用的），`encoder_target` 是本"
                         "專案的 `‖E(x′) − E(y)‖²`。**兩者對位移場的行為不同**"
                         "，這正是本支要分開量的東西")
    ap.add_argument("--target", type=Path, default=Path("data/targets/gray.png"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ip2p = IP2PWrapper(dtype=torch.float32)
    hits = sorted((args.data / args.image).glob("*.png"))
    if not hits:
        raise SystemExit(f"{args.data / args.image} 底下沒有影像")
    x = load_image_tensor(hits[0], ip2p.device, size=RESOLUTION)

    if args.loss == "latent_norm":
        def loss_fn(z):
            return ip2p.encode_image(z).flatten().norm(p=2)
    else:
        from src.baselines.encoder_target import make_encoder_target_loss
        y_target = load_image_tensor(args.target, ip2p.device, size=RESOLUTION)
        loss_fn = make_encoder_target_loss(ip2p, y_target)

    rows = []
    for alpha in args.alphas:
        p = WarpParam(radius=args.radius, grid=args.grid)
        p.reset(x, 0)
        c = p.c
        for i in range(args.steps):
            y = p.render(x)
            L = loss_fn(y)
            g, = torch.autograd.grad(L, [c])
            gabs = float(g.detach().abs().mean())
            gzero = float((g.detach() == 0).float().mean())
            with torch.no_grad():
                c.sub_(alpha * torch.sign(g))
            p.project()
            if i % max(1, args.steps // 6) == 0 or i == args.steps - 1:
                rows.append({
                    "loss_name": args.loss, "radius": args.radius,
                    "alpha_px": alpha, "step": i,
                    "loss": round(float(L.detach()), 4),
                    "grad_absmean": f"{gabs:.3e}",
                    "grad_zero_frac": round(gzero, 4),
                    "c_absmean": round(float(c.detach().abs().mean()), 5),
                    "c_absmax": round(float(c.detach().abs().max()), 5),
                    "pixel_absmean": round(
                        float((p.render(x).detach() - x).abs().mean()), 6),
                })
                print(rows[-1], flush=True)
        write_csv(args.out / f"step_probe_{args.loss}.csv", rows)

    print(f"\n表：{args.out / f'step_probe_{args.loss}.csv'}")


if __name__ == "__main__":
    main()
