"""探測本機 GPU 能跑多大的**含梯度**防禦訓練。

E0 的成本模型量的是 512²：峰值約 9.95 GB（開 UNet 與 VAE checkpoint），
E29 在 H100 上實測 10.3 GB。本機只有 4 GB，512² 的訓練跑不動是已知的
（`docs/NEXT_SESSION.md` §7）。

未知的是**縮小解析度後能跑到多大**。若 256² 或 320² 跑得動，就能在本機
先跑一輪縮小版的 E31 網格，把三個 defense_mode 的相對行為看出來，再決定
雲端那一輪的設定——那是零雲端成本的去風險。

縮小版不能取代 512² 的正式結果：VAE 的下採樣倍率固定為 8，256² 的 latent
是 32²，注意力層的空間解析度全部減半，cross-attention 的綁定結構不同。
本腳本的用途是可行性與成本，不是結論。

不設任何降級退路：OOM 就讓它 OOM 並記下該解析度不可行。

執行：python scripts/e31_train_probe.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.defense.objective import LossConfig
from src.defense.optimize import OptimConfig, optimize
from src.models.sd import SDWrapper
from src.purify.ops import Purifier
from src.residual.site_pixel import PixelResidual
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--sizes", default="256,320,384")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n_edit", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本探針要量的是 GPU 峰值記憶體，CPU 上沒有意義")
    device = get_device()
    p = torch.cuda.get_device_properties(0)
    print(f"[probe] {p.name} {p.total_memory / 2**30:.1f} GB "
          f"torch {torch.__version__}", flush=True)

    sd = SDWrapper(args.model)
    for size in [int(s) for s in args.sizes.split(",")]:
        g = torch.Generator().manual_seed(20260728)
        x01 = torch.rand(1, 3, size, size, generator=g).to(device)
        module = PixelResidual(size=size, channels=3, max_rank=16,
                               const_rank=16, seed=20260728).to(device)
        cfg = OptimConfig(steps=args.steps, lr=0.03, k_inv=10,
                          n_edit=args.n_edit, strength=0.5,
                          guidance_scale=7.5, prompt_edit="a wrecked car",
                          unet_ckpt=True, vae_ckpt=True, log_every=1)
        loss_cfg = LossConfig(tau_lpips=0.10, beta_linf=0.0, alpha_lpips=0.0,
                              margin=1.0)
        reset_peak_memory()
        t0 = time.perf_counter()
        res = optimize(sd, module, x01, cfg, loss_cfg, [Purifier("identity")])
        dt = time.perf_counter() - t0
        print(f"[probe] size={size}  {args.steps} 步 {dt:.1f}s "
              f"（{dt / args.steps:.2f} s/step）  peak={peak_memory_mb():.0f} MB "
              f"shift={res.history[-1]['edit_shift']:.4f}", flush=True)
        del module, res, x01
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
