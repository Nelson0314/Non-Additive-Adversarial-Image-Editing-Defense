"""探測本機 GPU 能否跑無梯度的 512² SDEdit。

E29 §2 量到的 10.3 GB 峰值是**含梯度的訓練**（H100、TF32 開）。無梯度推論是
另一件事，未量過。若本機 4 GB 跑得動，E31 的比對頁、劣化階梯的來源影像與
部分評測就不必占用雲端 GPU 時間。

本腳本不做任何降級退路：OOM 就讓它 OOM 並印出峰值。用 try/except 吞掉再落回
CPU，會讓「跑得動」與「跑不動但被掩蓋」分不出來。要另外量 fp16 或別的解析度，
用 --dtype / --size 另跑一次並在 log 中分開記錄。

執行：python scripts/e31_local_probe.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.sd import SDWrapper
from src.utils.device import peak_memory_mb, reset_peak_memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本探針要量的是 GPU 峰值記憶體，CPU 上沒有意義")
    p = torch.cuda.get_device_properties(0)
    print(f"[probe] {p.name} {p.total_memory / 2**30:.1f} GB "
          f"sm{p.major}{p.minor} torch {torch.__version__}", flush=True)

    sd = SDWrapper(args.model)
    device = torch.device("cuda")
    x = torch.rand(1, 3, args.size, args.size, device=device)
    emb = sd.encode_text("a wrecked car after an accident")
    emb_u = sd.encode_text("")
    lat = sd.latent_shape(args.size, args.size)

    reset_peak_memory()
    t0 = time.perf_counter()
    with torch.no_grad():
        n = sd.sample_edit_noise(torch.empty(lat, device=device), seed=0)
        y = sd.sdedit(x, emb, n, args.n_edit, strength=args.strength,
                      guidance_scale=args.guidance_scale, emb_uncond=emb_u)
    dt = time.perf_counter() - t0
    print(f"[probe] 無梯度 {args.size}² SDEdit（n_edit={args.n_edit}、"
          f"w={args.guidance_scale}）完成 {dt:.1f}s "
          f"peak={peak_memory_mb():.0f} MB  輸出 {tuple(y.shape)}", flush=True)
    print("[probe] 此處沒有 OOM 即代表 E31 的比對頁與評測可在本機跑，"
          "雲端只需負擔訓練")


if __name__ == "__main__":
    main()
