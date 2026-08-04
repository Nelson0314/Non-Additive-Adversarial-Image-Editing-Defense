"""本機能不能跑 cross-attention 目標的訓練？

    python scripts/l4_crossattn_probe.py

## 為什麼要問這一題

`e31_train_probe.py` 量到本機（RTX 2050 4 GB）跑 `optimize()` 在 256² 是
178 s/step，即 H100 在 512² 上的 75 倍，因而判定「本機無法執行任何規模可用的
訓練」（LEDGER 6.7）。**但那個判定只涵蓋 `optimize()`。**

`optimize_crossattn()` 的成本結構不同。E0 的成本模型是

    秒 ≈ 1.05 + 0.384·k_inv + 0.304·n_edit

`optimize()` 每步要跑一條 `n_edit = 10` 步的 SDEdit 鏈（第三項），site PF
不走反演故第二項為零。`optimize_crossattn()` **完全不跑那條鏈**，改為
`attn_timesteps` 次單步 UNet 前向（預設 4），即第三項由 10 步的鏈變成
4 次單步前向。理論上省一個量級。

若省得夠多，本機就能跑「site PF + crossattn 在真實 SD 上有沒有效」這一題，
而該題至今**從未有過任何資料**：`runs/` 中 59 個有記錄的 `env.json` 全部是
`untargeted`，`targeted` 與 `crossattn` 從未在真實 SD 上產生過資料
（LEDGER 5.2）。這是零雲端成本就能填的一個洞。

## 量什麼

三個條件在兩個解析度上各跑幾步，報每步秒數與峰值記憶體：

| 條件 | 每步做什麼 |
|---|---|
| `optimize` | 一條 n_edit = 10 步的 SDEdit 鏈（現況） |
| `optimize_crossattn` | attn_timesteps 次單步 UNet 前向 |
| `optimize_encoder` | 一次 VAE 編碼 |

不設任何降級退路：OOM 就讓它 OOM 並記下該組合不可行。
縮小解析度**不能取代 512² 的正式結果**——VAE 的下採樣倍率固定為 8，
256² 的 latent 是 32²，cross-attention 的空間解析度全部減半，綁定結構不同。
本腳本量的是可行性與成本，不是結論。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.defense.objective import LossConfig
from src.defense.optimize import (
    OptimConfig, optimize, optimize_crossattn, optimize_encoder,
)
from src.models.sd import SDWrapper
from src.purify.ops import Purifier
from src.residual.site_pixel_full import FullRankPixelResidual
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory

ARMS = {
    "crossattn": optimize_crossattn,
    "encoder": optimize_encoder,
    "untargeted": optimize,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--sizes", default="256,512")
    ap.add_argument("--arms", default="crossattn,encoder,untargeted")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--attn_timesteps", type=int, default=4)
    ap.add_argument("--out", default="runs/logs/l4_crossattn_probe.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本探針要量的是 GPU 峰值記憶體，CPU 上沒有意義")
    device = get_device()
    p = torch.cuda.get_device_properties(0)
    print(f"[probe] {p.name} {p.total_memory / 2**30:.1f} GB "
          f"torch {torch.__version__}", flush=True)

    sd = SDWrapper(args.model)
    rows = []
    for size in [int(s) for s in args.sizes.split(",")]:
        for arm in [a.strip() for a in args.arms.split(",")]:
            g = torch.Generator().manual_seed(20260804)
            x01 = torch.rand(1, 3, size, size, generator=g).to(device)
            module = FullRankPixelResidual(
                size=size, channels=3, seed=20260804).to(device)
            cfg = OptimConfig(
                steps=args.steps, lr=0.008, n_edit=args.n_edit,
                strength=0.3, guidance_scale=7.5,
                prompt_edit="a woman", attn_mode="suppress",
                attn_timesteps=args.attn_timesteps,
                unet_ckpt=True, vae_ckpt=True, log_every=1,
            )
            loss_cfg = LossConfig(tau_lpips=0.10, beta_linf=0.0,
                                  alpha_lpips=0.0, margin=1.0)
            reset_peak_memory()
            t0 = time.perf_counter()
            try:
                ARMS[arm](sd, module, x01, cfg, loss_cfg,
                          [Purifier("identity")])
                dt = time.perf_counter() - t0
                r = {"size": size, "arm": arm, "steps": args.steps,
                     "seconds": round(dt, 2),
                     "s_per_step": round(dt / args.steps, 2),
                     "peak_mb": round(peak_memory_mb(), 1), "ok": True}
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # OOM 是有效的量測結果，不是錯誤。記下該組合不可行並繼續。
                #
                # 必須連 RuntimeError 一起接：實測 512² 的 crossattn 在
                # piq.LPIPS 內部丟出的是 `torch.AcceleratorError: CUDA error:
                # out of memory`，那是 RuntimeError 的子類而**不是**
                # torch.cuda.OutOfMemoryError，只接後者會讓整支腳本中止，
                # 前面已量到的幾行也一起丟掉。
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                r = {"size": size, "arm": arm, "steps": args.steps,
                     "seconds": None, "s_per_step": None,
                     "peak_mb": round(peak_memory_mb(), 1), "ok": False,
                     "error": str(e).split("\n")[0][:160]}
            rows.append(r)
            print(f"[probe] size={size:<4} arm={arm:<11} "
                  + (f"{r['s_per_step']:>7.2f} s/step  peak={r['peak_mb']:.0f} MB"
                     if r["ok"] else f"OOM  peak={r['peak_mb']:.0f} MB"),
                  flush=True)
            del module, x01
            torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"gpu": p.name, "torch": torch.__version__,
         "n_edit": args.n_edit, "attn_timesteps": args.attn_timesteps,
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] 寫出 {out}")

    ok = [r for r in rows if r["ok"]]
    if ok:
        base = {(r["size"]): r for r in ok if r["arm"] == "untargeted"}
        print("\n相對 untargeted 的加速（同一解析度）：")
        for r in ok:
            b = base.get(r["size"])
            if b and r["arm"] != "untargeted":
                print(f"  size={r['size']:<4} {r['arm']:<11} "
                      f"{b['s_per_step'] / r['s_per_step']:.1f}×")


if __name__ == "__main__":
    main()
