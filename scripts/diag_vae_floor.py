"""量測走生成路徑的條件（N3／site apa）在 SDXL 1024² 上的保真度下限。

存在理由。段 1 的 N3 階段一（LoRA 保真對齊）跑滿 200 步後停在
LPIPS 0.2250（bird_03）／0.3260（cat_02），而本輪訓練的失真預算
τ_train = 0.35。這兩個數字之間的關係決定 N3 能不能做：

* 若 **VAE 來回本身**就是 0.22–0.33，階段一已經在下限上，預算在 φ 起作用
  之前就被吃掉，N3 在此 τ 下結構上沒有空間，處置是改參考點或換位置。
* 若來回誤差遠低於此（例如 0.05），階段一只是沒收斂，處置是調步數／學習率。

兩者的處置完全不同，故必須實測。`site_apa.py` 檔頭記的 LPIPS 0.1434 /
27.51 dB 是 **SD v1.4、512²** 的量測值，換到 SDXL、1024²、bf16 之後不能沿用。

量兩段，因為它們是兩個不同的下限，會分別出錯：

| op | 內容 | 對應 |
|---|---|---|
| `vae_roundtrip` | `decode(encode(x))` | 只要經過 VAE 就付的代價，BDIA 不作用於此 |
| `recon` | `G(x; φ=0)`，即 inversion + 去噪整條 | 生成路徑的實際起點 |

`recon` 掃 `(k_inv, t_max, exact_inversion)`。這正是 `generator.py` 的
`t_max` 註解要求「由呼叫端依 E0c 的量測結果指定，不可沿用預設值」的那份
量測，而段 0（`run_calibration`）並不產生它——它只量 micro_bench、
precision_equiv、strength、editable、warp_reach 與學習率。b1 因此以
`t_max=None`（走滿 [0,999]）與 DDIM 反演執行，而該註解記載那個設定下
k_inv=10 的重建已達 LPIPS 0.70。

用法：

    python scripts/diag_vae_floor.py --images bird_03 cat_02 dog_03 \
        --out <輸出目錄> [--precision bf16] [--recon]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.defense.generator import DefenseGenerator
from src.experiment.executors import load_lo_aligned, write_csv
from src.metrics.suite import MetricSuite
from src.models.sd import SDXLWrapper
from src.residual.site_apa import build_apa

PRECISION = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--precision", default="bf16", choices=list(PRECISION))
    ap.add_argument("--recon", action="store_true",
                    help="加量 G(x; φ=0) 的 (k_inv, t_max, exact_inversion) 掃描")
    ap.add_argument("--k-inv", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--t-max", type=int, nargs="+", default=[999, 500, 300, 200],
                    help="inversion 的 timestep 上限；999 等同程式的 None")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    sd = SDXLWrapper(args.model, dtype=PRECISION[args.precision])
    suite = MetricSuite(device=sd.device)
    entries = load_lo_aligned(args.data, args.resolution, sd.device,
                              ids=args.images, n=None, seed=0)

    rows = []
    for e in entries:
        with torch.no_grad():
            # `decode_latent` 回傳計算精度（bf16）的張量，而 `x01` 是 fp32。
            # 指標套件不做隱式轉型（piq 的 SSIM 直接以 RuntimeError 中止），
            # 且指標本來就該在 fp32 上算——比較的對象是影像，不是計算過程。
            xr = sd.decode_latent(sd.encode_image(e.x01)).float().clamp(0, 1)
        fid = suite.pairwise(e.x01, xr)
        row = {"image_id": e.image_id, "op": "vae_roundtrip",
               "dtype": args.precision, **fid}
        rows.append(row)
        print(f"[{e.image_id}] vae_roundtrip({args.precision}) "
              f"lpips={fid['lpips']:.4f} psnr={fid['psnr']:.2f} "
              f"ssim={fid.get('ssim', float('nan')):.4f}", flush=True)

    if args.recon:
        rows += _recon_sweep(sd, suite, entries, args)

    write_csv(args.out / "vae_floor.csv", rows)
    print(f"→ {args.out / 'vae_floor.csv'}")
    return 0


def _recon_sweep(sd, suite, entries, args) -> list:
    """G(x; φ=0) 對 (k_inv, t_max, exact_inversion) 的重建誤差。

    模塊建起來後**停用**。`generator.py` 的能力檢查明文允許停用
    （「停用的定義就是行為與模塊不存在完全一致」，且 `x_base = G(x; φ=0)`
    正是靠這條路徑取得），故這裡量到的就是生成路徑本身的代價，不含 φ。

    掃描逐格重建模塊：`build_apa` 的 `steps` 必須等於 `k_inv`
    （`LatentResidual` 逐步持有一組殘差，不符時在 `eps_hook` 就拋出）。
    """
    lat = sd.latent_shape(args.resolution, args.resolution)
    rows = []
    for k_inv in args.k_inv:
        for t_max in args.t_max:
            for exact in (False, True):
                module = build_apa(
                    sd.unet, steps=k_inv, latent_size=int(lat[-1]),
                    latent_channels=int(lat[1]), seed=0,
                ).to(sd.device)
                module.disable()
                gen = DefenseGenerator(
                    sd, module, k_inv=k_inv,
                    # 程式以 None 表示走滿 [0, 999]；掃描表用 999 這一格代表它
                    t_max=(None if t_max >= 999 else t_max),
                    exact_inversion=exact,
                )
                try:
                    for e in entries:
                        with torch.no_grad():
                            xg = gen.generate(
                                e.x01, gen.prepare(e.x01)).float().clamp(0, 1)
                        fid = suite.pairwise(e.x01, xg)
                        rows.append({
                            "image_id": e.image_id, "op": "recon",
                            "dtype": args.precision, "k_inv": k_inv,
                            "t_max": t_max, "exact_inversion": exact, **fid})
                        print(f"[{e.image_id}] recon k_inv={k_inv} "
                              f"t_max={t_max} exact={exact} "
                              f"lpips={fid['lpips']:.4f} "
                              f"psnr={fid['psnr']:.2f}", flush=True)
                finally:
                    module.remove()
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
