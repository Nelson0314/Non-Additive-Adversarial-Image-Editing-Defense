"""E0 — 成本實測 — spec §7.1。

目的：取得真實記憶體與時間數字，作為 k_inv、n_edit、EOT 取樣數的依據。
在拿到這些數字之前不填任何超參數。

量測對象是 site L 的完整反向路徑（spec §5.3），即最深的計算圖：

    L → 編輯鏈(n_edit) → VAE.decode → x_def → VAE.decode → 去噪鏈(k_inv) → φ

執行：
    python scripts/e0_cost.py --model CompVis/stable-diffusion-v1-4 --out runs/e0

判準：至少一組 (k_inv, n_edit) 可在 32 GB 內完成反向傳播。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# 讓腳本在任何 cwd 下都能 import src，本機與 TWCC 皆不需設 PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.defense.generator import DefenseGenerator
from src.models.sd import SDWrapper
from src.residual.site_latent import LatentResidual
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory


def measure(sd, x01, k_inv, n_edit, use_ckpt, strength, seed, warmup_done, vae_ckpt=False):
    """跑一次完整的 forward + backward，回傳量測結果 dict。"""
    device = get_device()
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])

    module = LatentResidual(
        steps=k_inv, channels=lat[1], size=lat[-1], max_rank=32, const_rank=8, seed=seed
    ).to(device)
    gen = DefenseGenerator(sd, module, k_inv=k_inv)

    emb_edit = sd.encode_text("a photo").detach()
    noise = sd.sample_edit_noise(torch.empty(lat, device=device), seed=seed)

    # y_orig 對 φ 為常數，先算好（spec §5.1）
    with torch.no_grad():
        y_orig = sd.sdedit(x01, emb_edit, noise, n_edit, strength=strength)

    reset_peak_memory()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    ctx = gen.prepare(x01, prompt_def="")
    x_def = gen.generate(x01, ctx, use_ckpt=use_ckpt, vae_ckpt=vae_ckpt)
    y_def = sd.sdedit(
        x_def, emb_edit, noise, n_edit,
        strength=strength, use_ckpt=use_ckpt, vae_ckpt=vae_ckpt,
    )

    # E0 只需要一個能撐起完整計算圖的目標，非最終 loss
    loss = -(y_def - y_orig).pow(2).mean()
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    grad_ok = (
        module.tensor.V.grad is not None and module.tensor.V.grad.abs().sum().item() > 0
    )
    peak = peak_memory_mb()

    del module, gen, ctx, x_def, y_def, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "k_inv": k_inv,
        "n_edit": n_edit,
        "unet_ckpt": int(use_ckpt),
        "vae_ckpt": int(vae_ckpt),
        "peak_mb": round(peak, 1),
        "seconds": round(dt, 3),
        "grad_reaches_phi": int(grad_ok),
        "oom": 0,
        "warmup": int(not warmup_done),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--out", default="runs/e0")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--grid", default="5,10,20", help="k_inv 與 n_edit 的掃描值")
    ap.add_argument(
        "--combos", default="u1v1,u1v0",
        help="checkpoint 組合，uXvY 表示 UNet=X、VAE=Y。E0 已證實 u0 於 512 必 OOM",
    )
    ap.add_argument("--image", default=None, help="留空則用隨機影像（成本與內容無關）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    grid = [int(v) for v in args.grid.split(",")]

    print(f"[E0] device={device} model={args.model} size={args.size}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[E0] GPU={p.name} total={p.total_memory / 1024**3:.1f} GB")

    sd = SDWrapper(args.model)

    if args.image:
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
        x01 = T.ToTensor()(img).unsqueeze(0).to(device)
    else:
        g = torch.Generator().manual_seed(args.seed)
        x01 = torch.rand(1, 3, args.size, args.size, generator=g).to(device)

    combos = [
        (bool(int(c[1])), bool(int(c[3])))
        for c in (v.strip() for v in args.combos.split(","))
    ]

    rows = []
    warmup_done = False
    for use_ckpt, vae_ckpt in combos:
        for k_inv in grid:
            for n_edit in grid:
                tag = (
                    f"k_inv={k_inv:>2} n_edit={n_edit:>2} "
                    f"u={int(use_ckpt)} v={int(vae_ckpt)}"
                )
                try:
                    row = measure(
                        sd, x01, k_inv, n_edit, use_ckpt,
                        args.strength, args.seed, warmup_done,
                        vae_ckpt=vae_ckpt,
                    )
                    warmup_done = True
                    print(
                        f"[E0] {tag}  peak={row['peak_mb']:>9.1f} MB"
                        f"  {row['seconds']:>7.3f} s  grad={row['grad_reaches_phi']}"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row = {
                        "k_inv": k_inv, "n_edit": n_edit,
                        "unet_ckpt": int(use_ckpt), "vae_ckpt": int(vae_ckpt),
                        "peak_mb": -1, "seconds": -1, "grad_reaches_phi": 0,
                        "oom": 1, "warmup": 0,
                    }
                    print(f"[E0] {tag}  OOM")
                rows.append(row)

    csv_path = out / "e0_cost.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    env = {
        "model": args.model,
        "size": args.size,
        "strength": args.strength,
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_mem_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            if torch.cuda.is_available() else None
        ),
    }
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    ok = [r for r in rows if r["oom"] == 0 and r["grad_reaches_phi"] == 1]
    print(f"\n[E0] 可行組合 {len(ok)}/{len(rows)}，結果寫入 {csv_path}")
    if not ok:
        print("[E0] 判準未達成：無任何組合可完成反向傳播")
    else:
        best = max(ok, key=lambda r: (r["k_inv"] + r["n_edit"], -r["peak_mb"]))
        print(
            f"[E0] 最大可行組合 k_inv={best['k_inv']} n_edit={best['n_edit']} "
            f"u={best['unet_ckpt']} v={best['vae_ckpt']} peak={best['peak_mb']} MB "
            f"{best['seconds']} s/iter"
        )


if __name__ == "__main__":
    main()
