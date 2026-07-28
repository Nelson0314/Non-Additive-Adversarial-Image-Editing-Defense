"""E0b — 記憶體歸因 — spec §7.1 的延伸。

E0 的主結果是：開啟 UNet gradient checkpointing 後，peak 記憶體在
(k_inv, n_edit) 由 (5,5) 變到 (20,20) 的過程中只從 18665.4 MB 動到
18667.3 MB。步數變動四倍而記憶體幾乎不動，代表 peak 由某個與步數無關的
項主導，UNet 激活已不是瓶頸。

本腳本逐項量測以歸因，不以推測代替量測。量測項目：

    weights      模型載入後的常駐量
    vae_encode   單次 VAE encode 的前向+反向
    vae_decode   單次 VAE decode 的前向+反向
    unet_step    單步 UNet（開 checkpoint）的前向+反向
    full_*       完整 E0 路徑，UNet/VAE checkpoint 四種組合

關鍵預期（若成立即證實歸因）：整條計算圖上有三次 VAE 呼叫（x_def 的
decode、sdedit 的 encode、sdedit 的 decode）。不做 VAE checkpoint 時三者
的激活必須同時留存，peak 為總和；做了之後反向一次只重算一塊，peak 降為
最大值。故 full_vae1 應顯著低於 full_vae0，而單次 vae_decode 不受影響。

執行：source env.sh && python scripts/e0_breakdown.py --out runs/e0b
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.defense.generator import DefenseGenerator
from src.models.sd import SDWrapper
from src.residual.site_latent import LatentResidual
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def probe(name, fn):
    """跑一次 fn 並回傳 peak 與耗時。fn 內部自行完成 forward + backward。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reset_peak_memory()
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    dt = time.perf_counter() - t0
    peak = peak_memory_mb()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[E0b] {name:16s} peak={peak:>9.1f} MB  {dt:>7.3f} s")
    return {"case": name, "peak_mb": round(peak, 1), "seconds": round(dt, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--out", default="runs/e0b")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--k_inv", type=int, default=10)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    sd = SDWrapper(args.model)
    _sync()
    weights_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    print(f"[E0b] 常駐權重 {weights_mb:.1f} MB")

    lat = sd.latent_shape(args.size, args.size)
    g = torch.Generator().manual_seed(args.seed)
    x01 = torch.rand(1, 3, args.size, args.size, generator=g).to(device)
    emb = sd.encode_text("a photo").detach()
    noise = sd.sample_edit_noise(torch.empty(lat, device=device), seed=args.seed)

    rows = [{"case": "weights", "peak_mb": round(weights_mb, 1), "seconds": 0.0}]

    # ---- 單一元件 ----

    def vae_encode():
        x = x01.clone().requires_grad_(True)
        sd.encode_image(x).pow(2).mean().backward()

    def vae_decode():
        z = torch.randn(lat, device=device, requires_grad=True)
        sd.decode_latent(z).pow(2).mean().backward()

    def vae_decode_ckpt():
        z = torch.randn(lat, device=device, requires_grad=True)
        sd.decode_latent(z, use_ckpt=True).pow(2).mean().backward()

    def unet_step():
        z = torch.randn(lat, device=device, requires_grad=True)
        t = torch.tensor(500)
        sd._eps(z, t, emb, use_ckpt=True).pow(2).mean().backward()

    rows.append(probe("vae_encode", vae_encode))
    rows.append(probe("vae_decode", vae_decode))
    rows.append(probe("vae_decode_ckpt", vae_decode_ckpt))
    rows.append(probe("unet_step_ckpt", unet_step))

    # ---- 完整路徑，UNet/VAE checkpoint 四組合 ----

    with torch.no_grad():
        y_orig = sd.sdedit(x01, emb, noise, args.n_edit, strength=args.strength)

    def make_full(unet_ck, vae_ck):
        def run():
            module = LatentResidual(
                steps=args.k_inv, channels=lat[1], size=lat[-1],
                max_rank=32, const_rank=8,
            ).to(device)
            gen = DefenseGenerator(sd, module, k_inv=args.k_inv)
            ctx = gen.prepare(x01, prompt_def="")
            x_def = gen.generate(x01, ctx, use_ckpt=unet_ck, vae_ckpt=vae_ck)
            y_def = sd.sdedit(
                x_def, emb, noise, args.n_edit,
                strength=args.strength, use_ckpt=unet_ck, vae_ckpt=vae_ck,
            )
            (-(y_def - y_orig).pow(2).mean()).backward()
            assert module.tensor.V.grad.abs().sum().item() > 0, "梯度未抵達 φ"

        return run

    for unet_ck in (True, False):
        for vae_ck in (True, False):
            name = f"full_u{int(unet_ck)}_v{int(vae_ck)}"
            try:
                rows.append(probe(name, make_full(unet_ck, vae_ck)))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[E0b] {name:16s} OOM")
                rows.append({"case": name, "peak_mb": -1, "seconds": -1})

    csv_path = out / "e0_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "peak_mb", "seconds"])
        w.writeheader()
        w.writerows(rows)

    (out / "env.json").write_text(
        json.dumps(
            {
                "model": args.model, "size": args.size,
                "k_inv": args.k_inv, "n_edit": args.n_edit,
                "strength": args.strength, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[E0b] 寫入 {csv_path}")


if __name__ == "__main__":
    main()
