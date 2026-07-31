"""E17 重建地板的拆解 —— A 族還有沒有機會。

**背景**

A 族（latent ε、文字嵌入、權重 LoRA）全部經過 `decode(...encode(x))`，因此
全部繼承同一個重建地板。實測：DDIM 路徑 φ=0 時 LPIPS 0.194；BDIA 精確反演
把「反演」那一半消掉後（latent 來回誤差降 5 個數量級，見 tests），剩下的
是純 VAE 來回的 0.143。加性像素位置實際運作在 **0.063**，故 A 族在 φ 還沒
作用前就已用掉 2.3 倍的失真預算。

**這個地板不是威脅模型逼出來的。** 威脅模型只規定攻擊方用 stable diffusion；
防禦方的 G 要用什麼機器產生 x_def 沒有限制。故「換掉 G 裡的 decode」完全
合法，攻擊方仍是拿 stock SD 編輯 x_def。

**本實驗要回答的**

0.143 之中，encoder 側（選了一個 decoder 重建不好的 z）與 decoder 側
（容量本身不足）各佔多少？拆開才知道該往哪個方向投資。

四個 arm：

- `roundtrip`  : decode(encode(x))，現況地板
- `latent_opt` : z 由 encode(x) 起步，直接對 z 做 Adam 最小化重建損失。
                 這消掉的是 **encoder 側**誤差——若它就能把地板砍一半，
                 換 decoder 的必要性大減，且不需要任何新模型。
- `asym_free`  : Asymmetric VQGAN（arXiv 2306.04632）的 decoder，
                 **mask 全為 1**（整張都算「要生成的區域」），條件分支拿不到
                 任何原圖資訊。這是唯一誠實的比較點：純粹「更重的 decoder」。
- `asym_leak`  : 同上但 **mask 全為 0**，條件分支可以完整看到原圖。
                 這是退化對照，用來顯示條件分支能把重建做到多好——但那個
                 好是「把原圖抄回來」，φ 的效果會被一併洗掉。**這一列不可
                 當成可用結果**，列出來是為了讓洩漏的幅度是可見的。

判準：任何 arm 把 LPIPS 壓到 0.063 以下，A 族才重新有討論空間。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.metrics.suite import MetricSuite
from src.models.sd import SDWrapper
from src.utils.device import get_device
from src.utils.artifacts import save_image, save_json
from scripts.run_defense import load_images

ASYM_ID = "cross-attention/asymmetric-autoencoder-kl-x-1-5"


def roundtrip(sd, x01):
    with torch.no_grad():
        return sd.decode_latent(sd.encode_image(x01))


def latent_opt(sd, x01, lpips_fn, steps, lr):
    """z* = argmin_z L(decode(z), x)，由 encode(x) 起步。

    損失用 MSE + LPIPS 兩項：只用 MSE 會得到模糊解（逐像素平方誤差對高頻
    不敏感），只用 LPIPS 會允許整體色偏。兩者相加是重建文獻的常見組合。
    """
    z = sd.encode_image(x01).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    hist = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        rec = sd.decode_latent(z)
        mse = (rec - x01).pow(2).mean()
        lp = lpips_fn(rec.clamp(0, 1), x01)
        loss = mse + 0.1 * lp
        loss.backward()
        opt.step()
        if i % 25 == 0 or i == steps - 1:
            hist.append({"step": i, "loss": float(loss),
                         "mse": float(mse), "lpips": float(lp)})
    with torch.no_grad():
        return sd.decode_latent(z).detach(), hist


def asym(vae, sd, x01, leak: bool):
    """Asymmetric VQGAN 的 decoder。latent 由 **SD 原廠 encoder** 產生。

    刻意不用 asym 自己的 encoder：論文本身就是「只重訓 decoder，vanilla
    encoder 與 StableDiffusion 不變」，而我們要的正是 latent 空間不動——
    site L 注入的 ε 就活在那個空間裡，換掉 encoder 等於換掉注入機制。

    mask 語意沿用 inpainting：1 表示「這塊要生成」。全 1 使條件分支拿不到
    原圖；全 0 使它完全看得到原圖（退化對照）。
    """
    with torch.no_grad():
        z = sd.encode_image(x01)
        img = (x01 * 2.0 - 1.0).to(vae.dtype)
        mask = torch.zeros_like(img[:, :1]) if leak else torch.ones_like(img[:, :1])
        out = vae.decode(z / sd.scaling_factor, image=img, mask=mask).sample
        return ((out + 1.0) / 2.0).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--data", default="data/dayn_testset")
    ap.add_argument("--out", default="runs/e17_vae_floor")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_asym", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)
    # 與訓練用的可微 LPIPS 取同一個實作（piq），否則「最佳化的量」與
    # 「回報的量」定義不同，拆解出來的數字無法與既有結果對照
    import piq
    lpips_fn = piq.LPIPS().to(device)
    images = load_images(Path(args.data), args.size, device, args.limit)
    print(f"[e17] {len(images)} 張影像，latent_opt {args.steps} 步 lr={args.lr}",
          flush=True)

    vae = None
    if not args.skip_asym:
        from diffusers import AsymmetricAutoencoderKL
        t = time.perf_counter()
        vae = AsymmetricAutoencoderKL.from_pretrained(
            ASYM_ID, torch_dtype=torch.float32).to(device).eval()
        vae.requires_grad_(False)
        print(f"[e17] 載入 {ASYM_ID} 用時 {time.perf_counter() - t:.0f}s",
              flush=True)

    rows = []
    for name, x01, _ in images:
        cell = out / name
        cell.mkdir(parents=True, exist_ok=True)
        save_image(x01, cell / "orig.png")

        arms = {}
        t0 = time.perf_counter()
        arms["roundtrip"] = (roundtrip(sd, x01), None)
        r, hist = latent_opt(sd, x01, lpips_fn, args.steps, args.lr)
        arms["latent_opt"] = (r, hist)
        if vae is not None:
            arms["asym_free"] = (asym(vae, sd, x01, leak=False), None)
            arms["asym_leak"] = (asym(vae, sd, x01, leak=True), None)

        for arm, (rec, hist) in arms.items():
            m = suite.pairwise(x01, rec)
            row = {"image": name, "arm": arm,
                   "lpips": m["lpips"], "psnr": m["psnr"], "ssim": m["ssim"],
                   "linf": float((rec - x01).abs().max()),
                   "under_target": m["lpips"] < 0.063}
            rows.append(row)
            save_image(rec, cell / f"{arm}.png")
            if hist:
                save_json(hist, cell / f"{arm}_history.json")
            print(f"  {name:10s} {arm:11s} lpips={m['lpips']:.4f} "
                  f"psnr={m['psnr']:.2f} linf={row['linf']:.4f}", flush=True)
        print(f"  {name} 用時 {time.perf_counter() - t0:.0f}s", flush=True)

    save_json(rows, out / "results.json")

    print("\n" + "=" * 62)
    print("平均（目標：LPIPS < 0.063，即加性像素位置實際運作的失真點）")
    print("=" * 62)
    summary = {}
    for arm in ["roundtrip", "latent_opt", "asym_free", "asym_leak"]:
        v = [r for r in rows if r["arm"] == arm]
        if not v:
            continue
        lp = sum(r["lpips"] for r in v) / len(v)
        ps = sum(r["psnr"] for r in v) / len(v)
        summary[arm] = {"lpips": lp, "psnr": ps, "n": len(v)}
        base = summary.get("roundtrip", {}).get("lpips", lp)
        print(f"  {arm:11s} lpips={lp:.4f}  psnr={ps:5.2f}  "
              f"相對現況={100 * lp / base:5.1f}%  "
              f"{'達標' if lp < 0.063 else '未達標'}")
    save_json(summary, out / "summary.json")
    print("\n注意：asym_leak 是退化對照，其 decoder 可完整看到原圖，"
          "重建好是因為把原圖抄回來，不可當成可用結果。")


if __name__ == "__main__":
    main()
