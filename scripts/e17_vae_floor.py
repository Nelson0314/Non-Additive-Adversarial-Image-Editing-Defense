"""E17 重建誤差下限的拆解 —— A 類還有沒有機會。

背景

A 類（latent ε、文字嵌入、權重 LoRA）全部經過 `decode(...encode(x))`，因此
全部繼承同一個重建誤差下限。實測：DDIM 路徑 φ=0 時 LPIPS 0.194；BDIA 精確反演
把「反演」那一半消掉後（latent 來回誤差降 5 個數量級，見 tests），剩下的
是純 VAE 來回的 0.143。加性像素位置實際運作在 0.063，故 A 類在 φ 還沒
作用前就已用掉 2.3 倍的失真預算。

這個下限不是威脅模型逼出來的。威脅模型只規定攻擊方用 stable diffusion；
防禦方的 G 要用什麼機器產生 x_def 沒有限制。故「換掉 G 裡的 decode」完全
合法，攻擊方仍是拿 stock SD 編輯 x_def。

本實驗要回答的

0.143 之中，encoder 側（選了一個 decoder 重建不好的 z）與 decoder 側
（容量本身不足）各佔多少？拆開才知道該往哪個方向投資。

四個 arm：

- `roundtrip`  : decode(encode(x))，現況下限
- `latent_opt` : z 由 encode(x) 起步，直接對 z 做 Adam 最小化重建損失。
                 這消掉的是 encoder 側誤差——若它就能把下限砍一半，
                 換 decoder 的必要性大減，且不需要任何新模型。
- `asym_free`  : Asymmetric VQGAN（arXiv 2306.04632）的 decoder，
                 mask 全為 1（整張都算「要生成的區域」），條件分支拿不到
                 任何原圖資訊。這是唯一誠實的比較點：純粹「更重的 decoder」。
- `asym_leak`  : 同上但 mask 全為 0，條件分支可以完整看到原圖。
                 這是退化對照，用來顯示條件分支能把重建做到多好——但那個
                 好是「把原圖抄回來」，φ 的效果會被一併洗掉。這一列不可
                 當成可用結果，列出來是為了讓洩漏的幅度是可見的。

判準：任何 arm 把 LPIPS 壓到 0.063 以下，A 類才重新有討論空間。

---

2026-07-31 修改（E18/E19），before/after 紀錄

E18（lr × 步數掃描）測得步數不是瓶頸：lr=0.005 下 400→1600 步（4 倍算力）
LPIPS 只再降 3.3%（0.0886 → 0.0857），軌跡已平坦。故改動兩處參數：

1. `latent_opt` 的損失權重
   - before：`loss = mse + 0.1 * lp`，0.1 寫死於函式內（原第 72 行）
   - after：`loss = mse + lam * lp`，`lam` 由 `--lam` 給，預設 0.1，
     即不帶旗標時與 E17 完全相同，E17 的數據仍可重現
   - 原因：把關指標是 LPIPS，但優化的量以 MSE 為主，等於沒有對準要優化
     的目標。`lam` 成為可掃描的軸。

2. 新增 arm `latent_opt_asym`（`--stack_asym`）
   - before：`latent_opt` 只走 SD 原廠 decoder；asym decoder 只有不做
     最佳化的 `asym_free` 一條件
   - after：`latent_opt` 接受 `decode` 參數，可改走任何 decoder；
     `--stack_asym` 時把兩者疊起來
   - 原因：E17 中 latent_opt 降 47%、asym_free 降 10%，兩者作用在不同
     環節（encoder 側選點 vs decoder 容量），疊加是尚未測過的組合。

3. 把關由單一 LPIPS 改為多指標
   - before：`under_target = m["lpips"] < 0.063`
   - after：`passes` 要求 LPIPS < 0.063 且 PSNR 不低於 `roundtrip`、
     DISTS 不高於 `roundtrip`；`dists` 一併寫入結果
   - 原因：直接對 LPIPS 做梯度下降有對抗性風險——可能只是攻擊了 LPIPS
     網路，分數降了但人眼品質沒變好。E15 已出現過「LPIPS 判為相等、實際
     不等」這個情形（site S 與 site P 在匹配 LPIPS 時差 9 dB PSNR）。
     單一 LPIPS 不足以把關，此處與 src/metrics/suite.py 的設計主張一致。
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

# 加性像素位置（site P）實際運作的失真點。A 類要重新進場的門檻。
TARGET_LPIPS = 0.063


def roundtrip(sd, x01):
    with torch.no_grad():
        return sd.decode_latent(sd.encode_image(x01))


def latent_opt(sd, x01, lpips_fn, steps, lr, lam=0.1, decode=None):
    """z* = argmin_z L(decode(z), x)，由 encode(x) 起步。

    損失用 MSE + LPIPS 兩項：只用 MSE 會得到模糊解（逐像素平方誤差對高頻
    不敏感），只用 LPIPS 會允許整體色偏。兩者相加是重建文獻的常見組合。
    `lam` 為 LPIPS 項的權重（見模組 docstring 的 E18/E19 修改紀錄）。

    `decode` 為 None 時走 SD 原廠 decoder；傳入 callable 則改用它。
    latent 空間不變——site L 注入的 ε 活在那個空間裡，換 decoder 合法，
    換 encoder 等於換掉注入機制。

    MSE 用未 clamp 的 `rec`：clamp 會讓落在 [0,1] 之外的像素梯度歸零，
    那正是需要被拉回範圍內的那些像素。LPIPS 則必須 clamp，因為其輸入
    定義域是 [0,1]。
    """
    dec = decode if decode is not None else sd.decode_latent
    z = sd.encode_image(x01).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    hist = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        rec = dec(z)
        mse = (rec - x01).pow(2).mean()
        lp = lpips_fn(rec.clamp(0, 1), x01)
        loss = mse + lam * lp
        loss.backward()
        opt.step()
        if i % 25 == 0 or i == steps - 1:
            hist.append({"step": i, "loss": float(loss),
                         "mse": float(mse), "lpips": float(lp)})
    with torch.no_grad():
        return dec(z).detach(), hist


def asym_decoder(vae, sd, x01):
    """把 Asymmetric VQGAN 的 decoder 包成 `latent_opt` 可用的 z → [0,1] 函式。

    mask 全為 1（整張都算「要生成的區域」），條件分支拿不到任何原圖資訊，
    與 `asym(..., leak=False)` 同一設定。不 clamp：交給 `latent_opt` 決定，
    否則會在最佳化途中截斷梯度。
    """
    img = (x01 * 2.0 - 1.0).to(vae.dtype)
    mask = torch.ones_like(img[:, :1])

    def dec(z):
        out = vae.decode(z / sd.scaling_factor, image=img, mask=mask).sample
        return (out + 1.0) / 2.0

    return dec


def asym(vae, sd, x01, leak: bool):
    """Asymmetric VQGAN 的 decoder。latent 由 SD 原廠 encoder 產生。

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
    ap.add_argument("--lam", type=float, default=0.1,
                    help="latent_opt 損失中 LPIPS 項的權重。預設 0.1 = E17 原設定")
    ap.add_argument("--stack_asym", action="store_true",
                    help="加跑 latent_opt_asym：latent 最佳化走 asym decoder")
    args = ap.parse_args()

    if args.stack_asym and args.skip_asym:
        ap.error("--stack_asym 需要 asym decoder，不能與 --skip_asym 併用")

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
        r, hist = latent_opt(sd, x01, lpips_fn, args.steps, args.lr, args.lam)
        arms["latent_opt"] = (r, hist)
        if vae is not None:
            arms["asym_free"] = (asym(vae, sd, x01, leak=False), None)
            arms["asym_leak"] = (asym(vae, sd, x01, leak=True), None)
            if args.stack_asym:
                r, hist = latent_opt(sd, x01, lpips_fn, args.steps, args.lr,
                                     args.lam, decode=asym_decoder(vae, sd, x01))
                arms["latent_opt_asym"] = (r, hist)

        # roundtrip 是現況下限，也是「不退步」的參照點：任何 arm 若把 LPIPS
        # 壓下去卻讓 PSNR 或 DISTS 比下限還差，那個進步就不可信。
        ref = suite.pairwise(x01, arms["roundtrip"][0])

        for arm, (rec, hist) in arms.items():
            m = suite.pairwise(x01, rec)
            row = {"image": name, "arm": arm,
                   "lpips": m["lpips"], "psnr": m["psnr"], "ssim": m["ssim"],
                   "dists": m["dists"],
                   "linf": float((rec - x01).abs().max()),
                   "lpips_ok": m["lpips"] < TARGET_LPIPS,
                   "psnr_ok": m["psnr"] >= ref["psnr"],
                   "dists_ok": m["dists"] <= ref["dists"]}
            row["passes"] = row["lpips_ok"] and row["psnr_ok"] and row["dists_ok"]
            rows.append(row)
            save_image(rec, cell / f"{arm}.png")
            if hist:
                save_json(hist, cell / f"{arm}_history.json")
            print(f"  {name:10s} {arm:15s} lpips={m['lpips']:.4f} "
                  f"psnr={m['psnr']:.2f} dists={m['dists']:.4f} "
                  f"linf={row['linf']:.4f} {'通過' if row['passes'] else ''}",
                  flush=True)
        print(f"  {name} 用時 {time.perf_counter() - t0:.0f}s", flush=True)

    save_json(rows, out / "results.json")

    print("\n" + "=" * 78)
    print(f"平均（把關：LPIPS < {TARGET_LPIPS}，且 PSNR 與 DISTS 皆不差於 roundtrip）")
    print("=" * 78)
    summary = {"config": {"steps": args.steps, "lr": args.lr, "lam": args.lam}}
    for arm in ["roundtrip", "latent_opt", "asym_free", "asym_leak",
                "latent_opt_asym"]:
        v = [r for r in rows if r["arm"] == arm]
        if not v:
            continue
        lp = sum(r["lpips"] for r in v) / len(v)
        ps = sum(r["psnr"] for r in v) / len(v)
        ds = sum(r["dists"] for r in v) / len(v)
        npass = sum(1 for r in v if r["passes"])
        summary[arm] = {"lpips": lp, "psnr": ps, "dists": ds,
                        "n": len(v), "n_pass": npass}
        base = summary.get("roundtrip", {}).get("lpips", lp)
        print(f"  {arm:15s} lpips={lp:.4f}  psnr={ps:5.2f}  dists={ds:.4f}  "
              f"相對現況={100 * lp / base:5.1f}%  "
              f"通過 {npass}/{len(v)}")
    save_json(summary, out / "summary.json")
    print("\n注意：asym_leak 是退化對照，其 decoder 可完整看到原圖，"
          "重建好是因為把原圖抄回來，不可當成可用結果。")


if __name__ == "__main__":
    main()
