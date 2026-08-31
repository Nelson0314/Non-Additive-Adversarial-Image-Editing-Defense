"""UNet 對「哪個頻帶、哪個噪聲步」有多敏感。**單卡，只前向。**

為什麼要重量一次
────────────────────────────────────────────────────────────────────
模糊的「可贏上界 1.54 倍」是 `模糊存活率 × VAE encoder 敏感度` 的乘積極大值
（`runs/fixedpoint_blur/`）。但真正的讀出端不是 encoder，是
**encoder → UNet 在某個 t 的 ε 預測**。

而擴散模型對頻率的敏感度**強烈依賴 t**：加進去的是白噪聲（頻譜平坦），自然
影像的頻譜約 `1/f`，所以噪聲水準越高，只有越低的頻率還露在噪聲之上。文獻
兩面都有：**DiffusionGuard**（ICLR 2025）只在 `t = T` 評估損失，換到 JPEG 與
crop-and-resize 的穩健性；**SimAC**（CVPR 2024）反面量到低 t 的梯度遠大於高 t。

合起來：**「模糊拿走的高頻載體」與「只在低 t 才有用的頻帶」可能是同一件事。**
若成立，正確的上界是 `H_σ(r)² × S_UNet(r, t)`，而 1.54 倍那個數字不涵蓋它——
**不是矛盾，是換了一個算子。**

量什麼
────────────────────────────────────────────────────────────────────
對每一個 `(帶, t)` 格點：把**同一個 θ 的隨機相位旋轉限制在該帶**（與
`encoder_frequency_response.py` 完全同一個探針），量

    unet_move(r, t) = ‖ eps(z_t, E_img(x_def), null) − eps(z_t, E_img(x), null) ‖

即「把單位預算放在該帶，UNet 在該噪聲步的反應變多少」。並列 `latent_move`
（同一個擾動在 encoder 上的位移）與 `dists_cost`（人眼代價），三者放在同一列
才讀得出「換了讀出端之後排序有沒有變」。

**用隨機相位而不是最佳化的相位**：這裡要的是**通道的性質**，不是某個解的
性質。最佳化會把答案繞回「損失喜歡什麼」。理由與既有探針相同。

**`z_t` 錨在原圖上**（`diffuse_src`）：用防禦圖當錨會讓取樣軌跡隨擾動漂移，
量到的就不是同一個 t 了。與 `image_guidance_loss` 的 `zt_mode="diffuse_src"`
同一條路。

**噪聲固定**：同一個 `(帶, t)` 上乾淨與防禦兩側必須用**同一份 eps**，否則量到
的是取樣變異不是敏感度。

用法：
    python scripts/unet_band_sensitivity.py --images <名稱> ... \\
        --out runs/unet_band_sensitivity/probe.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLUTION = 512
R_MAX = math.sqrt(2.0)
# 與 `runs/encoder_frequency_response/` 同一組邊界，好逐帶對照。
N_BANDS = 8
# 噪聲步。**取整條排程的六個切點**，不是只取兩端——若敏感度隨 t 單調移動，
# 中間點才看得出移動的形狀。
T_FRACS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)


def bands(n: int):
    e = [R_MAX * i / n for i in range(n + 1)]
    return list(zip(e[:-1], e[1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--theta", type=float, default=1.0,
                    help="隨機相位的標準差，與既有探針同一個值")
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    from apa_baseline import load_dataset
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper
    from src.residual.texture_rephase import PhaseResidual
    from src.utils.io import load_image_tensor, write_csv

    ip2p = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=ip2p.device)
    dev = ip2p.device
    unet = ip2p.unet
    # **共用 image_guidance_loss 的那兩個零件，不另抄一份**：抄出來會在模組
    # 改動時悄悄分岔，而症狀只是「量到的是另一個東西」。
    from src.defense.fixedpoint_loss import _null_embedding, _scheduler_of
    abar = _scheduler_of(ip2p).alphas_cumprod.to(device=dev, dtype=torch.float32)
    n_steps = len(abar)
    null_emb = _null_embedding(ip2p)

    keep = set(args.images)
    dataset = [d for d in load_dataset(args.data, prompt_index=0)
               if d["name"] in keep]
    if not dataset:
        raise SystemExit(f"{args.data} 底下沒有符合 --images 的影像")

    rows = []
    for item in dataset:
        x = load_image_tensor(item["path"], dev, size=RESOLUTION)
        with torch.no_grad():
            z_src = ip2p.encode_image(x).detach()
            z_img_clean = ip2p.image_latents(x).detach()
            z0 = ip2p.encode_image(x)

        for i, (lo, hi) in enumerate(bands(N_BANDS)):
            m = PhaseResidual(size=RESOLUTION, block=args.block, r_min=lo,
                              r_max=hi if hi < R_MAX else float("inf"),
                              theta_max=math.pi,
                              energy_quantile=0.0).to(device=dev, dtype=x.dtype)
            m.prepare_gates(x)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
            with torch.no_grad():
                m.theta.copy_((torch.randn(m.theta.shape, generator=gen)
                               * args.theta).clamp(-math.pi, math.pi)
                              .to(device=dev, dtype=x.dtype))
                x_def = m.pixel_residual(x).clamp(0, 1)
                dists = float(suite.pairwise(x, x_def)["dists"])
                latent_move = float((ip2p.encode_image(x_def) - z0)
                                    .flatten().norm())
                z_img_def = ip2p.image_latents(x_def)

            for tf in T_FRACS:
                step = max(0, min(n_steps - 1, int(round(tf * (n_steps - 1)))))
                # **兩側同一份 eps**，否則量到的是取樣變異不是敏感度。
                g2 = torch.Generator(device="cpu").manual_seed(
                    args.seed + 1000 * i + step)
                eps = torch.randn(z_src.shape, generator=g2,
                                  dtype=torch.float32).to(device=dev,
                                                          dtype=z_src.dtype)
                a = abar[step].to(z_src.dtype)
                z_t = z_src * a.sqrt() + eps * (1.0 - a).sqrt()
                tt = torch.tensor([step], device=dev, dtype=torch.long)
                emb = null_emb.to(z_t.dtype)
                with torch.no_grad():
                    e_clean = unet(torch.cat([z_t, z_img_clean], dim=1), tt,
                                   encoder_hidden_states=emb).sample
                    e_def = unet(torch.cat([z_t, z_img_def], dim=1), tt,
                                 encoder_hidden_states=emb).sample
                    move = float((e_def - e_clean).flatten().norm())
                rows.append({
                    "image": item["name"], "block": args.block,
                    "theta": args.theta, "r_lo": round(lo, 4),
                    "r_hi": round(hi, 4), "t_frac": tf, "t_step": step,
                    "unet_move": round(move, 5),
                    "latent_move": round(latent_move, 4),
                    "dists_cost": round(dists, 5),
                    "move_per_dists": round(move / dists, 4) if dists > 0 else "",
                })
            print(f"  {item['name']} 帶 {i + 1}/{N_BANDS} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    import statistics as st
    print(f"\n{args.out}：{len(rows)} 列\n")
    print("unet_move（越大＝該帶在該噪聲步越有槓桿）")
    print(f"{'帶':>14}" + "".join(f"{'t=' + str(tf):>11}" for tf in T_FRACS))
    for lo, hi in bands(N_BANDS):
        cells = ""
        for tf in T_FRACS:
            v = [float(r["unet_move"]) for r in rows
                 if float(r["r_lo"]) == round(lo, 4) and r["t_frac"] == tf]
            cells += f"{st.fmean(v):>11.3f}" if v else f"{'-':>11}"
        print(f"{f'{lo:.2f}-{hi:.2f}':>14}{cells}")

    print("\n對照：同一個擾動在 VAE encoder 上的位移（與 t 無關）")
    for lo, hi in bands(N_BANDS):
        v = [float(r["latent_move"]) for r in rows
             if float(r["r_lo"]) == round(lo, 4)]
        d = [float(r["dists_cost"]) for r in rows
             if float(r["r_lo"]) == round(lo, 4)]
        if v:
            print(f"{f'{lo:.2f}-{hi:.2f}':>14}  latent_move {st.fmean(v):>8.3f}"
                  f"   DISTS {st.fmean(d):>7.4f}")

    print("\n**要看的是：高 t 那幾欄的峰值是不是往低頻移動。**"
          "若是，模糊的可贏上界要用 S_UNet(r, t) 重算，1.54 倍不涵蓋它。")


if __name__ == "__main__":
    main()
