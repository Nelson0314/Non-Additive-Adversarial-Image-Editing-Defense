"""DJSMA 在**它自己的威脅模型**上的重現：ImageNet 分類 ＋ 六個 CNN。

論文：Chen et al., *JPEG compression-resistant adversarial attack with
invisible watermark embedding*, The Imaging Science Journal 2026,
doi:10.1080/13682199.2026.2644653。無公開程式碼；由使用者提供的掃描 PDF
逐頁判讀後實作，見 `src/baselines/dct_watermark.py`。

重現目標（Table 1／2／3／5）
────────────────────────────────────────────────────────────────────
設定：ImageNet、τ=1500、μ=1、攻擊區 E345（第 3–5 條反對角）、
      對抗樣本再壓一次 JPEG（Q=75，另報 Q=85）後才送進模型。

Table 1  壓縮後的分類正確率（%），越低代表攻擊越成功
         原圖        Vgg19 64.8  Inceptionv3 66  ResNet101 71.9
                     ResNet152 73.3  SqueezeNet 52.1  ShuffleNet 63.3
         已嵌浮水印  63.4 / 65.8 / 71.7 / 72.2 / 48.5 / 59.9
         **本方法**  1.2 / 4.6 / 3.9 / 4.6 / 0.6 / 0.8
         攻擊成功率定義為「已嵌浮水印的正確率 − 對抗樣本的正確率」，
         例如 VGG19 是 63.4 − 1.2 = 62.2%

Table 2  平均 PSNR（dB）：已嵌浮水印 37.81；本方法 36.01–37.11
Table 3  平均 SSIM：已嵌浮水印 0.981；本方法 0.941–0.966
Table 5  Q=85 時 Vgg19/Inceptionv3/ResNet101 的正確率 3.9/6.5/5.5、
         PSNR 40.16/40.42/39.53、SSIM 0.98/0.982/0.972、單張 0.8–1.1 秒

本專案與論文的差異（必須寫進報表）
────────────────────────────────────────────────────────────────────
1. **沒有浮水印那一階段。** J-UNIWARD ＋ 三元 STC ＋ RS 碼未實作（本專案的
   威脅模型不需要復原訊息）。故我們的 PSNR／SSIM 是相對**原圖**，論文的是
   相對**已嵌浮水印的影像**，參照點不同。
2. **張數**：預設 200（論文未載張數）。
3. **目標標籤的選法**與**嵌入端的 JPEG 品質**論文未載，為本專案指定。
4. 資料來源是 HF 鏡像 `mrm8488/ImageNet1K-val`，非官方 ILSVRC 下載點。

用法：
    python scripts/djsma_repro.py --out runs/djsma_repro --models vgg19 resnet101
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.dct_watermark import (  # noqa: E402
    PAPER_EVAL_QUALITY, PAPER_MU, PAPER_TAU, DJSMASpec, run_djsma,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advdrop_repro import (  # noqa: E402
    MEAN, STD, jpeg_roundtrip, load_split,
)

# 論文 §Experimental results 的六個模型。torchvision 的對應權重列在後面。
MODELS = {
    "vgg19": ("vgg19", "VGG19_Weights"),
    "inception_v3": ("inception_v3", "Inception_V3_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
    "resnet152": ("resnet152", "ResNet152_Weights"),
    "squeezenet": ("squeezenet1_1", "SqueezeNet1_1_Weights"),
    "shufflenet": ("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights"),
}
# 論文 Table 1「已嵌浮水印」那一列——本專案沒有浮水印，故拿**原圖**當參照，
# 兩者不可混為一談。這裡只作為報表上的對照欄。
PAPER_ACC = {"vgg19": 1.2, "inception_v3": 4.6, "resnet101": 3.9,
             "resnet152": 4.6, "squeezenet": 0.6, "shufflenet": 0.8}
PAPER_WM_ACC = {"vgg19": 63.4, "inception_v3": 65.8, "resnet101": 71.7,
                "resnet152": 72.2, "squeezenet": 48.5, "shufflenet": 59.9}


def build_model(name: str, device):
    import torchvision.models as tvm

    fn, wname = MODELS[name]
    weights = getattr(tvm, wname).IMAGENET1K_V1
    m = getattr(tvm, fn)(weights=weights).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    # inception_v3 的原生輸入是 299；torchvision 的權重在 224 上仍可用，
    # 但正確率會低一些。**這是本專案的簡化**，報表要註明。
    return lambda x01: m((x01 - mean) / std)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/imagenet_advdrop"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["vgg19", "resnet101"],
                    choices=sorted(MODELS))
    ap.add_argument("--eval-quality", type=int, nargs="+",
                    default=[PAPER_EVAL_QUALITY])
    ap.add_argument("--tau", type=int, default=PAPER_TAU)
    ap.add_argument("--mu", type=int, default=PAPER_MU)
    ap.add_argument("--q-embed", type=float, default=0.75,
                    help="嵌入端的 JPEG 品質。**論文未載**，本專案指定")
    ap.add_argument("--target", choices=("least_likely", "random"),
                    default="least_likely",
                    help="目標標籤的選法。**論文未載**，本專案指定")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1,
                    help="把影像等距切成幾片，本次跑第 --shard 片。切片是為了"
                         "讓一張卡吃一整批——1500 次迭代整批一起跑，逐張跑會"
                         "慢一個數量級")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_all, y_all, prov = load_split(args.data)
    if args.limit:
        x_all, y_all = x_all[:args.limit], y_all[:args.limit]
    if args.shards > 1:
        idx = torch.arange(len(x_all))[args.shard::args.shards]
        x_all, y_all = x_all[idx], y_all[idx]
    n = len(x_all)
    suite = MetricSuite(device=device)
    print(f"{n} 張、{device}、來源 {prov['repo']}", flush=True)

    rows = []
    for name in args.models:
        logits = build_model(name, device)

        def predict(z, bs=args.batch):
            with torch.no_grad():
                return torch.cat([logits(z[i:i + bs].to(device)).argmax(1).cpu()
                                  for i in range(0, len(z), bs)])

        spec = DJSMASpec(name=f"djsma_{name}", tau=args.tau, mu=args.mu,
                         q_embed=args.q_embed, target=args.target)
        t0 = time.time()
        advs = []
        for i in range(0, n, args.batch):
            xb = x_all[i:i + args.batch].to(device)
            advs.append(run_djsma(xb, spec, logits_fn=logits).x_def.cpu())
            print(f"  [{name}] {min(i + args.batch, n)}/{n}", flush=True)
        adv = torch.cat(advs)
        secs = (time.time() - t0) / n

        # 保真：相對**原圖**（論文相對已嵌浮水印的影像，參照點不同）
        pw = [suite.pairwise(x_all[i:i + 1].to(device), adv[i:i + 1].to(device))
              for i in range(n)]
        psnr = sum(p["psnr"] for p in pw) / n
        ssim = sum(p["ssim"] for p in pw) / n
        lpips = sum(p["lpips"] for p in pw) / n

        for q in args.eval_quality:
            acc_clean = float((predict(jpeg_roundtrip(x_all, q)) == y_all)
                              .double().mean())
            acc_adv = float((predict(jpeg_roundtrip(adv, q)) == y_all)
                            .double().mean())
            row = {"model": name, "eval_quality": q, "n": n,
                   "shard": args.shard, "shards": args.shards,
                   "tau": args.tau, "mu": args.mu, "q_embed": args.q_embed,
                   "target": args.target,
                   "acc_clean_jpeg": round(acc_clean * 100, 2),
                   "acc_adv_jpeg": round(acc_adv * 100, 2),
                   "succ_rate": round((acc_clean - acc_adv) * 100, 2),
                   "paper_acc_adv": PAPER_ACC.get(name),
                   "paper_acc_watermarked": PAPER_WM_ACC.get(name),
                   "psnr_vs_orig": round(float(psnr), 3),
                   "ssim_vs_orig": round(float(ssim), 4),
                   "lpips_vs_orig": round(float(lpips), 4),
                   "seconds_per_image": round(secs, 2)}
            rows.append(row)
            write_csv(args.out / "repro.csv", rows)
            print(f"{name} Q={q}：乾淨 {row['acc_clean_jpeg']:.1f}% → 對抗 "
                  f"{row['acc_adv_jpeg']:.1f}%（論文 {PAPER_ACC.get(name)}），"
                  f"成功率 {row['succ_rate']:.1f}%，PSNR {row['psnr_vs_orig']:.2f} "
                  f"SSIM {row['ssim_vs_orig']:.3f}，{row['seconds_per_image']:.1f}s/張",
                  flush=True)

    print()
    print("表：" + str(args.out / "repro.csv"))


if __name__ == "__main__":
    main()
