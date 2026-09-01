"""AdvDrop 在**它自己的威脅模型**上的重現：ImageNet 分類 ＋ ResNet-50。

為什麼要先做這一步：本專案要把 AdvDrop 當對照組，而它原本不是為擴散編輯
防護寫的。先在原生場景對上它自己報的數字，才知道移植過去之後量到的東西
是方法本身，不是移植錯誤。

重現目標（arXiv:2108.09034，2026-08-21 由論文全文抄出）
────────────────────────────────────────────────────────────────────
§4.1  2000 張**已被正確分類**的 ImageNet 影像、ResNet-50、未定向 50 步、
      定向 500 步、`q_init = 1`、約束 `|q - q_init|_inf < eps`

Table 1  未定向成功率  eps=20: 98.55±0.26  eps=60: 99.85±0.08  eps=100: 100.00±0.00
         定向成功率    eps=20: 97.20±0.37  eps=60: 99.45±0.16  eps=100: 99.95±0.05
Table 2  eps=100 在防禦下  無防禦 100.00  JPEG-30 82.60  Bit-6 95.65
§4.5     量化方法消融  硬四捨五入 5.00±0.98  本方法 97.20±0.37

本專案的差異（必須寫進報表）
────────────────────────────────────────────────────────────────────
1. **張數**：預設 200 而非 2000（使用者 2026-08-21 指示「看得出趨勢就好」）。
   成功率是比例，200 張的標準誤約 ±1.5%，論文報的是 ±0.05-0.37%。
2. **資料來源是 HF 鏡像** `mrm8488/ImageNet1K-val`，非官方 ILSVRC 下載點。
3. **軟四捨五入的 alpha 退火**沿用官方程式碼的 `[0.1, 1e-20]`；論文只說線性
   遞減，未給端點。

用法：
    python scripts/advdrop_repro.py --out runs/advdrop_repro
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.advdrop import (  # noqa: E402
    PAPER_EPS_SWEEP, PAPER_Q_INIT, PAPER_TARGETED_STEPS,
    PAPER_UNTARGETED_STEPS, AdvDropSpec, run_advdrop,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_split(root: Path):
    import numpy as np

    prov = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    xs, ys = [], []
    for rec in prov["images"]:
        a = np.asarray(Image.open(root / (rec["name"] + ".png")).convert("RGB"),
                       dtype="float32") / 255.0
        xs.append(torch.from_numpy(a).permute(2, 0, 1))
        ys.append(int(rec["label"]))
    return torch.stack(xs), torch.tensor(ys), prov


def build_model(device):
    from torchvision.models import ResNet50_Weights, resnet50

    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return lambda x01: m((x01 - mean) / std)


def jpeg_roundtrip(x01: torch.Tensor, quality: int,
                   subsampling: int = -1) -> torch.Tensor:
    """真正的 JPEG 編解碼（PIL），不是可微分近似。防禦端不需要梯度。"""
    import numpy as np

    out = []
    for xi in x01.cpu():
        arr = (xi.permute(1, 2, 0).numpy() * 255).round().clip(0, 255)
        im = Image.fromarray(arr.astype("uint8"))
        buf = io.BytesIO()
        # subsampling: -1 = PIL 依品質自選（品質 30 時是 4:2:0），
        # 0 = 4:4:4（不對色度下採樣）。論文未寫用哪一種。
        im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
        buf.seek(0)
        a = np.asarray(Image.open(buf).convert("RGB"), dtype="float32") / 255.0
        out.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(out).to(x01.device)


def bit_depth(x01: torch.Tensor, bits: int) -> torch.Tensor:
    """Feature squeezing 的位元深度縮減（Table 2 的 Bit-6 那一欄）。"""
    levels = 2 ** bits - 1
    return torch.round(x01 * levels) / levels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/imagenet_advdrop"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=list(PAPER_EPS_SWEEP))
    ap.add_argument("--step-size", type=float, nargs="+", default=[1.0],
                    help="量化表每步移動多少。論文式 (7) 是 sign 更新、隱含 1，"
                         "但 50 步 x 1 走不完 eps=100 的區間（實測 q 平均只到 "
                         "7.9），故開放成掃描變因")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--targeted", action="store_true",
                    help="另跑定向攻擊（500 步，貴 10 倍）")
    ap.add_argument("--targeted-seed", type=int, default=20260821)
    ap.add_argument("--succ-def", choices=("label", "defended"), default="label",
                    help="成功率的定義。label = f(x') != y（真實標籤，本專案原本"
                         "的作法）；defended = f(D(x')) != f(D(x))（與**防禦後的"
                         "乾淨預測**比）。後者會自動扣掉『算子本身把乾淨影像分錯』"
                         "那一塊，是論文 Table 2 的 JPEG-30 欄比我們高四倍最可能"
                         "的解釋——論文未明說用哪一個")
    ap.add_argument("--color", choices=("rgb", "ycbcr"), default="rgb",
                    help="量化做在哪個色彩空間。rgb = 官方程式碼實際做的；"
                         "ycbcr = 變數命名與論文 Figure 4 指向的。JPEG 對色度"
                         "做 4:2:0，兩者在 JPEG 防禦那一欄可能差很多")
    ap.add_argument("--jpeg-subsampling", type=int, default=-1,
                    help="-1 = PIL 依品質自選（品質 30 是 4:2:0）；0 = 4:4:4")
    ap.add_argument("--hard-quant", action="store_true",
                    help="用真正的 round() 取代軟四捨五入（論文 §4.5 的消融，"
                         "它報 5.00±0.98%）。round 的梯度處處為零，量化表完全"
                         "不會動，剩下的成功率全部來自 q_init 本身的損傷。"
                         "**注意**：把 alpha 設到 1e-20 不等於這件事——那時"
                         "phi_diff 仍可微，實測成功率 1.000")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_all, y_all, prov = load_split(args.data)
    if args.limit:
        x_all, y_all = x_all[:args.limit], y_all[:args.limit]
    logits = build_model(device)
    suite = MetricSuite(device=device)
    n = len(x_all)
    print(str(n) + " 張、" + device + "、資料來源 " + prov["repo"])

    def predict(z):
        with torch.no_grad():
            return torch.cat([
                logits(z[i:i + args.batch].to(device)).argmax(1).cpu()
                for i in range(0, len(z), args.batch)])

    if not (predict(x_all) == y_all).all():
        raise SystemExit("資料集裡有本來就分錯的影像，過濾沒生效")

    def one(eps, ss, targeted):
        steps = PAPER_TARGETED_STEPS if targeted else PAPER_UNTARGETED_STEPS
        spec = AdvDropSpec(
            name="advdrop", q_init=PAPER_Q_INIT, eps=eps, steps=steps,
            step_size=ss, color=args.color, hard_round=args.hard_quant,
            modified_from_paper=(args.hard_quant or args.color != "rgb"),
            modification_note="；".join(
                ([] if not args.hard_quant else
                 ["以硬四捨五入取代軟四捨五入，重現 §4.5 的消融"])
                + ([] if args.color == "rgb" else
                   ["在 YCbCr 上量化；官方程式碼是 RGB"])))
        t0 = time.time()
        advs, tgts = [], []
        for i in range(0, n, args.batch):
            x = x_all[i:i + args.batch].to(device)
            y = y_all[i:i + args.batch].to(device)
            if targeted:
                g = torch.Generator().manual_seed(args.targeted_seed + i)
                off = torch.randint(1, 1000, y.shape, generator=g)
                t = ((y.cpu() + off) % 1000).to(device)
                tgts.append(t.cpu())

                def loss_fn(xa, t=t):
                    # 論文式 (2) 定向：-log p_{y_adv}
                    return -torch.log_softmax(logits(xa), 1).gather(
                        1, t[:, None]).sum()
            else:
                def loss_fn(xa, y=y):
                    # 論文式 (2) 未定向：log p_y
                    return torch.log_softmax(logits(xa), 1).gather(
                        1, y[:, None]).sum()

            advs.append(run_advdrop(None, x, spec, loss_fn=loss_fn).x_def.cpu())
        adv = torch.cat(advs)
        tgt = torch.cat(tgts) if targeted else None

        def succ(z, defence=None):
            """`defence` 給定時是把同一個算子也套到乾淨影像上的那個函式。

            `--succ-def defended` 之下比的是「防禦後的預測有沒有被改掉」，
            分母因此不含算子自己造成的錯誤。
            """
            pred = predict(z)
            if targeted:
                return float((pred == tgt).double().mean())
            if args.succ_def == "defended" and defence is not None:
                ref = predict(defence(x_all))
                return float((pred != ref).double().mean())
            return float((pred != y_all).double().mean())

        lp = sum(suite.pairwise(x_all[i:i + 1].to(device),
                                adv[i:i + 1].to(device))["lpips"]
                 for i in range(n)) / n
        return {"eps": eps, "step_size": ss, "targeted": targeted,
                "steps": steps, "n": n,
                "succ_def": args.succ_def,
                "hard_quant": args.hard_quant,
                "succ_none": round(succ(adv), 4),
                "color": args.color,
                "jpeg_subsampling": args.jpeg_subsampling,
                "succ_jpeg30": round(
                    succ(jpeg_roundtrip(adv, 30, args.jpeg_subsampling),
                         lambda z: jpeg_roundtrip(z, 30, args.jpeg_subsampling)), 4),
                "succ_bit6": round(succ(bit_depth(adv, 6),
                                        lambda z: bit_depth(z, 6)), 4),
                "lpips": round(float(lp), 5),
                "seconds": round(time.time() - t0, 1)}

    rows = []
    modes = [False, True] if args.targeted else [False]
    for eps in args.eps:
        for ss in args.step_size:
            for targeted in modes:
                row = one(eps, ss, targeted)
                rows.append(row)
                write_csv(args.out / "repro.csv", rows)
                print("eps=%g 步長=%g %s：無防禦 %.3f　JPEG-30 %.3f　"
                      "Bit-6 %.3f　LPIPS %.4f　(%.0fs)"
                      % (eps, ss, "定向" if targeted else "未定向",
                         row["succ_none"], row["succ_jpeg30"],
                         row["succ_bit6"], row["lpips"], row["seconds"]),
                      flush=True)

    print()
    print("表：" + str(args.out / "repro.csv"))


if __name__ == "__main__":
    main()
