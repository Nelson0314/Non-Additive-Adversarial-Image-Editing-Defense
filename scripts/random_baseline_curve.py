"""隨機擾動在各個失真上「免費」取得多少免疫效果？

    python scripts/random_baseline_curve.py --size 512 --images horse_00,man_00

不需要訓練，只有 SDEdit 與指標。

## 這在回答什麼

2026-08-04 實測到一件事（LEDGER 1.18）：在擾動 LPIPS 0.14 上，**把高斯雜訊
二分搜尋到同一個 LPIPS，得到的 Δsiglip 是 −0.0186 ± 0.0053，依 E25 的規則
同樣判為語意失敗**——而最佳化解是 −0.0309。隨機方向就拿到 60% 的效果。

PhotoGuard 的 Table 6 有一列隨機雜訊對照，但它匹配的是**振幅**
（「of the same intensity」），結論是隨機「not effective」。**匹配感知失真
的版本沒有人做過**，而兩者的差別不小：`p17` 量到同一張影像上 PGD 解的
LPIPS 是 0.2935、同振幅隨機 sign 只有 0.1497（LEDGER 2.19、1.24）。

本腳本把它量成一條曲線：**Δsiglip、Δniqe 與 Table 1 五指標，各自作為擾動
LPIPS 的函數**。它同時是三件事：

1. 一條**免費基線**。任何方法的成績都應該減掉同失真的隨機值才有意義。
2. `runs/gate_suppress/` 的**解析度對照**。閘門跑在 256²（本機唯一跑得動的
   解析度），而 SD v1.4 是在 512² 上訓練的；本腳本可在 512² 上重跑同一個
   隨機條件，看該現象是不是解析度造成的。
3. **κ = 0.06 的另一種讀法**。文獻的標準預算對應 LPIPS ≈ 0.58（LEDGER 2.18），
   本腳本的最高一級就取在那裡：如果隨機雜訊在該失真上就有可觀的 Δsiglip，
   那麼在該預算上宣稱的免疫效果有多少是方法帶來的，就是一個公開的問題。

## 判準與規則

`Δsiglip = SigLIP(y_def, prompt) − SigLIP(y_ref, prompt)`，負值代表較不服從
prompt。判定沿用 E25 的規則（平均為負且 |mean| > sd），故 `--eval_seeds`
必須 ≥ 2——n = 1 時 sd 恆為 0，任何判定自動成立（LEDGER 1.4）。

兩條分支共用逐元素相同的 ε，且 `y_ref` 逐 (影像, 種子) 只算一次、各失真等級
共用：它不依賴擾動，重算只是重複耗用時間。
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image, save_json  # noqa: E402
from src.metrics.ray_scale import (  # noqa: E402
    lpips_against, scale_to_lpips as _scale,
)
from src.utils.device import get_device  # noqa: E402

EVAL_SEED_OFFSET = 10_000
TABLE1 = ("psnr", "ssim", "vif_p", "fsim", "lpips")
# 最高一級取 0.55：κ = 0.06 對應 LPIPS ≈ 0.58（LEDGER 2.18），
# 而本專案跑過的最大預算是 0.10。中間各級用來看曲線的形狀。
LEVELS = (0.05, 0.10, 0.20, 0.35, 0.55)


def scale_to_lpips(suite, x01, noise, target, tol=0.005, iters=24):
    """把雜訊縮放到指定的 LPIPS。回傳 (x_pert, 實際 LPIPS)。

    薄包裝，實作與理由見 `src/metrics/ray_scale.py`。
    """
    x, got, _ = _scale(lpips_against(suite, x01), x01, noise, target,
                       tol=tol, iters=iters)
    return x, got


def load(data: Path, name: str, size: int, device):
    import yaml
    from PIL import Image
    import torchvision.transforms as T

    spec = yaml.safe_load((data / "prompts.yaml").read_text(encoding="utf-8"))
    cls = name.rsplit("_", 1)[0]
    p = data / cls / f"{name}.png"
    if not p.exists():
        raise SystemExit(f"{p} 不存在")
    img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device), spec[cls]["prompts"][0]


def main():
    ap = argparse.ArgumentParser(description="隨機擾動的免費基線曲線")
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--out", default="runs/random_baseline")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--images", default="horse_00,man_00")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--levels", default=",".join(str(v) for v in LEVELS))
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--eval_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    if args.eval_seeds < 2:
        raise SystemExit("--eval_seeds 必須 ≥ 2（LEDGER 1.4）")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out / "protocol.json")
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)
    levels = [float(v) for v in args.levels.split(",")]
    res_path, sum_path = out / "results.csv", out / "summary.csv"

    for name in [s.strip() for s in args.images.split(",") if s.strip()]:
        x01, prompt = load(ROOT / args.data, name, args.size, device)
        emb = sd.encode_text(prompt).detach()
        emb_u = sd.encode_text("").detach()
        lat = sd.latent_shape(args.size, args.size)
        kw = dict(strength=args.strength, guidance_scale=args.guidance,
                  emb_uncond=emb_u)

        # y_ref 不依賴擾動，逐種子只算一次、各失真等級共用
        refs = []
        with torch.no_grad():
            for i in range(args.eval_seeds):
                s = args.seed + EVAL_SEED_OFFSET + i
                nz = sd.sample_edit_noise(torch.empty(lat, device=device), seed=s)
                refs.append((s, nz, sd.sdedit(x01, emb, nz, args.n_edit, **kw)))
        print(f"\n=== {name} / {prompt!r} / {len(refs)} 個參照編輯 ===",
              flush=True)

        g = torch.Generator(device="cpu").manual_seed(args.seed)
        noise = torch.randn(x01.shape, generator=g).to(device)
        for tgt in levels:
            t0 = time.perf_counter()
            try:
                xp, got = scale_to_lpips(suite, x01, noise, tgt)
            except RuntimeError as e:
                print(f"  [LPIPS {tgt}] 跳過：{e}", flush=True)
                continue
            save_image(xp, out / f"{name}__lpips{tgt}.png")
            rows = []
            with torch.no_grad():
                for s, nz, y_ref in refs:
                    y_def = sd.sdedit(xp, emb, nz, args.n_edit, **kw)
                    m = suite.full(y_ref, y_def, prompt=prompt)
                    rows.append({"image": name, "target_lpips": tgt,
                                 "eval_seed": s,
                                 **{f"edit_{k}": v for k, v in m.items()}})
            n = len(rows)
            dsig = [r["edit_siglip_b"] - r["edit_siglip_a"] for r in rows]
            dniq = [r["edit_niqe_b"] - r["edit_niqe_a"] for r in rows]
            ms = sum(dsig) / n
            ss = (sum((v - ms) ** 2 for v in dsig) / n) ** 0.5
            mn = sum(dniq) / n
            sn = (sum((v - mn) ** 2 for v in dniq) / n) ** 0.5
            pert = suite.pairwise(x01, xp)
            srow = {
                "image": name, "prompt": prompt, "target_lpips": tgt,
                "pert_lpips": got, "pert_linf": pert["linf"],
                "pert_psnr": pert["psnr"], "pert_rms": pert.get("rms"),
                "n_seeds": n,
                "dsiglip_mean": ms, "dsiglip_sd": ss,
                "dniqe_mean": mn, "dniqe_sd": sn,
                "semantic_fail": bool(ms < 0 and abs(ms) > ss),
                **{f"edit_{k}": sum(r[f"edit_{k}"] for r in rows) / n
                   for k in TABLE1},
                "seconds": round(time.perf_counter() - t0, 1),
            }
            for path, data in ((res_path, rows), (sum_path, [srow])):
                new = not path.exists()
                with open(path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                    if new:
                        w.writeheader()
                    w.writerows(data)
            print(f"  LPIPS {got:.4f}（目標 {tgt}）  L∞ {pert['linf']:.4f}  "
                  f"Δsiglip {ms:+.5f} ± {ss:.5f}  Δniqe {mn:+.3f} ± {sn:.3f}  "
                  f"語意失敗={srow['semantic_fail']}  "
                  f"編輯 LPIPS {srow['edit_lpips']:.4f}  "
                  f"（{srow['seconds']:.0f}s）", flush=True)

    print(f"\n完成。{sum_path}", flush=True)
    print("讀法：這是任何方法都應該減掉的免費基線。若某個失真上的隨機值已經"
          "接近方法的值，該方法在該預算上的貢獻就不成立（LEDGER 1.18）")


if __name__ == "__main__":
    main()
