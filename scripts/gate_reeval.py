"""用更多評測種子重評已經存檔的防禦圖。不重新訓練。

    python scripts/gate_reeval.py --run runs/gate_suppress --eval_seeds 20

## 為什麼這一支存在

`runs/gate_suppress/` 的閘門用 5 個評測種子，量到最佳化條件對同失真隨機條件的
配對差為 −0.01487 ± 0.01400（Cohen d = −1.06）。依該標準差，
α = 0.05 雙尾、power 80% 需要 **n ≈ 7**，power 90% 需要 **n ≈ 10**
（LEDGER 1.23）。**現在是 5，差一點點。**

補足樣本數**不需要重新訓練**：`x_def` 與隨機對照都已經以 PNG 存檔，
缺的只是更多組評測噪聲。訓練是 65 s/step × 60 步，評測是每個種子兩次
SDEdit——後者便宜一個量級，且本機跑得動。

**兩個條件必須用同一組 ε。** 那是配對分析成立的前提，也是把所需樣本數由
約 48 降到約 7 的原因。此處以同一個迴圈餵給兩個條件來保證，不是靠約定。

## 讀 PNG 而不是重算

存檔的是 8-bit PNG，重新載入會有量化誤差。實測該誤差對 LPIPS 的影響在
1e-3 量級，而本腳本要分辨的效果量是 1.5e-2，差一個量級。腳本會把重新
載入後的擾動 LPIPS 印出來與原始 `summary.csv` 的值對照，**兩者不符就是
存檔或載入有問題，不要略過**。
"""

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.device import get_device  # noqa: E402

EVAL_SEED_OFFSET = 10_000
TABLE1 = ("psnr", "ssim", "vif_p", "fsim", "lpips")


def load_png(path: Path, size: int, device):
    from PIL import Image
    import torchvision.transforms as T

    if not path.exists():
        raise SystemExit(f"{path} 不存在")
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser(description="加評測種子重評已存檔的防禦圖")
    ap.add_argument("--run", default="runs/gate_suppress")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--eval_seeds", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    run = ROOT / args.run
    out = ROOT / (args.out or f"{args.run}_reeval")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)

    # 從既有 summary.csv 取得要重評哪些 (影像, site)，以及原始的擾動 LPIPS
    src = run / "summary.csv"
    if not src.exists():
        raise SystemExit(f"{src} 不存在，沒有可重評的格")
    cells = {}
    with open(src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cells.setdefault((r["image"], r["site"]), {})[r["arm"]] = r

    res_path, sum_path = out / "results.csv", out / "summary.csv"
    for (name, site), arms in sorted(cells.items()):
        prompt = next(iter(arms.values()))["prompt"]
        x01 = load_png(run / f"{name}__orig.png", args.size, device)
        paths = {"opt": run / f"{name}__{site}__def.png",
                 "rand": run / f"{name}__{site}__rand.png"}
        xs = {a: load_png(p, args.size, device) for a, p in paths.items()
              if a in arms}
        print(f"\n=== {name} / site {site} / {prompt!r} / "
              f"{args.eval_seeds} 個種子 ===", flush=True)
        for a, xa in xs.items():
            got = suite.pairwise(x01, xa)["lpips"]
            was = float(arms[a]["pert_lpips"])
            print(f"  [{a}] 重新載入後擾動 LPIPS {got:.4f}"
                  f"（原 {was:.4f}，差 {abs(got - was):.4f}）", flush=True)
            if abs(got - was) > 0.01:
                raise SystemExit(
                    f"{a} 條件重新載入後的擾動 LPIPS 與原值差 "
                    f"{abs(got - was):.4f}，超過 8-bit 量化該有的量級。"
                    "存檔或載入有問題，不可繼續——匹配失真是整個比較的前提")

        emb = sd.encode_text(prompt).detach()
        emb_u = sd.encode_text("").detach()
        lat = sd.latent_shape(args.size, args.size)
        kw = dict(strength=args.strength, guidance_scale=args.guidance,
                  emb_uncond=emb_u)

        # 兩個條件共用同一組 ε，逐種子在同一個迴圈裡餵給兩邊
        per_arm = {a: [] for a in xs}
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(args.eval_seeds):
                s = args.seed + EVAL_SEED_OFFSET + i
                nz = sd.sample_edit_noise(torch.empty(lat, device=device),
                                          seed=s)
                y_ref = sd.sdedit(x01, emb, nz, args.n_edit, **kw)
                for a, xa in xs.items():
                    y_def = sd.sdedit(xa, emb, nz, args.n_edit, **kw)
                    m = suite.full(y_ref, y_def, prompt=prompt)
                    per_arm[a].append({
                        "image": name, "site": site, "arm": a, "eval_seed": s,
                        **{f"edit_{k}": v for k, v in m.items()}})
                if (i + 1) % 5 == 0:
                    print(f"  ...{i + 1}/{args.eval_seeds} 個種子，"
                          f"{time.perf_counter() - t0:.0f}s", flush=True)

        srows = []
        for a, rows in per_arm.items():
            n = len(rows)
            dsig = [r["edit_siglip_b"] - r["edit_siglip_a"] for r in rows]
            dniq = [r["edit_niqe_b"] - r["edit_niqe_a"] for r in rows]
            ms, mn = st.mean(dsig), st.mean(dniq)
            ss, sn = st.pstdev(dsig), st.pstdev(dniq)
            srows.append({
                "image": name, "site": site, "arm": a, "prompt": prompt,
                "n_seeds": n,
                "pert_lpips": suite.pairwise(x01, xs[a])["lpips"],
                "dsiglip_mean": ms, "dsiglip_sd": ss,
                "dniqe_mean": mn, "dniqe_sd": sn,
                "semantic_fail": bool(ms < 0 and abs(ms) > ss),
                **{f"edit_{k}": st.mean(r[f"edit_{k}"] for r in rows)
                   for k in TABLE1},
            })
            print(f"  [{a}] Δsiglip {ms:+.5f} ± {ss:.5f}   "
                  f"Δniqe {mn:+.4f} ± {sn:.4f}   "
                  f"編輯 LPIPS {srows[-1]['edit_lpips']:.4f}   "
                  f"語意失敗={srows[-1]['semantic_fail']}", flush=True)

        # 配對檢定。兩個條件共用 ε，故逐種子相減才是正確的比較（LEDGER 1.23）
        if len(per_arm) == 2:
            o = [r["edit_siglip_b"] - r["edit_siglip_a"] for r in per_arm["opt"]]
            r_ = [r["edit_siglip_b"] - r["edit_siglip_a"]
                  for r in per_arm["rand"]]
            d = [a - b for a, b in zip(o, r_)]
            m, sd_ = st.mean(d), st.stdev(d)
            t = m / (sd_ / len(d) ** 0.5) if sd_ > 0 else float("nan")
            n_need = (1.96 + 0.84) ** 2 * (sd_ / abs(m)) ** 2 if m else float("inf")
            print(f"  [配對] 最佳化 − 隨機 = {m:+.5f} ± {sd_:.5f}"
                  f"（n={len(d)}）  t = {t:+.3f}  Cohen d = {m / sd_:+.3f}",
                  flush=True)
            print(f"  [配對] 同向的種子 {sum(1 for v in d if v < 0)}/{len(d)}；"
                  f"power 80% 所需 n ≈ {n_need:.0f}", flush=True)
            srows.append({"image": name, "site": site, "arm": "paired_diff",
                          "prompt": prompt, "n_seeds": len(d),
                          "dsiglip_mean": m, "dsiglip_sd": sd_,
                          "pert_lpips": "", "dniqe_mean": "", "dniqe_sd": "",
                          "semantic_fail": "",
                          **{f"edit_{k}": "" for k in TABLE1}})

        for path, data in ((res_path, [r for rows in per_arm.values()
                                       for r in rows]),
                           (sum_path, srows)):
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(data)

    print(f"\n完成。{sum_path}", flush=True)


if __name__ == "__main__":
    main()
