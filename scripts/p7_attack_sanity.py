"""E26 —— 攻擊端本身有沒有在做文字引導編輯？

這個腳本存在的理由。使用者判讀 `runs/p5_semantic_axis/compare.html` 後
回報：「連原始圖片被文字編輯都沒有成功，然後有無防禦的圖編輯後長的都差不多，
一樣都很爛。」若成立，則 E2–E23 全部是在防禦一個不存在的攻擊。

程式碼層面的根因是明確的：`src/models/sd.py` 的 `_eps` 只以條件嵌入呼叫
一次 UNet，全專案 `grep guidance|uncond|cfg_scale|do_classifier` 沒有任何命中。
Stable Diffusion v1.x 是在 classifier-free guidance 下訓練也在其下使用的，
w = 1 時 prompt 對輸出的影響極弱。

本腳本以本機快取的真實 SD v1.4（4.27 GB，CPU）直接量出 w 的影響：

- 對每張影像跑 SDEdit，掃 `guidance_scale ∈ {1.0, 3.0, 7.5}`（strength 0.5）
  以及 `strength ∈ {0.5, 0.7}`（w=7.5），其餘設定與 E15/E21/E23 相同。
- 量三件事：對 prompt 的 CLIP / SigLIP 對齊、對原圖的 LPIPS、以及影像本身。
- 判準與 E25-1 相同：編輯成功要求對齊度相對原圖顯著上升。

輸出 `runs/p7_attack_sanity/{probe.csv, *.png, compare.html}`。

CPU 成本：每個 UNet 前向在 512² 約數秒，w>1 的組態每步兩次前向。全部組態
約數百次前向，屬「跑一晚」等級，不需要 GPU。
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "runs" / "p7_attack_sanity"

# (標籤, guidance_scale, strength)。第一列即現行設定，是被檢驗的對象。
CONFIGS = [
    ("w1.0_s0.5（現行設定）", 1.0, 0.5),
    ("w3.0_s0.5", 3.0, 0.5),
    ("w7.5_s0.5（標準攻擊）", 7.5, 0.5),
    ("w7.5_s0.7", 7.5, 0.7),
]

SEED = 20260728 + 10_000     # 與 run_defense.py 的 EVAL_SEED_OFFSET 一致


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--limit", type=int, default=2, help="影像數，CPU 上預設 2 張")
    args = ap.parse_args()

    from src.models.sd import SDWrapper
    from src.metrics.suite import MetricSuite

    OUT.mkdir(parents=True, exist_ok=True)
    # SDWrapper 自己決定裝置（get_device）；本機沒有 CUDA 故為 CPU。
    sd = SDWrapper(args.model)
    device = sd.device
    suite = MetricSuite(device=device)
    print(f"[p7] device={device}  model={args.model}", flush=True)

    prompts = yaml.safe_load(
        (ROOT / "data/dayn_testset/prompts.yaml").read_text(encoding="utf-8"))

    # 無條件嵌入即空字串的 CLIP 編碼，與 diffusers 的做法一致
    emb_uncond = sd.encode_text("").detach()

    files = sorted((ROOT / "data/dayn_testset").rglob("*.png"))[:args.limit]
    rows = []
    for path in files:
        name, cls = path.stem, path.parent.name
        prompt = (prompts.get(cls) or ["a photo"])[0]
        img = Image.open(path).convert("RGB").resize(
            (args.size, args.size), Image.LANCZOS)
        x = torch.from_numpy(
            __import__("numpy").asarray(img, dtype="float32") / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        Image.fromarray(
            (x[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        ).save(OUT / f"{name}__orig.png")

        emb = sd.encode_text(prompt).detach()
        sem_orig = suite.semantic(x, prompt)
        print(f"\n[{name}] prompt={prompt!r}  原圖 clip={sem_orig['clip']:.4f} "
              f"siglip={sem_orig['siglip']:.4f}", flush=True)

        lat = sd.latent_shape(args.size, args.size)
        noise = sd.sample_edit_noise(torch.empty(lat, device=device), seed=SEED)

        for label, w, strength in CONFIGS:
            t0 = time.perf_counter()
            with torch.no_grad():
                y = sd.sdedit(x, emb, noise, args.n_edit, strength=strength,
                              guidance_scale=w, emb_uncond=emb_uncond)
            sem = suite.semantic(y, prompt)
            pw = suite.pairwise(x, y)
            tag = f"w{w}_s{strength}"
            Image.fromarray(
                (y[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            ).save(OUT / f"{name}__{tag}.png")

            rows.append({
                "image": name, "prompt": prompt, "label": label,
                "guidance_scale": w, "strength": strength, "n_edit": args.n_edit,
                "clip_orig": sem_orig["clip"], "clip_edit": sem["clip"],
                "d_clip": sem["clip"] - sem_orig["clip"],
                "siglip_orig": sem_orig["siglip"], "siglip_edit": sem["siglip"],
                "d_siglip": sem["siglip"] - sem_orig["siglip"],
                "lpips_to_orig": pw["lpips"], "psnr_to_orig": pw["psnr"],
                "seconds": time.perf_counter() - t0,
            })
            print(f"  {label:22s} Δclip {rows[-1]['d_clip']:+.4f}  "
                  f"Δsiglip {rows[-1]['d_siglip']:+.4f}  "
                  f"LPIPS→原圖 {pw['lpips']:.4f}  {rows[-1]['seconds']:.0f}s",
                  flush=True)

    with (OUT / "probe.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n=== 彙總（逐組態對全部影像取平均）===")
    print(f"{'組態':24s}{'Δclip':>9s}{'Δsiglip':>10s}{'LPIPS→原圖':>12s}")
    for label, w, strength in CONFIGS:
        sub = [r for r in rows if r["label"] == label]
        n = len(sub)
        print(f"{label:24s}"
              f"{sum(r['d_clip'] for r in sub) / n:>+9.4f}"
              f"{sum(r['d_siglip'] for r in sub) / n:>+10.4f}"
              f"{sum(r['lpips_to_orig'] for r in sub) / n:>12.4f}")
    print(f"\n寫入 {OUT}")


if __name__ == "__main__":
    main()
