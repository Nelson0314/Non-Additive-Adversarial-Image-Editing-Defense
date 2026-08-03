"""以 SD v1.4 生成人物影像，作為 man／woman 兩類的資料。

    python scripts/generate_person_images.py --out data/_selected --per_class 4

為什麼人物用生成而非取用真實照片
─────────────────────────────────────────────────────────────────────

基準論文的註腳寫得很明確：「For ethical considerations, the human face image
is synthesized by the diffusion model」（Lo et al., CVPR 2024, 第 1 頁註 1）。
本專案採同一處置，理由有三：

1. **CC0 只放棄著作權，不處理肖像權。** 一張授權為 CC0 的人像照，其被攝者
   仍有人格權，而本專案的主題正是「對人臉影像施加擾動、再讓模型去編輯它」。
2. **與基準論文一致。** 對齊的目的是讓數字可以對讀，資料的取得方式屬於
   協定的一部分。
3. **完全可重現。** 每張圖的 prompt 與 seed 都寫進 `generation.json`，
   任何人都能從頭生成同一批影像，不需要下載任何檔案。

動物類不適用此考量，仍取 Commons 的 CC0 實拍（`scripts/fetch_cc0_images.py`）。

實作
─────────────────────────────────────────────────────────────────────

不呼叫 `SDWrapper.denoise`：該函式走的是 `_eps`，沒有 classifier-free
guidance，w = 1 下 SD v1.4 幾乎不服從 prompt（E26 §3），生出來的不會是人像。
此處直接走 `_eps_cfg`。**不修改 `denoise`**——那是既有實驗共用的路徑，為了
一支資料準備腳本去動它會改變別的東西的重跑條件。
"""

import argparse
import json
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sd import SDWrapper
from src.utils.artifacts import save_image
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory

# 每類的 prompt。刻意在年齡、族裔、髮型、場景上分散，避免整類都是同一種
# 長相——那會讓「四張影像」的有效樣本數遠低於張數。
PROMPTS = {
    "man": [
        "a photo of a man standing in a park, natural light, detailed face",
        "a portrait photo of an older man with a beard, indoor, soft light",
        "a photo of a young asian man on a city street, daylight",
        "a portrait photo of a black man wearing a jacket, outdoor",
    ],
    "woman": [
        "a photo of a woman standing in a park, natural light, detailed face",
        "a portrait photo of an older woman with gray hair, indoor, soft light",
        "a photo of a young asian woman on a city street, daylight",
        "a portrait photo of a black woman wearing a coat, outdoor",
    ],
}

NEGATIVE = "blurry, deformed, extra limbs, watermark, text, cartoon, painting"


@torch.no_grad()
def text2img(sd, prompt, negative, steps, guidance, size, seed, device):
    """DDIM text-to-image。回傳 (1,3,H,W) [0,1]。"""
    emb = sd.encode_text(prompt)
    emb_un = sd.encode_text(negative)
    lat = sd.latent_shape(size, size)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(lat, generator=g).to(device)

    abar = sd.alphas_cumprod(device)
    ts = torch.linspace(sd.num_train_timesteps - 1, 0, steps + 1).round().long()
    for i in range(steps):
        t, t_prev = ts[i], ts[i + 1]
        eps = sd._eps_cfg(z, t, emb, guidance, emb_un)
        pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
        z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps
    return sd.decode_latent(z).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_selected")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--per_class", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    device = get_device()
    sd = SDWrapper(args.model)
    out = Path(args.out)

    for cls, prompts in PROMPTS.items():
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        recs = []
        for i in range(args.per_class):
            prompt = prompts[i % len(prompts)]
            seed = args.seed + i
            reset_peak_memory()
            t0 = time.perf_counter()
            img = text2img(sd, prompt, NEGATIVE, args.steps, args.guidance,
                           args.size, seed, device)
            dt = time.perf_counter() - t0
            path = d / f"gen_{i:02d}.png"
            save_image(img, path)
            recs.append({
                "file": path.name, "prompt": prompt, "negative": NEGATIVE,
                "model": args.model, "steps": args.steps,
                "guidance_scale": args.guidance, "size": args.size,
                "seed": seed, "seconds": round(dt, 1),
                "peak_mb": round(peak_memory_mb(), 1),
            })
            print(f"  [{cls}] {path.name}  {dt:.0f}s  "
                  f"peak={peak_memory_mb():.0f}MB  {prompt[:50]}", flush=True)
        (d / "generation.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成。下一步：scripts/prepare_dataset.py --src {out} "
          f"--dst data/lo_aligned")


if __name__ == "__main__":
    main()
