"""以文字生成人像，供 teaser 案例挑選（不混入 data/pexels_portraits）。

模型：FLUX.2 [klein] 4B 蒸餾版（black-forest-labs/FLUX.2-klein-4B，bf16），4 步、guidance 1.0（與跨編輯器實驗的 klein 設定相同），512²。
第 i 張（i = 0 … --num−1）的種子為 --seed + i。
輸出：<--out-dir>/man/<名稱>_<i>.png 與 <--out-dir>/manifest.csv（image、prompt、seed、model、settings）。
`man/` 子目錄是 generate_defenses、generate_blurguard_defenses 讀取資料的目錄版面（兩者只掃類別子目錄），不代表影像內容的類別。

用法（遠端 baseline 專案根，需一張卡；取卡見 vendor/scripts/run_with_gpu_lease.sh）：
    python scripts/generate_synthetic_portraits.py --portrait-prompt "<prompt>" --num 8 --name <名稱>
"""
import argparse
import csv
import json
import time
from pathlib import Path

MODEL = "black-forest-labs/FLUX.2-klein-4B"
SETTINGS = {"num_inference_steps": 4, "guidance_scale": 1.0, "height": 512, "width": 512}
LAYOUT_CLASS = "man"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portrait-prompt", required=True, help="生成人像的文字")
    ap.add_argument("--seed", type=int, default=20260812, help="第一張的種子（預設主種子 20260812）")
    ap.add_argument("--num", type=int, default=1, help="張數")
    ap.add_argument("--name", required=True, help="輸出名稱（小寫 snake_case），預設輸出到 artifacts/synthetic_portraits/<名稱>/")
    ap.add_argument("--out-dir", type=Path, default=None, help="覆寫輸出目錄")
    args = ap.parse_args()
    if args.num < 1:
        ap.error("--num 須 ≥ 1")
    if not args.name.replace("_", "").isalnum() or args.name != args.name.lower():
        ap.error(f"--name 須為小寫 snake_case：{args.name}")
    out = args.out_dir or Path("artifacts/synthetic_portraits") / args.name
    if (out / "manifest.csv").exists():
        raise SystemExit(f"{out} 已有 manifest.csv；換一個 --name 或 --out-dir")

    import torch
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    (out / LAYOUT_CLASS).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.num):
        seed = args.seed + i
        t = time.time()
        img = pipe(prompt=args.portrait_prompt, generator=torch.Generator("cuda").manual_seed(seed), **SETTINGS).images[0]
        name = f"{args.name}_{i:02d}"
        img.save(out / LAYOUT_CLASS / f"{name}.png")
        rows.append({"image": name, "prompt": args.portrait_prompt, "seed": seed, "model": MODEL,
                     "settings": json.dumps(SETTINGS), "seconds": round(time.time() - t, 2)})
        print(f"[PORTRAIT] {name} seed={seed}", flush=True)
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[ALLDONE] {len(rows)} 張 → {out}", flush=True)


if __name__ == "__main__":
    main()
