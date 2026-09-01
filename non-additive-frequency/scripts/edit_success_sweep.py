"""編輯成功掃描：未防禦的原圖在不同 SDEdit 設定下，編輯結果照不照 prompt 走。

使用者 2026-08-17 的驗收條件是「一切實驗都必須在編輯成功的前提下」。現行的
strength 0.55 會改圖但不服從 prompt（FND-024／029／030），故必須先定出一組
會服從的設定，才有資格談防禦。

本腳本**不跑任何防禦**，只跑 `SDEdit(原圖)`，逐設定存圖並量 CLIP 與 SigLIP
對 prompt 的相似度。判準以人眼為主，數值只是輔助。

    python scripts/edit_success_sweep.py --out runs/editsweep --data data/set0817
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import (  # noqa: E402
    EDIT_SEED, EDIT_STEPS, MODEL_NAME, RESOLUTION, load_dataset,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

# (strength, guidance)。0.55/7.5 是現行設定，留著當對照。
SETTINGS = [(0.55, 7.5), (0.70, 7.5), (0.80, 7.5), (0.80, 12.0), (0.90, 12.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--prompt-index", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]

    rows = []
    for item in dataset:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(x01, args.out / f"{item['name']}__orig.png")
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
        print(f"\n########## {item['name']} 「{item['prompt']}」 ##########",
              flush=True)
        for strength, guidance in SETTINGS:
            with torch.no_grad():
                edit = sd.sdedit(x01, emb, noise, EDIT_STEPS, strength=strength,
                                 guidance_scale=guidance, emb_uncond=emb_u)
            sem = suite.semantic(edit, item["prompt"])
            base = suite.semantic(x01, item["prompt"])
            tag = f"s{strength:g}_g{guidance:g}"
            save_image(edit, args.out / f"{item['name']}__{tag}__edit.png")
            rows.append({
                "image": item["name"], "prompt": item["prompt"],
                "strength": strength, "guidance": guidance, "tag": tag,
                "clip_orig": round(base["clip"], 4),
                "clip_edit": round(sem["clip"], 4),
                "clip_gain": round(sem["clip"] - base["clip"], 4),
                "siglip_orig": round(base["siglip"], 4),
                "siglip_edit": round(sem["siglip"], 4),
                "siglip_gain": round(sem["siglip"] - base["siglip"], 4),
                "lpips_vs_orig": round(
                    float(suite.pairwise(x01, edit)["lpips"]), 4),
            })
            write_csv(args.out / "results.csv", rows)
            print(f"  {tag}  CLIP {base['clip']:.4f} -> {sem['clip']:.4f} "
                  f"({sem['clip'] - base['clip']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
