"""替淨化圖庫的每一張編輯圖算語意分數（CLIP-T 與 SigLIP-T）。

`phase_retention.py` 只量位移量（LPIPS），語意那一軸只在未淨化的評測裡有
（`results.csv` 的 `edit_clip_*`／`edit_siglip_*`）。本腳本補上淨化之後的：
對 `hb5_purify_gallery.py` 產生的每一張 `*__edit.png` 算它對該影像 prompt 的
CLIP 與 SigLIP 相似度。

**不跑任何擴散**——只有兩個文字-影像模型的前向，故成本是分鐘量級。
輸出一張長表：image, condition, purifier, clip, siglip。

未淨化那一格（`purifier == identity`）與 `results.csv` 的 `edit_clip_def`
不會逐位相同：圖庫只跑 `EDIT_SEED` 一個種子，而 identity 的防禦圖是從
PNG 讀回來的，差一次 8 位元量化。兩者的用途不同，不要互相取代。

    python scripts/semantic_scores.py --gallery runs/s0817/purified \\
        --data data/set0817 --out runs/s0817/semantic.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import RESOLUTION, load_dataset  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

STEM = re.compile(r"^(?P<image>.+?_\d+)__(?P<cond>.+?)__(?P<pur>.+?)__edit\.png$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt-index", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.gallery.glob("*__edit.png"))
    if not files:
        raise FileNotFoundError(f"{args.gallery} 底下沒有 *__edit.png")

    dataset = {d["name"]: d for d in load_dataset(args.data,
                                                 prompt_index=args.prompt_index)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suite = MetricSuite(device=device)

    # 原圖本身對 prompt 的分數：正規化用的基準線，每張影像只算一次。
    base = {}
    for name, item in dataset.items():
        x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
        s = suite.semantic(x01, item["prompt"])
        base[name] = (s["clip"], s["siglip"])
        print(f"{name:14s} 原圖 CLIP {s['clip']:.4f}  SigLIP {s['siglip']:.4f}",
              flush=True)

    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        m = STEM.match(f.name)
        if m is None:
            raise ValueError(f"檔名不符合預期的三段式：{f.name}")
        image, cond, pur = m["image"], m["cond"], m["pur"]
        if image not in dataset:
            raise KeyError(f"資料集裡沒有 {image}")
        x = load_image_tensor(f, device, size=RESOLUTION)
        s = suite.semantic(x, dataset[image]["prompt"])
        c0, s0 = base[image]
        rows.append({
            "image": image, "condition": cond, "purifier": pur,
            "clip": round(s["clip"], 5), "siglip": round(s["siglip"], 5),
            "clip_orig_image": round(c0, 5), "siglip_orig_image": round(s0, 5),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    write_csv(args.out, rows)
    print(f"\n{len(rows)} 列 → {args.out}（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
