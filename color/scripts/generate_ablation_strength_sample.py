"""消融降強度實驗（教授 11/25 意見第 1 點延伸）的影像樣本：crosseditor100 抽 30 張。

來源：baseline runtime/pexels_portraits/crosseditor100/images.txt（100 張）。影像 ID 以字串排序後，
以 numpy.random.default_rng(20260812).choice(100, 30, replace=False) 取索引，依抽出順序寫出。
輸出：--output（一行一個影像 ID）。來源不是 100 張即失敗。
用法（遠端）：python color/scripts/generate_ablation_strength_sample.py --images <images.txt> --output <txt>
"""
import argparse
from pathlib import Path

import numpy as np

SEED = 20260812
N_SAMPLE = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    images = sorted(args.images.read_text(encoding="utf-8").split())
    if len(images) != 100 or len(set(images)) != 100:
        raise SystemExit(f"{args.images} 應為 100 張不重複影像，實為 {len(images)} 張")
    idx = np.random.default_rng(SEED).choice(len(images), N_SAMPLE, replace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{images[i]}\n" for i in idx), encoding="utf-8", newline="\n")
    print(f"[SAMPLE] {N_SAMPLE} 張 → {args.output}")


if __name__ == "__main__":
    main()
