"""編輯結果對指令文字的 CLIP-S：cos(E_img(編輯結果), E_txt(指令))，CLIP 與 SigLIP 各一欄。

影像與文字經 `MetricSuite.semantic_multi`（core `metrics/suite.py`；CLIP＝openai/clip-vit-base-patch32、
SigLIP＝google/siglip-base-patch16-224；影像以 load_image_tensor 讀 512²，再依各模型的處理器縮放與正規化），
與 `measure_semantic_identity` 的 clip_pair、clip_dir 同一條前處理。輸出為未加權的餘弦（CLIPScore 原文另乘 2.5 並截 0，此處不做）。

計畫來源（可混用）：
    --semantic-csv  measure_semantic_identity 的輸出（condition、image、prompt_index、purifier、edit_png）；
                    指令取 --data-root 的 prompts.yaml 之 edits.ip2p[prompt_index]
    --edits-csv     編輯表（arm、image、prompt_index、prompt、png，可含 purifier），指令取 prompt 欄
輸出欄：condition、image、prompt_index、purifier、prompt、clip_s、siglip_s、png、device。

用法（專案根）：
    python scripts/measure_clip_score.py --semantic-csv <CSV>... --data-root data/pexels_portraits --output-csv <CSV> --device cuda
"""
import argparse
import csv
from pathlib import Path

import torch
import yaml

from immunization_core.io import load_image_tensor
from immunization_core.metrics.suite import MetricSuite

RESOLUTION = 512


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semantic-csv", nargs="*", default=[], type=Path)
    ap.add_argument("--edits-csv", nargs="*", default=[], type=Path)
    ap.add_argument("--data-root", type=Path, default=None, help="prompts.yaml 所在的資料集根目錄（--semantic-csv 需要）")
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    if not args.semantic_csv and not args.edits_csv:
        ap.error("需要 --semantic-csv 或 --edits-csv")
    plan = []
    if args.semantic_csv:
        if args.data_root is None:
            ap.error("--semantic-csv 需要 --data-root")
        prompts = yaml.safe_load((args.data_root / "prompts.yaml").read_text(encoding="utf-8"))["edits"]["ip2p"]
        for path in args.semantic_csv:
            for r in read(path):
                plan.append((r["condition"], r["image"], r["prompt_index"], r["purifier"],
                             prompts[int(r["prompt_index"])], r["edit_png"]))
    for path in args.edits_csv:
        for r in read(path):
            plan.append((r["arm"], r["image"], r["prompt_index"], r.get("purifier", "none"), r["prompt"], r["png"]))
    missing = [p for *_, p in plan if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"缺 {len(missing)} 張編輯結果，例如 {missing[:3]}")

    device = torch.device(args.device)
    suite = MetricSuite(device=device)
    rows = []
    for k, (condition, image, index, purifier, prompt, png) in enumerate(plan, 1):
        scores = suite.semantic_multi(load_image_tensor(png, device, size=RESOLUTION), [prompt])[prompt]
        rows.append({"condition": condition, "image": image, "prompt_index": index, "purifier": purifier,
                     "prompt": prompt, "clip_s": round(scores["clip"], 5), "siglip_s": round(scores["siglip"], 5),
                     "png": png, "device": args.device})
        if k % 500 == 0:
            print(f"[PROGRESS] {k}/{len(plan)}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_csv.with_suffix(".csv.incoming")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(args.output_csv)
    print(f"[ALLDONE] {args.output_csv}（{len(rows)} 格）", flush=True)


if __name__ == "__main__":
    main()
