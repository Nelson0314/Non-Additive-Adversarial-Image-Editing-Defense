"""inpainting 的抗淨化 retention，跑在 `inpaint_edit.py` 的輸出上。

定義與 `phase_retention.py`（img2img）相同：
`retention = effect(淨化) / effect(identity)`，分母塌陷時標 `usable=False`。
三處差別都來自威脅模型：

1. **攻擊走 `sd.inpaint`**（9 通道權重、無 strength、跑滿 `INPAINT_STEPS`）。
2. **`effect` 只在遮罩內量。** `sd.inpaint` 每一步把遮罩外貼回，兩條分支在
   該處正好差一個防禦擾動；淨化過的分支同理。整張圖算 LPIPS 會把失真與
   淨化的痕跡都算成防禦效果。
3. **場景與 prompt 由 run 目錄決定。** `inpaint_edit.py` 的 `results.csv`
   已記下每一列的 `scenario` 與 `prompt`，遮罩存在同一個目錄，故此處不重新
   查表——第一版的協定錯誤正是出在兩邊各自決定 prompt。

防禦圖仍取自 img2img 批次（防禦加在整張圖上，遮罩只在編輯時進來），
故 `--defended` 指向 `runs/hb5`。

用法：
    python scripts/inpaint_retention.py --run runs/ip2/background \
        --defended runs/hb5 runs/hb5_pgc --out runs/ip2/retention_bg.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from inpaint_edit import (  # noqa: E402
    INPAINT_GUIDANCE, INPAINT_SEED, INPAINT_STEPS, MODEL_NAME, RESOLUTION,
    find_def, load_mask, masked_compare,
)
from phase_retention import label, purifier_set  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDInpaintWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="inpaint_edit.py 的輸出目錄（含 results.csv 與遮罩）")
    ap.add_argument("--defended", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--gallery", type=Path, default=None)
    args = ap.parse_args()

    with (args.run / "results.csv").open(encoding="utf-8") as fh:
        cells = [{"image": r["image"], "condition": r["condition"],
                  "prompt": r["prompt"], "scenario": r["scenario"]}
                 for r in csv.DictReader(fh)]
    if args.images:
        keep = set(args.images)
        cells = [c for c in cells if c["image"] in keep]
    if args.conditions:
        keep = set(args.conditions)
        cells = [c for c in cells if c["condition"] in keep]
    if not cells:
        raise SystemExit("沒有符合條件的格")

    sd = SDInpaintWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    purifiers = purifier_set(sd, seed=0)
    seeds = [INPAINT_SEED + k for k in range(args.seeds)]
    if args.gallery:
        args.gallery.mkdir(parents=True, exist_ok=True)
    orig_cache: dict = {}

    def repaint(x01, prompt, mask, seed):
        emb, emb_u = sd.encode_text(prompt), sd.uncond_prompt()
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=seed)
        with torch.no_grad():
            return sd.inpaint(x01.clamp(0, 1), mask, emb, noise, INPAINT_STEPS,
                              guidance_scale=INPAINT_GUIDANCE, emb_uncond=emb_u)

    rows = []
    for cell in cells:
        img = cell["image"]
        x01 = load_image_tensor(args.run / f"{img}__orig.png", sd.device,
                                size=RESOLUTION)
        mask = load_mask(args.run / f"{img}__mask.png", sd.device)
        p = find_def(args.defended, img, cell["condition"])
        if p is None:
            raise FileNotFoundError(f"{img} / {cell['condition']} 缺防禦圖")
        x_def = load_image_tensor(p, sd.device, size=RESOLUTION)

        for seed in seeds:
            key = (img, seed)
            if key not in orig_cache:
                orig_cache[key] = masked_compare(
                    repaint(x01, cell["prompt"], mask, seed), x01, mask)

        print(f"=== {img} / {cell['condition']} / {cell['scenario']} ===", flush=True)
        t0 = time.time()
        effects: dict = {}
        for pur in purifiers:
            name = label(pur)
            x_pur = pur.evaluate(x_def)
            vals = []
            for seed in seeds:
                y = repaint(x_pur, cell["prompt"], mask, seed)
                vals.append(float(suite.pairwise(
                    orig_cache[(img, seed)],
                    masked_compare(y, x01, mask))["lpips"]))
                if args.gallery and seed == seeds[0]:
                    stem = f"{img}__{cell['condition']}__{name}"
                    save_image(x_pur.clamp(0, 1), args.gallery / f"{stem}__pur.png")
                    save_image(y, args.gallery / f"{stem}__gen.png")
            effects[name] = vals

        base = effects["identity"]
        bm = statistics.fmean(base)
        bs = statistics.stdev(base) if len(base) > 1 else float("nan")
        usable = bool(bs == bs and bm >= 3.0 * bs)
        if not usable:
            print(f"    分母塌陷：mean={bm:.4f} sd={bs:.4f}", flush=True)
        for name, vals in effects.items():
            rows.append({
                "image": img, "condition": cell["condition"],
                "scenario": cell["scenario"], "purifier": name,
                "effect_mean": round(statistics.fmean(vals), 5),
                "effect_sd": round(statistics.stdev(vals), 5) if len(vals) > 1 else "",
                "effect_identity_mean": round(bm, 5),
                "effect_identity_sd": round(bs, 5) if bs == bs else "",
                "retention": round(statistics.fmean(vals) / bm, 5) if bm > 0 else "",
                "usable": usable,
                "seconds": round(time.time() - t0, 1),
            })
        write_csv(args.out, rows)
        print("    " + ", ".join(f"{k}={statistics.fmean(v):.4f}"
                                 for k, v in effects.items()), flush=True)

    print(f"\n表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
