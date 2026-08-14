"""inpainting 威脅模型的抗淨化 retention，跑在已存的防禦圖上。

與 `phase_retention.py`（img2img）同一個定義：
`retention = effect(淨化) / effect(identity)`，分母塌陷時標 `usable=False`。
兩處差別都來自威脅模型：

1. **攻擊走 `sd.inpaint`**（9 通道權重、無 strength、跑滿 `INPAINT_STEPS`）。
2. **`effect` 只在遮罩內量。** `sd.inpaint` 每一步把遮罩外貼回，故兩條分支
   在遮罩外正好差一個防禦擾動；整張圖算 LPIPS 會把失真算成效果
   （`inpaint_baseline.masked_compare` 的理由）。淨化過的分支同理——淨化
   改動的也包含遮罩外，那同樣不是防禦效果。

不重跑攻擊：只讀 `*__def.png`。

用法：
    python scripts/inpaint_retention.py --run runs/ip5/dog_03 --seeds 3
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

from inpaint_baseline import (  # noqa: E402
    DATA, INPAINT_GUIDANCE, INPAINT_SEED, INPAINT_STEPS, MASKS, MODEL_NAME,
    RESOLUTION, load_dataset, load_mask, masked_compare,
)
from phase_retention import label, purifier_set  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDInpaintWrapper  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--masks", type=Path, default=MASKS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, default=None,
                    help="給定時把淨化後的圖與其重畫結果存下來（純供人眼）")
    args = ap.parse_args()

    cells = []
    for run in args.run:
        with (run / "results.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cells.append({"image": r["image"], "condition": r["condition"],
                              "run": run})
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
    dataset = {d["name"]: d for d in load_dataset(args.data, args.masks)}
    if args.gallery:
        args.gallery.mkdir(parents=True, exist_ok=True)
    orig_cache: dict = {}

    def repaint(x01, item, mask, seed):
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=seed)
        with torch.no_grad():
            return sd.inpaint(x01.clamp(0, 1), mask, emb, noise, INPAINT_STEPS,
                              guidance_scale=INPAINT_GUIDANCE, emb_uncond=emb_u)

    rows = []
    for cell in cells:
        item = dataset[cell["image"]]
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        mask = load_mask(item["mask"], sd.device)
        def_png = cell["run"] / f"{cell['image']}__{cell['condition']}__def.png"
        if not def_png.exists():
            raise FileNotFoundError(f"缺少防禦圖 {def_png}")
        x_def = load_image_tensor(def_png, sd.device, size=RESOLUTION)

        for seed in seeds:
            key = (cell["image"], seed)
            if key not in orig_cache:
                orig_cache[key] = masked_compare(
                    repaint(x01, item, mask, seed), x01, mask)

        print(f"=== {cell['image']} / {cell['condition']} ===", flush=True)
        t0 = time.time()
        effects: dict = {}
        for p in purifiers:
            name = label(p)
            x_pur = p.evaluate(x_def)
            vals = []
            for seed in seeds:
                y = repaint(x_pur, item, mask, seed)
                vals.append(float(suite.pairwise(
                    orig_cache[(cell["image"], seed)],
                    masked_compare(y, x01, mask))["lpips"]))
                if args.gallery and seed == seeds[0]:
                    from src.utils.artifacts import save_image
                    stem = f"{cell['image']}__{cell['condition']}__{name}"
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
                "image": cell["image"], "condition": cell["condition"],
                "purifier": name,
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
