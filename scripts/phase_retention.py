"""主讀數：抗淨化的衰減率（retention），跑在 A 臂已存的防禦圖上。

規格 §4。`retention = effect(淨化) / effect(identity)`，`effect` 取
`LPIPS(未防禦的編輯, 防禦後的編輯)`——與 `apa_baseline.evaluate` 的
`edit_lpips` 同一個量，只是換了輸入的淨化算子。

**分母塌陷時不可解讀**（`METRICS.md` §6）：`effect(identity)` 的多 seed 平均
低於三倍標準差時，該列標 `usable=False` 並排除在任何統計之外。這不是保守，
是 FND-009 記過的事——分母本身在雜訊裡時，比值沒有意義。

不重跑攻擊：A 臂已把 `*__def.png` 存下來，本腳本只讀圖。故換淨化算子集合
或加 seed 都不需要重新訓練。

用法：
    python scripts/phase_retention.py --run runs/phaseA_full --seeds 3
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

from apa_baseline import (  # noqa: E402
    EDIT_GUIDANCE, EDIT_SEED, EDIT_STEPS, EDIT_STRENGTH, MODEL_NAME,
    RESOLUTION, load_dataset,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.purify import ops as purify_ops  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

# 七個既有算子 ＋ C&R 串接。強度沿用 `main_set` 與 `eval_sweep` 的既有值，
# 不另立新數字——換了強度就不能與既有的 runs/ 比。
def purifier_set(sd, seed: int):
    cands = [
        purify_ops.Purifier("identity"),
        purify_ops.Purifier("blur", 1.0),
        purify_ops.Purifier("noise", 0.05, seed=seed),
        purify_ops.Purifier("quantize", 16),
        purify_ops.Purifier("jpeg", 75),
        purify_ops.Purifier("jpeg", 30),
        purify_ops.Purifier("crop_resize", purify_ops.CROP_FRACTION_DIA),
        purify_ops.Purifier("jpeg_then_resize", purify_ops.CR_JPEG_QUALITY),
        purify_ops.Purifier("adverse_cleaner"),
        purify_ops.Purifier("impress", sd=sd, seed=seed),
    ]
    kept, skipped = [], []
    for p in cands:
        (kept if p.available else skipped).append(p)
    if skipped:
        print(f"[purify] 相依不齊，跳過：{[p.kind for p in skipped]}", flush=True)
    return kept


def label(p) -> str:
    return p.kind if not p.strength else f"{p.kind}{p.strength:g}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run / "retention.csv")

    with (args.run / "results.csv").open(encoding="utf-8") as fh:
        cells = [
            {"image": r["image"], "condition": r["condition"],
             "budget": r["budget_target"]}
            for r in csv.DictReader(fh)
        ]
    if not cells:
        raise SystemExit(f"{args.run / 'results.csv'} 沒有任何列")

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    purifiers = purifier_set(sd, seed=0)
    seeds = [EDIT_SEED + k for k in range(args.seeds)]

    dataset = {d["name"]: d for d in load_dataset(args.data)}
    edit_orig_cache: dict = {}

    def edit(x01, item, seed):
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=seed)
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        with torch.no_grad():
            return sd.sdedit(x01.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=EDIT_STRENGTH, guidance_scale=EDIT_GUIDANCE,
                             emb_uncond=emb_u)

    rows = []
    for cell in cells:
        item = dataset[cell["image"]]
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        tag = f"{cell['condition']}__d{float(cell['budget']):g}"
        def_png = args.run / f"{cell['image']}__{tag}__def.png"
        if not def_png.exists():
            raise FileNotFoundError(f"缺少防禦圖 {def_png}")
        x_def = load_image_tensor(def_png, sd.device, size=RESOLUTION)

        for seed in seeds:
            key = (cell["image"], seed)
            if key not in edit_orig_cache:
                edit_orig_cache[key] = edit(x01, item, seed)

        print(f"=== {cell['image']} / {tag} ===", flush=True)
        t0 = time.time()
        effects: dict = {}
        for p in purifiers:
            name = label(p)
            vals = []
            x_pur = p.evaluate(x_def)
            for seed in seeds:
                e = edit(x_pur, item, seed)
                vals.append(float(
                    suite.pairwise(edit_orig_cache[(cell["image"], seed)], e)["lpips"]))
            effects[name] = vals

        base = effects["identity"]
        base_mean = statistics.fmean(base)
        base_sd = statistics.stdev(base) if len(base) > 1 else float("nan")
        usable = bool(base_sd == base_sd and base_mean >= 3.0 * base_sd)
        if not usable:
            print(f"    分母塌陷：effect(identity) mean={base_mean:.4f} "
                  f"sd={base_sd:.4f}，本格的 retention 不可用", flush=True)

        for name, vals in effects.items():
            mean = statistics.fmean(vals)
            rows.append({
                "image": cell["image"], "condition": cell["condition"],
                "budget_target": cell["budget"], "purifier": name,
                "effect_mean": round(mean, 5),
                "effect_sd": round(statistics.stdev(vals), 5) if len(vals) > 1 else "",
                "effect_identity_mean": round(base_mean, 5),
                "effect_identity_sd": round(base_sd, 5) if base_sd == base_sd else "",
                "retention": round(mean / base_mean, 5) if base_mean > 0 else "",
                "usable": usable,
                "seconds": round(time.time() - t0, 1),
            })
        write_csv(out, rows)
        print(f"    {', '.join(f'{k}={statistics.fmean(v):.4f}' for k, v in effects.items())}",
              flush=True)

    print(f"\n表：{out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
