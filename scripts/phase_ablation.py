"""A 臂：參數化消融——加性 δ 對紋理重相位 θ 對隨機相位，同一個損失。

規格：`docs/superpowers/specs/2026-08-13-texture-rephasing-design.md` §3。

損失、更新規則、步數、種子、預算對齊程序全部固定，**唯一的變因是參數化**：

    add          φ = δ，L∞ 投影
    phase        φ = θ，site F（紋理重相位）
    phase_rand   同幅度的隨機 θ，即 RPN 本身，不最佳化

每個條件對半徑二分搜尋，使最終的 DISTS 落在目標上，故三者在同一個預算軸
的同一點上比較。相位有構造上的天花板（|θ| ≤ π），達不到的預算點標成
`unreachable` 而非 failed（與 FND-001 同型）。

評測直接沿用 `apa_baseline.evaluate`，不另寫一份——兩份評測會慢慢分岔而
沒有症狀，既有 runs/ 會靜默變得不可比。

用法：
    python scripts/phase_ablation.py --out runs/phaseA --data data/lo_aligned \
        --images horse_00 man_00 bird_03
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import (  # noqa: E402
    MODEL_NAME, RESOLUTION, TARGET_IMAGE, evaluate, load_dataset,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.param_pgd import (  # noqa: E402
    AdditiveParam, PhaseParam, RandomPhaseParam, fit_to_budget,
)
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

CONDITIONS = ("add", "phase", "phase_rand")

# 兩個預算點。0.075 是現行非加性條件的水位、0.04 是加性的（FND-026 的表）。
# 單點容易是巧合，故取兩點看趨勢。
BUDGETS = (0.04, 0.075)

# 加性的半徑上界取 32/255：Mist 在 [0,1] 上的等價 eps 是 16/255，留一倍
# 餘裕讓二分搜尋在上界內找得到 DISTS 0.075。
ADD_RADIUS_HI = 32.0 / 255.0
ADD_RADIUS_LO = 0.5 / 255.0
PHASE_RADIUS_LO = 0.05


def build(name: str, seed: int):
    if name == "add":
        return AdditiveParam(radius=ADD_RADIUS_HI), ADD_RADIUS_LO, ADD_RADIUS_HI
    if name == "phase":
        return PhaseParam(size=RESOLUTION), PHASE_RADIUS_LO, math.pi
    if name == "phase_rand":
        return RandomPhaseParam(size=RESOLUTION), PHASE_RADIUS_LO, math.pi
    raise ValueError(f"未知條件 {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS))
    ap.add_argument("--target", type=Path, default=Path(TARGET_IMAGE))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    y_target = load_image_tensor(args.target, sd.device, size=RESOLUTION)
    loss_fn = make_encoder_target_loss(sd, y_target)

    def dists_of(a, b):
        return float(suite.pairwise(a.clamp(0, 1), b)["dists"])

    dataset = load_dataset(args.data)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]

    rows = []
    for item in dataset:
        item["path01"] = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)

        for budget in args.budgets:
            for cond in args.conditions:
                tag = f"{cond}__d{budget:g}"
                print(f"=== {item['name']} / {tag} ===", flush=True)
                t0 = time.time()
                param, lo, hi = build(cond, args.seed)
                res = fit_to_budget(
                    item["path01"], param, loss_fn, dists_of, budget,
                    lo=lo, hi=hi, steps=args.steps, seed=args.seed,
                    rounds=args.rounds,
                )
                fit = res.history[-1]
                metrics, eo, ed = evaluate(sd, suite, aes, item, res.x_def)
                for sub, img in (("def", res.x_def), ("edit_orig", eo),
                                 ("edit_def", ed)):
                    save_image(img, args.out / f"{item['name']}__{tag}__{sub}.png")

                row = {
                    "image": item["name"], "condition": cond,
                    "budget_target": budget,
                    "budget_reached": round(float(fit["reached"]), 5),
                    "unreachable": bool(fit["unreachable"]),
                    "radius": round(float(fit["radius"]), 5),
                    "total_seconds": round(time.time() - t0, 1),
                    **metrics,
                }
                if cond in ("phase", "phase_rand"):
                    row["amp_dev"] = round(
                        param.module.amplitude_deviation(item["path01"]), 5)
                    row["active_fraction"] = round(param.module.active_fraction(), 4)
                rows.append(row)
                print(row, flush=True)
                write_csv(args.out / "results.csv", rows)

    print(f"\n表：{args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
