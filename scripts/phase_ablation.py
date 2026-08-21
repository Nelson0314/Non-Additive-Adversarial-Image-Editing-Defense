"""像素臂：參數化消融——加性 δ 對紋理重相位 θ 對隨機相位，同一個損失。

規格：`docs/superpowers/specs/2026-08-13-texture-rephasing-design.md` §3。

損失、更新規則、步數、種子、預算對齊程序全部固定，**唯一的變因是參數化**：

    add          φ = δ，L∞ 投影
    phase        φ = θ，紋理重相位
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
    EDIT_STRENGTH, MODEL_NAME, RESOLUTION, TARGET_IMAGE, evaluate, load_dataset,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.param_pgd import (  # noqa: E402
    AdditiveParam, PhaseParam, RandomPhaseParam, fit_to_budget, run_param_pgd,
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

# 人眼門檻（使用者 2026-08-13 於失真掃描頁上劃定）。給 --human-threshold 時
# 不做預算對齊，直接用這兩個半徑——**每個條件各自在自己的可接受上限上**，
# 這才是「匹配人眼可辨失真」的字面意思。以任何單一指標對齊都做不到這件事：
# 兩個門檻在十六項指標上沒有一項落在同一個值（最接近的 LPIPS 也差 1.28 倍）。
HUMAN_RADIUS = {"phase": 1.30, "phase_rand": 1.30, "add": 1.2 / 255.0}

# 加性的半徑上界取 32/255：Mist 在 [0,1] 上的等價 eps 是 16/255，留一倍
# 餘裕讓二分搜尋在上界內找得到 DISTS 0.075。
ADD_RADIUS_HI = 32.0 / 255.0
ADD_RADIUS_LO = 0.5 / 255.0
PHASE_RADIUS_LO = 0.05


def build(name: str, seed: int, block: int = 32, r_min: float = 0.12,
          quantile: float = 0.5, gl_iters: int = 0, pixel_gate_sigma: float = 0.0,
          gain_ratio: float = 0.0, r_max: float = float("inf")):
    """`block`／`r_min`／`quantile` 是相位算子的三個構造設定。

    預設值是 現行定案（`docs/METHOD.md` §4）。開放成參數是為了掃描
    「約束落在哪個頻帶、哪些區塊」對效果與失真的取捨——三者都改變**閘**，
    也就是改變擾動被允許出現的位置，不改變損失或更新規則。
    """
    if name == "add":
        return AdditiveParam(radius=ADD_RADIUS_HI), ADD_RADIUS_LO, ADD_RADIUS_HI
    if name == "phase":
        return (PhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                           r_max=r_max,
                           energy_quantile=quantile, gl_iters=gl_iters,
                           pixel_gate_sigma=pixel_gate_sigma),
                PHASE_RADIUS_LO, math.pi)
    if name in ("phase_gain", "gain_only"):
        # 2026-08-21 的改動一：幅度譜也可學。`gain_only` 把 theta 凍結在 0，
        # 用來分辨「幅度單獨有沒有用」與「兩者是否相加」。
        #
        # **上界不是 pi**：相位是週期量所以封頂在 pi，增益不是，這正是加它的
        # 主要理由。上界取 8.0 是本專案指定的掃描上限，沒有出處——超過那裡
        # exp(8) ~ 3000 倍，影像早就毀了，掃上去只是浪費機時。
        if gain_ratio <= 0:
            raise ValueError(f"{name} 需要 gain_ratio > 0，收到 {gain_ratio}")
        return (PhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                           r_max=r_max,
                           energy_quantile=quantile, gl_iters=gl_iters,
                           pixel_gate_sigma=pixel_gate_sigma,
                           gain_ratio=gain_ratio,
                           phase_on=(name == "phase_gain")),
                PHASE_RADIUS_LO, 8.0)
    if name == "phase_rand":
        return (RandomPhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                                 r_max=r_max,
                                 energy_quantile=quantile, gl_iters=gl_iters),
                PHASE_RADIUS_LO, math.pi)
    raise ValueError(f"未知條件 {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS))
    ap.add_argument("--human-threshold", action="store_true",
                    help="不做預算對齊，直接用 HUMAN_RADIUS 的人眼門檻半徑")
    ap.add_argument("--target", type=Path, default=Path(TARGET_IMAGE))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="用 prompts.yaml 的第幾個編輯 prompt（0 改內容、1 改場景）")
    ap.add_argument("--block", type=int, default=32, help="重疊區塊邊長")
    ap.add_argument("--r-min", type=float, default=0.12, help="徑向頻率閘的下限")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="紋理閘的梯度能量參考分位數")
    ap.add_argument("--gl-iters", type=int, default=0,
                    help="Griffin-Lim 迭代投影的輪數。>0 時把 STFT 一致性投影"
                         "誤差壓下去，用來判別效果來自相位重排還是新造的能量"
                         "（FND-040／049）。0 = 關閉，與既有批次逐位相同")
    ap.add_argument("--phase-radius", type=float, default=None,
                    help="覆寫人眼門檻的相位半徑（只在 --human-threshold 下有效）")
    ap.add_argument("--add-radius", type=float, default=None,
                    help="覆寫人眼門檻的加性半徑 eps，單位是 [0,1]（只在 "
                         "--human-threshold 下有效）。人眼門檻是 1.2/255 = 0.0047")
    ap.add_argument("--edit-strength", type=float, default=EDIT_STRENGTH,
                    help="SDEdit 的 strength。像素臂的攻擊與強度無關，這個旗標"
                         "只影響評測，供強度掃描使用")
    ap.add_argument("--purify-aware", choices=("none", "jpeg"), default="none",
                    help="DEC-027：把可微分 JPEG 放進最佳化迴圈的前向，讓擾動"
                         "自己找壓縮活得下來的位置。品質沿 95→50 的課程排程。"
                         "**改變的是防禦圖，不是評測**——交出去的仍是未經淨化的"
                         "防禦圖，且條件標籤會加上 `_pa` 以免與既有批次混淆")
    ap.add_argument("--tag-suffix", type=str, default="",
                    help="附加在條件標籤後，讓同一個 --out 下的多組設定不互相覆寫檔名")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    y_target = load_image_tensor(args.target, sd.device, size=RESOLUTION)
    loss_fn = make_encoder_target_loss(sd, y_target)

    def dists_of(a, b):
        return float(suite.pairwise(a.clamp(0, 1), b)["dists"])

    transform = None
    if args.purify_aware == "jpeg":
        from src.defense.purify_aware import make_jpeg_transform

        transform = make_jpeg_transform(args.steps)
        args.tag_suffix = args.tag_suffix + "_pa"
        print(f"[purify-aware] JPEG 課程排程進入最佳化迴圈，"
              f"標籤加上 _pa（{args.steps} 步）", flush=True)

    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]

    rows = []
    for item in dataset:
        item["path01"] = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)

        budgets = ["human"] if args.human_threshold else args.budgets
        for budget in budgets:
            for cond in args.conditions:
                tag = (f"{cond}__human" if budget == "human"
                       else f"{cond}__d{budget:g}") + args.tag_suffix
                print(f"=== {item['name']} / {tag} ===", flush=True)
                t0 = time.time()
                param, lo, hi = build(cond, args.seed, block=args.block,
                                      r_min=args.r_min, quantile=args.quantile,
                                      gl_iters=args.gl_iters)
                if budget == "human":
                    r_human = HUMAN_RADIUS[cond]
                    if args.phase_radius is not None and cond in ("phase", "phase_rand"):
                        r_human = args.phase_radius
                    if args.add_radius is not None and cond == "add":
                        r_human = args.add_radius
                    param.set_radius(r_human)
                    res = run_param_pgd(item["path01"], param, loss_fn,
                                        steps=args.steps, seed=args.seed,
                                        transform=transform)
                    fit = {"unreachable": False, "target": r_human,
                           "reached": dists_of(res.x_def, item["path01"]),
                           "radius": param.radius}
                else:
                    res = fit_to_budget(
                        item["path01"], param, loss_fn, dists_of, budget,
                        lo=lo, hi=hi, steps=args.steps, seed=args.seed,
                        rounds=args.rounds, transform=transform,
                    )
                    fit = res.history[-1]
                metrics, eo, ed = evaluate(sd, suite, aes, item, res.x_def,
                                           strength=args.edit_strength)
                for sub, img in (("def", res.x_def), ("edit_orig", eo),
                                 ("edit_def", ed)):
                    save_image(img, args.out / f"{item['name']}__{tag}__{sub}.png")

                row = {
                    "image": item["name"], "condition": cond,
                    "prompt_index": args.prompt_index, "prompt": item["prompt"],
                    "block": args.block, "r_min": args.r_min,
                    "quantile": args.quantile, "target_image": str(args.target),
                    "budget_target": budget,
                    "budget_mode": "human" if budget == "human" else "dists",
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
