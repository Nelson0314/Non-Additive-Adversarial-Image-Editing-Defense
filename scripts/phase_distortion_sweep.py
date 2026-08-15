"""失真掃描：把 紋理重相位 與加性各自的半徑掃過一整排，供人眼定門檻。

規格：`docs/superpowers/specs/2026-08-13-texture-rephasing-design.md` §4 的
(a)(c) 兩項處置。2026-08-13 的 像素臂顯示 **DISTS 與 LPIPS 都低估了相位擾動的
可見失真**，而 PSNR 與人眼一致；因此在定約束之前必須先由人眼在一整排半徑上
劃線，不能沿用既有的預算軸。

本腳本**不做編輯評測**，只產生防禦圖與失真指標——門檻審查問的是「這張圖看
起來壞了沒有」，與防禦強度無關，混在一起跑只會讓它變慢。

指標分三組（全部照報，不挑選）：

| 組 | 項目 | 為什麼在這裡 |
|---|---|---|
| 既有 | psnr／linf／ssim／vif_p／fsim／lpips／dists／acutance_ratio／rms／frac_gt_16_255 | `MetricSuite.pairwise`，與既有 runs/ 可比 |
| 新增 | haarpsi／gmsd／ms_ssim／mdsi | 對**結構破壞**敏感，正是 LPIPS／DISTS 漏掉的那一類 |
| 無參考 | niqe／brisque | 「看起來自不自然」，不需要原圖 |

新增那四項寫在本腳本而非 `MetricSuite.pairwise`：後者的欄位集合是既有全部
批次的 CSV schema，加欄會讓舊批與新批的欄位不一致而沒有症狀。

用法：
    python scripts/phase_distortion_sweep.py --out runs/phase_sweep \
        --data data/lo_aligned
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

from apa_baseline import MODEL_NAME, RESOLUTION, TARGET_IMAGE, load_dataset  # noqa: E402
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.param_pgd import AdditiveParam, PhaseParam, run_param_pgd  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

# 相位的上界是 π（週期量，構造上的天花板）。取幾何間距而非等距：可見失真
# 對半徑是加速上升的，等距會把大半的格子浪費在看不出差別的低端。
THETA_GRID = (0.15, 0.25, 0.40, 0.60, 0.90, 1.30, 1.80, 2.40, math.pi)
# 加性的 ε（[0,1] 尺度）。文獻常用點 8/255 與 16/255 都落在格上。
EPS_GRID = tuple(v / 255.0 for v in (0.5, 1, 2, 3, 5, 8, 12, 18, 26))


def extra_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """對結構破壞敏感的四項。a 是原圖、b 是待評影像。

    HaarPSI 與 MDSI 值域為「愈高愈好」，GMSD 為「愈低愈好」，MS-SSIM 愈高愈好。
    不做正規化或翻轉——翻轉之後就不是那篇論文報的那個數了。
    """
    import piq

    a = a.float().clamp(0, 1)
    b = b.float().clamp(0, 1)
    return {
        "haarpsi": float(piq.haarpsi(a, b, data_range=1.0)),
        "gmsd": float(piq.gmsd(a, b, data_range=1.0)),
        "ms_ssim": float(piq.multi_scale_ssim(a, b, data_range=1.0)),
        "mdsi": float(piq.mdsi(a, b, data_range=1.0)),
    }


def no_reference(x: torch.Tensor, suite: MetricSuite) -> dict:
    import piq

    return {
        "niqe": round(suite.niqe(x), 4),
        "brisque": round(float(piq.brisque(x.float().clamp(0, 1), data_range=1.0)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--images", nargs="+",
                    default=["bird_03", "cat_02", "dog_03",
                             "horse_00", "man_00", "woman_03"])
    ap.add_argument("--conditions", nargs="+", default=["phase", "add"])
    ap.add_argument("--target", type=Path, default=Path(TARGET_IMAGE))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    loss_fn = make_encoder_target_loss(
        sd, load_image_tensor(args.target, sd.device, size=RESOLUTION))

    dataset = [d for d in load_dataset(args.data) if d["name"] in args.images]
    rows = []
    for item in dataset:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(x01, args.out / f"{item['name']}__orig.png")
        base_nr = no_reference(x01, suite)
        print(f"\n########## {item['name']} "
              f"（原圖 niqe={base_nr['niqe']} brisque={base_nr['brisque']}）"
              f" ##########", flush=True)

        for cond in args.conditions:
            grid = THETA_GRID if cond == "phase" else EPS_GRID
            for radius in grid:
                t0 = time.time()
                param = (PhaseParam(size=RESOLUTION, radius=radius) if cond == "phase"
                         else AdditiveParam(radius=radius))
                res = run_param_pgd(x01, param, loss_fn,
                                    steps=args.steps, seed=args.seed)
                x_def = res.x_def.clamp(0, 1)
                tag = f"{cond}__r{radius:.4f}"
                save_image(x_def, args.out / f"{item['name']}__{tag}__def.png")

                row = {"image": item["name"], "condition": cond,
                       "radius": round(radius, 5),
                       "seconds": round(time.time() - t0, 1)}
                row.update({k: round(v, 5) for k, v in suite.pairwise(x01, x_def).items()})
                row.update({k: round(v, 5) for k, v in extra_metrics(x01, x_def).items()})
                row.update(no_reference(x_def, suite))
                row["niqe_orig"] = base_nr["niqe"]
                row["brisque_orig"] = base_nr["brisque"]
                if cond == "phase":
                    row["amp_dev"] = round(param.module.amplitude_deviation(x01), 5)
                    row["active_fraction"] = round(param.module.active_fraction(), 4)
                rows.append(row)
                print(f"  {cond:6s} r={radius:.4f}  dists={row['dists']:.4f} "
                      f"lpips={row['lpips']:.4f} psnr={row['psnr']:.2f} "
                      f"haarpsi={row['haarpsi']:.4f} gmsd={row['gmsd']:.4f} "
                      f"niqe={row['niqe']:.2f}", flush=True)
                write_csv(args.out / "sweep.csv", rows)

    print(f"\n表：{args.out / 'sweep.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
