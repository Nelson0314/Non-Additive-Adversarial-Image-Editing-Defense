"""DEC-026 的三個對照組驅動：AdvDrop、BlurGuard，以及可選的針對淨化最佳化。

產出與 `apa_baseline.py` 相容的 `results.csv` 與 `{image}__{condition}__def.png`，
故 `phase_retention.py` 可以直接吃這個目錄跑抗淨化。

三種跑法
────────────────────────────────────────────────────────────────────
`--mode paper`（預設）
    照各自論文的原生設定與更新規則跑（`run_advdrop`／`run_blurguard`）。
    這是「重現該篇」的跑法。

`--mode aligned`
    走本專案共用的 `run_param_pgd` ＋ `encoder_target` 損失 ＋ `fit_to_budget`，
    把最終 DISTS 釘到 `--budget`。這是「與相位臂同預算比較」的跑法。
    **AdvDrop 沒有零點**（量化表壓到下界仍會丟資訊），故目標 DISTS 可能低於
    它的地板；`unreachable` 欄會標出來。

`--purify-aware jpeg`
    只在 `aligned` 模式下有效：把可微分 JPEG 放進最佳化迴圈（DEC-027）。
    課程排程由品質 95 走到 50。**這會改變的是防禦圖，不是評測**——交出去的
    仍是未經淨化的防禦圖。

BlurGuard 的遮罩
────────────────────────────────────────────────────────────────────
需要 SAM。`--sam-ckpt` 未給時該條件直接跳過並在標準輸出寫明，**不會**改用
任何替代分割（見 `src/baselines/blurguard.py` 的模組 docstring）。遮罩會存到
`{out}/masks/{image}/`，重跑時若已存在就直接讀，不重算。

用法
    python scripts/freq_baselines_run.py --out runs/freqbase/g0 \\
        --conditions advdrop --mode paper --edit-strength 0.7 --images cat_00
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402

from apa_baseline import (  # noqa: E402
    EDIT_GUIDANCE, EDIT_SEED, EDIT_STEPS, EDIT_STRENGTH, MODEL_NAME,
    RESOLUTION, head_keep, load_dataset,
)
from src.baselines.advdrop import (  # noqa: E402
    PAPER_Q_MIN, AdvDropParam, AdvDropSpec, run_advdrop,
)
from src.baselines.blurguard import (  # noqa: E402
    SPEC_PAPER as BG_SPEC, BlurGuardParam, check_partition, run_blurguard,
    sam_masks,
)
from src.baselines.diffusionguard import (  # noqa: E402
    SPEC_PAPER as DG_SPEC, make_early_step_loss, run_diffusionguard,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.param_pgd import AdditiveParam, fit_to_budget  # noqa: E402
from src.defense.purify_aware import make_jpeg_transform  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

CONDITIONS = ("advdrop", "blurguard", "diffusionguard")
# aligned 模式的搜尋區間。AdvDrop 的半徑是量化表上界（下界 5），BlurGuard 的
# 是像素 L∞——兩者單位不同，故各有一組。
SEARCH_RANGE = {
    "advdrop": (PAPER_Q_MIN + 1e-3, 60.0),
    "blurguard": (0.0, 0.25),
    "diffusionguard": (0.0, 0.25),      # 像素 L∞，與 blurguard 同一把尺
}


def _load_masks(sd, x01, item, out: Path, sam_ckpt):
    """讀或產生 BlurGuard 的遮罩。已存在就不重算。"""
    d = out / "masks" / item["name"]
    if d.exists():
        files = sorted(d.glob("mask*.pt"), key=lambda p: int(p.stem[4:]))
        masks = {p.stem: torch.load(p, map_location=x01.device) for p in files}
        check_partition(masks)
        return masks
    masks = sam_masks(x01, ckpt=sam_ckpt)
    d.mkdir(parents=True, exist_ok=True)
    for k, v in masks.items():
        torch.save(v.cpu(), d / f"{k}.pt")
    return {k: v.to(x01.device) for k, v in masks.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--mode", choices=("paper", "aligned"), default="paper")
    ap.add_argument("--budget", type=float, default=0.0349,
                    help="aligned 模式的 DISTS 目標。預設是 FND-055 的相位臂值")
    ap.add_argument("--steps", type=int, default=100,
                    help="只作用在 aligned 模式；paper 模式用各篇自己的步數")
    ap.add_argument("--purify-aware", choices=("none", "jpeg"), default="none")
    ap.add_argument("--edit-strength", type=float, default=EDIT_STRENGTH)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--sam-ckpt", type=Path, default=None,
                    help="BlurGuard 的 SAM 檢查點。未給時跳過 blurguard")
    ap.add_argument("--target", type=Path, default=Path("data/targets/gray.png"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.purify_aware != "none" and args.mode != "aligned":
        raise SystemExit("--purify-aware 只在 --mode aligned 下有效")

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    items = load_dataset(args.data)
    if args.images:
        keep = set(args.images)
        items = [d for d in items if d["name"] in keep]

    y_target = (load_image_tensor(args.target, sd.device, size=RESOLUTION)
                if args.mode == "aligned" else None)
    transform = (make_jpeg_transform(args.steps)
                 if args.purify_aware == "jpeg" else None)

    def edit(x01, item, seed):
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=seed)
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        with torch.no_grad():
            return sd.sdedit(x01.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=args.edit_strength,
                             guidance_scale=EDIT_GUIDANCE, emb_uncond=emb_u,
                             keep01=head_keep(item, x01))

    rows = []
    for item in items:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        e_orig = edit(x01, item, EDIT_SEED)

        for cond in args.conditions:
            if cond == "blurguard" and args.sam_ckpt is None:
                print(f"[skip] {item['name']} / blurguard：未給 --sam-ckpt，"
                      "本檔不提供替代分割", flush=True)
                continue
            t0 = time.time()
            masks = (_load_masks(sd, x01, item, args.out, args.sam_ckpt)
                     if cond == "blurguard" else None)

            if args.mode == "paper":
                if cond == "diffusionguard":
                    res = run_diffusionguard(sd, x01, item["prompt"],
                                             args.edit_strength, DG_SPEC,
                                             seed=0, log_every=100)
                    x_def, radius = res.x_def, DG_SPEC.eps_pixel01
                elif cond == "advdrop":
                    spec = AdvDropSpec(
                        modified_from_paper=True,
                        modification_note=("損失由分類交叉熵改為 mean(latent²)"
                                           "——本專案沒有分類器，威脅模型是擴散編輯"))
                    res = run_advdrop(sd, x01, spec, log_every=10)
                    x_def, radius = res.x_def, spec.q_size
                else:
                    res = run_blurguard(sd, x01, masks, BG_SPEC, log_every=25)
                    x_def, radius = res.x_def, BG_SPEC.eps_pixel01
                unreachable = False
            else:
                if cond == "advdrop":
                    param = AdvDropParam()
                elif cond == "blurguard":
                    param = BlurGuardParam(masks)
                else:
                    param = AdditiveParam()
                # DiffusionGuard 的新意在損失，不在參數化：對齊模式下它用
                # 一般的加性 δ ＋ 它自己的早期時間步損失。其餘兩個條件用
                # 本專案共用的 encoder-targeted 損失，以維持「唯一變因是
                # 參數化」。
                al_loss = (make_early_step_loss(sd, item["prompt"],
                                                args.edit_strength, seed=0)
                           if cond == "diffusionguard"
                           else make_encoder_target_loss(sd, y_target))
                lo, hi = SEARCH_RANGE[cond]
                out = fit_to_budget(
                    x01, param, al_loss,
                    lambda a, b: suite.pairwise(b, a)["dists"],
                    target=args.budget, lo=lo, hi=hi, steps=args.steps, seed=0,
                    transform=transform)
                x_def, radius = out.x_def, out.radius
                unreachable = bool(out.history[-1].get("unreachable", False))

            fid = suite.pairwise(x01, x_def)
            e_def = edit(x_def, item, EDIT_SEED)
            eff = float(suite.pairwise(e_orig, e_def)["lpips"])
            tag = f"{cond}_pa" if args.purify_aware != "none" else cond
            vutils.save_image(x_def.clamp(0, 1),
                              args.out / f"{item['name']}__{tag}__def.png")
            vutils.save_image(e_def.clamp(0, 1),
                              args.out / f"{item['name']}__{tag}__edit_def.png")
            vutils.save_image(e_orig.clamp(0, 1),
                              args.out / f"{item['name']}__{tag}__edit_orig.png")
            rows.append({
                "image": item["name"], "condition": tag, "budget_target": "",
                "mode": args.mode, "purify_aware": args.purify_aware,
                "radius": round(radius, 6), "unreachable": unreachable,
                "n_masks": len(masks) if masks else "",
                "fid_lpips": round(fid["lpips"], 5),
                "fid_dists": round(fid["dists"], 5),
                "fid_psnr": round(fid["psnr"], 3),
                "fid_ssim": round(fid["ssim"], 5),
                "fid_linf": round(fid["linf"], 5),
                "fid_rms": round(fid["rms"], 5),
                "edit_lpips": round(eff, 5),
                "edit_strength": args.edit_strength,
                "total_seconds": round(time.time() - t0, 1),
            })
            write_csv(args.out / "results.csv", rows)
            print(f"{item['name']:14s} {tag:14s} r={radius:.4f} "
                  f"lpips={fid['lpips']:.4f} dists={fid['dists']:.4f} "
                  f"effect={eff:.4f} ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("沒有任何條件跑成功——檢查 --conditions 與 --sam-ckpt")
    print(f"\n表：{args.out / 'results.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
