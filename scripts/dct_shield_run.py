"""DCT-Shield 的攻擊與保真度評測 — DEC-025 的頻率 baseline。

產出與 `apa_baseline.py` 相容的 `results.csv` 與 `{image}__{condition}__def.png`，
故 `phase_retention.py` 可以直接吃這個目錄跑抗淨化。

兩種跑法
────────────────────────────────────────────────────────────────────
`--mode paper`（預設）
    完全照論文：`run_dct_shield` 走 Algorithm 1、損失是 `‖E(x')‖₂`、
    步長 `(1−i/N)γ`、1000 步、`ε` 由 `--eps` 給（原生值 1）。
    這是「重現該篇」的跑法。

`--mode aligned`
    走本專案共用的 `run_param_pgd` ＋ `encoder_target` 損失 ＋
    `fit_to_budget`，把最終 DISTS 釘到 `--budget`。**ε 會被搜到 1 以下，
    論文 §4.2 的抗 JPEG 條件因此失效**，該列的 `modified_from_paper` 為真。
    這是「與相位臂同預算比較」的跑法。

兩種跑法的差別必須留在表上，不能混報：前者量的是那篇論文，後者量的是
「頻域加性這個參數化」。
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
from src.baselines.dct_shield import (  # noqa: E402
    PAPER_DEFAULT_QUALITY, PAPER_EPS, PAPER_JPEG_FIG_QUALITY, PAPER_STEPS,
    DCTShieldParam, DCTShieldSpec, DCTShieldYParam, run_dct_shield,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.param_pgd import fit_to_budget  # noqa: E402
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

CONDITIONS = ("dct_shield", "dct_shield_y")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--mode", choices=("paper", "aligned"), default="paper")
    ap.add_argument("--eps", type=float, default=PAPER_EPS)
    ap.add_argument("--steps", type=int, default=PAPER_STEPS)
    ap.add_argument("--budget", type=float, default=0.0349,
                    help="aligned 模式的 DISTS 目標。預設是 FND-055 的相位臂值")
    ap.add_argument("--edit-strength", type=float, default=EDIT_STRENGTH)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--target", type=Path, default=Path("data/targets/gray.png"),
                    help="aligned 模式的 encoder-targeted 目標")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    items = load_dataset(args.data)
    if args.images:
        keep = set(args.images)
        items = [d for d in items if d["name"] in keep]

    y_target = (load_image_tensor(args.target, sd.device, size=RESOLUTION)
                if args.mode == "aligned" else None)

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
            t0 = time.time()
            q = (PAPER_JPEG_FIG_QUALITY if cond.endswith("_y")
                 else PAPER_DEFAULT_QUALITY)
            if args.mode == "paper":
                spec = DCTShieldSpec(
                    name=cond, q_alg=q, eps=args.eps, steps=args.steps,
                    channels=("Y",) if cond.endswith("_y") else ("Y", "Cb", "Cr"),
                    modified_from_paper=args.eps < PAPER_EPS,
                    modification_note=("eps 低於論文的 1，抗 JPEG 條件失效"
                                       if args.eps < PAPER_EPS else ""),
                    source="arXiv:2504.17894 補充材料 Algorithm 1")
                res = run_dct_shield(sd, x01, spec, log_every=250)
                x_def, radius, unreachable = res.x_def, spec.eps, False
            else:
                param = (DCTShieldYParam(q_alg=q) if cond.endswith("_y")
                         else DCTShieldParam(q_alg=q))
                loss_fn = make_encoder_target_loss(sd, y_target)
                out = fit_to_budget(
                    x01, param, loss_fn,
                    lambda a, b: suite.pairwise(b, a)["dists"],
                    target=args.budget, lo=0.0, hi=4.0, steps=100, seed=0)
                x_def, radius = out.x_def, out.radius
                unreachable = bool(out.history[-1].get("unreachable", False))

            fid = suite.pairwise(x01, x_def)
            e_def = edit(x_def, item, EDIT_SEED)
            prot = suite.pairwise(e_orig, e_def)
            eff = float(prot["lpips"])
            vutils.save_image(x_def.clamp(0, 1),
                              args.out / f"{item['name']}__{cond}__def.png")
            vutils.save_image(e_def.clamp(0, 1),
                              args.out / f"{item['name']}__{cond}__edit_def.png")
            vutils.save_image(e_orig.clamp(0, 1),
                              args.out / f"{item['name']}__{cond}__edit_orig.png")
            rows.append({
                "image": item["name"], "condition": cond, "budget_target": "",
                "mode": args.mode, "q_alg": q, "radius": round(radius, 6),
                "steps": args.steps, "unreachable": unreachable,
                # 統一指標清單（DEC-028）：兩個半邊各報五項成對指標。
                # `fid_` 是 fidelity，不是 FID——後者是分布指標，由
                # `scripts/fid_batch.py` 另算成 `frechet` 欄。
                **standard_row("fid_", fid),
                **standard_row("edit_", prot),
                "fid_linf": round(fid["linf"], 5),
                "fid_rms": round(fid["rms"], 5),
                "edit_strength": args.edit_strength,
                "total_seconds": round(time.time() - t0, 1),
            })
            write_csv(args.out / "results.csv", rows)
            print(f"{item['name']:14s} {cond:14s} eps={radius:.4f} "
                  f"lpips={fid['lpips']:.4f} dists={fid['dists']:.4f} "
                  f"effect={eff:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n表：{args.out / 'results.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
