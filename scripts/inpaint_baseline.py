"""inpainting 威脅模型：site F 與四個加性方法的抗編輯評測。

與 `apa_baseline.py`（img2img）的差別全部來自威脅模型，不是實作偏好：

1. **權重是 9 通道的 `runwayml/stable-diffusion-inpainting`**，
   PhotoGuard-c／AdvPaint／PromptFlare 三篇的原生設定。它們在 img2img 下
   標著 `modified_from_paper=True`，換到這裡就不用移植了。
2. **沒有 strength。** 由純噪聲起跑、跑滿 `INPAINT_STEPS`。
3. **遮罩外的擾動才活得下來。** `SDWrapper.mask_latents` 算的是
   `encode(x01 * (1 - mask))`，落在重畫區的擾動在進入 UNet 之前就被歸零。
   故本腳本提供 `_out` 變體，把擾動的容量集中到 `keep = 1 - mask`。

`effect` 只在遮罩內量（見 `masked_compare`）
────────────────────────────────────────────────────────────────────
`sd.inpaint` 每一步把遮罩外貼回，故未防禦分支貼回 `x`、防禦分支貼回
`x_def`——**兩者在遮罩外正好差一個防禦擾動 δ**。直接對整張圖算 LPIPS 會把
δ 自己算成「防禦效果」，那是把失真當成效果報。做法是兩條分支的遮罩外都
換成同一張原圖，於是比較只落在生成內容上。

用法：
    python scripts/inpaint_baseline.py --out runs/ip_pre --images dog_03 \
        --conditions phase add
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from apa_baseline import TARGET_IMAGE  # noqa: E402
from src.baselines import advpaint, photoguard, promptflare  # noqa: E402
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.baselines.pgd import run_pgd  # noqa: E402
from src.defense.param_pgd import (  # noqa: E402
    AdditiveParam, PhaseParam, RandomPhaseParam, run_param_pgd,
)
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDInpaintWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

MODEL_NAME = "runwayml/stable-diffusion-inpainting"
RESOLUTION = 512
DATA = Path("data/lo_inpaint")
MASKS = Path("data/lo_masks")

# 攻擊方的設定。inpainting 沒有 strength，跑滿 50 步是 pipeline 的預設。
INPAINT_STEPS = 50
INPAINT_GUIDANCE = 7.5
INPAINT_SEED = 20260814

# 人眼門檻（FND-035）。這兩個半徑是在 img2img 的失真掃描上劃定的，而失真是
# 算子對影像做的事，與下游模型無關，故直接沿用。**但三張圖經過重裁與上採樣**
# （`data/lo_masks/recrop.csv`），紋理被抹平，同一個 θ 的實際失真會偏低——
# 逐圖的 DISTS 照報，不假設它與 hb5 相同。
HUMAN_RADIUS = {"phase": 1.30, "phase_out": 1.30, "phase_rand": 1.30,
                "add": 1.2 / 255.0, "add_out": 1.2 / 255.0}

PARAM_CONDS = tuple(HUMAN_RADIUS)
PGD_CONDS = ("photoguard_c", "advpaint", "promptflare")
CONDITIONS = PARAM_CONDS + PGD_CONDS


def load_dataset(root: Path, masks: Path) -> list:
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for c in sorted(spec):
        for img in sorted((root / c).glob("*.png")):
            m = masks / f"{img.stem}.png"
            if not m.exists():
                raise FileNotFoundError(f"{img.stem} 沒有遮罩：{m}")
            out.append({"name": img.stem, "class": spec[c]["content"],
                        "path": img, "mask": m, "content": spec[c]["content"],
                        "prompt": spec[c]["prompts"][0]})
    return out


def load_mask(path: Path, device) -> torch.Tensor:
    """(1,1,H,W)，**1 表示攻擊方要重畫的區域**（與 `mask_latents` 同一約定）。"""
    m = load_image_tensor(path, device, size=RESOLUTION)[:, :1]
    if not torch.isin(m, torch.tensor([0.0, 1.0], device=m.device)).all():
        raise ValueError(
            f"{path} 不是二值遮罩。人工繪製的遮罩必須是 0/1——中間值在"
            "`mask_latents` 的最近鄰下採樣後既不算保留也不算重畫")
    return m


def build_param(cond: str, keep: torch.Tensor):
    """`_out` 變體把容量限制在 `keep` 內；其餘完全相同。"""
    k = keep if cond.endswith("_out") else None
    if cond.startswith("add"):
        return AdditiveParam(radius=HUMAN_RADIUS[cond], keep=k)
    cls = RandomPhaseParam if cond == "phase_rand" else PhaseParam
    return cls(size=RESOLUTION, radius=HUMAN_RADIUS[cond], keep=k)


def run_additive(sd, item, cond, x01, mask, seed):
    spec = {"photoguard_c": photoguard.SPEC, "advpaint": advpaint.SPEC,
            "promptflare": promptflare.SPEC}[cond]
    kw = {"mask": mask, "use_ckpt": True, "vae_ckpt": True}
    if cond == "advpaint":
        # 攻擊 prompt 沒有原始碼可依（`advpaint.prepare` 拒絕預設值）。
        # 本專案為 prompt-free，故餵空字串並在報表標為我方設定。
        kw["prompt"] = ""
    return run_pgd(sd, x01, spec, seed=seed,
                   grad_mask=mask if spec.grad_outside_mask else None, **kw)


def masked_compare(y: torch.Tensor, x01: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """把遮罩外換成原圖，使後續的比較只落在生成內容上（見檔頭）。"""
    return y * mask + x01 * (1.0 - mask)


def evaluate(sd, suite, aes, item, x01, mask, x_def):
    m = suite.pairwise(x01, x_def)
    a = aes.measure(x_def)
    fid = {"lpips": m["lpips"], "ssim": m["ssim"], "dists": m["dists"],
           "psnr": m["psnr"], "linf": m["linf"],
           "clip_img": aes.clip_image_similarity(x01, x_def),
           "nima": a["nima"], "cnniqa": a["cnniqa"]}

    emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
    z_like = sd.encode_image(x01)
    noise = sd.sample_edit_noise(z_like, seed=INPAINT_SEED)
    with torch.no_grad():
        y_orig = sd.inpaint(x01, mask, emb, noise, INPAINT_STEPS,
                            guidance_scale=INPAINT_GUIDANCE, emb_uncond=emb_u)
        y_def = sd.inpaint(x_def.clamp(0, 1), mask, emb, noise, INPAINT_STEPS,
                           guidance_scale=INPAINT_GUIDANCE, emb_uncond=emb_u)
    co = masked_compare(y_orig, x01, mask)
    cd = masked_compare(y_def, x01, mask)
    so, sd_ = suite.semantic(y_orig, item["prompt"]), suite.semantic(y_def, item["prompt"])
    row = {**{f"fid_{k}": round(float(v), 4) for k, v in fid.items()},
           "gen_lpips": round(float(suite.pairwise(co, cd)["lpips"]), 4),
           "gen_lpips_full": round(float(suite.pairwise(y_orig, y_def)["lpips"]), 4),
           "clip_orig": round(so["clip"], 4), "clip_def": round(sd_["clip"], 4),
           "clip_drop": round(so["clip"] - sd_["clip"], 4),
           "mask_coverage": round(float(mask.mean()), 4)}
    return row, y_orig, y_def


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--masks", type=Path, default=MASKS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--target", type=Path, default=Path(TARGET_IMAGE))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDInpaintWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    y_target = load_image_tensor(args.target, sd.device, size=RESOLUTION)
    loss_fn = make_encoder_target_loss(sd, y_target)

    dataset = load_dataset(args.data, args.masks)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]

    rows = []
    for item in dataset:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        mask = load_mask(item["mask"], sd.device)
        keep = 1.0 - mask
        save_image(x01, args.out / f"{item['name']}__orig.png")
        save_image(mask.expand(-1, 3, -1, -1), args.out / f"{item['name']}__mask.png")
        print(f"\n########## {item['name']} ({item['class']}) "
              f"重畫區 {float(mask.mean()):.3f} ##########", flush=True)

        for cond in args.conditions:
            print(f"=== {item['name']} / {cond} ===", flush=True)
            t0 = time.time()
            extra = {}
            if cond in PARAM_CONDS:
                param = build_param(cond, keep)
                r = run_param_pgd(x01, param, loss_fn, steps=args.steps,
                                  seed=args.seed)
                x_def = r.x_def
                if cond.startswith("phase"):
                    extra["active_fraction"] = round(param.module.active_fraction(), 4)
                    extra["amp_dev"] = round(param.module.amplitude_deviation(x01), 5)
            else:
                x_def = run_additive(sd, item, cond, x01, mask, args.seed).x_adv01
            x_def = x_def.detach()

            metrics, yo, yd = evaluate(sd, suite, aes, item, x01, mask, x_def)
            for sub, img in (("def", x_def.clamp(0, 1)), ("gen_orig", yo),
                             ("gen_def", yd)):
                save_image(img, args.out / f"{item['name']}__{cond}__{sub}.png")
            row = {"image": item["name"], "condition": cond,
                   "radius": HUMAN_RADIUS.get(cond, ""),
                   "total_seconds": round(time.time() - t0, 1),
                   **metrics, **extra}
            rows.append(row)
            print(row, flush=True)
            write_csv(args.out / "results.csv", rows)

    print(f"\n表：{args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
