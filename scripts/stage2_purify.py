"""stage2：淨化後比較 — SPEC.md §6.3、STRUCTURE.md §4.4。

流程（淨化 → 編輯 → 測編輯劣化）：對 stage1 的每張受保護影像
  1. purified = purify(protected, method, strength)
  2. edited_pur = edit(purified, prompt, seed)   # 與 stage1 同 seed
  3. m_purified = compute_all(edited_pur, edited_orig)   # edited_orig 重用 stage1 輸出
  4. drop = (m_clean − m_purified) / m_clean（逐指標；方向見 METRIC_HIGHER_IS_BETTER）

輸出 experiments/stage2/<timestamp>/：results.csv、purify_strength_curve.png、summary.md。
generation/edit 設定一律取 stage1 之 config_snapshot.yaml，確保條件一致。
GrIDPure 需 pixel-space checkpoint（--gridpure-model）；--gridpure-fake 以
Gaussian blur 代替淨化器，僅供本地流程驗證。執行前先印出成本估算（--dry-run 只估算）。

用法：python scripts/stage2_purify.py --stage1-dir experiments/stage1/<ts> [--dry-run]
      [--purify-methods jpeg,blur] [--gridpure-model path/256x256_diffusion_uncond.pt]
"""

import argparse
import time
from pathlib import Path

from common import REPO_ROOT, load_yaml, model_tag, sample_mask  # noqa: E402

from src.edit import edit_image
from src.metrics.quality import compute_all
from src.models.sd_wrapper import SDWrapper
from src.purify import purify
from src.purify.lightweight import gaussian_blur
from src.utils.io import (
    load_csv, load_image, load_json, make_run_dir, save_config_snapshot,
    save_csv, save_env_json, save_image, save_json, save_summary,
)
from src.utils.seed import set_seed

METRIC_KEYS = ["psnr", "ssim", "vifp", "fsim", "lpips", "clip"]


class FakePurifier:
    """本地流程驗證用：以 Gaussian blur 假冒 grid 淨化器（無 checkpoint 環境）。"""

    def __init__(self):
        self.pure_steps = 0

    def set_pure_steps(self, n):
        self.pure_steps = n

    def __call__(self, patch):
        return gaussian_blur(patch, sigma=1.0)


def build_ops(purify_cfg: dict, only: list[str] = None):
    """回傳 [(family, label, strength, numeric_strength)]。gridpure 之 strength 為 dict。"""
    ops = []
    for q in purify_cfg["jpeg"]["quality"]:
        ops.append(("jpeg", f"jpeg_q{q}", q, q))
    for s in purify_cfg["blur"]["sigma"]:
        ops.append(("blur", f"blur_s{s}", s, s))
    for r in purify_cfg["crop_resize"]["ratio"]:
        ops.append(("crop_resize", f"crop_r{r}", r, r))
    ops.append(("advclean_bf", "advclean_bf", None, None))
    ops.append(("advclean_bfgf", "advclean_bfgf", None, None))
    gp = purify_cfg["gridpure"]
    iter_n = next(s["iterations"] for s in gp["settings"] if s["name"] == "iterative")
    for st in gp["settings"]:
        ops.append(("gridpure", f"gridpure_{st['name']}",
                    {"pure_steps": st["pure_steps"], "iterations": st["iterations"]},
                    st["pure_steps"]))
    for ps in gp.get("pure_steps_scan", []):
        ops.append(("gridpure", f"gridpure_scan{ps}",
                    {"pure_steps": ps, "iterations": iter_n}, ps))
    if only:
        ops = [op for op in ops if op[0] in only or op[1] in only]
    return ops


def main():
    parser = argparse.ArgumentParser(description="stage2 淨化後比較")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--purify-methods", default=None, help="家族或標籤子集，逗號分隔")
    parser.add_argument("--gridpure-model", default=None, help="256x256_diffusion_uncond.pt 路徑")
    parser.add_argument("--gridpure-fake", action="store_true", help="本地流程驗證用假淨化器")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="僅列印成本估算")
    args = parser.parse_args()

    s1 = Path(args.stage1_dir)
    manifest = load_json(s1 / "manifest.json")
    cfgs = load_yaml(s1 / "config_snapshot.yaml")  # 與 stage1 完全一致之設定
    base = cfgs["base"]
    set_seed(base["runtime"]["seed"])

    clean = {}
    for r in load_csv(s1 / "results.csv"):
        key = (r["method"], r["model"], r["edit_method"], r["image_id"], r["prompt_idx"])
        clean[key] = {k: float(r[k]) for k in METRIC_KEYS if r.get(k)}

    only = args.purify_methods.split(",") if args.purify_methods else None
    ops = build_ops(cfgs["purify"], only)

    purifier = None
    if args.gridpure_fake:
        purifier = FakePurifier()
    elif args.gridpure_model:
        from src.purify.gridpure import load_guided_diffusion_purifier
        purifier = load_guided_diffusion_purifier(
            args.gridpure_model, pure_steps=10, repo_dir=str(REPO_ROOT / "external" / "GrIDPure"))
    else:
        skipped = [op[1] for op in ops if op[0] == "gridpure"]
        ops = [op for op in ops if op[0] != "gridpure"]
        if skipped:
            print(f"[警告] 未提供 --gridpure-model / --gridpure-fake，略過: {skipped}")

    samples = manifest["samples"][: args.max_images] if args.max_images else manifest["samples"]
    methods = manifest["methods"]
    n_edits = (len(samples) * len(methods) * len(ops) * len(manifest["eval_models"])
               * len(manifest["edit_methods"]) * 2 * len(next(iter(manifest["seeds"].values()))))
    n_gridpure = len([op for op in ops if op[0] == "gridpure"]) * len(samples) * len(methods)
    print(f"成本估算：編輯呼叫 {n_edits} 次；GrIDPure 淨化 {n_gridpure} 次"
          f"（真實 checkpoint 下單張 512×512 於 V100 約 2 分鐘"
          f" → 約 {n_gridpure * 2 / 60:.1f} 小時，另加編輯時間）")
    if args.dry_run:
        return

    run_dir = make_run_dir("stage2")
    print(f"run_dir={run_dir}")

    # --- 淨化階段（與模型/編輯無關，逐 (方法, 影像, op) 一次）---
    for mkey in methods:
        for sample in samples:
            sid = sample["image_id"].replace("/", "__")
            protected = load_image(s1 / manifest["protected"][mkey][sample["image_id"]]["path"])
            for family, label, strength, _ in ops:
                t0 = time.time()
                out = purify(protected, family, strength,
                             config=cfgs["purify"], purifier=purifier)
                save_image(run_dir / "purified" / mkey / label / f"{sid}.png", out)
                if family == "gridpure":
                    print(f"purify {mkey} {sid} {label}: {time.time() - t0:.0f}s")

    # --- 編輯與指標 ---
    rows = []
    for model_name in manifest["eval_models"]:
        sd_eval = SDWrapper(model_name)
        tag = model_tag(model_name)
        for edit_m in manifest["edit_methods"]:
            for sample in samples:
                sid = sample["image_id"].replace("/", "__")
                mask = sample_mask(sample, base) if edit_m == "inpaint" else None
                for pi, prompt in enumerate(sample["edit_prompts"]):
                    seeds = manifest["seeds"][f"{sample['image_id']}|p{pi}"]
                    edited_origs = {
                        seed: load_image(s1 / "edited_orig" / tag / edit_m /
                                         f"{sid}_p{pi}_s{seed}.png")
                        for seed in seeds
                    }
                    for mkey in methods:
                        for family, label, strength, num_strength in ops:
                            purified = load_image(run_dir / "purified" / mkey / label / f"{sid}.png")
                            per_seed = []
                            for seed in seeds:
                                ep = edit_image(purified, prompt, seed, edit_m,
                                                mask=mask, config=base, sd=sd_eval)
                                per_seed.append(compute_all(ep, edited_origs[seed]))
                            mean = {k: sum(d[k] for d in per_seed) / len(per_seed)
                                    for k in per_seed[0]}
                            ckey = (mkey, model_name, edit_m, sample["image_id"], str(pi))
                            row = {"purify": label, "purify_family": family,
                                   "strength": num_strength, "method": mkey,
                                   "model": model_name, "edit_method": edit_m,
                                   "image_id": sample["image_id"], "prompt_idx": pi, **mean}
                            for k, v in mean.items():
                                c = clean.get(ckey, {}).get(k)
                                if c:
                                    row[f"drop_{k}"] = (c - v) / c
                            rows.append(row)
                    print(f"eval {tag} {edit_m} {sample['image_id']} p{pi}: done")
        del sd_eval

    save_csv(run_dir / "results.csv", rows)
    _plot(rows, run_dir)
    save_config_snapshot(run_dir, cfgs)
    save_env_json(run_dir)
    save_json(run_dir / "source.json", {"stage1_dir": str(s1)})
    _summary(run_dir, rows, args)
    print(f"stage2 完成：{len(rows)} 列 → {run_dir}")


def _plot(rows, run_dir):
    """淨化強度 vs 防禦效果（lpips，越高越好）曲線，逐家族一子圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in ("jpeg", "blur", "crop_resize", "gridpure")
                if any(r["purify_family"] == f for r in rows)]
    if not families:
        return
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 4), squeeze=False)
    for ax, family in zip(axes[0], families):
        frows = [r for r in rows if r["purify_family"] == family and r["strength"] is not None]
        for mkey in sorted({r["method"] for r in frows}):
            pts = {}
            for r in frows:
                if r["method"] == mkey:
                    pts.setdefault(r["strength"], []).append(r["lpips"])
            xs = sorted(pts)
            ax.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], marker="o", label=mkey)
        ax.set_title(family)
        ax.set_xlabel("strength")
        ax.set_ylabel("LPIPS (higher = better defense)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "purify_strength_curve.png", dpi=150)
    plt.close(fig)


def _summary(run_dir, rows, args):
    additive = {"pg_enc", "pg_diff"}
    lines = [
        "# stage2 摘要（實測值）", "",
        "- drop = (clean − purified) / clean，逐指標；lpips 之 drop 越大代表淨化越有效"
        "（防禦被移除越多）。方向定義見 METRIC_HIGHER_IS_BETTER。",
        "- 核心比較：加性（pg_*）vs 非加性（advdiff/apa/hybrid）之 drop。"
        "待驗證假設（非預設結論）：非加性之 drop 較小。", "",
        "| purify | 加性 mean drop_lpips | 非加性 mean drop_lpips |", "|---|---|---|",
    ]
    labels = list(dict.fromkeys(r["purify"] for r in rows))
    for label in labels:
        def _mean(pred):
            vals = [r["drop_lpips"] for r in rows
                    if r["purify"] == label and pred(r["method"]) and "drop_lpips" in r]
            return f"{sum(vals) / len(vals):.4f}" if vals else "-"
        lines.append(f"| {label} | {_mean(lambda m: m in additive)} |"
                     f" {_mean(lambda m: m not in additive)} |")
    save_summary(run_dir, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
