"""stage1：乾淨情況比較 — SPEC.md §6.2、STRUCTURE.md §4.3。

方法矩陣 = {pg_enc, pg_diff, advdiff, apa, hybrid} × eval_models × {sdedit, inpaint}，
每 (影像, prompt) 以 seed 協定取 n_seeds 個 seed，原圖與受保護影像同 seed 編輯，
指標對 seed 取平均。保護一律以 protect_model 生成（與評測模型解耦）。

輸出 experiments/stage1/<timestamp>/：
- results.csv（method, model, edit_method, image_id, prompt_idx, 指標, peak_memory_mb, elapsed_sec）
- protected/<method>/<id>.png、edited_orig/<model>/<edit>/<id>_p<i>_s<seed>.png（stage2 重用）
- manifest.json（seeds 與路徑，stage2 依據）
校準檢查點：pg_enc / pg_diff 數值須與 SPEC §2.7 DAYN Table 1 之 Encoder/Diffusion 欄接近。

用法：python scripts/stage1_clean.py [--smoke] [--methods pg_enc,advdiff] [--model M] [--no-fid]
"""

import argparse
import time

import torch

from common import apply_smoke, load_configs, model_tag, sample_mask  # noqa: E402

from src.data.dataset import load_dataset
from src.edit import edit_image
from src.edit.seed_protocol import search_valid_seeds
from src.metrics.quality import ClipScorer, compute_all, compute_fid
from src.models.sd_wrapper import SDWrapper
from src.protect import METHOD_KEYS, build_protection
from src.utils.io import (
    load_image, make_run_dir, save_config_snapshot, save_csv, save_env_json,
    save_image, save_json, save_summary,
)
from src.utils.seed import set_seed


def safe_id(image_id: str) -> str:
    return image_id.replace("/", "__")


_inception = None


def inception_feats(image01: torch.Tensor) -> torch.Tensor:
    """piq InceptionV3 特徵（FID 用），(1,3,H,W)[0,1] → (1,D)。"""
    global _inception
    from piq.feature_extractors import InceptionV3

    if _inception is None:
        _inception = InceptionV3()
        _inception.eval()
    with torch.no_grad():
        feats = _inception(image01.float().clamp(0, 1))
    return feats[0].squeeze(-1).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="stage1 乾淨情況比較")
    parser.add_argument("--methods", default=",".join(METHOD_KEYS))
    parser.add_argument("--model", default=None, help="只評測單一模型（覆蓋 eval_models）")
    parser.add_argument("--protect-model", default=None, help="覆蓋 protect_model（本地測試用）")
    parser.add_argument("--edit-methods", default=None, help="如 sdedit,inpaint")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--clip-model", default=None, help="CLIP score 模型名；未給則略過 clip")
    args = parser.parse_args()

    cfgs = load_configs()
    protect_model = args.protect_model or cfgs["base"]["model"]["protect_model"]
    sd_protect = SDWrapper(protect_model)
    if args.smoke:
        apply_smoke(cfgs, sd_protect)
    base = cfgs["base"]
    set_seed(base["runtime"]["seed"])

    methods = args.methods.split(",")
    eval_models = [args.model] if args.model else base["model"]["eval_models"]
    edit_methods = (args.edit_methods.split(",") if args.edit_methods
                    else base["edit"]["methods"])
    n_seeds = args.n_seeds or base["edit"]["n_seeds"]
    data = load_dataset(base, max_images=args.max_images)
    run_dir = make_run_dir("stage1")
    clip_scorer = ClipScorer(args.clip_model) if args.clip_model else None
    print(f"stage1: protect={protect_model}, eval={eval_models}, methods={methods}, "
          f"edits={edit_methods}, images={len(data)}, seeds={n_seeds}, run_dir={run_dir}")

    # --- 保護階段（protect_model；與評測模型無關，僅執行一次）---
    prot_info = {}
    for mkey in methods:
        method = build_protection(mkey, sd_protect, cfgs["additive"], cfgs["nonadditive"])
        prot_info[mkey] = {}
        for sample in data:
            t0 = time.time()
            protected = method.protect(sample["image"], sample["concept"])
            elapsed = time.time() - t0
            path = run_dir / "protected" / mkey / f"{safe_id(sample['image_id'])}.png"
            save_image(path, protected)
            prot_info[mkey][sample["image_id"]] = {
                "path": str(path.relative_to(run_dir)),
                "elapsed_sec": round(elapsed, 2),
                "peak_memory_mb": method.peak_memory_mb(),
            }
            print(f"protect {mkey} {sample['image_id']}: {elapsed:.0f}s")

    # --- seed 協定（於原圖上搜尋，與評測模型解耦前提：threshold=null 時決定性）---
    seeds_map = {}
    for sample in data:
        for pi, prompt in enumerate(sample["edit_prompts"]):
            seeds_map[f"{sample['image_id']}|p{pi}"] = search_valid_seeds(
                sample["image"], prompt, n_seeds=n_seeds, config=base,
                sd=sd_protect, clip_scorer=clip_scorer,
            )

    if protect_model not in eval_models:
        del sd_protect  # 釋放記憶體後再載評測模型
        sd_protect = None

    # --- 評測階段 ---
    rows, fid_feats = [], {}
    for model_name in eval_models:
        sd_eval = sd_protect if (sd_protect and model_name == protect_model) else SDWrapper(model_name)
        tag = model_tag(model_name)
        for edit_m in edit_methods:
            for sample in data:
                sid = safe_id(sample["image_id"])
                mask = sample_mask(sample, base) if edit_m == "inpaint" else None
                for pi, prompt in enumerate(sample["edit_prompts"]):
                    seeds = seeds_map[f"{sample['image_id']}|p{pi}"]
                    edited_origs = {}
                    for seed in seeds:
                        eo = edit_image(sample["image"], prompt, seed, edit_m,
                                        mask=mask, config=base, sd=sd_eval)
                        edited_origs[seed] = eo
                        save_image(run_dir / "edited_orig" / tag / edit_m /
                                   f"{sid}_p{pi}_s{seed}.png", eo)
                    for mkey in methods:
                        protected = load_image(run_dir / prot_info[mkey][sample["image_id"]]["path"])
                        per_seed = []
                        for seed in seeds:
                            ep = edit_image(protected, prompt, seed, edit_m,
                                            mask=mask, config=base, sd=sd_eval)
                            per_seed.append(compute_all(ep, edited_origs[seed], prompt, clip_scorer))
                            if not args.no_fid:
                                key = (mkey, tag, edit_m)
                                fid_feats.setdefault(key, {"prot": [], "orig": []})
                                fid_feats[key]["prot"].append(inception_feats(ep))
                                fid_feats[key]["orig"].append(inception_feats(edited_origs[seed]))
                        mean = {k: sum(d[k] for d in per_seed) / len(per_seed) for k in per_seed[0]}
                        rows.append({
                            "method": mkey, "model": model_name, "edit_method": edit_m,
                            "image_id": sample["image_id"], "prompt_idx": pi, **mean,
                            "peak_memory_mb": prot_info[mkey][sample["image_id"]]["peak_memory_mb"],
                            "elapsed_sec": prot_info[mkey][sample["image_id"]]["elapsed_sec"],
                        })
                    print(f"eval {tag} {edit_m} {sample['image_id']} p{pi}: done")
        if sd_eval is not sd_protect:
            del sd_eval

    # --- FID（資料集層級，逐組計算）---
    fid_rows = []
    for (mkey, tag, edit_m), feats in fid_feats.items():
        fid_rows.append({
            "method": mkey, "model": tag, "edit_method": edit_m,
            "fid": compute_fid(torch.stack(feats["prot"]), torch.stack(feats["orig"])),
        })
    if fid_rows:
        save_csv(run_dir / "fid.csv", fid_rows)

    save_csv(run_dir / "results.csv", rows)
    save_json(run_dir / "manifest.json", {
        "protect_model": protect_model, "eval_models": eval_models,
        "edit_methods": edit_methods, "methods": methods, "smoke": args.smoke,
        "samples": [{"image_id": s["image_id"], "concept": s["concept"],
                     "edit_prompts": s["edit_prompts"]} for s in data],
        "seeds": seeds_map, "protected": prot_info,
    })
    save_config_snapshot(run_dir, cfgs)
    save_env_json(run_dir)
    _summary(run_dir, rows, args)
    print(f"stage1 完成：{len(rows)} 列 → {run_dir}")


def _summary(run_dir, rows, args):
    metric_keys = [k for k in ("psnr", "ssim", "vifp", "fsim", "lpips", "clip")
                   if rows and k in rows[0]]
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["model"], r["edit_method"]), []).append(r)
    lines = [
        "# stage1 摘要（實測值）", "",
        f"- smoke={args.smoke}（smoke 為流程驗證，數值不具比較意義）",
        "- 校準檢查點：pg_enc / pg_diff 之列須與 SPEC §2.7 DAYN Table 1（引用值）之 "
        "Encoder / Diffusion 欄比對，接近方可確認設定正確。", "",
        "| method | model | edit | " + " | ".join(metric_keys) + " |",
        "|---" * (3 + len(metric_keys)) + "|",
    ]
    for (m, mo, e), g in groups.items():
        means = [f"{sum(r[k] for r in g) / len(g):.4f}" for k in metric_keys]
        lines.append(f"| {m} | {mo} | {e} | " + " | ".join(means) + " |")
    save_summary(run_dir, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
