"""stage1：乾淨情況比較 — SPEC.md §6.2、STRUCTURE.md §4.3。

方法矩陣 = {pg_enc, pg_diff, advdiff, apa, hybrid} × eval_models × {sdedit, inpaint}，
每 (影像, prompt) 以 seed 協定取 n_seeds 個 seed，原圖與受保護影像同 seed 編輯，
指標對 seed 取平均。保護一律以 protect_model 生成（與評測模型解耦）。

輸出 experiments/stage1/<timestamp>/：
- results.csv（method, model, edit_method, image_id, prompt_idx, 指標, peak_memory_mb, elapsed_sec）
- protected/<method>/<id>.png、edited_orig/<model>/<edit>/<id>_p<i>_s<seed>.png（stage2 重用）
- manifest.json（seeds 與路徑，stage2 依據）
校準檢查點：pg_enc / pg_diff 數值須與 SPEC §2.7 DAYN Table 1 之 Encoder/Diffusion 欄接近。

穩健性：results.csv 逐筆寫入（每列 flush）；`--resume DIR` 續跑既有 run
目錄——已完成之保護影像、edited_orig 與結果列自動跳過。
FID 需完整樣本：續跑時若有列被跳過則略過 FID（如需 FID 請完整重跑該組）。

用法：python scripts/stage1_clean.py [--smoke] [--methods pg_enc,advdiff] [--model M]
      [--no-fid] [--resume experiments/stage1/<ts>]
"""

import argparse
import time
from pathlib import Path

import torch

from common import (  # noqa: E402
    IncrementalCsv, apply_smoke, load_configs, model_tag, sample_mask,
    used_placeholder_mask,
)

from src.data.dataset import load_dataset
from src.edit import edit_image_batch
from src.edit.seed_protocol import search_valid_seeds
from src.metrics.quality import ClipScorer, compute_all, compute_fid
from src.models.sd_wrapper import SDWrapper
from src.protect import METHOD_KEYS, build_protection
from src.utils.io import (
    load_image, load_json, make_run_dir, save_config_snapshot, save_csv,
    save_env_json, save_image, save_json, save_summary,
)
from src.utils.seed import set_seed

KEY_FIELDS = ["method", "model", "edit_method", "image_id", "prompt_idx"]


def safe_id(image_id: str) -> str:
    return image_id.replace("/", "__")


_inception = None


def inception_feats(image01: torch.Tensor) -> torch.Tensor:
    """piq InceptionV3 特徵（FID 用），(1,3,H,W)[0,1] → (D,)。"""
    global _inception
    from piq.feature_extractors import InceptionV3

    if _inception is None:
        _inception = InceptionV3()
        _inception.eval()
    with torch.no_grad():
        feats = _inception(image01.float().clamp(0, 1))
    return feats[0].squeeze(-1).squeeze(-1).squeeze(0)


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
    parser.add_argument("--with-clip", action="store_true",
                        help="計算 CLIP score（模型取 base.yaml metrics.clip_model）")
    parser.add_argument("--clip-model", default=None, help="覆蓋 CLIP 模型名（隱含 --with-clip）")
    parser.add_argument("--resume", default=None, help="續跑既有 run 目錄")
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
    run_dir = Path(args.resume) if args.resume else make_run_dir("stage1")
    results = IncrementalCsv(run_dir / "results.csv", KEY_FIELDS)
    clip_name = args.clip_model or (base["metrics"]["clip_model"] if args.with_clip else None)
    clip_scorer = ClipScorer(clip_name) if clip_name else None
    print(f"stage1: protect={protect_model}, eval={eval_models}, methods={methods}, "
          f"edits={edit_methods}, images={len(data)}, seeds={n_seeds}, run_dir={run_dir}"
          + (f"（續跑，已有 {len(results.rows)} 列）" if args.resume else ""))

    # --- 保護階段（protect_model；與評測模型無關）。續跑：影像與紀錄俱在則跳過 ---
    info_path = run_dir / "protect_info.json"
    prot_info = load_json(info_path) if info_path.exists() else {}
    for mkey in methods:
        method = None
        prot_info.setdefault(mkey, {})
        for sample in data:
            path = run_dir / "protected" / mkey / f"{safe_id(sample['image_id'])}.png"
            if path.exists() and sample["image_id"] in prot_info[mkey]:
                continue
            if method is None:
                method = build_protection(mkey, sd_protect, cfgs["additive"], cfgs["nonadditive"])
            t0 = time.time()
            protected = method.protect(sample["image"], sample["concept"])
            elapsed = time.time() - t0
            save_image(path, protected)
            prot_info[mkey][sample["image_id"]] = {
                "path": str(path.relative_to(run_dir)),
                "elapsed_sec": round(elapsed, 2),
                "peak_memory_mb": method.peak_memory_mb(),
            }
            save_json(info_path, prot_info)  # 逐筆落盤（斷點續跑）
            print(f"protect {mkey} {sample['image_id']}: {elapsed:.0f}s")

    # --- seed 協定（於原圖上搜尋；threshold=null 時決定性，續跑結果一致）---
    seeds_map = {}
    for sample in data:
        for pi, prompt in enumerate(sample["edit_prompts"]):
            seeds_map[f"{sample['image_id']}|p{pi}"] = search_valid_seeds(
                sample["image"], prompt, n_seeds=n_seeds, config=base,
                sd=sd_protect, clip_scorer=clip_scorer,
            )

    if protect_model not in eval_models:
        del sd_protect
        sd_protect = None

    # --- 評測階段 ---
    fid_feats, skipped_rows = {}, 0
    for model_name in eval_models:
        sd_eval = sd_protect if (sd_protect and model_name == protect_model) else SDWrapper(model_name)
        tag = model_tag(model_name)
        for edit_m in edit_methods:
            for sample in data:
                sid = safe_id(sample["image_id"])
                mask = sample_mask(sample, base) if edit_m == "inpaint" else None
                for pi, prompt in enumerate(sample["edit_prompts"]):
                    pending = [m for m in methods if not results.done({
                        "method": m, "model": model_name, "edit_method": edit_m,
                        "image_id": sample["image_id"], "prompt_idx": pi})]
                    skipped_rows += len(methods) - len(pending)
                    if not pending:
                        continue
                    seeds = seeds_map[f"{sample['image_id']}|p{pi}"]
                    edited_origs, missing = {}, []
                    for seed in seeds:
                        eo_path = (run_dir / "edited_orig" / tag / edit_m /
                                   f"{sid}_p{pi}_s{seed}.png")
                        if eo_path.exists():
                            edited_origs[seed] = load_image(eo_path)
                        else:
                            missing.append((seed, eo_path))
                    if missing:
                        eouts = edit_image_batch(
                            [sample["image"]] * len(missing), [prompt] * len(missing),
                            [s for s, _ in missing], edit_m,
                            masks=[mask] * len(missing) if mask is not None else None,
                            config=base, sd=sd_eval)
                        for (seed, eo_path), eo in zip(missing, eouts):
                            edited_origs[seed] = eo
                            save_image(eo_path, eo)
                    for mkey in pending:
                        protected = load_image(run_dir / prot_info[mkey][sample["image_id"]]["path"])
                        eps_list = edit_image_batch(
                            [protected] * len(seeds), [prompt] * len(seeds), list(seeds),
                            edit_m, masks=[mask] * len(seeds) if mask is not None else None,
                            config=base, sd=sd_eval)
                        per_seed = []
                        for seed, ep in zip(seeds, eps_list):
                            per_seed.append(compute_all(ep, edited_origs[seed], prompt, clip_scorer))
                            if not args.no_fid:
                                key = (mkey, tag, edit_m)
                                fid_feats.setdefault(key, {"prot": [], "orig": []})
                                fid_feats[key]["prot"].append(inception_feats(ep))
                                fid_feats[key]["orig"].append(inception_feats(edited_origs[seed]))
                        mean = {k: sum(d[k] for d in per_seed) / len(per_seed) for k in per_seed[0]}
                        results.append({
                            "method": mkey, "model": model_name, "edit_method": edit_m,
                            "image_id": sample["image_id"], "prompt_idx": pi, **mean,
                            "peak_memory_mb": prot_info[mkey][sample["image_id"]]["peak_memory_mb"],
                            "elapsed_sec": prot_info[mkey][sample["image_id"]]["elapsed_sec"],
                        })
                    print(f"eval {tag} {edit_m} {sample['image_id']} p{pi}: done")
        if sd_eval is not sd_protect:
            del sd_eval

    # --- FID（資料集層級；需完整樣本，續跑跳過任何列時不輸出）---
    if not args.no_fid:
        if skipped_rows:
            print(f"[警告] 續跑跳過 {skipped_rows} 列，FID 樣本不完整，略過 FID 輸出；"
                  "如需 FID 請對該組完整重跑")
        else:
            fid_rows = [{"method": mk, "model": tg, "edit_method": em,
                         "fid": compute_fid(torch.stack(f["prot"]), torch.stack(f["orig"]))}
                        for (mk, tg, em), f in fid_feats.items()]
            if fid_rows:
                save_csv(run_dir / "fid.csv", fid_rows)

    save_json(run_dir / "manifest.json", {
        "protect_model": protect_model, "eval_models": eval_models,
        "edit_methods": edit_methods, "methods": methods, "smoke": args.smoke,
        "placeholder_mask": used_placeholder_mask(data, edit_methods),
        "mask_spec": base["edit"].get("inpaint_mask"),
        "samples": [{"image_id": s["image_id"], "concept": s["concept"],
                     "edit_prompts": s["edit_prompts"]} for s in data],
        "seeds": seeds_map, "protected": prot_info,
    })
    save_config_snapshot(run_dir, cfgs)
    save_env_json(run_dir)
    _summary(run_dir, results.rows, args, base, data, edit_methods)
    print(f"stage1 完成：{len(results.rows)} 列 → {run_dir}")


def _summary(run_dir, rows, args, base, data, edit_methods):
    metric_keys = [k for k in ("psnr", "ssim", "vifp", "fsim", "lpips", "clip")
                   if rows and k in rows[0]]
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["model"], r["edit_method"]), []).append(r)
    lines = [
        "# stage1 摘要（實測值）", "",
        f"- smoke={args.smoke}（smoke 為流程驗證，數值不具比較意義）",
        "- 校準檢查點：pg_enc / pg_diff 之 **sdedit（img2img）** 列須與 SPEC §2.7 "
        "DAYN Table 1（引用值）之 Encoder / Diffusion 欄比對，接近方可確認設定正確。"
        "（論文核驗：Table 1 為 image editing（img2img）情境、20 seeds 平均；"
        "inpainting 於 DAYN 僅質性比較，勿以 inpaint 列比對。）",
    ]
    if base["data"].get("is_placeholder"):
        lines.append("- **placeholder 資料集**：結果與真實資料集不可直接比較。")
    if used_placeholder_mask(data, edit_methods):
        lines.append(f"- **placeholder 遮罩**（{base['edit'].get('inpaint_mask')}）："
                     "遮罩幾何實質影響保護難度，inpaint 結果與真實資料集結果不可直接比較，"
                     "真實資料到手後須重跑。")

    def table(rows_pred):
        out = ["| method | model | edit | " + " | ".join(metric_keys) + " |",
               "|---" * (3 + len(metric_keys)) + "|"]
        for (m, mo, e), g in groups.items():
            if not rows_pred(e):
                continue
            means = [f"{sum(float(r[k]) for r in g) / len(g):.4f}" for k in metric_keys]
            out.append(f"| {m} | {mo} | {e} | " + " | ".join(means) + " |")
        return out if len(out) > 2 else []

    # v5（SPEC §2.7）：img2img 有 DAYN 錨點（主要條件）；inpainting 無錨點（補充條件）
    anchored = table(lambda e: e == "sdedit")
    if anchored:
        lines += ["", "## 有 DAYN 錨點之條件：sdedit（img2img，主要條件）", "",
                  "校準比對限本表之 pg_enc / pg_diff 列。"] + [""] + anchored
    unanchored = table(lambda e: e != "sdedit")
    if unanchored:
        lines += ["", "## 無 DAYN 錨點之條件：inpaint（補充條件）", "",
                  "DAYN 對 inpainting 僅質性比較，無數值錨點；本表僅能以本專案"
                  "自行實作之 PhotoGuard（pg_enc / pg_diff 列）為對照，"
                  "不可與 DAYN Table 1 比對。"] + [""] + unanchored
    save_summary(run_dir, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
