"""baseline 在**原始強度**下擋不擋得住這個威脅模型（k=1，不做射線縮放）。

存在理由
──────────────────────────────────────────────────────────────────────
`v14_bird_03` 的段 3 顯示七個條件在 τ=0.20 的 `effect_siglip` 全部在雜訊裡
（−0.007 ~ +0.001），**包含三篇已發表的加性方法**。三篇獨立的攻擊同時失效
不像是方法的問題，故必須分辨兩種可能：

1. **協定把它們壓小了。** 段 2 把每個條件縮放到 LPIPS = τ，而三篇原生就
   運作在更高的失真上（實測 bird_03：mist 0.6212、photoguard_c 0.4025、
   dia_r 0.2972），於是 τ=0.20 對應的 `scale_k` 是 0.180 / 0.469 / 0.594
   ——**跑在原始設定的 18%–59%**。
2. **威脅模型本身超出它們的能力。** SDEdit strength=0.6 會把影像加噪到
   t≈600/1000 再去噪，對抗訊號在該過程中大量流失。多數免疫論文評測的是
   inpainting 或更低的 strength。

兩者的處置完全不同：前者要調整比較預算，後者要重新檢視威脅模型。本腳本
直接量第 2 點——**用段 1 存下的 `x_def.png`（k=1，原始強度）跑一次評測**。

方法
──────────────────────────────────────────────────────────────────────
與 `eval_executor` 走同一條路徑：同一個 `_sdedit`、同一個
`eval_noise_seed`、同一個 `MetricSuite.semantic`，淨化固定為 identity。
對照側直接讀 `control/` 已存的編輯結果，不重跑——它與防禦側必須逐元素
共用同一組噪聲，重跑會引入一個與防禦無關的差異。

用法
──────────────────────────────────────────────────────────────────────
    python scripts/diag_native_strength.py --batch v14_bird_03 \
        --runs-root ~/wacv_runs --gpu-tag RTX-3090 --precision fp32 \
        --model CompVis/stable-diffusion-v1-4 --wrapper sd --resolution 512 \
        --images bird_03 --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import executors, grid          # noqa: E402
import run_stage as rs                              # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # 與 `run_stage.py` 共用旗標定義，避免兩份不一致
    ap.add_argument("--batch", required=True)
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--gpu-tag", required=True)
    ap.add_argument("--precision", required=True,
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--wrapper", default="auto",
                    choices=["auto", "sd", "sdxl"])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--spec-version", type=int, default=1)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--n-seeds", type=int, default=grid.N_SEEDS)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    # `build_resources` 需要的其餘欄位取程式預設；本腳本不訓練，故那些
    # 旋鈕不影響結果。明寫而非依賴 argparse 的預設，是為了讓「這裡用了
    # 什麼」在讀程式時看得到。
    for k, v in dict(mist_target="", target_image="data/targets/gray.png",
                     diffpure_ckpt="", train_n_edit=10, n_eot=1, k_inv=10,
                     t_max=None, exact_inversion=False, purify_mode="all",
                     max_steps=250, align_steps=200, stop_patience=20,
                     stop_min_steps=25, attn_timesteps=4, shared_tokens=[0],
                     tau_train=grid.TRAIN_TAU,
                     tau_acut=executors.RunConfig.tau_acut,
                     tau_chroma=executors.RunConfig.tau_chroma,
                     beta_linf=executors.RunConfig.beta_linf,
                     tau_linf=executors.RunConfig.tau_linf,
                     warp_grid_size=32, warp_max_disp=1.5,
                     warp_resample="bicubic", apa_lora_rank=8,
                     apa_latent_max_rank=32, apa_latent_const_rank=8,
                     random_init_std=0.5, lr_grid="1e-4,1e-3,5e-3,2e-2,1e-1",
                     probe_steps=12, edit_effect_threshold=0.0).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    batch_dir = args.runs_root / args.batch
    res = rs.build_resources(args, batch_dir, load_model=True)
    conds = grid.resolve_conditions(args.conditions)

    rows = []
    for entry in res.images.values():
        img = entry.image_id
        prompt = entry.prompts[0]
        ctrl_dir = executors.control_dir(res, img, grid.IDENTITY)
        for cond in conds:
            xdef_png = res.cell_dir(cond, img) / "x_def.png"
            if not xdef_png.exists():
                print(f"  [skip] {cond}/{img}：{xdef_png} 不存在", flush=True)
                continue
            x_def = executors.load_image_tensor(xdef_png, res.device)
            fid = res.suite.pairwise(entry.x01, x_def)
            for s in range(args.n_seeds):
                ctrl_png = ctrl_dir / f"edit_seed{s}.png"
                if not ctrl_png.exists():
                    raise FileNotFoundError(
                        f"{ctrl_png} 不存在：對照側必須已跑完。兩側共用同一組"
                        "編輯噪聲，重跑對照會引入與防禦無關的差異")
                y_ctrl = executors.load_image_tensor(ctrl_png, res.device)
                y_def, _ = executors._sdedit(res, x_def, prompt, s)
                a = res.suite.semantic(y_ctrl, prompt)
                b = res.suite.semantic(y_def, prompt)
                rows.append({
                    "image_id": img, "condition": cond, "seed": s,
                    "scale_k": 1.0, "prompt": prompt,
                    "fid_lpips": fid["lpips"], "fid_psnr": fid["psnr"],
                    "fid_linf": fid["linf"],
                    "siglip_ctrl": a["siglip"], "siglip_def": b["siglip"],
                    "clip_ctrl": a["clip"], "clip_def": b["clip"],
                    "effect_siglip": a["siglip"] - b["siglip"],
                    "effect_clip": a["clip"] - b["clip"],
                    "edit_lpips": float(
                        res.suite.pairwise(y_ctrl, y_def)["lpips"]),
                })
                print(f"  {cond:<14} {img} seed{s}  "
                      f"effect_siglip={rows[-1]['effect_siglip']:+.4f}  "
                      f"lpips(def,orig)={fid['lpips']:.4f}", flush=True)
                del y_def, y_ctrl
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out = args.out or (batch_dir / "native_strength.csv")
    executors.write_csv(out, rows)
    print(f"\n寫出 {out}（{len(rows)} 列）")

    import statistics
    print("\n=== 原始強度（k=1）下的 effect_siglip ===")
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r["effect_siglip"])
    for c, v in sorted(by.items()):
        sd = statistics.stdev(v) if len(v) > 1 else float("nan")
        print(f"  {c:<16} mean={statistics.fmean(v):+.4f}  sd={sd:.4f}  "
              f"n={len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
