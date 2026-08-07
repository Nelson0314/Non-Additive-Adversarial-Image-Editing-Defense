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
import sys
from dataclasses import replace as dc_replace
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
    ap.add_argument("--strengths", default=None,
                    help="逗號分隔的 strength 掃描點，例 0.2,0.3,0.4,0.5,0.6。"
                         "給定時對照側逐點重算，不沿用 control/ 的產物")
    ap.add_argument("--source", default="native",
                    help="native 取段 1 的 x_def.png（k=1，各方法自己的工作"
                         "點）；給 τ 值則取段 2 的 x_def_tau<τ>.png")
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

    strengths = ([float(x) for x in args.strengths.split(",")]
                 if args.strengths else [args.strength])
    src = args.source
    tag = "x_def.png" if src == "native" else f"x_def_tau{src}.png"

    rows = []
    for entry in res.images.values():
        img = entry.image_id
        prompt = entry.prompts[0]
        # 防禦圖先讀進來：它與 strength 無關，每個 strength 重讀是純浪費
        defs = {}
        for cond in conds:
            p = res.cell_dir(cond, img) / tag
            if not p.exists():
                print(f"  [skip] {cond}/{img}：{p} 不存在", flush=True)
                continue
            x = executors.load_image_tensor(p, res.device)
            defs[cond] = (x, res.suite.pairwise(entry.x01, x))

        for st in strengths:
            # **對照側必須在同一個 strength 上重算。** `control/` 存的是
            # `--strength` 那一點的結果，拿它去比另一個 strength 的防禦側，
            # 量到的差異主要來自 strength 本身。
            res.cfg = dc_replace(res.cfg, strength=st)
            ctrls = {}
            for s in range(args.n_seeds):
                y, _ = executors._sdedit(res, entry.x01, prompt, s)
                ctrls[s] = (y, res.suite.semantic(y, prompt))

            for cond, (x_def, fid) in defs.items():
                for s in range(args.n_seeds):
                    y_ctrl, a = ctrls[s]
                    y_def, _ = executors._sdedit(res, x_def, prompt, s)
                    b = res.suite.semantic(y_def, prompt)
                    rows.append({
                        "image_id": img, "condition": cond, "seed": s,
                        "strength": st, "source": src, "prompt": prompt,
                        "fid_lpips": fid["lpips"], "fid_psnr": fid["psnr"],
                        "fid_linf": fid["linf"],
                        "siglip_ctrl": a["siglip"], "siglip_def": b["siglip"],
                        "clip_ctrl": a["clip"], "clip_def": b["clip"],
                        "effect_siglip": a["siglip"] - b["siglip"],
                        "effect_clip": a["clip"] - b["clip"],
                        # 攻擊方在這個 strength 上「本來能拿多少」。防禦效果
                        # 必須對它正規化才跨 strength 可比：strength 越低，
                        # 編輯本身移動得越少，同樣的絕對值代表的比例不同。
                        "siglip_orig": res.suite.semantic(
                            entry.x01, prompt)["siglip"],
                        "edit_lpips": float(
                            res.suite.pairwise(y_ctrl, y_def)["lpips"]),
                    })
                    print(f"  st={st:.2f} {cond:<14} {img} seed{s}  "
                          f"effect={rows[-1]['effect_siglip']:+.4f}  "
                          f"lpips(def,orig)={fid['lpips']:.4f}", flush=True)
                    del y_def
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del ctrls
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = args.out or (batch_dir / "native_strength.csv")
    executors.write_csv(out, rows)
    print(f"\n寫出 {out}（{len(rows)} 列）")

    import statistics
    if len(strengths) > 1:
        print("\n=== effect_siglip 隨 strength（正 = 防禦有效）===")
        print(f"{'條件':<16}" + "".join(f"{st:>11.2f}" for st in strengths)
              + f"{'原生LPIPS':>12}")
        for c in conds:
            if c not in {r["condition"] for r in rows}:
                continue
            line = f"{c:<16}"
            for st in strengths:
                v = [r["effect_siglip"] for r in rows
                     if r["condition"] == c and r["strength"] == st]
                line += f"{statistics.fmean(v):>+11.4f}" if v else f"{'-':>11}"
            fl = [r["fid_lpips"] for r in rows if r["condition"] == c]
            line += f"{statistics.fmean(fl):>12.4f}"
            print(line)

        print("\n=== 攻擊方在該 strength 上開出的區間（siglip_edit − siglip_orig）===")
        for st in strengths:
            v = [r["siglip_ctrl"] - r["siglip_orig"] for r in rows
                 if r["strength"] == st]
            if v:
                print(f"  strength={st:.2f}  gap={statistics.fmean(v):+.4f}")

    print("\n=== 逐條件的 effect_siglip ===")
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
