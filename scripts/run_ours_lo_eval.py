"""把本專案的殘差模塊放進 Lo et al. 的評測路徑，取得可與基準並排的數字。

    python scripts/run_ours_lo_eval.py --data data/lo_aligned \
        --out runs/ours_lo_PF --sites PF --eval_seeds 20

為什麼不直接用 run_defense.py
─────────────────────────────────────────────────────────────────────

`run_defense.py` 有自己的評測路徑（淨化掃描、對齊階段、泛化掃描），且

1. 它讀 `prompts.yaml` 的格式是 `{類別: [prompt, ...]}`，而 `data/lo_aligned`
   是 `{類別: {content: ..., prompts: [...]}}`。`list(dict)` 會把 `content`
   與 `prompts` 兩個**鍵名**當成 prompt 餵進去，沒有任何症狀。
2. 它每格只評一個噪聲種子。基準是 20 個種子的平均，SDEdit 對種子高度敏感，
   n=1 對 n=20 的比較讀不出東西——那正是 E29 之前一連串判定問題的來源。

本腳本改為直接沿用 `run_lo_baseline` 的 `reference_edits` 與 `evaluate`，
確保「未防禦的編輯」與「防禦後的編輯」兩端的取樣、種子偏移、步數、guidance
與基準逐字相同。只有產生 `x_def` 的那一步不同，那正是要比較的東西。

失真並未匹配
─────────────────────────────────────────────────────────────────────

本專案以 LPIPS 綁定約束（τ_lpips），基準以 L∞ ≤ κ 硬投影。實測基準在
κ = 0.06 上的擾動 LPIPS 為 0.49–0.54，是本專案 τ = 0.10 的 4–6 倍。
**兩者的數字因此不是同一條軸上的點**，本腳本不做匹配、也不假裝有匹配：
`pert_lpips` 與 `pert_linf` 逐格寫出，讀表時必須併看。匹配失真的掃描是
後續工作（見規格 §7）。

超參數
─────────────────────────────────────────────────────────────────────

沿用 `runs/e29c_P_tau0.10/env.json`：本專案唯一在有效威脅模型（攻擊方
guidance = 7.5）下校準過的一組。唯一的更動是 strength 0.5 → 0.3，理由是
必須與 `run_lo_baseline` 的評測設定相同，否則量到的是兩個不同的攻擊。
"""

import argparse
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defense.linf_attack import LinfAttackConfig  # noqa: E402
from src.defense.objective import LossConfig  # noqa: E402
from src.defense.optimize import OptimConfig, optimize  # noqa: E402
from src.purify.ops import default_train_set  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image, save_json  # noqa: E402
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory  # noqa: E402
from scripts.run_defense import build_module  # noqa: E402
from scripts.run_lo_baseline import (  # noqa: E402
    TABLE1,
    append_csv,
    completed_pairs,
    evaluate,
    load_dataset,
    reference_edits,
)


def main():
    ap = argparse.ArgumentParser(
        description="本專案的殘差模塊，走 Lo et al. 的評測路徑"
    )
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--sites", default="PF",
                    help="逗號分隔。P 加性低秩／PF 加性全秩／S 空間變形／C 色度")
    ap.add_argument("--rank", type=int, default=16,
                    help="site S 與 C 會把它重新解讀為控制網格邊長")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt_index", type=int, default=0)
    # ---- 沿用 e29c 的校準值 ----
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tau_lpips", type=float, default=0.10)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--defense_mode", default="untargeted",
                    choices=["untargeted", "targeted"])
    ap.add_argument("--target_image", default="",
                    help="有目標模式的目標影像；基準的兩個 PhotoGuard 變體用灰圖")
    # site S 專用。預設值改為 bicubic：OptimConfig 的預設 bilinear 只為了讓
    # E13–E19 可重現，E20 §5.2 量出 bicubic 把銳利度保留率由 85.0% 拉到
    # 99.9%，e21 的實測也顯示同一 τ 下 bicubic 的編輯 LPIPS 0.2473 對
    # bilinear 的 0.0931。拿 bilinear 跑等於自願讓非加性臂變弱。
    ap.add_argument("--warp_resample", default="bicubic",
                    choices=["bilinear", "bicubic"])
    ap.add_argument("--warp_max_disp", type=float, default=1.5,
                    help="site S 的位移硬上界，單位像素。與 tau_lpips 併列"
                         "記錄：本位置的失真預算是位移量而非 L∞")
    # ---- 必須與 run_lo_baseline 相同，否則量到的是另一個攻擊 ----
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--eval_seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.defense_mode == "targeted" and not args.target_image:
        raise SystemExit(
            "--defense_mode targeted 需要 --target_image。"
            "有目標模式的損失是 d(編輯結果, 目標)，沒有目標就沒有目標函數"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)

    data = load_dataset(Path(args.data), args.size, device, args.limit)
    if not data:
        raise SystemExit(f"{args.data} 底下沒有任何 PNG")

    y_target = None
    if args.target_image:
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(args.target_image).convert("RGB").resize(
            (args.size, args.size), Image.LANCZOS)
        y_target = T.ToTensor()(img).unsqueeze(0).to(device)

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    purifiers = default_train_set()
    res_path, sum_path = out / "results.csv", out / "summary.csv"
    done = completed_pairs(sum_path)
    if done and not args.resume:
        raise SystemExit(
            f"{sum_path} 已有 {len(done)} 格結果。要接續請加 --resume；"
            "要重跑請換一個 --out。此處不自動覆寫"
        )
    save_json(vars(args), out / "protocol.json")
    if done:
        print(f"接續模式：已完成 {len(done)} 格，將略過", flush=True)

    for name, x01, prompts, content in data:
        prompt = prompts[args.prompt_index]
        todo = [s for s in sites if (name, s) not in done]
        if not todo:
            print(f"=== {name}：全部 site 已完成，略過 ===", flush=True)
            continue

        # 評測用的設定物件。與 run_lo_baseline 的欄位逐一對應——這是兩邊
        # 數字可比的唯一依據，改動任何一項都會讓比較失效。
        eval_cfg = LinfAttackConfig(
            strength=args.strength, guidance_scale=args.guidance,
            n_edit=args.n_edit, prompt_edit=prompt, seed=args.seed,
        )
        t_ref = time.perf_counter()
        refs = reference_edits(sd, x01, prompt, eval_cfg, args.eval_seeds)
        print(f"\n=== {name}：{len(refs)} 個種子的未防禦編輯，"
              f"{time.perf_counter()-t_ref:.0f}s ===", flush=True)

        for site in todo:
            cfg = OptimConfig(
                steps=args.steps, lr=args.lr, n_edit=args.n_edit,
                strength=args.strength, guidance_scale=args.guidance,
                prompt_edit=prompt, seed=args.seed,
                stop_on_plateau=True,
                warp_resample=args.warp_resample,
                warp_max_disp=args.warp_max_disp,
            )
            loss_cfg = LossConfig(
                margin=args.margin, defense_mode=args.defense_mode,
                tau_lpips=args.tau_lpips,
            )
            print(f"\n=== {name} / site {site} / τ_lpips={args.tau_lpips} / "
                  f"{prompt!r} ===", flush=True)
            reset_peak_memory()
            module = build_module(
                site, args.rank, cfg, sd, args.size, args.seed).to(device)
            try:
                res = optimize(sd, module, x01, cfg, loss_cfg, purifiers,
                               y_target=y_target)
            finally:
                # site W 把 hook 掛在共用的 UNet 上，不卸除會滲進下一格，
                # 症狀是「另一個 site 的結果莫名被改動」。
                if hasattr(module, "remove"):
                    module.remove()

            x_def = res.x_def.detach().clamp(0, 1)
            save_image(x_def, out / f"{name}__{site}__def.png")
            save_json(res.history, out / f"{name}__{site}__history.json")

            pert = suite.pairwise(x01, x_def)
            ev = evaluate(sd, suite, refs, x_def, prompt, eval_cfg,
                          out, f"{name}__{site}")
            rows = [{
                "image": name, "attack": site, "content": content,
                "prompt": prompt, "tau_lpips": args.tau_lpips,
                "steps": args.steps, "steps_done": res.steps_done,
                "stop_reason": res.stop_reason,
                "defense_mode": args.defense_mode, "rank": args.rank,
                # site S 的失真預算是位移量，不是 L∞ 或 τ_lpips，不記就
                # 無從得知該格實際被綁在哪裡
                "warp_resample": args.warp_resample,
                "warp_max_disp": args.warp_max_disp,
                "strength": args.strength, "guidance_scale": args.guidance,
                "attack_seconds": round(res.seconds, 1),
                "peak_mb": round(peak_memory_mb(), 1),
                **{f"pert_{k}": v for k, v in pert.items()},
                **r,
            } for r in ev]
            n = len(ev)
            srow = {
                "image": name, "attack": site, "n_seeds": n,
                **{f"edit_{k}": sum(r[f"edit_{k}"] for r in ev) / n
                   for k in TABLE1},
                "pert_linf": pert["linf"], "pert_lpips": pert["lpips"],
                "steps_done": res.steps_done,
                "seconds": round(res.seconds, 1),
                "peak_mb": round(peak_memory_mb(), 1),
            }
            append_csv(res_path, rows)
            append_csv(sum_path, [srow])
            print(f"  [{site}] 擾動 LPIPS {pert['lpips']:.4f} / L∞ "
                  f"{pert['linf']:.4f}  →  編輯 LPIPS "
                  f"{srow['edit_lpips']:.4f}  PSNR {srow['edit_psnr']:.2f}"
                  f"  （{res.steps_done} 步，{res.seconds:.0f}s）", flush=True)

    print(f"\n完成。{sum_path}", flush=True)


if __name__ == "__main__":
    main()
