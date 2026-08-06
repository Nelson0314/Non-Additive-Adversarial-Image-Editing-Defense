"""實驗主驅動 — `docs/RUNBOOK_2026-08-05.md` §3。

    python scripts/run_stage.py <段> --batch b1 [選項]

段：`calib` / `train` / `rayscale` / `eval` / `report`，依序執行。

## 三層的分工

| 層 | 位置 | 職責 |
|---|---|---|
| 格點 | `src/experiment/grid.py` | 哪些格存在、哪些結構上不適用 |
| 骨架 | `src/experiment/runner.py` | 哪些格要跑、跑過了沒有、失敗怎麼記 |
| 計算 | `src/experiment/executors.py` | 實際的訓練、縮放、評測、彙整 |
| 驅動 | 本檔 | 把三者接起來，並決定每段的前置條件 |

前三層各有測試。本檔的責任只有「接線」，故它刻意不含任何計算邏輯——
任何在這裡出現的數值處理，都是一段沒有測試涵蓋的程式。

## 段與段之間的前置條件

- `train` 需要 `calib/calibration.json`（學習率只有校準表一個入口）
- `rayscale` 需要段 1 的 `phi.pt`
- `eval` 需要段 2 的 `phi_tau{τ}.pt`，**且需要 φ=0 對照**。對照是
  `grid.control_cells()` 的格，跨 9 個條件共用，故它與 `eval` 一起跑：
  本檔先跑完 `control` 再跑 `eval`，順序由此保證。
- `report` 需要 `_cells/` 裡的 eval 紀錄

缺前置條件時executor 會以 `FileNotFoundError` 指出缺哪一個檔，不會靜默跳過。

## 為什麼 --dry-run 值得單獨存在

它在**耗掉任何機時之前**回答「這次會跑多久、續跑判定有沒有生效」。
沒有它，這兩個問題要等跑完才知道，而雲端容器會被刪除、實驗無法重跑。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import executors, grid                    # noqa: E402
from src.experiment.runner import plan_report, run_stage      # noqa: E402
from src.utils.progress import ProgressWriter                 # noqa: E402

STAGES = ("calib", "train", "rayscale", "eval", "report")

# `control`（φ=0 的同淨化對照）不是獨立的 CLI 段——它與 eval 一起跑，
# 因為兩者共用同一批淨化後的影像。但它**必須出現在乾跑報告裡**：
# 它有 300 格（N=3 時），漏掉會讓「這次要跑多久」少算一截。
REPORTED = ("calib", "train", "rayscale", "control", "eval", "report")

# 精度旗標與 torch dtype 的對應。SDWrapper 會依 `resolve_precision` 決定
# VAE 要不要留在 fp32（fp16 下不留會讓 SDXL 的 VAE 溢位成全黑圖）。
PRECISION = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def build_env(args) -> dict:
    """`env.json` 的內容。卡別與精度是必填——它們進 `config_hash`，
    是「兩張卡不可混跑」的程式化保證。"""
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    return {
        "gpu": args.gpu_tag,
        "precision": args.precision,
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "commit": commit,
        "spec_version": args.spec_version,
        "model": args.model,
        "resolution": args.resolution,
    }


def run_config(args) -> executors.RunConfig:
    """CLI → 計算層設定。這裡不做任何預設值的推導：每一項都有明確來源。"""
    return executors.RunConfig(
        resolution=args.resolution,
        guidance=args.guidance,
        strength=args.strength,
        steps=args.steps,
        seed=args.seed,
        train_n_edit=args.train_n_edit,
        n_eot=args.n_eot,
        k_inv=args.k_inv,
        t_max=args.t_max,
        exact_inversion=args.exact_inversion,
        purify_mode=args.purify_mode,
        max_steps=args.max_steps,
        align_steps=args.align_steps,
        stop_patience=args.stop_patience,
        stop_min_steps=args.stop_min_steps,
        attn_timesteps=args.attn_timesteps,
        shared_tokens=tuple(args.shared_tokens),
        tau_train=args.tau_train,
        tau_acut=args.tau_acut,
        tau_chroma=args.tau_chroma,
        beta_linf=args.beta_linf,
        tau_linf=args.tau_linf,
        warp_grid_size=args.warp_grid_size,
        warp_max_disp=args.warp_max_disp,
        warp_resample=args.warp_resample,
        apa_lora_rank=args.apa_lora_rank,
        apa_latent_max_rank=args.apa_latent_max_rank,
        apa_latent_const_rank=args.apa_latent_const_rank,
        random_init_std=args.random_init_std,
        lr_grid=tuple(float(v) for v in args.lr_grid.split(",")),
        probe_steps=args.probe_steps,
        edit_effect_threshold=args.edit_effect_threshold,
        target_image=args.target_image,
        mist_target=args.mist_target,
        diffpure_ckpt=args.diffpure_ckpt,
    )


def base_config(args) -> dict:
    """整批共用的設定。格點自己的四個軸由 `cell_config` 疊上去。

    三份參數字典都不是空的：凡是會改變數值結果的計算層旋鈕都放進去，
    使「改了設定卻沿用舊結果」在雜湊層就被擋下。漏掉任何一項的症狀是
    完全沒有症狀——輸出仍是一張合理的圖，只是它不是這次設定跑出來的。
    `module_params` 承載參數化容量（控制點數、LoRA 秩），
    `optim_params` 承載最佳化旋鈕（步數、停止準則）。
    """
    cfg = run_config(args)
    return {
        "spec_version": args.spec_version,
        "model": args.model,
        "resolution": args.resolution,
        "guidance": args.guidance,
        "steps": args.steps,
        "strength": args.strength,
        "gpu": args.gpu_tag,
        "precision": args.precision,
        "loss_params": cfg.loss_params(),
        "module_params": cfg.module_params(),
        "optim_params": cfg.optim_params(),
        # 學習率由校準表決定，不由 CLI 給，故此處恆為 None。實際採用的值
        # 寫進每格的 `meta.json`（`lr` 欄），使事後查得到。
        "lr": None,
    }


def load_entries(args, device) -> list:
    """回傳 `ImageEntry` 清單。

    本輪用 `data/lo_aligned/`（25 張 CC0 真實照片）而非 PIE-Bench：遠端機器
    連不上 HuggingFace，取不到後者。`n` 是樣本數的唯一入口，`--images`
    明給時覆蓋它。
    """
    return executors.load_lo_aligned(
        args.data, args.resolution, device,
        ids=args.images, n=(None if args.images else args.n), seed=args.seed,
    )


def build_resources(args, batch_dir: Path, load_model: bool = True
                    ) -> executors.Resources:
    """載入權重與指標模型，組出跨格共用的 `Resources`。

    校準表在此**盡力載入**：段 0 本身要產生它，故不存在時不視為錯誤；
    但段 1 之後任何一次取學習率都會經 `Resources.require_calib()` 拋出。
    這與「沒有校準表就用預設值」是兩件事——後者才是本專案要消滅的路徑。

    `load_model=False` 供段 4：它只讀 `_cells/` 的逐格紀錄，載入 SDXL 是
    數分鐘的純浪費。此時 `sd` 與影像都是空的，任何需要它們的路徑都會以
    `AttributeError`／`KeyError` 當場失敗，不會靜默算出一個沒有模型的結果。
    """
    import torch

    from src.metrics.suite import MetricSuite
    from src.models.sd import SDWrapper, SDXLWrapper
    from src.utils.calibration import Calibration

    if not load_model:
        return executors.Resources(
            sd=None, suite=None, batch_dir=batch_dir,
            base_config=base_config(args), cfg=run_config(args),
        )

    dtype = getattr(torch, PRECISION[args.precision])
    wrapper = SDXLWrapper if args.wrapper == "sdxl" else SDWrapper
    if args.wrapper == "auto":
        wrapper = SDXLWrapper if "xl" in args.model.lower() else SDWrapper
    print(f"[env] 載入 {wrapper.__name__}({args.model}) dtype={dtype}",
          flush=True)
    sd = wrapper(args.model, dtype=dtype)
    suite = MetricSuite(device=sd.device)

    entries = load_entries(args, sd.device)
    print(f"[env] 影像 {len(entries)} 張：{[e.image_id for e in entries]}",
          flush=True)

    calib_path = batch_dir / "calib" / "calibration.json"
    calib = Calibration.load(calib_path) if calib_path.exists() else None

    y_target = None
    if args.target_image:
        y_target = executors.load_image_tensor(
            Path(args.target_image), sd.device)
        if y_target.shape[-1] != args.resolution:
            import torch.nn.functional as F

            y_target = F.interpolate(
                y_target, size=(args.resolution, args.resolution),
                mode="bicubic", antialias=True).clamp(0, 1)

    return executors.Resources(
        sd=sd, suite=suite, batch_dir=batch_dir,
        base_config=base_config(args), cfg=run_config(args),
        images={e.image_id: e for e in entries},
        calib=calib, y_target=y_target,
    )


def load_images(args) -> list:
    """乾跑與格點列舉用的影像 id 清單。**不載入影像本身。**

    乾跑必須在載入 SDXL 之前就能回答「這次要跑多久」，故此處只讀
    `prompts.yaml` 的目錄結構。
    """
    if args.images:
        return list(args.images)
    import yaml

    root = Path(args.data)
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    by_group = {cls: sorted(p.stem for p in (root / cls).glob("*.png"))
                for cls in sorted(spec)}
    picked, round_idx = [], 0
    while len(picked) < args.n:
        progressed = False
        for cls in sorted(by_group):
            if len(picked) >= args.n:
                break
            if round_idx < len(by_group[cls]):
                picked.append(by_group[cls][round_idx])
                progressed = True
        if not progressed:
            raise ValueError(
                f"{root} 只有 {len(picked)} 張影像，少於要求的 n={args.n}")
        round_idx += 1
    return picked


def _print_warnings(warns) -> None:
    if not warns:
        return
    print("\n[preflight] 以下事項會影響本批的結果，請先處理：", file=sys.stderr)
    for w in warns:
        print(f"  - {w}", file=sys.stderr)
    print("", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WACV 實驗主驅動")
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--batch", required=True, help="批次名，例 b1")
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))

    # 卡別與精度必填：它們進 config_hash，換卡會使全部格點視為未完成
    ap.add_argument("--gpu-tag", required=True,
                    help="例 Tesla V100-SXM2-32GB 或 RTX-5090")
    ap.add_argument("--precision", required=True,
                    choices=["fp32", "fp16", "bf16"])

    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--wrapper", default="auto", choices=["auto", "sd", "sdxl"],
                    help="auto 依 model 名稱含不含 xl 判斷")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50, help="攻擊方的去噪步數")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--spec-version", type=int, default=1)

    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--n", type=int, default=3, help="樣本數的唯一入口")
    ap.add_argument("--images", nargs="*", help="直接指定影像 id，覆蓋 --n")
    ap.add_argument("--seed", type=int, default=20260805)

    # ---- 防禦訓練 ----
    g = ap.add_argument_group("訓練")
    g.add_argument("--train-n-edit", type=int, default=10,
                   help="訓練期代理編輯鏈的步數，與評測期的 --steps 分開")
    g.add_argument("--n-eot", type=int, default=1)
    g.add_argument("--k-inv", type=int, default=10)
    g.add_argument("--t-max", type=int, default=None,
                   help="inversion 的 timestep 上限，依段 0 的重建誤差量測指定")
    g.add_argument("--exact-inversion", action="store_true",
                   help="以 BDIA 取代 DDIM inversion，只影響走生成路徑的條件")
    g.add_argument("--purify-mode", default="all", choices=["rotate", "all"])
    g.add_argument("--max-steps", type=int, default=250,
                   help="N1／N2／N3 階段二的步數上限（開平台停止時）")
    g.add_argument("--align-steps", type=int, default=200,
                   help="N3 階段一（LoRA 保真對齊）的步數，APA 官方實作為 200")
    g.add_argument("--stop-patience", type=int, default=20)
    g.add_argument("--stop-min-steps", type=int, default=25)
    g.add_argument("--attn-timesteps", type=int, default=4)
    g.add_argument("--shared-tokens", type=int, nargs="+", default=[0],
                   help="N1 要把注意力質量導向的 token 格。預設 0（BOS）；"
                        "SDXL 上 BOS 的質量實測僅 7e-06，見 RUNBOOK §3")

    # ---- 保真約束 ----
    g = ap.add_argument_group("保真約束")
    g.add_argument("--tau-train", type=float, default=grid.TRAIN_TAU,
                   help="訓練所在的失真預算，其餘 τ 由段 2 的射線縮放取得")
    g.add_argument("--tau-acut", type=float,
                   default=executors.RunConfig.tau_acut,
                   help="鈍化 hinge 的門檻。預設值是在 τ_lpips=0.05 上由人眼"
                        "判讀定出的絕對值，本輪訓練在 0.35，須重新判讀")
    g.add_argument("--tau-chroma", type=float,
                   default=executors.RunConfig.tau_chroma,
                   help="色度偏壓 hinge 的門檻。同上，須重新判讀")
    g.add_argument("--beta-linf", type=float,
                   default=executors.RunConfig.beta_linf,
                   help="L∞ hinge 的係數。預設 0：L∞ 對非加性參數化不具鑑別力，"
                        "開著它綁定約束就不是 τ_LPIPS 而是 L∞")
    g.add_argument("--tau-linf", type=float,
                   default=executors.RunConfig.tau_linf)

    # ---- 參數化 ----
    g = ap.add_argument_group("參數化")
    g.add_argument("--warp-grid-size", type=int, default=32)
    g.add_argument("--warp-max-disp", type=float, default=1.5,
                   help="位移場的硬上界（像素）。段 0 的 warp_reach.csv 會量出"
                        "該上界下可達的最大 LPIPS，低於 --tau-train 時段 2 會拋出")
    g.add_argument("--warp-resample", default="bicubic",
                   choices=["bilinear", "bicubic"])
    g.add_argument("--apa-lora-rank", type=int, default=8)
    g.add_argument("--apa-latent-max-rank", type=int, default=32)
    g.add_argument("--apa-latent-const-rank", type=int, default=8)
    g.add_argument("--random-init-std", type=float, default=0.5,
                   help="R（同失真隨機對照）的位移場初始標準差")

    # ---- 段 0 ----
    g = ap.add_argument_group("段 0 校準")
    g.add_argument("--lr-grid", default="1e-4,1e-3,5e-3,2e-2,1e-1")
    g.add_argument("--probe-steps", type=int, default=12)
    g.add_argument("--edit-effect-threshold", type=float, default=0.0,
                   help="SigLIP(編輯,target) − SigLIP(原圖,target) 的下限")

    # ---- 外部檔案 ----
    g = ap.add_argument_group("外部檔案")
    g.add_argument("--target-image", default="data/targets/gray.png",
                   help="targeted 模式的目標影像（N2 取 LPIPS、N3 取 MSE）")
    g.add_argument("--mist-target", default="",
                   help="Mist 的 MIST.png。缺少時該條件的每一格都會明確失敗")
    g.add_argument("--diffpure-ckpt", default="")

    ap.add_argument("--dry-run", action="store_true",
                    help="只列出會跑哪些格，不執行也不寫入")
    ap.add_argument("--force", action="store_true",
                    help="忽略續跑判定重跑全部格子")
    args = ap.parse_args(argv)

    batch_dir = args.runs_root / args.batch
    images = load_images(args)
    plan = grid.plan(images, n_seeds=grid.N_SEEDS)

    if args.dry_run:
        # 乾跑不取寫入鎖：它是唯讀的，且可能與正在跑的批次並存
        w = ProgressWriter(batch_dir, env=build_env(args), take_lock=False)
        rep = plan_report(plan, w, base_config(args))
        print(f"batch {args.batch}   影像 {len(images)}   條件 {len(grid.CONDITIONS)}")
        print(f"{'stage':<10}{'todo':>8}{'resumable':>11}{'skipped':>9}{'total':>8}")
        total_todo = 0
        for st in REPORTED:
            r = rep.get(st)
            if r:
                total_todo += r["todo"]
                print(f"{st:<10}{r['todo']:>8}{r['resumable']:>11}"
                      f"{r['skipped']:>9}{r['total']:>8}")
        print(f"{'合計':<10}{total_todo:>8}")
        if grid.EXCLUDED:
            print("\n未納入本輪的方法：")
            for name, reason in grid.EXCLUDED.items():
                print(f"  {name}: {reason[:70]}…")
        return 0

    with ProgressWriter(batch_dir, env=build_env(args)) as w:
        (batch_dir / "env.json").write_text(
            json.dumps(build_env(args), indent=2, ensure_ascii=False),
            encoding="utf-8")
        # 段 4 只讀 `_cells/` 的逐格紀錄，載入 SDXL 是數分鐘的純浪費。
        res = build_resources(args, batch_dir,
                              load_model=(args.stage != "report"))
        ctx = {"res": res}

        # 段 0 與段 4 沒有格點：`grid.plan()` 不列它們，硬塞進格點框架只會
        # 得到一個「零格、永遠成功」的段。
        if args.stage == "calib":
            out = executors.run_calibration(res)
            print(f"\n[calib] 校準表寫入 {out['path']}")
            print(json.dumps(out["summary"], indent=2, ensure_ascii=False,
                             default=str))
            return 0
        if args.stage == "report":
            out = executors.run_report(res)
            print(f"\n[report] {out['path']}（{out['n_rows']} 列）")
            return 0

        _print_warnings(executors.preflight(res))
        executor = executors.make_executor(args.stage)
        failed = 0

        # φ=0 對照與 eval 一起跑：兩者共用同一批淨化後的影像，而對照
        # 跨 9 個條件共用，各條件各算一次就是 9 倍的重複計算。
        if args.stage == "eval":
            ctrl = executors.annotate_unavailable(plan["control"], res)
            cres = run_stage("control", ctrl, executors.make_executor("control"),
                             w, base_config(args), ctx=ctx, force=args.force)
            print(f"\n[control] done={cres.done} failed={cres.failed} "
                  f"skipped={cres.skipped} resumed={cres.resumed}")
            if cres.aborted:
                print(cres.abort_reason, file=sys.stderr)
                return 3
            failed += cres.failed

        cells = plan.get(args.stage, [])
        if args.stage == "eval":
            cells = executors.annotate_unavailable(cells, res)
        res_stage = run_stage(args.stage, cells, executor, w,
                              base_config(args), ctx=ctx, force=args.force)
        print(f"\n[{res_stage.stage}] done={res_stage.done} "
              f"failed={res_stage.failed} skipped={res_stage.skipped} "
              f"resumed={res_stage.resumed}")
        if res_stage.aborted:
            print(res_stage.abort_reason, file=sys.stderr)
            return 3
        return 1 if (failed + res_stage.failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
