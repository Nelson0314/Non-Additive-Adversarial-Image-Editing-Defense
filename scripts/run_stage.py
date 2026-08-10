"""實驗主驅動 — `docs/RUNBOOK.md` §3。

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

from src.defense import objective                             # noqa: E402
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
    """CLI → 計算層設定。這裡不做任何預設值的推導：每一項都有明確來源。

    `tau_acut` / `tau_chroma` 的比例導出在 `main` 完成（要印出導出的值），
    故到這裡時它們必為具體數值。仍在此擋一次 `None`：讓它流下去的話
    `loss_params` 會把 `None` 寫進 `config_hash`，而那是一個合法的雜湊，
    整批跑完才會在 hinge 的比較上以 TypeError 出現。
    """
    missing = [k for k in ("tau_acut", "tau_chroma")
               if getattr(args, k) is None]
    if missing:
        raise ValueError(
            f"{missing} 尚未解析；呼叫 run_config 之前必須先做比例導出"
            "（run_stage.main 的 [thresholds] 那一段）"
        )
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
        attn_mask_tau=args.attn_mask_tau,
        attn_mask_timesteps=args.attn_mask_timesteps,
        shared_tokens=tuple(args.shared_tokens),
        conditions=grid.resolve_conditions(args.conditions),
        tau_plan=tau_plan_of(args),
        tau_train=args.tau_train,
        tau_acut=args.tau_acut,
        tau_chroma=args.tau_chroma,
        beta_linf=args.beta_linf,
        tau_linf=args.tau_linf,
        warp_grid_size=args.warp_grid_size,
        warp_max_disp=args.warp_max_disp,
        warp_resample=args.warp_resample,
        warp_mask_gate=args.warp_mask_gate,
        apa_lora_rank=args.apa_lora_rank,
        apa_latent_max_rank=args.apa_latent_max_rank,
        apa_latent_const_rank=args.apa_latent_const_rank,
        random_init_std=args.random_init_std,
        recon=args.recon,
        recon_objective=args.recon_objective,
        recon_a1_steps=args.recon_a1_steps,
        recon_a1_lr=args.recon_a1_lr,
        recon_a2_steps=args.recon_a2_steps,
        recon_a2_lr=args.recon_a2_lr,
        recon_floor_ratio=args.recon_floor_ratio,
        recon_gamma_acut=args.recon_gamma_acut,
        recon_acut_band=args.recon_acut_band,
        recon_w_pixel=args.recon_w_pixel,
        recon_resp_scale=args.recon_resp_scale,
        lr_grid=tuple(float(v) for v in args.lr_grid.split(",")),
        probe_steps=args.probe_steps,
        edit_effect_threshold=args.edit_effect_threshold,
        target_image=args.target_image,
        mist_target=args.mist_target,
        diffpure_ckpt=args.diffpure_ckpt,
    )


def tau_plan_of(args) -> grid.TauPlan:
    """本批的失真預算軸。**唯一的導出點。**

    `run_config`（進 `RunConfig.tau_plan`，段 2 的射線縮放讀它）與 `main`
    （列舉格點）各自呼叫 `grid.tau_plan_for` 的話，加一個模式就得改兩處，
    而漏改的症狀是格點按相對軸列舉、縮放卻按絕對軸求解，兩邊都不報錯。
    """
    if args.budget_delta is None:
        return grid.tau_plan_for(args.tau_train, args.full_purify_taus)
    if args.full_purify_taus is not None:
        raise SystemExit(
            "--budget-delta 與 --full-purify-taus 不可並用："
            "相對軸上只有一個預算點，完整淨化組必然就在該點")
    return grid.budget_tau_plan(args.budget_delta, args.budget_metric)


def base_config(args) -> dict:
    """整批共用的設定。格點自己的四個軸由 `cell_config` 疊上去。

    三份參數字典都不是空的：凡是會改變數值結果的計算層旋鈕都放進去，
    使「改了設定卻沿用舊結果」在雜湊層就被擋下。漏掉任何一項的症狀是
    完全沒有症狀——輸出仍是一張合理的圖，只是它不是這次設定跑出來的。
    `module_params` 承載參數化容量（控制點數、LoRA 秩），
    `optim_params` 承載最佳化旋鈕（步數、停止準則）。
    """
    cfg = run_config(args)
    out = {
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
    # 遮罩設定**只在 inpainting 威脅模型下出現**。
    #
    # `config_hash` 吃的是整個 dict，多一個鍵就會改變**每一格**的雜湊。無條件
    # 加入的話，img2img 的既有批次一旦續跑就會把已完成的格全部判為未完成——
    # 2026-08-07 當下正有三個分片在跑，其中一個還剩約五小時。故此鍵的存在
    # 與否本身就承載「這批是哪一種威脅模型」，而 img2img 的雜湊逐位不變。
    if args.masks:
        # 遮罩的**內容**進雜湊，不只是目錄名：換一張遮罩就是換一個攻擊，
        # 舊結果不可沿用。取整組摘要而非逐影像，故改一張會使全部格的雜湊
        # 改變——刻意的保守方向，寧可多重跑，不可靜默沿用（`masks_digest`）。
        from src.data.masks import mask_files, masks_digest

        out["masks"] = {"dir": args.masks.name,
                        "digest": masks_digest(mask_files(args.masks))}
    # 攻擊 prompt 換一個就是換一個攻擊，**必須**進雜湊，否則續跑會把用另一個
    # prompt 跑出來的格判為完成。與 `mask` 同一慣例：只在非預設值時出現，
    # 使 img2img 既有批次的每一格雜湊逐位不變。
    if args.prompt_index:
        out["prompt_index"] = args.prompt_index
    # 預算軸的定義換了，同一個 τ 標籤就不是同一個失真量。與 `masks`、
    # `prompt_index` 同一慣例：只在非預設（即相對模式）時出現，故絕對模式的
    # 既有批次雜湊逐位不變。
    if args.budget_delta is not None:
        out["budget"] = {"metric": args.budget_metric,
                         "delta": args.budget_delta, "relative": True}
    return out


def resolve_thresholds(args, verbose: bool = True) -> None:
    """把兩道 hinge 的門檻就地填成具體數值。**呼叫 `run_config` 之前必做。**

    不給時依 τ_train 等比例導出（2026-08-08 處置 A，見 `objective` 的
    「門檻的適用範圍」）。導出的是具體數值，之後照常進 `loss_params` 與
    `config_hash`；印出來使命令列沒寫的那兩個數在 log 上仍查得到。

    與 `build_parser` 同一個理由抽成函式：`scripts/tau_preview.py` 也要
    `build_resources`，而後者會經 `run_config`，那裡對 `None` 是硬拋的。
    抄一份導出規則出去，兩份就會分岔。
    """
    derived = objective.scaled_thresholds(args.tau_train)
    for key in ("tau_acut", "tau_chroma"):
        if getattr(args, key) is None:
            setattr(args, key, derived[key])
            if verbose:
                print(f"[thresholds] {key} = {derived[key]:.4g} "
                      f"（由 τ_train={args.tau_train} 依比例導出）", flush=True)


def load_entries(args, device) -> list:
    """回傳 `ImageEntry` 清單。

    本輪用 `data/lo_aligned/`（25 張 CC0 真實照片）而非 PIE-Bench：遠端機器
    連不上 HuggingFace，取不到後者。`n` 是樣本數的唯一入口，`--images`
    明給時覆蓋它。
    """
    return executors.load_lo_aligned(
        args.data, args.resolution, device,
        ids=args.images, n=(None if args.images else args.n), seed=args.seed,
        prompt_index=args.prompt_index,
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
    from src.models.sd import SDInpaintWrapper, SDWrapper, SDXLWrapper
    from src.utils.calibration import Calibration

    if not load_model:
        return executors.Resources(
            sd=None, suite=None, batch_dir=batch_dir,
            base_config=base_config(args), cfg=run_config(args),
        )

    dtype = getattr(torch, PRECISION[args.precision])
    wrapper = {"sdxl": SDXLWrapper, "sd_inpaint": SDInpaintWrapper,
               "sd": SDWrapper}.get(args.wrapper)
    if wrapper is None:                                   # auto
        name = args.model.lower()
        wrapper = (SDInpaintWrapper if "inpaint" in name
                   else SDXLWrapper if "xl" in name else SDWrapper)
    print(f"[env] 載入 {wrapper.__name__}({args.model}) dtype={dtype}",
          flush=True)
    sd = wrapper(args.model, dtype=dtype)
    suite = MetricSuite(device=sd.device)

    entries = load_entries(args, sd.device)
    print(f"[env] 影像 {len(entries)} 張：{[e.image_id for e in entries]}",
          flush=True)

    # 威脅模型與載入的權重必須一致。兩個方向都要擋：
    #
    # - 給了 `--masks` 卻載入一般權重 → `SDWrapper.edit` 會在第一格拋出，
    #   但那時已經載完模型、跑完段 0 的一部分。此處提前擋掉。
    # - 載入 inpainting 權重卻沒給 `--masks` → `edit` 會因為缺遮罩拋出。
    #   同樣提前擋。
    if bool(args.masks) != bool(sd.is_inpainting):
        raise SystemExit(
            f"威脅模型與權重不一致：--masks={args.masks!r} 而 "
            f"{args.model} 的 UNet in_channels="
            f"{sd.unet.config.in_channels}。inpainting 兩者都要給，"
            "img2img 兩者都不要給")

    if args.masks:
        # 遮罩是**攻擊方的設定**（他要重畫哪一塊），由人工繪製，逐影像一張
        # PNG（`scripts/draw_masks.py`，DEC-010）。此處只負責載入與驗證，
        # 不產生任何遮罩——缺檔即拋出，不落回自動產生的版本。
        from dataclasses import replace as dc_replace

        from src.data.masks import load_drawn_mask

        # `ImageEntry` 是 frozen 的（刻意：影像與 prompt 在批次中途被改掉，
        # 症狀是某幾格用了別的輸入而報表看不出來），故產生新的 entry 而非
        # 就地賦值。
        # 遮罩落盤存證到批次目錄。它決定攻擊方能改哪一塊，故是**結果的一
        # 部分**而不是中間狀態：涵蓋率不同，同一個防禦的效果就不同。
        # `runs/` 是唯一的證據來源（`CLAUDE.md`），而 `data/` 那份日後可能
        # 被重畫，兩者必須各存一份才對得起帳。
        mask_dir = batch_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        filled, rows = [], []
        for e in entries:
            src = args.masks / f"{e.image_id}.png"
            m = load_drawn_mask(src, args.resolution, sd.device)
            filled.append(dc_replace(e, mask=m))
            executors.save_image(m.expand(-1, 3, -1, -1),
                                 mask_dir / f"{e.image_id}_mask.png")
            # 遮罩疊在原圖上：判斷「它有沒有壓到 c_a」只能靠這張圖，
            # 純遮罩看不出對位。
            executors.save_image((e.x01 * (1.0 - 0.6 * m)).clamp(0, 1),
                                 mask_dir / f"{e.image_id}_overlay.png")
            cov = float(m.mean())
            rows.append({"image_id": e.image_id, "group": e.group,
                         "content": e.content, "coverage": cov,
                         "source": str(src)})
            print(f"  [mask] {e.image_id} c_a={e.content!r}（須在遮罩外）"
                  f" 涵蓋率={cov:.3f}  ← {src}", flush=True)
        executors.write_csv(mask_dir / "masks.csv", rows)
        entries = filled

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


def build_parser() -> argparse.ArgumentParser:
    """整份命令列定義。**與 `main` 分開**，使其他腳本能複用同一份。

    `scripts/tau_preview.py` 要用 `build_resources`，而後者吃的是這裡解析出
    來的 `args`。抄一份平行的參數表出去，兩份就會分岔——而分岔的症狀是
    某個旗標在主驅動上生效、在別的腳本上靜默失效。
    """
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
    # ---- inpainting 威脅模型 ----
    #
    # 給了 `--masks` 就是跑 inpainting。它同時決定 `base_config` 多不多一個
    # `masks` 鍵，故不設預設值：預設一個目錄等於讓每個 img2img 批次的雜湊
    # 也跟著變。權重必須是 inpainting 專用的（9 通道），否則
    # `SDWrapper.edit` 會在第一格就拋出。
    g = ap.add_argument_group("inpainting")
    g.add_argument("--masks", type=Path, default=None,
                   help="人工繪製的遮罩目錄，逐影像 <image_id>.png，"
                        "255 表示要重畫的區域。給定即切換到 inpainting 威脅"
                        "模型。用 scripts/draw_masks.py 產生（DEC-010）。"
                        "缺任何一張即拋出，不落回自動產生的遮罩")
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="本批用 prompts.yaml 的第幾個攻擊 prompt。"
                         "0 = 改掉 c_a（img2img 各批一律如此）、"
                         "1 = 保留 c_a 改動別處（Lo 的 inpainting 情境，"
                         "DEC-010）。非 0 時才進 config_hash，故 img2img "
                         "既有批次的每一格雜湊逐位不變")

    ap.add_argument("--wrapper", default="auto",
                    choices=["auto", "sd", "sdxl", "sd_inpaint"],
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
    g.add_argument("--tau-acut", type=float, default=None,
                   help=f"鈍化 hinge 的門檻。不給時由 --tau-train 依比例導出"
                        f"（{objective.ACUT_PER_TAU} × τ），錨點是 τ_lpips=0.05 "
                        f"上的人眼判讀 0.04")
    g.add_argument("--tau-chroma", type=float, default=None,
                   help=f"色度偏壓 hinge 的門檻。不給時同上依比例導出"
                        f"（{objective.CHROMA_PER_TAU} × τ）")
    g.add_argument("--budget-delta", type=float, default=None,
                   help="改用相對預算軸：τ 是**超出該格自己 φ=0 下限**的增量"
                        "（見 grid.budget_tau_plan）。給定時 τ 軸只有這一點，"
                        "且段 1 拒絕執行——本模式是評測預算，不是訓練預算，"
                        "φ 必須已由絕對模式訓練好")
    g.add_argument("--budget-metric", default="dists", choices=["lpips", "dists"],
                   help="相對預算軸量在哪個指標上。只在 --budget-delta 給定時"
                        "生效。預設 DISTS：LPIPS 是全圖平均，主體占比小時會"
                        "把主體被改寫的代價稀釋掉（docs/METRICS.md MET-dists）")
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
    g.add_argument("--warp-mask-gate", action="store_true",
                   help="位移場在遮罩內歸零，只擾動攻擊方不會覆寫的脈絡。"
                        "須與 --mask-mode 並用（inpainting 才有遮罩）")
    g.add_argument("--apa-lora-rank", type=int, default=8)
    g.add_argument("--apa-latent-max-rank", type=int, default=32)
    g.add_argument("--apa-latent-const-rank", type=int, default=8)
    g.add_argument("--random-init-std", type=float, default=0.5,
                   help="同失真隨機對照的初始標準差。R 用於位移場的 flow，"
                        "Ra 用於 apa 階段二的方向參數（射線縮放會把兩者都"
                        "拉到 τ，故此值只決定起點的方向分布）")
    g.add_argument("--recon", action="store_true",
                   help="A 段（DEC-016）：把 site apa 的階段一從 UNet 的 LoRA "
                        "換成 VAE 上的兩件事——A1 解 z* 使 decode(z*)≈x、"
                        "A2 只微調解碼器的 GroupNorm affine 與 conv bias。"
                        "開啟時 apa／Ra 的 --align-steps 自動歸零（階段一在 "
                        "UNet 上，碰不到 VAE 的重建誤差，實測 200 步只移動 "
                        "LPIPS 0.0015 卻花 910 秒）。不給時全部相關鍵不進 "
                        "config_hash，既有批次逐格雜湊不變")
    g.add_argument("--recon-objective", default="lpips",
                   choices=["lpips", "dists"],
                   help="A 段的感知項與停止判準。dists 目標實測會被鑽穿"
                        "（DISTS 掉 86–95% 而 LPIPS 反升 1–51%、PSNR 不動），"
                        "改它之前先看 runs/faA 的比對頁")
    g.add_argument("--recon-a1-steps", type=int, default=200)
    g.add_argument("--recon-a1-lr", type=float, default=0.02)
    g.add_argument("--recon-a2-steps", type=int, default=200)
    g.add_argument("--recon-a2-lr", type=float, default=2e-3)
    g.add_argument("--recon-floor-ratio", type=float, default=0.5,
                   help="A2 的停止目標＝本圖舊下限 × 此比例。這是硬停止條件"
                        "本身：解碼器背熟原圖會對 latent 擾動變遲鈍，防禦就"
                        "失去表達管道（recon 模組 docstring）")
    g.add_argument("--recon-gamma-acut", type=float, default=1.0,
                   help="A1／A2 雙邊銳利度 hinge 的係數。設 0 關閉，但關掉後"
                        "實測會拿變糊換感知分數（銳利度比 0.9935→0.7887）")
    g.add_argument("--recon-acut-band", type=float, default=0.05,
                   help="銳利度帶的半寬下限，實際取它與本圖舊下限自身偏差的"
                        "大者，故起點恆為可行")
    g.add_argument("--recon-w-pixel", type=float, default=0.5,
                   help="A 段損失中逐像素項的權重，逐目標校準："
                        "lpips 用 0.5、dists 用 0.15")
    g.add_argument("--recon-resp-scale", type=float, default=0.05,
                   help="latent 反應探針的擾動大小，以 ‖z‖ 的比例給定")
    g.add_argument("--attn-mask-tau", type=float, default=None,
                   help="apa（suppress_attn_ca）式 (4) 的遮罩門檻，作用在**峰值"
                        "正規化後**的 [0,1] 尺度上。論文未給值（Lo et al. "
                        "CVPR 2024），本專案選定並記錄。不給時 apa 直接拋出，"
                        "不回退到 0.5——它決定損失壓的是哪一塊。"
                        "**只在給定時進 config_hash**，故不影響既有批次")
    g.add_argument("--attn-mask-timesteps", type=int, default=0,
                   help="取遮罩時要平均幾個 timestep。0 表示沿用 "
                        "--attn-timesteps，即遮罩與施力落在同一組 t 上")

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

    ap.add_argument("--conditions", nargs="*", default=None,
                    help="只跑這些條件（預設全部）。用途是讓只影響部分條件的"
                         "旗標不必整批切換——`--purify-mode` 只作用於訓練期對"
                         "淨化算子取期望值的方式，baseline 與 R 都不受影響，"
                         "但它在 config_hash 內，整批切換會把已算完的 "
                         "photoguard_c 判成未完成。φ=0 的 control 格跨條件"
                         "共用，不隨本旗標改變")
    ap.add_argument("--full-purify-taus", type=float, nargs="*", default=None,
                    help="在哪些 τ 上跑完整的主組淨化算子。不給時：--tau-train "
                         "為預設值 0.20 的批次沿用模組常數 (0.20, 0.35)，"
                         "其餘批次只在訓練點上跑。**訓練點必須在其中**——"
                         "抗淨化（主張一）的分母就是訓練點上的效果，"
                         "不在其中的話那一批一個淨化格都不會有（TauPlan 會拒絕）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出會跑哪些格，不執行也不寫入")
    ap.add_argument("--force", action="store_true",
                    help="忽略續跑判定重跑全部格子")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # inpainting 沒有 strength。`--strength` 的預設是 0.6，不清掉的話它會
    # 進 `base_config`、進 `config_hash`、再被交給 `SDWrapper.edit` 而拋出
    # ——那時模型已經載完。清成 None 之後「這個威脅模型沒有 strength」這件事
    # 出現在雜湊與每一格的紀錄裡，而不是一個沿用下來卻不起作用的數。
    #
    # 明給 `--strength` 又給 `--masks` 是矛盾的指令，直接擋掉而不是
    # 靜默忽略其中一個。
    if args.masks:
        if "--strength" in (argv if argv is not None else sys.argv[1:]):
            raise SystemExit(
                "--masks 與 --strength 不可並用：inpainting 由純噪聲起跑、"
                "跑滿自己的排程，沒有 strength 這個參數（五篇 baseline 的"
                "原始碼裡也都沒有）")
        args.strength = None
    elif args.warp_mask_gate:
        # 沒有遮罩就沒有閘可加。靜默忽略會讓命令列上明寫的設定不生效，而
        # 那正是本專案要消滅的路徑——`config_hash` 屆時也記不出差別。
        raise SystemExit(
            "--warp-mask-gate 須與 --masks 並用：閘由遮罩產生，"
            "img2img 威脅模型沒有遮罩")

    resolve_thresholds(args)

    batch_dir = args.runs_root / args.batch
    images = load_images(args)
    conditions = grid.resolve_conditions(args.conditions)
    # 失真預算軸逐批次導出（2026-08-09）。四個常數必須一致地跟著批次走：
    # 只改 `--tau-train` 而不動其餘三個，訓練點上一個淨化格都不會有，
    # 而抗淨化是主張一。`--tau-train` 為預設值時本函式回傳模組常數本身，
    # 故 v14／v14r 的格點逐格不變（`tests/test_grid.py` 釘住）。
    if args.budget_delta is not None and args.stage == "train":
        raise SystemExit(
            "--budget-delta 與 `train` 段不可並用：相對預算軸是評測期的預算"
            "（τ = 超出該格 φ=0 下限的增量），而段 1 的綁定約束是絕對 "
            "τ_LPIPS。要在此模式下評測，先用絕對模式跑完段 1，"
            "本批直接從 rayscale 開始")
    tau_plan = tau_plan_of(args)
    if tau_plan is not grid.DEFAULT_TAU_PLAN:
        origin = (f"相對軸：τ = 超出各格 φ=0 下限的 {tau_plan.metric.upper()} 增量"
                  if tau_plan.relative
                  else f"絕對軸：τ = 達成的 {tau_plan.metric.upper()}"
                       f"（由 --tau-train={args.tau_train} 導出）")
        print(f"[taus] τ 軸 {tau_plan.taus}；主表與掃描組 "
              f"{tau_plan.main_tau}；完整淨化組 {tau_plan.full_purify_taus}"
              f"；{origin}", flush=True)
    # 這一份用**名目**下限。乾跑到此為止；真跑會在載入模型之後重建，見下方。
    plan = grid.plan(images, conditions=conditions, n_seeds=grid.N_SEEDS,
                     tau_plan=tau_plan)

    if args.dry_run:
        # 乾跑不取寫入鎖：它是唯讀的，且可能與正在跑的批次並存
        w = ProgressWriter(batch_dir, env=build_env(args), take_lock=False)
        rep = plan_report(plan, w, base_config(args))
        print(f"batch {args.batch}   影像 {len(images)}   "
              f"條件 {len(conditions)}{'' if len(conditions) == len(grid.CONDITIONS) else ' ' + str(list(conditions))}")
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

        # 以**實測**的逐影像重建下限重建格點。
        #
        # 上面那一份用的是 `grid.GENERATIVE_LPIPS_FLOOR` 這個名目常數，因為
        # 乾跑必須在載入 SDXL 之前就能回答「這次要跑多久」。到了這裡模型與
        # 影像都在，量一次 VAE 來回只要 0.4 秒／張（`micro_bench` 實測），
        # 沒有理由再用常數。
        #
        # 2026-08-07 加入。before：全程只有名目值，而實測下限逐影像差很多
        # （cat_02 是 0.2398，高於 τ=0.20）。`rayscale/N3/cat_02/tau0.2` 因此
        # 被送進 `solve_k`，二分 28 次後正確地拒絕並拋出，該格記為 failed，
        # 分片就此停住。改用實測值之後那一格是 skipped——結構上不適用，
        # 不是失敗。**跳過與否不進 `config_hash`**，故已完成的格不受影響。
        floors = executors.measure_recon_floors(res, batch_dir / "calib")
        print("  [floor] 生成路徑的逐影像重建下限（LPIPS）："
              + "、".join(f"{k}={v:.4f}" for k, v in sorted(floors.items())),
              flush=True)
        plan = grid.plan(images, conditions=conditions,
                         n_seeds=grid.N_SEEDS, floors=floors,
                         tau_plan=tau_plan)

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
