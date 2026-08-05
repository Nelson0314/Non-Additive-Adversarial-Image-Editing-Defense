"""實驗主驅動 — `docs/RUNBOOK_2026-08-05.md` §3。

    python scripts/run_stage.py <段> --batch b1 [選項]

段：`calib` / `train` / `rayscale` / `eval` / `report`，依序執行。

## 目前的完成度

**骨架已完成並有測試；計算層待在 GPU 上接線。**

可用（不需 GPU，已驗證）：

- `--dry-run` 列出每段有幾格要跑、幾格可續、幾格不適用
- 格點列舉、續跑判定、進度寫入、失敗處置（`src/experiment/`）
- `env.json` 的產出

待接線（需 SDXL 權重與 GPU）：各段的 `executor`。接線點是本檔的
`_executor_for()`，它目前對每一段拋出 `NotImplementedError` 並寫明缺什麼。
**這是刻意的**——寫了但從未執行過的計算層，其完成度是假的；
專案規範要求宣告完成前必須實際跑過並看到成功輸出。

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

from src.experiment import grid                              # noqa: E402
from src.experiment.runner import plan_report, run_stage     # noqa: E402
from src.utils.progress import ProgressWriter                # noqa: E402

STAGES = ("calib", "train", "rayscale", "eval", "report")

# `control`（φ=0 的同淨化對照）不是獨立的 CLI 段——它與 eval 一起跑，
# 因為兩者共用同一批淨化後的影像。但它**必須出現在乾跑報告裡**：
# 它有 300 格（N=3 時），漏掉會讓「這次要跑多久」少算一截。
REPORTED = ("calib", "train", "rayscale", "control", "eval", "report")


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


def base_config(args) -> dict:
    """整批共用的設定。格點自己的四個軸由 `cell_config` 疊上去。"""
    return {
        "spec_version": args.spec_version,
        "model": args.model,
        "resolution": args.resolution,
        "guidance": args.guidance,
        "steps": args.steps,
        "strength": args.strength,
        "gpu": args.gpu_tag,
        "precision": args.precision,
        "loss_params": {},
        "lr": None,
    }


def load_images(args) -> list:
    """回傳影像 id 清單。`--images` 明給時用它，否則由資料集分層抽樣。

    `n` 是唯一的樣本數入口。程式中不得出現字面值樣本數，
    使 N 由 3 擴到 150 只需改一個設定值。
    """
    if args.images:
        return list(args.images)
    from src.data.pie_bench import load, stratified_pick

    samples = load(args.data)
    return [s.key for s in stratified_pick(samples, args.n, seed=args.seed)]


def _executor_for(stage: str, args):
    """回傳該段的計算函式。

    **目前全部拋出。** 接線需要 SDXL 權重與 GPU，而未經執行的計算層
    其完成度是假的。每一段的訊息寫明該接什麼、依賴哪個模組。
    """
    needs = {
        "calib": ("段 0 校準：strength 掃描、編輯有效性過濾、逐條件學習率、"
                  "SDXL 的 attn_norm、fp16／bf16 對 fp32 的等價性驗證。"
                  "產出 calib/calibration.json 與 micro_bench.csv。"
                  "依賴 src/models/sd.py 的 SDXLWrapper 與真實權重"),
        "train": ("段 1 訓練：非加性條件走 src/defense/optimize.py 的 optimize()，"
                  "baseline 走 src/baselines/pgd.py 的 run_pgd()。"
                  "兩者的學習率都必須由 calibration.json 取得"),
        "rayscale": ("段 2 射線縮放：src/metrics/ray_scale.py 的二分搜尋，"
                     "把訓練好的 φ 落到各個 τ 上。只有前向、無梯度"),
        "eval": ("段 3 評測：淨化 → SDEdit → 指標。φ=0 對照跨條件共用，"
                 "見 grid.control_cells()"),
        "report": ("段 4 報表：grid.csv 彙整、compare.html（人眼比對頁，主判準）、"
                   "attention.html"),
    }

    def not_wired(cell, ctx):
        raise NotImplementedError(
            f"{stage} 的計算層尚未接線。\n  {needs[stage]}\n"
            "骨架（格點、續跑、進度）已完成並有測試，見 tests/test_runner.py。"
        )

    return not_wired


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
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50, help="攻擊方的去噪步數")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--spec-version", type=int, default=1)

    ap.add_argument("--data", type=Path, default=Path("data/pie_bench"))
    ap.add_argument("--n", type=int, default=3, help="樣本數的唯一入口")
    ap.add_argument("--images", nargs="*", help="直接指定影像 id，覆蓋 --n")
    ap.add_argument("--seed", type=int, default=20260805)

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

    if args.stage not in plan and args.stage not in ("calib", "report"):
        print(f"段 {args.stage} 沒有對應的格點", file=sys.stderr)
        return 2

    with ProgressWriter(batch_dir, env=build_env(args)) as w:
        (batch_dir / "env.json").write_text(
            json.dumps(build_env(args), indent=2, ensure_ascii=False),
            encoding="utf-8")
        cells = plan.get(args.stage, [])
        res = run_stage(args.stage, cells, _executor_for(args.stage, args),
                        w, base_config(args), force=args.force)
        print(f"\n[{res.stage}] done={res.done} failed={res.failed} "
              f"skipped={res.skipped} resumed={res.resumed}")
        if res.aborted:
            print(res.abort_reason, file=sys.stderr)
            return 3
        return 1 if res.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
