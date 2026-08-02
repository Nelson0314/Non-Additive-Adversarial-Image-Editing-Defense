"""量測本機器的每步成本，據以推算 E29／E30 的實際時間。

為什麼 Colab 需要這一支而 Lightning AI 不需要。E27 在 H100 上量到 2.47 s/step、
41.4 s/格評測，`docs/NEXT_SESSION.md` §6 的「2 小時／4.2 小時」是由這兩個數字
乘出來的。Colab 配到哪一張 GPU 不由使用者決定（T4／L4／A100 都可能），而這三者
的 fp32 吞吐相差數倍——同一份腳本在 A100 上是 5 小時、在 T4 上可能超過 12 小時的
連線上限。時間估錯的代價是網格跑到一半連線中斷。

作法是實測而非查表：跑兩個步數不同的極短 run，兩者相減消掉每格的固定成本
（k_inv 次反演、模組建構），斜率就是每步成本。評測成本另由一個開評測的 run
與同步數不開評測的 run 相減得到。

參考點（E27，H100 80GB，`runs/e27_evaltiming/`、`runs/e27d_C_lr0.3/`）：

    每步 2.47 s（147.3 s ÷ 60 步）
    評測 41.4 s（total_seconds 66.1 − seconds 24.7）
    峰值記憶體 10.3 GB —— 決定了 16 GB 的 T4 放得下

成本：約 20 個 512² 步加一次評測，視 GPU 快慢約 3–12 分鐘。

執行：
    python scripts/colab_probe.py
    python scripts/colab_probe.py --site P --out /tmp/wacv_probe
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# E27 在 H100 上的實測值，作為比值的分母。
H100_PER_STEP = 2.47
H100_EVAL = 41.4
H100_PEAK_MB = 10320.2

# 兩個探測步數。相減消掉固定成本，故兩者都取小；差 8 步足以讓斜率不被
# 計時雜訊主導（H100 上 8 步約 20 s）。
STEPS_LO = 4
STEPS_HI = 12

# E29／E30 的規模，與 `scripts/drivers/e29_calibration.sh`、`e30_grid.sh` 一致。
E29_CELLS = 8            # 4 個 run × --limit 2
E29_STEPS = 60
E30_CELLS = 36           # 2 site × 3 τ × 6 圖
E30_STEPS_EXPECT = 60    # 平台停止的預期步數
E30_STEPS_CAP = 150      # --steps 上限用滿


def run_one(py, out_dir, steps, evaluate, site, lr, size, model):
    """跑一個 run 並回傳 (優化秒數, 全程秒數)。

    優化秒數取自 summary.csv 的 `seconds`（`optimize()` 的牆鐘時間），
    全程秒數取自 env.json 的 `total_seconds`（含載入、存圖與評測）。
    兩者分開讀才能把評測成本從步進成本裡切出來。
    """
    cmd = [
        py, str(ROOT / "scripts" / "run_defense.py"),
        "--model", model, "--sites", site, "--ranks", "32" if site == "C" else "16",
        "--lr", str(lr), "--size", str(size), "--steps", str(steps),
        "--k_inv", "10", "--n_edit", "10", "--limit", "1",
        "--guidance_scale", "7.5", "--beta_linf", "0", "--tau_lpips", "0.05",
        "--margin", "1.0", "--alpha_lpips", "0", "--color_max_dev", "2.0",
        "--out", str(out_dir),
    ]
    if not evaluate:
        cmd.append("--no_eval")

    # 明確指定 UTF-8：run_defense.py 的輸出含中文，`text=True` 預設用地區
    # 編碼，在 Windows（cp950）下讀取子行程輸出會拋 UnicodeDecodeError。
    # 子行程的 PYTHONIOENCODING 一併設定，兩端才是同一個編碼。
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                          encoding="utf-8", env=child_env)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise SystemExit(f"run_defense.py 失敗（步數 {steps}，評測 {evaluate}）")

    import csv
    rows = list(csv.DictReader((out_dir / "summary.csv").open(encoding="utf-8")))
    env = json.loads((out_dir / "env.json").read_text(encoding="utf-8"))
    return float(rows[0]["seconds"]), float(env["total_seconds"]), wall, rows[0], env


def fmt_hours(seconds):
    h = seconds / 3600.0
    return f"{h:.1f} 小時" if h >= 1 else f"{seconds / 60:.0f} 分鐘"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=sys.executable, help="直譯器路徑")
    ap.add_argument("--out", default="/tmp/wacv_probe", help="探測輸出目錄（不入庫）")
    ap.add_argument("--site", default="C", choices=["C", "P"])
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    args = ap.parse_args()

    out = Path(args.out)
    print(f"=== 計時探測（site {args.site}，{args.size}²）===")
    print(f"直譯器 {args.py}")

    lo_sec, lo_total, lo_wall, lo_row, env = run_one(
        args.py, out / "lo", STEPS_LO, False, args.site, args.lr, args.size, args.model)
    print(f"  {STEPS_LO:>3d} 步（不評測）：優化 {lo_sec:6.1f} s   全程 {lo_total:6.1f} s")

    hi_sec, hi_total, hi_wall, hi_row, _ = run_one(
        args.py, out / "hi", STEPS_HI, False, args.site, args.lr, args.size, args.model)
    print(f"  {STEPS_HI:>3d} 步（不評測）：優化 {hi_sec:6.1f} s   全程 {hi_total:6.1f} s")

    ev_sec, ev_total, ev_wall, ev_row, _ = run_one(
        args.py, out / "ev", STEPS_LO, True, args.site, args.lr, args.size, args.model)
    print(f"  {STEPS_LO:>3d} 步（含評測）：優化 {ev_sec:6.1f} s   全程 {ev_total:6.1f} s")

    per_step = (hi_sec - lo_sec) / (STEPS_HI - STEPS_LO)
    # 每格除了步進以外的固定成本：反演、模組建構、存圖。由截距得到，
    # 而非另外假設一個值。
    fixed = lo_total - lo_sec + (lo_sec - STEPS_LO * per_step)
    eval_cost = ev_total - lo_total

    if per_step <= 0:
        raise SystemExit(
            f"斜率非正（{per_step:.3f} s/step）。兩次量測被雜訊淹沒或機器負載"
            f"不穩，重跑一次；持續為負代表計時欄位讀錯，不要用這個估計值。")

    gpu = env.get("gpu", "未知")
    peak = float(ev_row["peak_mb"])
    ratio = per_step / H100_PER_STEP

    print(f"\n=== 本機器 ===")
    print(f"  GPU              {gpu}")
    print(f"  torch            {env.get('torch')}")
    print(f"  每步             {per_step:6.2f} s      （H100 {H100_PER_STEP} s，{ratio:.1f}×）")
    print(f"  每格固定成本     {fixed:6.1f} s")
    print(f"  每格評測         {eval_cost:6.1f} s      （H100 {H100_EVAL} s，"
          f"{eval_cost / H100_EVAL:.1f}×）")
    print(f"  峰值記憶體       {peak:6.0f} MB    （H100 {H100_PEAK_MB:.0f} MB）")

    e29 = E29_CELLS * (E29_STEPS * per_step + fixed)
    e30_lo = E30_CELLS * (E30_STEPS_EXPECT * per_step + fixed + eval_cost)
    e30_hi = E30_CELLS * (E30_STEPS_CAP * per_step + fixed + eval_cost)

    print(f"\n=== 推算 ===")
    print(f"  E29 校準  {E29_CELLS} 格 × {E29_STEPS} 步（不評測）      {fmt_hours(e29)}")
    print(f"  E30 網格  {E30_CELLS} 格 × {E30_STEPS_EXPECT} 步（預期）  {fmt_hours(e30_lo)}")
    print(f"  E30 網格  {E30_CELLS} 格 × {E30_STEPS_CAP} 步（上限）    {fmt_hours(e30_hi)}")
    print(f"  合計（預期）                            {fmt_hours(e29 + e30_lo)}")
    print(f"  合計（上限）                            {fmt_hours(e29 + e30_hi)}")

    print(f"\n=== 判定 ===")
    # 記憶體：探測用的是與正式跑相同的 512²、相同 k_inv／n_edit，故峰值可直接
    # 外推。留 15% 餘裕，因為 E30 的 --steps 上限較高而 autograd 圖不隨步數成長，
    # 餘裕是給評測期的淨化算子與其他行程用的。
    import torch
    total_mb = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    if peak > total_mb * 0.85:
        print(f"  [記憶體] 峰值 {peak:.0f} MB 佔了 {peak / total_mb:.0%}，餘裕不足。")
    else:
        print(f"  [記憶體] 峰值 {peak:.0f} MB / {total_mb:.0f} MB，可跑。")

    # Colab 的連線上限：免費層約 12 小時且閒置會斷，Pro 開背景執行約 24 小時。
    # 這裡只以「單一 run 呼叫」為續跑單位判斷，因為 run_defense.py 沒有續跑
    # 旗標——中斷的那個 run 目錄要整個重跑。
    per_call_lo = e30_lo / 6      # E30 是 6 次 run 呼叫（3 τ × 2 site）
    per_call_hi = e30_hi / 6
    print(f"  [續跑單位] E30 的一次 run 呼叫（6 圖）= "
          f"{fmt_hours(per_call_lo)} ～ {fmt_hours(per_call_hi)}。")
    print(f"             run_defense.py 沒有續跑旗標，連線在呼叫中途斷掉，"
          f"該次呼叫要整個重跑。")
    if per_call_hi > 3 * 3600:
        print(f"  [風險] 單次呼叫上限超過 3 小時，中斷的代價過高。"
              f"建議換更快的 runtime，或接受只跑 τ=0.05 一組。")
    if e29 + e30_hi > 11 * 3600:
        print(f"  [風險] 上限情境超過 11 小時，免費層的連線上限接不完，"
              f"需分成多個 session 並在每次呼叫後推上 origin。")
    else:
        print(f"  [連線] 上限情境 {fmt_hours(e29 + e30_hi)}，單一 session 內可完成。")

    print(f"\n探測輸出在 {out}（不入庫）。本節輸出請留存到 runs/logs/。")


if __name__ == "__main__":
    sys.exit(main())
