"""逐格判定「到底是哪一道約束綁住了這一格」。

為什麼需要一個常設工具。匹配失真的比較只有在兩臂都被同一道約束
綁住、且都已在該約束下停下來時才成立。本專案已經連續踩到四個不同的綁定者，
每一次都是事後翻 `history.json` 才發現：

| 何時 | 真正的綁定者 | 症狀 |
|---|---|---|
| E13 | L∞ hinge | site S 的 LPIPS 遠低於 τ，L∞ 罰則 41.2 |
| E21–E23 | 步數 | τ=0.10 沒有一格碰到預算，末端 6/6 仍在上升 |
| E27 第一輪 | `max_dev` 硬上界 | site C 的 LPIPS hinge 0/60 步啟動 |
| E27 第三輪 | 防禦 hinge 的 margin | shift 逼近 0.5，飽和後不再花失真預算 |

`scripts/e22_binding_check.py` 只查 LPIPS 與鈍化兩道，且寫死了 E15/E21/E22
的 run 名稱。本腳本改為對任意 run 目錄運作，並把上表的四類全部納入判定。

執行：`python scripts/e27_binding_check.py runs/e27_C_tau0.05 [...]`
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 懲罰值鍵 → (人類可讀名稱, `loss` 區塊中對應的係數欄位)
#
# 每一道都必須填上自己的係數欄位。初版只給 L∞ 填了，其餘三道留 None，
# 於是 `gamma_psnr=0`（保留計算與記錄但不參與梯度）的 PSNR 罰則被當成綁定者
# ——實測把 site C 判成「PSNR hinge 56/60 步啟動」，而 PSNR 根本不在梯度裡。
# 判定「誰綁住這一格」的前提就是只看真的進了梯度的那幾道。
HINGES = [
    ("fid_pen_lpips", "LPIPS", "gamma_lpips"),
    ("fid_pen_acut", "鈍化", "gamma_acut"),
    ("fid_pen_linf", "L∞", "beta_linf"),
    ("fid_pen_psnr", "PSNR", "gamma_psnr"),
]


def analyse(run_dir: Path) -> list:
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    # 損失的係數與門檻在 `loss` 區塊裡（LossConfig 的完整傾印），不在頂層。
    # 頂層只有驅動腳本自己的參數。
    loss = env.get("loss", {})
    margin = loss.get("margin", env.get("margin", 0.5))
    tau = loss.get("tau_lpips")
    max_dev = env.get("color_max_dev")
    max_disp = env.get("warp_max_disp")

    summary = {}
    sm = run_dir / "summary.csv"
    if sm.exists():
        for r in csv.DictReader(sm.open(encoding="utf-8")):
            summary[r["image"]] = r

    out = []
    for hf in sorted(run_dir.glob("*/history.json")):
        h = json.loads(hf.read_text(encoding="utf-8"))
        cell = hf.parent.name
        image = cell.split("__")[0]
        n = len(h)
        shift = [x["edit_shift"] for x in h]

        engaged = {}
        for key, label, coef_field in HINGES:
            if coef_field is not None and loss.get(coef_field) == 0:
                continue          # 係數為零的 hinge 不構成約束
            engaged[label] = sum(1 for x in h if x.get(key, 0.0) > 0.0)

        # 防禦 hinge 是否飽和：偏移達到 margin 之後防禦項不再施力
        sat = sum(1 for s in shift if s >= margin)

        # 硬上界是否貼頂
        s = summary.get(image, {})
        cdev = float(s["cdev_mean"]) if s.get("cdev_mean") else None
        disp = float(s["disp_p99_px"]) if s.get("disp_p99_px") else None
        clamp = None
        if cdev is not None and max_dev:
            clamp = ("max_dev", cdev / max_dev)
        elif disp is not None and max_disp:
            clamp = ("max_disp", disp / max_disp)

        # 判定：依「誰先擋住」排序。硬上界與 margin 優先，因為它們一旦綁住，
        # 保真約束就永遠不會被碰到，該格的 τ 等於沒有作用。
        if clamp and clamp[1] > 0.9:
            verdict = f"硬上界 {clamp[0]}（用到 {clamp[1]:.0%}）"
        elif sat > n * 0.5:
            verdict = f"防禦 margin={margin}（{sat}/{n} 步飽和）"
        elif max(engaged.values(), default=0) == 0:
            verdict = "沒有任何約束啟動過 —— 步數不足或 lr 太小"
        else:
            top = max(engaged, key=engaged.get)
            verdict = f"{top} hinge（{engaged[top]}/{n} 步啟動）"

        out.append({
            "run": run_dir.name, "image": image, "steps": n,
            "tau_lpips": tau, "final_lpips": h[-1]["fid_lpips"],
            "final_acut": h[-1].get("fid_acut"),
            "best_shift": max(shift), "final_shift": shift[-1],
            "margin": margin, "saturated": f"{sat}/{n}",
            **{f"啟動_{k}": f"{v}/{n}" for k, v in engaged.items()},
            "clamp_frac": None if clamp is None else round(clamp[1], 3),
            "verdict": verdict,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 目錄")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        d = Path(r)
        if not d.is_absolute():
            d = ROOT / r
        if not (d / "env.json").exists():
            print(f"[跳過] {d} 沒有 env.json")
            continue
        rows.extend(analyse(d))

    if not rows:
        raise SystemExit("沒有可分析的 run")

    print(f"{'run':22s}{'圖':>10s}{'步':>4s}{'τ':>6s}{'末lpips':>9s}"
          f"{'最佳shift':>10s}{'飽和':>8s}  判定")
    for r in rows:
        print(f"{r['run']:22s}{r['image']:>10s}{r['steps']:>4d}"
              f"{r['tau_lpips'] if r['tau_lpips'] is not None else '?':>6}"
              f"{r['final_lpips']:>9.4f}"
              f"{r['best_shift']:>10.4f}{r['saturated']:>8s}  {r['verdict']}")

    print("\n判定的優先序：硬上界 > 防禦 margin > 保真 hinge。前兩者一旦綁住，"
          "\nτ 對該格就完全不起作用，該格不可與其他 τ 的格子並列比較。")


if __name__ == "__main__":
    sys.exit(main())
