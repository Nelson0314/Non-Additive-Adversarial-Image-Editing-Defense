"""E21/E22 —— 到底是什麼在綁住每一格？

E22 的動機是「bicubic 臂被 `max_disp` 綁住」，但實測放寬四倍後位移與 LPIPS
幾乎完全不變（平均位移 0.518→0.515、LPIPS 0.0375→0.0375）。**該假設是錯的。**

於是改問一個更基本的問題：每一格的 25 步訓練，究竟有沒有讓任何一道約束
真正啟動過？三道 hinge 的懲罰值逐步都記在 `history.json` 裡：

- `fid_pen_lpips`：LPIPS 超過 τ_lpips 的部分
- `fid_pen_acut`：鈍化超過 τ_acut 的部分
- `fid_pen_linf`：L∞ 超過 τ_linf 的部分（E14 起 β=0，不進梯度）

若某一格的三者全程為零，則綁住它的既不是保真約束也不是位移上界，而是
**步數**——那一格量到的不是「該方法在此失真預算下的能力」，而是「該方法
在 25 步內走到哪裡」。兩者是完全不同的量，且後者不可用於跨 site 比較，
因為兩個 site 的 φ 單位不同、每步的有效移動距離也不同。
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

GROUPS = [
    ("e15_S", ["0.02", "0.05", "0.10"]),
    ("e15_P", ["0.02", "0.05", "0.10"]),
    ("e21_Sbic", ["0.02", "0.05", "0.10"]),
    ("e21_Sbil", ["0.02", "0.05", "0.10"]),
    ("e21_P", ["0.02", "0.05", "0.10"]),
    ("e22_Sbic_d6", ["0.02", "0.05", "0.10"]),
]

PENS = ["fid_pen_lpips", "fid_pen_acut"]


def scan(run: Path, tau: float) -> dict:
    """逐圖檢查三道 hinge 是否啟動過，以及軌跡末端是否仍在上升。"""
    out = {"n": 0, "lpips_active": 0, "acut_active": 0,
           "final_lpips": [], "tail_slope": [], "final_shift": []}
    for h in sorted(run.glob("*/history.json")):
        hist = json.load(h.open(encoding="utf-8"))
        if not hist:
            continue
        out["n"] += 1
        # 「啟動過」= 任何一步的懲罰值大於零
        if any(r.get("fid_pen_lpips", 0.0) > 0 for r in hist):
            out["lpips_active"] += 1
        if any(r.get("fid_pen_acut", 0.0) > 0 for r in hist):
            out["acut_active"] += 1
        out["final_lpips"].append(hist[-1]["fid_lpips"])
        out["final_shift"].append(hist[-1].get("edit_shift", float("nan")))
        # 末端斜率：最後五步的 LPIPS 平均增量。仍為正代表還沒收斂。
        tail = [r["fid_lpips"] for r in hist[-6:]]
        out["tail_slope"].append((tail[-1] - tail[0]) / max(len(tail) - 1, 1))
    return out


def main() -> None:
    print(f"{'run':>16s} {'τ':>5s} {'n':>3s} {'LPIPS hinge':>12s} {'鈍化 hinge':>11s} "
          f"{'末端LPIPS':>10s} {'預算用掉':>9s} {'末端斜率':>10s} {'仍在上升':>9s}")
    print("-" * 100)
    for prefix, taus in GROUPS:
        for tau in taus:
            run = RUNS / f"{prefix}_tau{tau}"
            if not run.is_dir():
                continue
            s = scan(run, float(tau))
            if not s["n"]:
                continue
            fl = float(np.mean(s["final_lpips"]))
            slope = float(np.mean(s["tail_slope"]))
            rising = sum(x > 1e-5 for x in s["tail_slope"])
            print(f"{prefix:>16s} {tau:>5s} {s['n']:>3d} "
                  f"{s['lpips_active']:>7d}/{s['n']:<4d} "
                  f"{s['acut_active']:>6d}/{s['n']:<4d} "
                  f"{fl:>10.4f} {100 * fl / float(tau):>8.0f}% "
                  f"{slope:>+10.5f} {rising:>6d}/{s['n']}")
    print()
    print("「LPIPS hinge」「鈍化 hinge」= 25 步中曾經啟動過的影像數。")
    print("「預算用掉」= 末端 LPIPS 佔 τ 的比例。遠低於 100% 且 hinge 從未啟動，")
    print("代表該格是**步數受限**，不是失真預算受限。")
    print("「仍在上升」= 最後五步 LPIPS 仍在增加的影像數；全數上升即尚未收斂。")


if __name__ == "__main__":
    main()
