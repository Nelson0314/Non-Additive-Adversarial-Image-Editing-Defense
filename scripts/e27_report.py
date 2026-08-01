"""E27 主網格的彙整 —— 修好攻擊端之後的第一次對等比較。

與 E15/E21 的報告腳本不同之處，全部來自 E25–E26：

1. 主判準是語意軸，不是 `net_lpips`。後者量的是「輸出移動了多少」，
   E25 已證明它不代表「編輯有沒有失敗」（726 格語意失敗 0 格）。此處以
   SigLIP 為主（它通過了 E25 §1.1 的對照，CLIP 沒有），`net_lpips` 併列
   但降為輔助。
2. 逐格檢查 `stop_reason`。空字串代表用盡步數上限而非收斂，該格
   不可用於跨 site 比較（E21–E23 §5.4）。本腳本明確把這些格子標出來，
   並在計算比值時排除它們。
3. site S 不在網格內（使用者 2026-08-01 決定）。非加性一側是 site C。

執行：`python scripts/e27_report.py --prefix e30`

run 目錄的前綴是參數而非寫死的字串。校準與主網格會落在不同前綴（E29 校準、
E30 主網格），改字串常數等於每跑一輪就要改一次腳本，而改過的腳本無法再重現
前一輪的表。
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TAUS = ["0.02", "0.05", "0.10"]
ARMS = [("C", "非加性（色度）"), ("P", "加性基準")]

# 無淨化那一格。identity 只以訓練種子跑，未見種子的無淨化格是 blur 強度 0。
BASE = ("blur", 0.0)


def load(run: str):
    path = RUNS / run / "results.csv"
    if not path.exists():
        return None
    return list(csv.DictReader(path.open(encoding="utf-8")))


def cell_summary(rows):
    """逐 cell 取無淨化那一列，回傳各量的陣列與收斂情形。"""
    base = [r for r in rows
            if r["purify"] == BASE[0] and float(r["strength"]) == BASE[1]
            and r["noise_split"] == "heldout"]
    out = {
        "n": len(base),
        "net": np.array([float(r["net_lpips"]) for r in base]),
        "dsiglip": np.array([float(r["edit_siglip_b"]) - float(r["edit_siglip_a"])
                             for r in base]),
        "dclip": np.array([float(r["edit_clip_b"]) - float(r["edit_clip_a"])
                           for r in base]),
        "defimg_lpips": np.array([float(r["defimg_lpips"]) for r in base]),
        "acut": np.array([float(r["defimg_acutance_ratio"]) for r in base]),
        "steps": np.array([int(r["steps_done"]) for r in base]),
        "converged": np.array([bool(r["stop_reason"].strip()) for r in base]),
        "images": [r["image"] for r in base],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e30",
                    help="run 目錄的前綴，目錄名為 <prefix>_<site>_tau<τ>")
    args = ap.parse_args()

    table, missing = {}, []
    for tau in TAUS:
        for site, _ in ARMS:
            run = f"{args.prefix}_{site}_tau{tau}"
            rows = load(run)
            if rows is None:
                missing.append(run)
                continue
            table[(tau, site)] = cell_summary(rows)

    if missing:
        print(f"[{args.prefix}] 尚未產生的 run：{', '.join(missing)}\n")
    if not table:
        raise SystemExit(f"沒有任何 {args.prefix}_* 的資料，無法彙整")

    print("=== 主表（無淨化、未見種子）===")
    print(f"{'τ':>6s}{'臂':>4s}{'n':>3s}{'收斂':>6s}{'步數':>7s}"
          f"{'net_lpips':>12s}{'Δsiglip':>11s}{'保真LPIPS':>11s}{'銳利度':>9s}")
    for tau in TAUS:
        for site, label in ARMS:
            s = table.get((tau, site))
            if s is None:
                continue
            print(f"{tau:>6s}{site:>4s}{s['n']:>3d}"
                  f"{int(s['converged'].sum()):>3d}/{s['n']:<2d}"
                  f"{s['steps'].mean():>7.0f}"
                  f"{s['net'].mean():>8.4f}±{s['net'].std():.4f}"
                  f"{s['dsiglip'].mean():>+8.4f}±{s['dsiglip'].std():.4f}"
                  f"{s['defimg_lpips'].mean():>11.4f}{s['acut'].mean():>9.3f}")

    print("\n=== 比值（只用兩臂都收斂的格子）===")
    print(f"{'τ':>6s}{'可用格數':>10s}{'net C/P':>10s}{'Δsiglip C/P':>14s}  判定")
    for tau in TAUS:
        c, p = table.get((tau, "C")), table.get((tau, "P"))
        if c is None or p is None:
            continue
        # 逐圖配對，且兩臂都必須收斂。跨 site 比較的前提是兩邊都已在同一道
        # 約束下停下來，這正是 E21–E23 §5.4 的要求。
        keep = c["converged"] & p["converged"]
        k = int(keep.sum())
        if k == 0:
            print(f"{tau:>6s}{0:>10d}{'—':>10s}{'—':>14s}  無可用格子（兩臂未同時收斂）")
            continue
        rn = c["net"][keep].mean() / p["net"][keep].mean()
        ds_c, ds_p = c["dsiglip"][keep].mean(), p["dsiglip"][keep].mean()
        # Δsiglip 越負代表防禦越有效，故比值取「誰比較負」而非直接相除：
        # 兩者可能異號，相除會給出無意義的數字。
        verdict = ("兩臂皆未達語意失敗" if ds_c >= 0 and ds_p >= 0 else
                   "C 較有效" if ds_c < ds_p else "P 較有效")
        rs = f"{ds_c:+.4f} vs {ds_p:+.4f}"
        print(f"{tau:>6s}{k:>10d}{rn:>10.2f}{rs:>14s}  {verdict}")

    print("\n=== 未收斂的格子（用盡步數上限，不可用於比較）===")
    any_unconv = False
    for (tau, site), s in sorted(table.items()):
        bad = [img for img, ok in zip(s["images"], s["converged"]) if not ok]
        if bad:
            any_unconv = True
            print(f"  τ={tau} site {site}：{', '.join(bad)}"
                  f"（步數 {s['steps'].max()}）")
    if not any_unconv:
        print("  無。全部格子都在上限內收斂。")

    print("\n判準說明：Δsiglip < 0 才是防禦讓編輯不服從 prompt。net_lpips 只是"
          "\n輸出移動了多少，E25 已證明它不代表編輯失敗（726 格語意失敗 0 格），"
          "\n故它在此為輔助欄位，不作為主判準。")


if __name__ == "__main__":
    sys.exit(main())
