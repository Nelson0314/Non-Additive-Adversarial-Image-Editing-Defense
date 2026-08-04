"""逐預算算出各種「通過／擋下」分組的分離度，供決定門檻怎麼定。

## 為什麼需要這一支

`p14_budget_thresholds.py` 用一組固定的條件歸屬（沿用 E20 §5.2 與 E28 §1 的
人眼判定）算門檻。但那個判定是在 **τ_lpips = 0.05 的量級**上做出來的——
使用者當時看的是「0.4–0.6 px 的位移」的比對頁。同一個歸屬在別的預算上不
一定還成立：τ=0.28 時空間變形位置的位移是當初的數倍，早已不是「不明顯」。

`runs/p14_budget_thresholds/arms.csv` 的實測顯示分離度確實隨預算塌掉。
以 person_00 為例：

| 預算 | noise（通過） | warp_bicubic（通過） | warp_bilinear（擋下） | blur（擋下） |
|---|---|---|---|---|
| 0.05 | 0.0122 | 0.0446 | 0.0967 | 0.1521 |
| 0.15 | 0.0482 | 0.1014 | 0.1195 | 0.3059 |
| 0.28 | 0.1616 | 0.1650 | 0.1650 | 0.5765 |

到 0.28 時 `noise` 與 `warp_bilinear` 已經分不開。

本腳本不替使用者做決定，只把幾組候選歸屬的分離度攤開：**門檻要定在哪裡是
實驗設計，須先討論**（`CLAUDE.md`）。

執行：python scripts/p14_separation.py
成本：只讀 CSV，秒級。
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 候選分組。每一組後面附它隱含的主張。
GROUPINGS = {
    "現行（E20/E28 的歸屬）": {
        "acut": (("noise", "warp_bicubic"), ("blur", "warp_bilinear")),
        "chroma": (("noise", "warp_bicubic"), ("chroma",)),
        "why": "沿用 τ=0.05 上的人眼判定，假設它在所有預算上都成立",
    },
    "只保 noise 為通過側": {
        "acut": (("noise",), ("blur", "warp_bilinear")),
        "chroma": (("noise",), ("chroma",)),
        "why": "最小的可辯護通過側——約束不得懲罰加性基準自己的失真型態；"
               "空間變形位置的人眼判定只在低位移下做過，不外推",
    },
    "只擋 blur": {
        "acut": (("noise", "warp_bicubic"), ("blur",)),
        "chroma": (("noise", "warp_bicubic"), ("chroma",)),
        "why": "約束的職責只有一條：擋住以模糊換分。site S 已在死路清單上，"
               "空間變形位置不再是受測的參數化，只是參照失真",
    },
    "noise 對 blur": {
        "acut": (("noise",), ("blur",)),
        "chroma": (("noise",), ("chroma",)),
        "why": "上兩者的交集，分離度最大但參照最少",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="runs/p14_budget_thresholds/arms.csv")
    ap.add_argument("--out", default="runs/p14_budget_thresholds/separation.csv")
    args = ap.parse_args()

    with open(ROOT / args.arms, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    budgets = sorted({float(r["budget"]) for r in rows})
    images = sorted({r["image"] for r in rows})
    print(f"[sep] {len(rows)} 列，{len(images)} 張影像，預算 {budgets}\n")

    def mean_of(budget, arm, col):
        sel = [float(r[col]) for r in rows
               if abs(float(r["budget"]) - budget) < 1e-9 and r["arm"] == arm]
        return sum(sel) / len(sel)

    out_rows = []
    for name, g in GROUPINGS.items():
        print(f"=== {name} ===")
        print(f"    {g['why']}")
        print(f"    {'預算':<7} {'acut 通過≤':>10} {'acut 擋下≥':>10} {'比值':>7}"
              f"   {'chroma 通過≤':>12} {'chroma 擋下≥':>12} {'比值':>7}")
        for b in budgets:
            r = {"grouping": name, "budget": b}
            for axis in ("acut", "chroma"):
                p_arms, b_arms = g[axis]
                hi_pass = max(mean_of(b, a, axis) for a in p_arms)
                lo_block = min(mean_of(b, a, axis) for a in b_arms)
                r[f"{axis}_pass_max"] = round(hi_pass, 4)
                r[f"{axis}_block_min"] = round(lo_block, 4)
                r[f"{axis}_ratio"] = round(lo_block / hi_pass, 3) if hi_pass > 0 else float("inf")
                r[f"{axis}_ok"] = lo_block > hi_pass
            out_rows.append(r)
            flag_a = "" if r["acut_ok"] else "  ← 分不開"
            flag_c = "" if r["chroma_ok"] else "  ← 分不開"
            print(f"    {b:<7} {r['acut_pass_max']:>10.4f} {r['acut_block_min']:>10.4f} "
                  f"{r['acut_ratio']:>7.2f}   {r['chroma_pass_max']:>12.4f} "
                  f"{r['chroma_block_min']:>12.4f} {r['chroma_ratio']:>7.2f}"
                  f"{flag_a}{flag_c}")
        print()

    with open(ROOT / args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[sep] 寫出 {ROOT / args.out}")
    print("[sep] 比值 = 擋下側下緣 / 通過側上緣。小於 1 代表該預算上這個指標"
          "分不出兩群，門檻無論定在哪裡都會誤判其中一側。")


if __name__ == "__main__":
    main()
