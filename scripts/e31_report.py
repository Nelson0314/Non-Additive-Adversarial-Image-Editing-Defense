"""E31 主網格的彙整與 ISR 判定。

判定邏輯不在此處重寫，直接呼叫 `p12_isr_rejudge.judge_cell`：同一個判準在
兩個地方各寫一次遲早會分歧，而本專案的結論全部建立在判準上。

正式判準所在的位置是「乾淨攻擊」那一格——未經淨化、且用未見過的噪聲種子
（`noise_split=heldout`）。用訓練時見過的種子是對防禦最有利的條件，那一列
一併列出但不作為判定。

執行：python scripts/e31_report.py --degrade_tau <p11 定出的值>
成本：只讀 CSV，秒級。
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.p12_isr_rejudge import collect, judge_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e31_")
    ap.add_argument("--degrade_metric", default="niqe")
    ap.add_argument("--degrade_tau", type=float, required=True,
                    help="p11 的人眼階梯定出的劣化門檻。必填——用預設值跑出來"
                         "的判定沒有定錨，不可寫進報告")
    ap.add_argument("--out", default="runs/e31_report")
    args = ap.parse_args()

    runs = sorted(p for p in (ROOT / "runs").iterdir()
                  if p.is_dir() and p.name.startswith(args.prefix)
                  and (p / "results.csv").exists())
    if not runs:
        raise FileNotFoundError(
            f"找不到任何 {args.prefix}* 且含 results.csv 的 run。"
            "主網格尚未跑完，或跑的是 --no_eval"
        )
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in runs:
        cells = collect(run, args.degrade_metric)
        key = ("identity", "0.0", "heldout")
        if key not in cells:
            raise KeyError(
                f"{run.name} 沒有 (identity, 0.0, heldout) 這一格。"
                f"正式判準就在該格，缺席時不可用其他格代替"
            )
        sem, deg = cells[key]
        r = judge_cell(sem, deg, args.degrade_tau)

        # 收斂與否：stop_reason 為空代表用盡上限而非收斂，該格量到的是
        # 「走到哪裡」不是能力（E21–E23 §5.4），不可用於跨格比較。
        with open(run / "summary.csv", encoding="utf-8") as f:
            srows = list(csv.DictReader(f))
        converged = all(s.get("stop_reason") for s in srows)
        rows.append({
            "run": run.name,
            "converged": converged,
            "steps_done": ",".join(s["steps_done"] for s in srows),
            "final_lpips": ",".join(f"{float(s['final_lpips']):.4f}"
                                    for s in srows),
            **r,
        })

    with open(out / "e31_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':<34} {'收斂':<4} {'n':<3} {'ΔSigLIP':>9} {'Δ劣化':>8} "
          f"{'語意':<4} {'劣化':<4} ISR")
    for r in rows:
        print(f"{r['run']:<34} {'是' if r['converged'] else '否':<4} "
              f"{r['n']:<3} {r['mean_dsiglip']:>+9.4f} "
              f"{r['mean_ddegrade']:>+8.3f} "
              f"{'是' if r['semantic_fail'] else '否':<4} "
              f"{'是' if r['degrade_fail'] else '否':<4} "
              f"{'**是**' if r['isr'] else '否'}")
    n = sum(r["isr"] for r in rows)
    print(f"\n[e31] {len(rows)} 格中 ISR 成立 {n} 格"
          f"（劣化門檻 {args.degrade_tau}，指標 {args.degrade_metric}）")
    if n == 0:
        print("[e31] 正對照未建立。這對應規格 §9 的第 1 種預期否定結果："
              "預算（已含 0.28）與 strength（已含 0.3）兩個替代解釋都已排除")
    print(f"[e31] 寫出 {out / 'e31_summary.csv'}")


if __name__ == "__main__":
    main()
