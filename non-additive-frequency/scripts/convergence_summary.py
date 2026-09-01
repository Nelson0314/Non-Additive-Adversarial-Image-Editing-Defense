"""收斂曲線的彙整。**不跑 GPU、不重算任何數字**，只讀 `trace.csv`。

判收斂看的是 `eval` 欄——一組**固定**抽樣的評估值。訓練用的損失每一步重抽
`(t, eps)`，逐步值本來就會抖（實測 0.16–0.61），那是取樣變異不是參數在漂。

輸出兩份：
    curve.csv     逐 tag 逐步的 eval 中位數與四分位距（跨影像）
    summary.csv   逐 tag 的收尾狀態：停在第幾步、為什麼停、最後 10% 的斜率

**不下任何「有沒有效」的判斷**（`CLAUDE.md`）。

用法：
    python scripts/convergence_summary.py --runs runs/ip2p_ig_converge \
        --out runs/ip2p_ig_converge/convergence
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import write_csv  # noqa: E402


def read_rows(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tail_slope(pairs: List[tuple], frac: float = 0.1) -> float:
    """最後 `frac` 段的**相對**變化：`(首 − 末) / 首`。正值＝仍在下降。

    取相對值而不是絕對斜率，因為不同損失的量級差兩個數量級
    （`latent_norm` 約 70–100，`image_guidance` 約 0.003–0.16），
    絕對斜率跨損失不可比。
    """
    if len(pairs) < 4:
        return float("nan")
    cut = max(2, int(len(pairs) * frac))
    head, tail = pairs[-2 * cut:-cut], pairs[-cut:]
    a, b = st.mean(v for _, v in head), st.mean(v for _, v in tail)
    return (a - b) / abs(a) if a else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="含各 tag 子目錄的批次目錄")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    curve, summary = [], []
    for root in args.runs:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            trace = read_rows(d / "trace.csv")
            res = read_rows(d / "results.csv")
            if not trace:
                continue
            by_step: Dict[int, List[float]] = {}
            per_image: Dict[str, List[tuple]] = {}
            for r in trace:
                if not r.get("eval"):
                    continue
                s, v = int(r["step"]), float(r["eval"])
                by_step.setdefault(s, []).append(v)
                per_image.setdefault(r.get("image", ""), []).append((s, v))
            if not by_step:
                continue
            # **早停之後向前填補**：一張圖停了，它的損失就停在最終值，不是
            # 從統計裡消失。少了這一步，中位數是在「還沒停的那幾張」上算的，
            # 影像陸續退出時中位數會**往上跳**，畫出來是一條假的上升曲線。
            steps = sorted(by_step)
            last = {}
            for s in steps:
                for img, pts in per_image.items():
                    hit = [v for st_, v in pts if st_ == s]
                    if hit:
                        last[img] = hit[0]
                vs = sorted(last.values())
                curve.append({
                    "tag": d.name, "step": s, "n_images": len(vs),
                    "n_live": len(by_step[s]),
                    "eval_median": round(st.median(vs), 8),
                    "eval_min": round(vs[0], 8), "eval_max": round(vs[-1], 8),
                })
            slopes = [tail_slope(sorted(v)) for v in per_image.values()
                      if len(v) >= 4]
            stops = [r.get("stop_reason", "") for r in res]
            steps_done = [int(r["stopped_at"]) for r in res
                          if r.get("stopped_at", "").isdigit()]
            summary.append({
                "tag": d.name,
                "n_images": len(per_image),
                "loss": res[0].get("loss", "") if res else "",
                "radius": res[0].get("radius", "") if res else "",
                "max_steps": res[0].get("defense_steps", "") if res else "",
                "last_eval_step": max(by_step),
                # 停在哪裡、為什麼停。**早停與跑滿要分得出來**。
                "early_stopped": sum(1 for s in stops if s == "early_stop"),
                "ran_full": sum(1 for s in stops if s == "max_steps"),
                "median_stopped_at": (int(st.median(steps_done))
                                      if steps_done else ""),
                # 最後 10% 的相對變化：正值＝還在下降，越接近 0 越平。
                "tail_change_median": (round(st.median(slopes), 6)
                                       if slopes else ""),
                "tail_change_max": (round(max(slopes), 6) if slopes else ""),
            })

    write_csv(args.out / "curve.csv", curve)
    write_csv(args.out / "summary.csv", summary)
    print(f"曲線 {args.out / 'curve.csv'}（{len(curve)} 列）")
    print(f"收尾 {args.out / 'summary.csv'}（{len(summary)} 列）\n")
    if summary:
        hdr = ("tag", "loss", "n_images", "last_eval_step",
               "median_stopped_at", "early_stopped", "ran_full",
               "tail_change_median")
        print("  ".join(f"{h:>17s}" for h in hdr))
        for r in summary:
            print("  ".join(f"{str(r.get(h, '')):>17s}" for h in hdr))


if __name__ == "__main__":
    main()
