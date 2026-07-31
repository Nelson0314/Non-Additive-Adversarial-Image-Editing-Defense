"""E18/E19 報告 — latent_opt 的 lr、步數、lam 與 decoder 疊加。

從 runs/e18_lopt_lr*/ 與 runs/e19_*/ 讀取。步數維度不另開 run：
latent_opt_history.json 每 25 步記一次 LPIPS，任意步數的表現由軌跡讀出。

執行：python scripts/e18_report.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET = 0.063
STEP_MARKS = [100, 200, 400, 800, 1200, 1600]


def _at_step(hist, step):
    """軌跡中不大於 step 的最後一筆。步數不是 25 的倍數時取最接近的較早點。"""
    ok = [e for e in hist if e["step"] <= step]
    return ok[-1]["lpips"] if ok else None


def step_table(run: Path):
    """單一 run 的步數依賴：各影像在 STEP_MARKS 的 LPIPS 平均。"""
    hists = [json.load(open(p))
             for p in sorted(run.glob("*/latent_opt_history.json"))]
    if not hists:
        return None
    out = {}
    for s in STEP_MARKS:
        vals = [v for h in hists if (v := _at_step(h, s)) is not None]
        if vals:
            out[s] = sum(vals) / len(vals)
    return out


def load_summary(run: Path):
    p = run / "summary.json"
    return json.load(open(p)) if p.exists() else None


def main():
    runs_dir = Path("runs")
    runs = sorted([p for p in runs_dir.glob("e18_lopt_lr*") if p.is_dir()])
    runs += sorted([p for p in runs_dir.glob("e19_*") if p.is_dir()])
    if not runs:
        print("找不到 runs/e18_lopt_lr* 或 runs/e19_*，先跑實驗")
        return

    print("=" * 78)
    print("步數依賴（latent_opt 的 LPIPS 平均，由軌跡讀出，非另開 run）")
    print("=" * 78)
    head = "run".ljust(24) + "".join(f"{s:>9}" for s in STEP_MARKS)
    print(head)
    print("-" * len(head))
    for r in runs:
        t = step_table(r)
        if not t:
            continue
        line = r.name.ljust(24)
        for s in STEP_MARKS:
            line += f"{t[s]:>9.4f}" if s in t else " " * 9
        print(line)
        first, last = t.get(STEP_MARKS[2]), t[max(t)]
        if first and last and first > 0:
            print(" " * 24 + f"400→{max(t)} 再降 {100 * (first - last) / first:.1f}%")

    print()
    print("=" * 78)
    print(f"把關（LPIPS < {TARGET}，且 PSNR 與 DISTS 皆不差於 roundtrip）")
    print("=" * 78)
    hdr = ("run".ljust(24) + "arm".ljust(17) + "lpips".rjust(8)
           + "psnr".rjust(8) + "dists".rjust(9) + "通過".rjust(8))
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        s = load_summary(r)
        if not s:
            continue
        cfg = s.get("config", {})
        tag = r.name
        if cfg:
            tag += f" (lam={cfg.get('lam')})"
        for arm, v in s.items():
            if arm == "config":
                continue
            npass = v.get("n_pass")
            mark = f"{npass}/{v['n']}" if npass is not None else "—"
            print(tag.ljust(24) + arm.ljust(17)
                  + f"{v['lpips']:>8.4f}{v['psnr']:>8.2f}"
                  + (f"{v['dists']:>9.4f}" if "dists" in v else " " * 9)
                  + mark.rjust(8))
            tag = ""

    print()
    print("參照：E17 現況地板 roundtrip 0.1434 / 27.51 dB；"
          f"site P 實際運作於 LPIPS {TARGET}")


if __name__ == "__main__":
    main()
