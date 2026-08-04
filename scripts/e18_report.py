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
    print("=" * 78)
    print("是哪一項擋下來的（逐影像通過數，n/總數）")
    print("=" * 78)
    hdr2 = ("run".ljust(24) + "arm".ljust(17)
            + "lpips<門檻".rjust(12) + "psnr 不退".rjust(11)
            + "dists 不退".rjust(12))
    print(hdr2)
    print("-" * len(hdr2))
    for r in runs:
        p = r / "results.json"
        if not p.exists():
            continue
        rows = json.load(open(p))
        if not rows or "dists_ok" not in rows[0]:
            continue  # 舊格式（E17/E18 首格）沒有逐項旗標
        tag = r.name
        for arm in ["roundtrip", "latent_opt", "asym_free", "asym_leak",
                    "latent_opt_asym"]:
            v = [x for x in rows if x["arm"] == arm]
            if not v:
                continue
            n = len(v)
            print(tag.ljust(24) + arm.ljust(17)
                  + f"{sum(x['lpips_ok'] for x in v)}/{n}".rjust(12)
                  + f"{sum(x['psnr_ok'] for x in v)}/{n}".rjust(11)
                  + f"{sum(x['dists_ok'] for x in v)}/{n}".rjust(12))
            tag = ""

    print()
    print("參照：E17 現況下限 roundtrip 0.1434 / 27.51 dB；"
          f"site P 實際運作於 LPIPS {TARGET}")
    print("註：roundtrip 是自身的參照點，其 psnr/dists 必然「不退」，"
          "該列的這兩欄恆為滿分，不具資訊。")


if __name__ == "__main__":
    main()
