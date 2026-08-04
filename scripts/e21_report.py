"""E21 —— 新約束下的三個設定的比較，並與 E15 逐格對照。

E15 的宣稱是「匹配 LPIPS 後 site S 領先 site P 約 1.15×」。E20 證明那個匹配
只對齊 LPIPS：site S 在該點丟掉 15% 的高頻，而加上鈍化約束後其運作點 0/6
可行。E21 在新可行域內重測，三個設定為：

- `Sbic`：site S + bicubic 重取樣。假設它能在保住銳利度的前提下維持防禦。
- `Sbil`：site S + bilinear（E15 的設定）。對照組，用來量「E15 那部分效果
  有多少是靠由模糊換來的」。
- `P`：加性基準。E20 實測它在新約束下 6/6 可行，故其結果應與 E15 幾乎相同；
  若不同，代表管線被改壞了，這是本報告的完整性檢查。

淨額（net）的定義沿用 E15：`edit_lpips − ctrl_lpips`，即扣掉淨化本身
造成的偏移後歸因於防禦的部分。取未見種子（heldout）。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "e21_report"

TAUS = ["0.02", "0.05", "0.10"]
ARMS = [("Sbic", "site S bicubic"), ("Sbil", "site S bilinear"), ("P", "site P 加性")]


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


# 沿用 e15_report.py 的定義，兩份報告必須可直接對照：未見種子下「無淨化」
# 的那一格不叫 identity——eval_sweep() 的每個算子都含強度 0 的對照，故
# blur@0.0 與 noise@0.0 就是恆等。標成 identity 的是訓練種子那一列。
NOPURIFY = {("blur", "0.0"), ("noise", "0.0")}


def net_of(run: Path) -> dict:
    """該 run 的平均淨額與失真，取無淨化、未見種子那一格。

    取無淨化是因為本實驗問的是防禦本身的強度，不是耐淨化性；後者是另一個
    獨立問題。取未見種子是因為訓練種子下的 shift 含過擬合（site P 實測
    3.30–3.56 倍）。
    """
    rows = list(csv.DictReader((run / "results.csv").open(encoding="utf-8")))
    sel = [r for r in rows
           if (r["purify"], r["strength"]) in NOPURIFY
           and r["noise_split"] == "heldout"]
    if not sel:
        raise ValueError(
            f"{run} 沒有無淨化／未見種子的列，無法取淨額。"
            f"實際的 (purify,strength,split) 組合："
            f"{sorted({(r['purify'], r['strength'], r['noise_split']) for r in rows})}")
    per_img = {}
    for r in sel:
        per_img.setdefault(r["image"], []).append(r)
    out = {"net": [], "lpips": [], "image": [], "disp_mean": [], "disp_max": []}
    for img, rs in sorted(per_img.items()):
        # blur@0.0 與 noise@0.0 是同一件事的兩次量測，取平均而非任選一個
        out["net"].append(float(np.mean([float(r["net_lpips"]) for r in rs])))
        out["lpips"].append(float(np.mean([float(r["defimg_lpips"]) for r in rs])))
        out["image"].append(img)
        # 位移統計逐格相同，取第一列即可。site P 沒有這兩欄。
        out["disp_mean"].append(float(rs[0].get("disp_mean_px") or "nan"))
        out["disp_max"].append(float(rs[0].get("disp_max_px") or "nan"))
    return out


# `max_disp` 逐分量夾在 ±1.5，故位移量值的上界是 1.5√2。E15 實測數張
# 影像的 disp_max_px 正好等於此值，即該處兩個分量都飽和。bicubic 要達到
# 同一 LPIPS 需要更大的位移（E20 §5.1：+54%），上界可能綁得更緊，那會使
# 兩個條件不對等。故本報告必須列出貼頂比例，不能只報平均。
DISP_CAP = 1.5 * np.sqrt(2.0)


def fidelity_of(run: Path) -> dict:
    """由留存影像重算銳利度與局部鈍化偏差，不從 CSV 轉抄。"""
    from src.metrics.acutance import acutance
    from src.metrics.local_acutance import local_acutance

    acut, dev = [], []
    for d in sorted(run.iterdir()):
        o, x = d / "orig.png", d / "defended.png"
        if not (d.is_dir() and o.exists() and x.exists()):
            continue
        a, b = load(o), load(x)
        acut.append(acutance(a, b)["acutance_ratio"])
        dev.append(local_acutance(a, b)["local_acutance_dev"])
    if not acut:
        raise FileNotFoundError(f"{run} 沒有任何 orig/defended 配對")
    return {"acutance": acut, "local_dev": dev}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from src.defense.objective import LossConfig

    tau_acut = LossConfig().tau_acut

    rows, lines = [], []
    for tau in TAUS:
        for arm, label in ARMS:
            run = RUNS / f"e21_{arm}_tau{tau}"
            if not run.is_dir():
                print(f"[跳過] {run} 尚未產生")
                continue
            n = net_of(run)
            f = fidelity_of(run)
            rows.append({
                "tau": tau, "arm": arm, "label": label,
                "n": len(n["net"]),
                "net_mean": float(np.mean(n["net"])),
                "net_std": float(np.std(n["net"])),
                "lpips_mean": float(np.mean(n["lpips"])),
                "acut_mean": float(np.mean(f["acutance"])),
                "dev_mean": float(np.mean(f["local_dev"])),
                "dev_max": float(np.max(f["local_dev"])),
                "feasible": int(sum(d <= tau_acut for d in f["local_dev"])),
                "disp_mean": float(np.nanmean(n["disp_mean"])),
                "disp_max": float(np.nanmax(n["disp_max"])) if not np.all(np.isnan(n["disp_max"])) else float("nan"),
                "disp_at_cap": int(np.nansum(np.abs(np.array(n["disp_max"]) - DISP_CAP) < 1e-3)),
            })

    if not rows:
        raise SystemExit("沒有任何 e21_* run 目錄，實驗尚未開始或已失敗")

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'τ':>5s} {'條件':>16s} {'net':>16s} {'LPIPS':>8s} "
           f"{'銳利度':>8s} {'鈍化':>8s} {'可行':>6s} {'位移':>8s} {'貼頂':>5s}")
    print(hdr)
    print("-" * 92)
    lines.append(hdr)
    for r in rows:
        line = (f"{r['tau']:>5s} {r['label']:>16s} "
                f"{r['net_mean']:>9.4f}±{r['net_std']:<6.4f} "
                f"{r['lpips_mean']:>8.4f} {r['acut_mean']:>8.3f} "
                f"{r['dev_mean']:>8.4f} {r['feasible']:>3d}/{r['n']} "
                f"{r['disp_mean']:>8.3f} {r['disp_at_cap']:>3d}/{r['n']}")
        print(line)
        lines.append(line)

    # ---- 與 E15 對照：同 τ 同 site，看新約束改變了什麼 ----
    print()
    print("=== 與 E15 對照（僅 τ=0.05，E15 只有該 τ 的影像入庫）===")
    lines.append("")
    lines.append("=== 與 E15 對照（τ=0.05）===")
    for site, arm in (("S", "Sbil"), ("P", "P")):
        old = RUNS / f"e15_{site}_tau0.05"
        new = RUNS / f"e21_{arm}_tau0.05"
        if not (old.is_dir() and new.is_dir()):
            continue
        o, nw = net_of(old), net_of(new)
        of, nf = fidelity_of(old), fidelity_of(new)
        s = (f"site {site}: net {np.mean(o['net']):.4f} → {np.mean(nw['net']):.4f} "
             f"({100*(np.mean(nw['net'])/np.mean(o['net'])-1):+.1f}%)，"
             f"銳利度 {np.mean(of['acutance']):.3f} → {np.mean(nf['acutance']):.3f}")
        print(s)
        lines.append(s)

    # ---- 主結論：可行域內 S 對 P 的比值 ----
    print()
    lines.append("")
    for tau in TAUS:
        d = {r["arm"]: r for r in rows if r["tau"] == tau}
        if "P" not in d:
            continue
        for arm in ("Sbic", "Sbil"):
            if arm not in d:
                continue
            a = d[arm]
            ratio = a["net_mean"] / d["P"]["net_mean"] if d["P"]["net_mean"] else float("nan")
            ok = "可行" if a["feasible"] == a["n"] else f"僅 {a['feasible']}/{a['n']} 可行"
            s = f"τ={tau} {arm} / P = {ratio:.2f}×（{ok}）"
            print(s)
            lines.append(s)

    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("比值只有在該條件全部影像可行時才可解讀；不可行的條件其 net 是靠")
    print("鈍化換來的，與 E15 的 1.15× 屬於同一類無效比較。")
    print(f"「貼頂」= disp_max_px 觸及 max_disp 的量值上界 {DISP_CAP:.3f}（逐分量夾 ±1.5）。")
    print("若 bicubic 條件貼頂張數明顯較多，代表它被位移上界綁住而非被保真約束綁住，")
    print("兩個條件即不對等，須放寬 max_disp 重跑才能比較。")


if __name__ == "__main__":
    main()
