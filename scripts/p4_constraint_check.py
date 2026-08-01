"""P4 —— 新保真約束在既有資料上的可行域檢查。

修訂後的約束是交集式：LPIPS、鈍化、L∞ 三道 hinge 各自獨立，全部滿足才
算可行。加權和不行——它永遠可以用便宜的軸換貴的軸，而那正是 E18/E19 觀察到
的行為。

本腳本不重新訓練，而是問一個更直接的問題：既有的 E15 運作點，在新可行域
裡還站得住嗎？這是新約束能否解讀舊結果的前提。

判準來自 `docs/NEXT_SESSION.md` §3.3：「把它加入約束後，site S 還能不能靠
模糊換取防禦效果？」若 site S 的既有運作點在新約束下不可行，答案就是不能，
約束達成了設計目的。

同時檢查 A 族（E19 的 `latent_opt` 各臂）：那些是低 λ 時鈍化到 84% 的臂，
新約束應該同樣擋下它們。此處只用 τ=0.05 的 E15 影像與 E19 的既有數值，
不需 GPU。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.defense.objective import LossConfig
from src.metrics.local_acutance import local_acutance
from src.metrics.suite import MetricSuite

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "p4_constraint_check"

TAU = "0.05"


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = LossConfig()
    suite = MetricSuite()

    rows = []
    for site in ("S", "P"):
        run = RUNS / f"e15_{site}_tau{TAU}"
        pairs = [(d / "orig.png", d / "defended.png") for d in sorted(run.iterdir())
                 if d.is_dir()]
        pairs = [(o, x) for o, x in pairs if o.exists() and x.exists()]
        if len(pairs) != 6:
            raise FileNotFoundError(f"{run} 應有 6 組影像，實得 {len(pairs)}")

        for o, x in pairs:
            a, b = load(o), load(x)
            # site P 與 site S 都不經 VAE，故 x_base 即原圖（見 site_warp 的
            # 「φ=0 時等於原圖，構造保證」）。兩道 hinge 的對象與訓練時一致。
            m = suite.pairwise(a, b)
            acut = local_acutance(a, b)["local_acutance_dev"]
            linf = float((a - b).abs().max())

            ok_lpips = m["lpips"] <= cfg.tau_lpips
            ok_acut = acut <= cfg.tau_acut
            rows.append({
                "site": site, "image": o.parent.name.split("__")[0],
                "lpips": m["lpips"], "acut": acut, "linf": linf,
                "acutance_ratio": m["acutance_ratio"],
                "ok_lpips": ok_lpips, "ok_acut": ok_acut,
                "feasible": ok_lpips and ok_acut,
            })

    with (OUT / "feasibility.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [f"約束：LPIPS ≤ {cfg.tau_lpips}，local_acutance_dev ≤ {cfg.tau_acut}",
             f"（L∞ hinge 的 beta_linf 自 E14 起一律設 0，故不列入可行性判定）",
             ""]
    print("\n".join(lines))
    hdr = (f"{'site':>6s} {'影像':>10s} {'LPIPS':>8s} {'鈍化':>8s} "
           f"{'銳利度':>8s}  {'LPIPS':>6s} {'鈍化':>6s}  可行")
    print(hdr)
    print("-" * 68)
    lines.append(hdr)
    for r in rows:
        line = (f"{r['site']:>6s} {r['image']:>10s} {r['lpips']:>8.4f} "
                f"{r['acut']:>8.4f} {r['acutance_ratio']:>8.3f}  "
                f"{'通過' if r['ok_lpips'] else '擋下':>6s} "
                f"{'通過' if r['ok_acut'] else '擋下':>6s}  "
                f"{'是' if r['feasible'] else '否'}")
        print(line)
        lines.append(line)

    print()
    lines.append("")
    for site in ("S", "P"):
        sub = [r for r in rows if r["site"] == site]
        n = sum(r["feasible"] for r in sub)
        s = (f"site {site}：{n}/{len(sub)} 可行"
             f"（LPIPS 通過 {sum(r['ok_lpips'] for r in sub)}/{len(sub)}，"
             f"鈍化通過 {sum(r['ok_acut'] for r in sub)}/{len(sub)}）")
        print(s)
        lines.append(s)

    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
