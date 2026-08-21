"""批次層級的 FID（DEC-028 的指標清單裡唯一的分布指標）。

**FID 吃兩組影像而不是一對**，所以它不可能出現在 `results.csv` 的逐列裡——
逐列是一張影像。本支讀兩組已存的 PNG，算出該批次的一個 FID，寫成
`frechet.csv` 供報表端合併。

欄名是 `frechet` 不是 `fid`：本專案的 `fid_` 前綴自 2026-07 起代表
**fidelity**（`fid_lpips`／`fid_psnr`…），撞名會讓既有的分析腳本讀錯欄。

兩個半邊各算一次（見 `src/metrics/standard.py`）：

    失真半邊     orig 對 def            越低越好
    防禦半邊     edit(orig) 對 edit(def) 越高越好

樣本數：`MetricSuite.FID_MIN_TRUSTED = 150`。低於此值仍會算，但預設拒絕寫出，
要寫必須明給 `--allow-small`，且該列會標 `trusted=False`。理由是 2048×2048 的
協方差在小樣本上有嚴重偏誤，7 張的 FID 不是「比較不準」而是不可解讀。

不需要 SD，只需要 Inception-V3，本機可跑。

用法：
    python scripts/fid_batch.py --run runs/sweep0820 --half both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

# 兩個半邊各自的檔名樣式。`{cond}` 由 --conditions 填。
#
# 條件與後綴之間可能還有一段預算標記——`phase_ablation` 寫的是
# `{image}__{cond}__human__def.png` 或 `__d{budget}__def.png`，而
# `dct_shield_run` 寫的是 `{image}__{cond}__def.png`。故樣式中間留 `*`，
# 再由 `_collect` 把 `edit_def` 從失真半邊剔掉——否則 `*def.png` 會同時
# 撈到編輯後的圖，兩組影像混在一起而 FID 仍算得出來，不會報錯。
HALVES = {
    "fidelity": ("*__orig.png", "*__{cond}__*def.png"),
    "protection": ("*__{cond}__*edit_orig.png", "*__{cond}__*edit_def.png"),
}


def _collect(dirs, pattern: str, half: str) -> List[Path]:
    hits = sorted(q for d in dirs for q in d.glob(pattern))
    if half == "fidelity":
        hits = [p for p in hits if not p.name.endswith("edit_def.png")]
    return hits


def _load(paths: List[Path], device) -> torch.Tensor:
    return torch.cat([load_image_tensor(p, device, size=RESOLUTION)
                      for p in paths], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="預設由 *__def.png 的檔名推出")
    ap.add_argument("--half", choices=("fidelity", "protection", "both"),
                    default="both")
    ap.add_argument("--allow-small", action="store_true",
                    help=f"樣本數低於 {MetricSuite.FID_MIN_TRUSTED} 時仍寫出，"
                         "該列會標 trusted=False")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run[0] / "frechet.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)

    conds = args.conditions
    if conds is None:
        found = set()
        for d in args.run:
            for p in d.glob("*__def.png"):
                found.add(p.name.split("__", 1)[1].rsplit("__def.png", 1)[0])
        conds = sorted(found)
    if not conds:
        raise SystemExit("找不到任何條件——檢查 --run 底下有沒有 *__def.png")

    halves = ["fidelity", "protection"] if args.half == "both" else [args.half]
    rows = []
    for cond in conds:
        for half in halves:
            pat_a, pat_b = (p.format(cond=cond) for p in HALVES[half])
            a = _collect(args.run, pat_a, half)
            b = _collect(args.run, pat_b, half)
            if len(a) < 2 or len(b) < 2:
                print(f"[跳過] {cond}/{half}：只找到 {len(a)}／{len(b)} 張",
                      flush=True)
                continue
            trusted = min(len(a), len(b)) >= MetricSuite.FID_MIN_TRUSTED
            if not trusted and not args.allow_small:
                print(f"[跳過] {cond}/{half}：n={min(len(a), len(b))} < "
                      f"{MetricSuite.FID_MIN_TRUSTED}，加 --allow-small 才寫出",
                      flush=True)
                continue
            val = suite.fid(_load(a, device), _load(b, device),
                            batch_size=args.batch_size)
            rows.append({"condition": cond, "half": half,
                         "n_a": len(a), "n_b": len(b),
                         "frechet": round(val, 3), "trusted": trusted})
            print(f"{cond:16s} {half:10s} n={len(a)}/{len(b)} "
                  f"frechet={val:.3f} trusted={trusted}", flush=True)

    if not rows:
        raise SystemExit("沒有任何一格算得出 FID")
    write_csv(out, rows)
    print(f"\n表：{out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
