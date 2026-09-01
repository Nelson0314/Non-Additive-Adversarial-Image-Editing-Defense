"""補算已存批次的 CIEDE2000 色差。**不跑 GPU，只讀已存的圖。**

存在理由：`fid_deltaE00` 是為色彩族新加的欄位，而 `ip2p_run.py` 的列在加上它
之前就跑過一批。防禦圖與原圖都還在磁碟上，色差是逐像素的確定量，重算與當初
量的是同一個東西——這不是重跑實驗，是補一欄。

**只補這一欄。** 其餘欄位一律不動：那些是量測結果，重算會混進不同的
函式庫版本與精度。

用法：

    python scripts/color_delta_e.py --src runs/ip2p_color --out runs/ip2p_color/delta_e.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.metrics.suite import _delta_e00  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="含各條件子目錄的批次目錄")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cpu")
    rows = []
    for d in sorted(p for p in args.src.iterdir() if p.is_dir()):
        csv_path = d / "results.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as fh:
            recs = list(csv.DictReader(fh))
        for r in recs:
            name, cond = r["image"], r["condition"]
            orig = d / f"{name}__orig.png"
            defended = d / f"{name}__{cond}__def.png"
            if not (orig.exists() and defended.exists()):
                # 圖不入版控，缺了就沒得補。逐筆印出來，不靜默略過。
                print(f"[skip] {d.name}/{name}：缺 {orig.name} 或 "
                      f"{defended.name}", flush=True)
                continue
            a = load_image_tensor(orig, dev, size=RESOLUTION)
            b = load_image_tensor(defended, dev, size=RESOLUTION)
            rows.append({
                "tag": d.name, "image": name, "condition": cond,
                "radius": r.get("radius", ""),
                "fid_dists": r.get("fid_dists", ""),
                "fid_lpips": r.get("fid_lpips", ""),
                "deltaE00": round(_delta_e00(a, b), 4),
            })
            print(f"{d.name:20s} {name:38s} dE00={rows[-1]['deltaE00']:.3f}",
                  flush=True)

    write_csv(args.out, rows)
    print(f"\n{'tag':20s}{'n':>4s}{'dE00 中位數':>14s}")
    for tag in sorted({r["tag"] for r in rows}):
        sel = [r["deltaE00"] for r in rows if r["tag"] == tag]
        print(f"{tag:20s}{len(sel):>4d}{statistics.median(sel):>14.3f}")
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
