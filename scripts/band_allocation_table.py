"""`runs/ip2p_band_allocation/purify` 的出表：扣地板的淨增益，並附可達比例。

**不跑 GPU。** 只讀 `phase_retention.py` 寫出的 CSV。

三件事寫在這裡而不是留給讀者：

1. **扣空白地板。** 淨化算子自己就會把編輯推開（裁切 10% 的地板實測 0.515），
   不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯推得比較開」
   這個平庸解釋（`docs/GOAL.md`）。逐（影像, 算子）配對相減，不是先平均再相減
   ——後者會被缺格的影像污染。

2. **附可達比例。** 主讀數 LPIPS 在兩張不相干的自然影像之間會飽和，十張兩兩
   配對（45 對）的中位數是 **0.772**（`runs/readout_ceiling/`）。所以每一欄的
   可達範圍是 `0.772 − 地板`，而不是 `0.772`。同一個絕對淨增益放在地板 0.05
   的欄與地板 0.52 的欄，意義差很多。

3. **缺格照實報，不補值。** 任何條件或算子在某張影像上缺讀數，該格從平均裡
   排除並在末尾列出，不靜默略過。

用法：
    python scripts/band_allocation_table.py [--src runs/ip2p_band_allocation/purify]
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

# LPIPS 在不相干自然影像之間的飽和值。出處：`runs/readout_ceiling/README.md`，
# 十張兩兩配對 45 對的中位數。**這是讀數的性質，不是判準。**
LPIPS_CEILING = 0.772

# 分片名。`all` 用在影像少到切不動分片的批次（`runs/ip2p_eot_ceiling` 只有
# 兩張），與 `retention_table.py` 的 `SHARDS` 同一組名字。
SHARDS = ("color", "object", "scene", "all")
PURIFIER_ORDER = ("identity", "jpeg75", "jpeg30", "blur1", "blur2",
                  "crop_resize0.1")


def tag_of(name: str) -> str:
    """`surv_r40_color.csv` → `surv_r40`。分片名不在清單裡就拋錯。"""
    stem = name[:-4] if name.endswith(".csv") else name
    for shard in SHARDS:
        if stem.endswith("_" + shard):
            return stem[: -len(shard) - 1]
    raise ValueError(f"{name} 的分片名不在 {SHARDS} 裡，無法還原條件標籤")


def read(src: Path) -> dict:
    """`{tag: {(image, purifier): effect}}`。"""
    out: dict = defaultdict(dict)
    for path in sorted(src.glob("*.csv")):
        tag = tag_of(path.name)
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    v = float(r["effect_mean"])
                except (KeyError, TypeError, ValueError):
                    continue
                out[tag][(r["image"], r["purifier"])] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path("runs/ip2p_band_allocation/purify"))
    args = ap.parse_args()

    data = read(args.src)
    if "floor" not in data:
        raise SystemExit(f"{args.src} 沒有 floor_*.csv，扣地板不可省略")
    floor = data.pop("floor")

    purs = [p for p in PURIFIER_ORDER
            if any(p == k[1] for d in data.values() for k in d)]
    tags = sorted(data)

    floor_mean = {}
    for p in purs:
        v = [x for (img, pp), x in floor.items() if pp == p]
        floor_mean[p] = st.mean(v) if v else float("nan")

    missing = []
    print("扣空白地板的淨增益（逐影像相減後平均）")
    print("條件".ljust(10) + "".join(p.rjust(15) for p in purs))
    for tag in tags:
        cells = []
        for p in purs:
            diffs = [v - floor[(img, p)]
                     for (img, pp), v in data[tag].items()
                     if pp == p and (img, p) in floor]
            n_have = sum(1 for k in data[tag] if k[1] == p)
            if len(diffs) < n_have:
                missing.append(f"{tag}/{p}：{n_have - len(diffs)} 格缺地板")
            cells.append(f"{st.mean(diffs):15.4f}" if diffs else " " * 15)
        print(tag.ljust(10) + "".join(cells))
    print("空白地板".ljust(8) + "".join(f"{floor_mean[p]:15.4f}" for p in purs))

    print()
    print(f"佔可達範圍的比例（可達 = {LPIPS_CEILING} − 地板）")
    print("條件".ljust(10) + "".join(p.rjust(15) for p in purs))
    for tag in tags:
        cells = []
        for p in purs:
            diffs = [v - floor[(img, p)]
                     for (img, pp), v in data[tag].items()
                     if pp == p and (img, p) in floor]
            room = LPIPS_CEILING - floor_mean[p]
            if not diffs or room <= 0:
                cells.append(" " * 15)
            else:
                cells.append(f"{100 * st.mean(diffs) / room:14.1f}%")
        print(tag.ljust(10) + "".join(cells))
    print("可達範圍".ljust(8)
          + "".join(f"{LPIPS_CEILING - floor_mean[p]:15.4f}" for p in purs))

    if missing:
        print("\n缺格（已從平均排除）：")
        for m in sorted(set(missing)):
            print("  " + m)


if __name__ == "__main__":
    main()
