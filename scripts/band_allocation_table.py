"""`runs/ip2p_band_allocation/purify` 的出表：總增益與淨增益並列。

**不跑 GPU。** 只讀 `phase_retention.py` 寫出的 CSV。

三件事寫在這裡而不是留給讀者：

1. **兩個絕對值並列，不換算成比例。** 每一格報

       總增益 = effect(p)              ← `phase_retention.py` 量到的位移本身
       淨增益 = effect(p) − 空白地板

   兩張表用同一組（影像, 算子）配對算，故逐格的差就是該格的地板。表尾另印
   一列空白地板的絕對值，讀者看得到差額從哪裡來。

2. **扣空白地板不可省略。** 淨化算子自己就會把編輯推開（裁切 10% 的地板實測
   0.515），不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯推得比較
   開」這個平庸解釋（`docs/GOAL.md`）。逐（影像, 算子）配對相減，不是先平均
   再相減——後者會被缺格的影像污染。

   **幾何類算子（`src/purify/ops.py` 的 `GEOMETRIC_KINDS`）的地板由構造為
   0**，總增益與淨增益因此相等。那一類的參照是 `編輯(p(原圖))` 而不是
   `編輯(原圖)`（見 `scripts/phase_retention.py` 的 docstring），兩側吃同一個
   算子，相減退化成恆等。讀到非 0 的幾何地板一律拋錯：那份地板是舊參照量的，
   與新協定不可並列。

3. **缺格照實報，不補值。** 任何條件或算子在某張影像上缺讀數，該格從平均裡
   排除並在末尾列出，不靜默略過。

位移的飽和值只作為讀表的參考，**不進任何算式**：主讀數 LPIPS 在兩張不相干的
自然影像之間飽和於 0.772（`runs/readout_ceiling/`，十張兩兩配對 45 對的
中位數），所以 0.6 附近的讀數已經接近這個量的上限。

用法：
    python scripts/band_allocation_table.py [--src runs/ip2p_band_allocation/purify]
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.purify.ops import label_is_geometric  # noqa: E402

# LPIPS 在不相干自然影像之間的飽和值。出處：`runs/readout_ceiling/README.md`，
# 十張兩兩配對 45 對的中位數。**只寫進表尾的說明，不進任何算式**：表上報的是
# 總增益與淨增益兩個絕對值，不是佔某個範圍的比例。
LPIPS_CEILING = 0.772

# 分片名。`all` 用在影像少到切不動分片的批次（`runs/ip2p_eot_ceiling` 只有
# 兩張），與 `retention_table.py` 的 `SHARDS` 同一組名字。
SHARDS = ("color", "object", "scene", "all")
PURIFIER_ORDER = ("identity", "jpeg75", "jpeg30", "blur1", "blur2",
                  "crop_resize0.1")

FOOTNOTE = (
    "位移是 LPIPS，在兩張不相干的自然影像之間飽和於 "
    f"{LPIPS_CEILING}（runs/readout_ceiling/，45 對的中位數），"
    "故 0.6 附近的讀數已接近這個量的上限；該值只作為飽和值的參考，"
    "不進表上任何算式。幾何類算子的參照是「同一個算子淨化過的原圖」"
    "（purified_orig），空白地板由構造為 0，總增益與淨增益相等。"
)


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


def paired(cells: dict, floor: dict, pur: str) -> list:
    """該（條件, 算子）格裡兩邊都有讀數的 `(effect, 地板)`。

    總增益與淨增益吃同一組配對，逐格的差因此就是該格的地板；缺地板的影像
    兩張表一起排除，不是只在其中一張裡消失。
    """
    return [(v, floor[(img, pur)]) for (img, pp), v in cells.items()
            if pp == pur and (img, pur) in floor]


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
        if label_is_geometric(p):
            bad = {img: x for (img, pp), x in floor.items() if pp == p and x != 0.0}
            if bad:
                raise SystemExit(
                    f"幾何類算子 {p} 的空白地板不是 0：{bad}。該地板是舊參照"
                    f"（`編輯(原圖)`）量的，與現行協定的 `編輯(p(原圖))` 不可"
                    f"並列——重跑 `phase_retention.py --floor`。")

    missing = []
    for tag in tags:
        for p in purs:
            n_have = sum(1 for k in data[tag] if k[1] == p)
            n_pair = len(paired(data[tag], floor, p))
            if n_pair < n_have:
                missing.append(f"{tag}/{p}：{n_have - n_pair} 格缺地板")

    def table(title: str, value) -> None:
        print(title)
        print("條件".ljust(10) + "".join(p.rjust(15) for p in purs))
        for tag in tags:
            cells = []
            for p in purs:
                vals = paired(data[tag], floor, p)
                cells.append(f"{st.mean([value(e, f) for e, f in vals]):15.4f}"
                             if vals else " " * 15)
            print(tag.ljust(10) + "".join(cells))
        print("空白地板".ljust(8)
              + "".join(f"{floor_mean[p]:15.4f}" for p in purs))
        print("參照".ljust(10)
              + "".join(("purified_orig" if label_is_geometric(p) else "orig")
                        .rjust(15) for p in purs))

    table("總增益 = effect(p)（逐影像平均；配對同下表）", lambda e, f: e)
    print()
    table("淨增益 = effect(p) - 空白地板（逐影像相減後平均）",
          lambda e, f: e - f)

    print()
    print(FOOTNOTE)

    if missing:
        print("\n缺格（已從兩張表一併排除）：")
        for m in sorted(set(missing)):
            print("  " + m)


if __name__ == "__main__":
    main()
