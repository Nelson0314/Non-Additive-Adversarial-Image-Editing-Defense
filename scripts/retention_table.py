"""抗淨化的頭對頭表：扣掉空白地板的**淨增益**。**不跑 GPU。**

主讀數不是 `retention` 比值
────────────────────────────────────────────────────────────────────
比值 `effect(淨化)/effect(identity)` 被分母支配（條件層 r = −0.83、逐圖層
−0.900，FND-037／039），而淨化算子自己就會把編輯推開五到九成（FND-043）。
故主讀數取

    淨增益(條件, 算子) = effect(條件, 算子) − effect(地板, 算子)

其中「地板」那一格的防禦圖就是原圖本身，量到的是算子自己造成的位移。
`retention` 仍照報，但**必須與絕對位移一併看**。

逐圖相減
────────────────────────────────────────────────────────────────────
地板逐圖不同（同一個模糊在不同影像上推開的量差好幾倍），故相減在**逐圖**
層級做，不是先各自平均再相減。缺地板的影像整格排除並回報，不靜默補零。

`usable = False` 的列（`effect(identity)` 低於三倍標準差）一律排除。

用法：
    python scripts/retention_table.py --src runs/ip2p_purify_headtohead \
        --out runs/ip2p_purify_headtohead/net_gain.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import write_csv  # noqa: E402

FLOOR_CONDITION = "none"        # phase_retention.py --floor 寫下的條件名
# 分片檔名是 `<tag>_<shard>.csv`。分片名是族群不是流水號（見
# scripts/purify_headtohead.sh），列在這裡好把 tag 還原出來。
SHARDS = ("color", "scene", "object", "all")


def tag_of(filename: str) -> str:
    """`ours_add_color.csv` → `ours_add`。

    **不可用 `split("_")[0]`**：那會把 `ours_add` 與 `ours_nonadd` 併成
    `ours`，而它們的 `condition` 欄都是 `phase_gain`，併起來不會有任何症狀。
    """
    stem = filename[:-4] if filename.endswith(".csv") else filename
    for shard in SHARDS:
        if stem.endswith("_" + shard):
            return stem[: -len(shard) - 1]
    raise ValueError(
        f"{filename} 的分片名不在 {SHARDS} 裡，無法還原條件標籤")


def read_all(src: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(src.rglob("*.csv")):
        with path.open(encoding="utf-8") as fh:
            head = fh.readline()
            if "purifier" not in head or "effect_mean" not in head:
                continue
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = path.name
                rows.append(r)
    return rows


def usable_rows(rows: List[dict]) -> Tuple[List[dict], int]:
    keep = [r for r in rows if str(r.get("usable", "True")).lower() != "false"]
    return keep, len(rows) - len(keep)


def net_gain(rows: List[dict]) -> Tuple[List[dict], List[str]]:
    """逐（條件, 算子）的淨增益。回傳表格與被排除的格子說明。"""
    floor: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if r["condition"] == FLOOR_CONDITION:
            floor[(r["image"], r["purifier"])] = float(r["effect_mean"])

    buckets: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
    dropped: List[str] = []
    for r in rows:
        cond = r["condition"]
        if cond == FLOOR_CONDITION:
            continue
        key = (r["image"], r["purifier"])
        if key not in floor:
            dropped.append(f"{cond}/{r['image']}/{r['purifier']}：缺地板")
            continue
        buckets.setdefault((tag_of(r["_file"]) + "|" + cond,
                            r["purifier"]), []).append(
            (r["image"], float(r["effect_mean"]), floor[key]))

    out = []
    for (tag, pur), vals in sorted(buckets.items()):
        gains = [e - f for _, e, f in vals]
        out.append({
            "condition": tag, "purifier": pur, "n_images": len(vals),
            "effect": round(statistics.fmean(e for _, e, _ in vals), 5),
            "floor": round(statistics.fmean(f for _, _, f in vals), 5),
            "net_gain": round(statistics.fmean(gains), 5),
            "net_gain_sd": round(
                statistics.stdev(gains) if len(gains) > 1 else 0.0, 5),
            "wins_over_floor": sum(1 for g in gains if g > 0),
        })
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: List[dict] = []
    for s in args.src:
        rows.extend(read_all(s))
    if not rows:
        raise SystemExit(f"{args.src} 底下沒有 retention 的 CSV")
    rows, skipped = usable_rows(rows)
    table, dropped = net_gain(rows)
    if not table:
        raise SystemExit("沒有可算的格子——地板那一批跑了嗎？")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, table)

    purs = sorted({r["purifier"] for r in table})
    conds = sorted({r["condition"] for r in table})
    by = {(r["condition"], r["purifier"]): r for r in table}
    print(f"usable=False 排除 {skipped} 列；缺地板排除 {len(dropped)} 格")
    head = f"{'條件':28s}" + "".join(f"{p:>18s}" for p in purs)
    print(head)
    print("-" * len(head))
    for c in conds:
        cells = ""
        for p in purs:
            r = by.get((c, p))
            cells += "" if r is None else f"{r['net_gain']:+18.4f}"
        print(f"{c:28s}{cells}")
    print(f"\n地板（算子自己造成的位移）")
    for p in purs:
        vals = [by[(c, p)]["floor"] for c in conds if (c, p) in by]
        if vals:
            print(f"  {p:20s}{statistics.fmean(vals):8.4f}")
    if dropped:
        print("\n被排除的格子：")
        for d in dropped[:20]:
            print(f"  {d}")
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
