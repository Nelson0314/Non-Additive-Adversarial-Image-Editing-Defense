"""把逐格看圖的判讀併進 helmet 掃描的 `sweep.csv`。

掃描本身只產影像與數值，四個判讀欄位留空。判讀是看圖做的，來源是
`review.csv`（人手填，與掃描結果分開存，改判讀不必重跑 GPU）。這支只做
合併與檢查，不推論任何欄位。

欄位的定義
    helmet_appeared  頭上有一個實心、可辨識為安全帽的殼體。只有一條薄帶、
                     或能透見頭髮的薄膜記 0。
    extra_person     畫面多出原圖沒有的人臉或人頭（含假人頭）。
    face_swapped     主體的臉不再是原圖那個人。**帽體把臉遮住時留空**，
                     那一格無法判定，留空與記 0 是兩件事。
    usable           前三欄的機械組合：有帽、沒有多出來的人、沒有換臉、
                     臉沒有被帽體遮住。

用法
    python scripts/ip2p_helmet_review.py \
        --sweep runs/ip2p_helmet_sweep/guidance_grid/sweep.csv \
        --review runs/ip2p_helmet_sweep/review.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ("helmet_appeared", "extra_person", "face_swapped", "usable")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def check(review: list[dict], sweep: list[dict]) -> dict[tuple[str, str], dict]:
    """判讀與掃描必須逐格一一對應，值只能是 0、1，或 face_swapped 的留空。"""
    indexed = {}
    for row in review:
        key = (row["cell"], row["image"])
        if key in indexed:
            raise SystemExit(f"判讀有重複的格子：{key}")
        for field in FIELDS:
            value = row[field].strip()
            if value in ("0", "1"):
                continue
            if value == "" and field == "face_swapped" and row["review_notes"].strip():
                continue
            raise SystemExit(
                f"{key} 的 {field} 是 {value!r}：只能填 0 或 1，"
                "face_swapped 留空時必須在 review_notes 說明為什麼無法判定")
        indexed[key] = row
    expected = {(row["cell"], row["image"]) for row in sweep}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise SystemExit(f"判讀與掃描對不起來。缺 {missing}；多 {extra}")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    sweep = read_csv(args.sweep)
    indexed = check(read_csv(args.review), sweep)

    merged = []
    for row in sweep:
        judgement = indexed[(row["cell"], row["image"])]
        merged.append(dict(row, review_status="reviewed",
                           review_notes=judgement["review_notes"],
                           **{field: judgement[field].strip() for field in FIELDS}))
    write_csv(args.sweep, merged)

    order = list(dict.fromkeys(row["cell"] for row in merged))
    width = max(len(cell) for cell in order)
    print(f"{'cell':<{width}}  s_t  s_i  帽  多人  換臉  可用")
    for cell in order:
        rows = [row for row in merged if row["cell"] == cell]
        counts = [sum(row[field] == "1" for row in rows) for field in FIELDS]
        first = rows[0]
        print(f"{cell:<{width}}  {first['s_t']:>3}  {first['s_i']:>3}"
              f"  {counts[0]:>2}  {counts[1]:>4}  {counts[2]:>4}  {counts[3]:>4}")


if __name__ == "__main__":
    main()
