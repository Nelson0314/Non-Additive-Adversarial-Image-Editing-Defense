"""把多個分片目錄併成一個批次目錄，供 `phase_retention.py` 讀。

平行跑必須分片：`write_csv` 每次呼叫都整份覆寫 `results.csv`，兩個行程寫
同一個目錄會互相蓋掉（**不是鎖的問題，是覆寫語意**）。故每張卡寫自己的
`--out`，最後由本腳本併起來。

`--pick` 可以只挑走來源目錄裡的一部分列（例如從 24 張的 `phaseA_human`
只取本批次的 5 張）；PNG 也只複製對應那幾列的。

用法：
    python scripts/merge_runs.py --out runs/hb5 \
        --src runs/hb5/g0 runs/hb5/g1 runs/hb5/g2 runs/hb5/g3 \
        --src-pick runs/phaseA_human \
        --images man_02 woman_02 dog_03 horse_03 cat_01
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_retention import cell_of  # noqa: E402

from src.utils.io import write_csv  # noqa: E402

# 每一列對應的三張圖。`orig` 不帶 tag，逐影像一張，重複複製是冪等的。
SUFFIXES = ("def", "edit_orig", "edit_def")


def read_rows(src: Path) -> list[dict]:
    path = src / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"分片缺少 {path}")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, nargs="*", default=[],
                    help="整個併入的分片目錄")
    ap.add_argument("--src-pick", type=Path, nargs="*", default=[],
                    help="只併入 --images 指定影像的來源目錄")
    ap.add_argument("--images", nargs="+", default=None,
                    help="--src-pick 的影像白名單")
    args = ap.parse_args()

    if args.src_pick and not args.images:
        raise SystemExit("--src-pick 需要 --images：不給白名單就等同 --src")
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    copied = missing = 0
    for src, keep in ([(s, None) for s in args.src]
                      + [(s, set(args.images)) for s in args.src_pick]):
        for row in read_rows(src):
            if keep is not None and row["image"] not in keep:
                continue
            cell = cell_of(row)
            names = [f"{cell['image']}__orig.png"] + [
                f"{cell['image']}__{cell['tag']}__{s}.png" for s in SUFFIXES]
            for name in names:
                srcp = src / name
                if not srcp.exists():
                    # def 圖缺了 retention 會停住，這裡先報出來而不是留到那時。
                    print(f"[缺圖] {srcp}", flush=True)
                    missing += 1
                    continue
                shutil.copy2(srcp, args.out / name)
                copied += 1
            rows.append(row)

    if missing:
        raise SystemExit(f"有 {missing} 張圖缺失，不產出 results.csv")
    write_csv(args.out / "results.csv", rows)
    conds = sorted({r["condition"] for r in rows})
    imgs = sorted({r["image"] for r in rows})
    print(f"併入 {len(rows)} 列、複製 {copied} 張圖 → {args.out}")
    print(f"  條件 {len(conds)}：{', '.join(conds)}")
    print(f"  影像 {len(imgs)}：{', '.join(imgs)}")


if __name__ == "__main__":
    main()
