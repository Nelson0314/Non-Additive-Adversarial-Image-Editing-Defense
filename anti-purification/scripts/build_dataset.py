"""由候選池與一份選單組出資料集目錄。**會覆寫輸出目錄的影像。**

輸入是 `scripts/screen_candidates.py` 的產物（`crops/` 與 `screen.csv`）加上
一份逐類的選單檔：每行一個 `screen.csv` 的 `image` 欄值，空行與 `#` 開頭略過。
選單由人看圖決定，這一支不挑。

輸出的版面與 `data/lo_aligned` 相同：每類一個子目錄、檔名 `<類別>_<序號>.png`、
`prompts.yaml` 留在原地不動（類別與編輯 prompt 不由這一支決定）。

留下什麼證據
────────────────────────────────────────────────────────────────────
- `provenance.json`：逐張的輸出檔名、來源檔、來源 sha256、裁切邊長、輸出尺寸，
  以及 `screen.csv` 量到的每一欄。**sha256 取的是候選池裡那個檔**，所以
  「這張圖是從哪來的」與「當時量到什麼」在同一列可以對上。
- 每類的 `attribution.json`：從候選池的同名檔案抄過來，只留被選中的那幾筆。
  授權與作者是引用時必須附的，不能只留在池子裡。

**舊影像會被刪掉。** 輸出目錄裡既有的 `*.png` 在寫入前全數移除——半新半舊的
資料集會讓後續批次的讀數無法歸因到哪一批影像，比整批重來更糟。

用法
    python scripts/build_dataset.py --screen runs/candidate_screen \\
        --select runs/candidate_screen/selection.txt --out data/lo_aligned
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def read_selection(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", type=Path, required=True,
                    help="screen_candidates.py 的輸出目錄")
    ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()

    rows = {r["image"]: r for r in csv.DictReader(
        open(args.screen / "screen.csv", encoding="utf-8"))}
    picks = read_selection(args.select)
    missing = [p for p in picks if p not in rows]
    if missing:
        raise SystemExit(f"選單裡有 screen.csv 沒有的項目：{missing}")

    by_class = {}
    for name in picks:
        by_class.setdefault(rows[name]["class"], []).append(name)
    bad = {c: len(v) for c, v in by_class.items() if len(v) != args.per_class}
    if bad:
        raise SystemExit(
            f"每類要 {args.per_class} 張，實際 {bad}。"
            "數量不齊的資料集會讓逐類的讀數不可比，故不接受。")

    for cls, names in sorted(by_class.items()):
        d = args.out / cls
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        pool_attr = {}
        records, attribution = [], []
        for i, name in enumerate(names):
            r = rows[name]
            crop = args.screen / "crops" / f"{name}.jpg"
            src = Path(r["path"])
            out_name = f"{cls}_{i:02d}.png"
            Image.open(crop).convert("RGB").save(d / out_name)
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            records.append({
                "output": f"{cls}/{out_name}", "candidate": name,
                "source": str(src).replace("\\", "/"), "source_sha256": sha,
                "output_size": 512,
                "subject_frac": float(r["subject_frac"]),
                "n_parts": int(r["n_parts"]),
                "largest_part_frac": float(r["largest_part_frac"]),
                "centre_offset": float(r["centre_offset"]),
                "sharpness": float(r["sharpness"]),
                "colour_count": int(r["colour_count"]),
                "edge_ratio": float(r["edge_ratio"]),
                "faces": r["faces"],
            })
            if not pool_attr:
                ap_path = src.parent / "attribution.json"
                pool_attr = {a["file"]: a for a in json.loads(
                    ap_path.read_text(encoding="utf-8"))}
            entry = dict(pool_attr[src.name])
            entry["output"] = f"{cls}/{out_name}"
            attribution.append(entry)
        (d / "attribution.json").write_text(
            json.dumps(attribution, ensure_ascii=False, indent=2),
            encoding="utf-8")
        args.out.mkdir(parents=True, exist_ok=True)
        prev = args.out / "provenance.json"
        allrec = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else []
        allrec = [x for x in allrec if not x.get("output", "").startswith(f"{cls}/")]
        allrec += records
        prev.write_text(json.dumps(allrec, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"{cls}: {len(records)} 張 -> {d}")

    stale = args.out / "masks"
    if stale.exists():
        shutil.rmtree(stale)
        print(f"刪掉 {stale}（遮罩對應舊影像，要重畫）")
    print("接著跑：python scripts/make_masks.py --data", args.out)


if __name__ == "__main__":
    main()
