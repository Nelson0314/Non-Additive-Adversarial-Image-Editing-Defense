"""主線頭對頭的人眼比對大圖：**靜態、原生 512、不縮放**。

五欄，就是一條完整的攻防路徑：

    原圖 │ 原圖的編輯 │ 防禦圖 │ 防禦圖被淨化後 │ 淨化後的編輯

「原圖的編輯」是判準的參照點——沒有它就看不出模型本來會把這張圖畫成什麼樣，
也就無從判斷防禦有沒有把它推開。最後一欄才是真正要判「擋下與否」的那一格。

**一律原生 512 不縮放**：判定看的是語意內容，縮放會把細節抹掉，邊緣案例會判錯。
輸出很大（五欄 × 十列約 2650 × 5450），這是刻意的。

**不跑 GPU、不重算任何數字**，只讀已存的 PNG：
`ip2p_run.py` 的 `{name}__orig.png` 與 `{name}__{cond}__edit_orig/def/edit_def.png`，
以及 `phase_retention.py --gallery` 的
`{name}__{cond}__{purifier}__pur.png` 與 `__edit_def.png`。

用法：
    python scripts/mainline_sheets.py --tag ours_pg_q --purifier jpeg75 \\
        --defense runs/ip2p_mainline --gallery runs/gallery_mainline \\
        --out runs/ip2p_mainline/sheets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

TILE = 512
PAD = 16
TITLE_BAND = 46
HEAD_BAND = 34
ROWLAB = 30
BG = 0.5

FONT = {
    "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"],
    "A": ["010", "101", "111", "101", "101"], "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"], "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"], "F": ["111", "100", "110", "100", "100"],
    "G": ["011", "100", "101", "101", "011"], "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"], "J": ["001", "001", "001", "101", "010"],
    "K": ["101", "110", "100", "110", "101"], "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"], "N": ["101", "111", "111", "111", "101"],
    "O": ["010", "101", "101", "101", "010"], "P": ["110", "101", "110", "100", "100"],
    "Q": ["010", "101", "101", "111", "011"], "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"], "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"], "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"], "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"], "Z": ["111", "001", "010", "100", "111"],
    "_": ["000", "000", "000", "000", "111"], "-": ["000", "000", "111", "000", "000"],
    ".": ["000", "000", "000", "000", "010"], "+": ["000", "010", "111", "010", "000"],
    "/": ["001", "001", "010", "100", "100"], " ": ["000", "000", "000", "000", "000"],
    "#": ["101", "111", "101", "111", "101"], ":": ["000", "010", "000", "010", "000"],
}


def draw_text(canvas, text, top, left, scale=3, value=0.0):
    x = left
    for ch in text.upper():
        rows = FONT.get(ch)
        if rows is None:
            x += 4 * scale
            continue
        for r, bits in enumerate(rows):
            for c, b in enumerate(bits):
                if b == "1":
                    y0, x0 = top + r * scale, x + c * scale
                    if y0 + scale <= canvas.shape[1] and x0 + scale <= canvas.shape[2]:
                        canvas[:, y0:y0 + scale, x0:x0 + scale] = value
        x += 4 * scale


def tile_of(path: Optional[Path]) -> torch.Tensor:
    """讀一格。缺檔時畫深灰並標明——留白與「這一格是黑圖」在視覺上分不開。"""
    if path is None or not path.exists():
        t = torch.full((3, TILE, TILE), 0.25)
        draw_text(t, "MISSING", TILE // 2 - 10, TILE // 2 - 90, scale=5, value=0.7)
        return t
    return load_image_tensor(path, torch.device("cpu")).clamp(0, 1)[0]


def build(rows: List[Dict], headers: List[str], title: str) -> torch.Tensor:
    """標題與欄名各佔一條——共用一條時字會疊在一起。"""
    cols = len(headers)
    top = TITLE_BAND + HEAD_BAND
    h = top + len(rows) * (ROWLAB + TILE + PAD) + PAD
    w = cols * TILE + (cols + 1) * PAD
    canvas = torch.full((3, h, w), BG)
    draw_text(canvas, title, 10, PAD, scale=5)
    for c, name in enumerate(headers):
        draw_text(canvas, name, TITLE_BAND + 8, (c + 1) * PAD + c * TILE, scale=4)
    y = top
    for row in rows:
        draw_text(canvas, row["label"], y + 4, PAD, scale=3)
        y += ROWLAB
        for c in range(cols):
            x0 = (c + 1) * PAD + c * TILE
            canvas[:, y:y + TILE, x0:x0 + TILE] = tile_of(row["paths"][c])
        y += TILE + PAD
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", nargs="+", required=True, help="要出圖的條件（可多個）")
    ap.add_argument("--purifier", default="jpeg75")
    ap.add_argument("--defense", type=Path, default=Path("runs/ip2p_mainline"))
    ap.add_argument("--gallery", type=Path, default=Path("runs/gallery_mainline"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    for tag in args.tag:
        d = args.defense / tag
        g = args.gallery / tag

        def one(pat: str, name: str, exclude_edit=False):
            hits = sorted(d.glob(pat.format(n=name)))
            if exclude_edit:
                hits = [p for p in hits if not p.name.endswith("__edit_def.png")]
            return hits[0] if hits else None

        rows = []
        for i, n in enumerate(names, 1):
            gh = sorted(g.glob(f"{n}__*__{args.purifier}__pur.png"))
            ge = sorted(g.glob(f"{n}__*__{args.purifier}__edit_def.png"))
            rows.append({
                "label": f"#{i:02d} {n[:34]}",
                "paths": [
                    one("{n}__orig.png", n),
                    one("{n}__*__edit_orig.png", n),
                    one("{n}__*__def.png", n, exclude_edit=True),
                    gh[0] if gh else None,
                    ge[0] if ge else None,
                ],
            })
        headers = ["ORIG", "EDIT OF ORIG", "DEFENDED",
                   f"DEFENDED +{args.purifier}", f"EDIT AFTER {args.purifier}"]
        sheet = build(rows, headers, f"{tag}  vs  {args.purifier}")
        out = args.out / f"{tag}__{args.purifier}.png"
        save_image(sheet, out)
        miss = sum(1 for r in rows for p in r["paths"] if p is None or not p.exists())
        print(f"寫出 {out}  {sheet.shape[2]}x{sheet.shape[1]}"
              + (f"  **缺 {miss} 格**" if miss else ""))


if __name__ == "__main__":
    main()
