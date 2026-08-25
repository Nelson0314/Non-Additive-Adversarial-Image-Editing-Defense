"""量化交付的人眼比對大圖：**靜態、原生 512、不縮放**。

`docs/GOAL.md` 的判準是「原圖還認得出來嗎」，而那是人眼判定、指標只是輔助。
本檔把同一張影像在四個條件下、淨化前後的編輯輸出排成一張大圖，讓「擋下與否」
可以逐張看過再下判定。

**不跑 GPU、不重算任何數字**，只讀已存的 PNG。

版面
────────────────────────────────────────────────────────────────────
每一列是一張影像，每一欄是一個階段：

    原圖 │ 未防禦的編輯 │ <條件 A 的編輯> │ <條件 B 的編輯> │ …

「未防禦的編輯」那一欄是判準的參照點——沒有它就看不出模型本來會把這張圖畫成
什麼樣，也就無從判斷防禦有沒有把它推開（`edit_contact_sheet.py` 的同一個理由）。

**一律原生 512 不縮放**：擋下與否的判定看的是語意內容，而縮放會把細節抹掉，
邊緣案例會判錯。輸出因此很大（八欄 × 十三列約 4200 × 7000 像素），這是刻意的。

用法：
    python scripts/deliver_jpeg_sheets.py \\
        --stage-dir ours_add=runs/ip2p_axis_necessity/b_pg_r20 \\
        --gallery-dir ours_add=runs/gallery_deliver_jpeg/ours_add \\
        --purifier jpeg75 --out runs/ip2p_deliver_jpeg/sheets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

TILE = 512
PAD = 16
TITLE_BAND = 46     # 大標題自己一條
HEAD_BAND = 34      # 欄名自己一條
ROWLAB = 30         # 每一列的列名
BG = 0.5            # 中性灰。環境亮度會改變感知對比，固定住才可比

# 5×7 點陣字，只夠畫這一頁需要的字元。沿用 edit_contact_sheet.py 的作法：
# 不引入字型相依，因為那會讓不同機器上跑出來的圖不同。
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


def draw_text(canvas: torch.Tensor, text: str, top: int, left: int,
              scale: int = 3, value: float = 0.0) -> None:
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
    """讀一格。**不縮放**——來源就是 512。缺檔時畫一格深灰並標明，不留白：
    留白與「這一格是黑圖」在視覺上分不開。"""
    if path is None or not path.exists():
        t = torch.full((3, TILE, TILE), 0.25)
        draw_text(t, "MISSING", TILE // 2 - 10, TILE // 2 - 90, scale=5, value=0.7)
        return t
    return load_image_tensor(path, torch.device("cpu")).clamp(0, 1)[0]


def build(rows: List[Dict], headers: List[str], title: str) -> torch.Tensor:
    """標題與欄名各佔一條，**不可共用同一條**——共用時字會疊在一起。
    （第一版把標題畫在 y=8、欄名畫在 y=LABEL−20=14，兩者重疊到看不清楚，
    是把圖印出來看才發現的。）"""
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
    ap.add_argument("--stage-dir", nargs="+", required=True,
                    help="tag=目錄，`ip2p_run.py` 的輸出（orig／edit_orig／edit_def）")
    ap.add_argument("--gallery-dir", nargs="+", default=[],
                    help="tag=目錄，`phase_retention.py --gallery` 的輸出")
    ap.add_argument("--cond", nargs="+", default=[],
                    help="tag=條件名，gallery 檔名裡的那一段（例如 phase_gain）")
    ap.add_argument("--purifier", default="jpeg75",
                    help="要看哪一個淨化算子之後的編輯")
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def parse(pairs):
        out = {}
        for p in pairs:
            if "=" not in p:
                raise SystemExit(f"要寫成 tag=值，收到 {p!r}")
            k, v = p.split("=", 1)
            out[k] = v
        return out

    stage = {k: Path(v) for k, v in parse(args.stage_dir).items()}
    gal = {k: Path(v) for k, v in parse(args.gallery_dir).items()}
    cond = parse(args.cond)
    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    tags = list(stage)
    args.out.mkdir(parents=True, exist_ok=True)

    def edit_def(tag: str, name: str) -> Optional[Path]:
        d = stage[tag]
        hits = sorted(d.glob(f"{name}__*__edit_def.png"))
        return hits[0] if hits else None

    def def_png(tag: str, name: str) -> Optional[Path]:
        """防禦圖。**不可用 `{name}__*__def.png`**：那個樣式也會匹配到
        `{name}__{cond}__edit_def.png`（`*` 吃得下 `{cond}__edit`），
        目前只是碰巧因為 `d` 排在 `e` 前面而取對，是靠排序的巧合。
        改成排除 `edit_` 開頭的那一段。"""
        hits = [p for p in sorted(stage[tag].glob(f"{name}__*__def.png"))
                if not p.name.endswith("__edit_def.png")]
        return hits[0] if hits else None

    def purified_edit(tag: str, name: str) -> Optional[Path]:
        if tag not in gal:
            return None
        c = cond.get(tag)
        if c is None:
            hits = sorted(gal[tag].glob(f"{name}__*__{args.purifier}__edit_def.png"))
            return hits[0] if hits else None
        return gal[tag] / f"{name}__{c}__{args.purifier}__edit_def.png"

    first = tags[0]
    # 檔名的兩種形狀（`ip2p_run.py` 的存檔那一段）：原圖是 `{name}__orig.png`，
    # 其餘三個都夾著條件名 `{name}__{cond}__{sub}.png`。**`edit_orig` 屬於後者**
    # ——第一版漏了條件那一段，於是「未防禦的編輯」整欄變成 MISSING，而那正是
    # 判準的參照點。是把圖印出來看才發現的。
    orig = {n: sorted(stage[first].glob(f"{n}__orig.png")) for n in names}
    eo = {n: sorted(stage[first].glob(f"{n}__*__edit_orig.png")) for n in names}

    # ---- 表一：淨化前後的編輯輸出，這是「擋下與否」的判定表 ----
    headers = ["ORIG", "NO DEFENSE"]
    for t in tags:
        headers += [f"{t}", f"{t} {args.purifier}"]
    rows = []
    for i, n in enumerate(names, 1):
        paths = [orig[n][0] if orig[n] else None, eo[n][0] if eo[n] else None]
        for t in tags:
            paths += [edit_def(t, n), purified_edit(t, n)]
        rows.append({"label": f"#{i:02d} {n[:34]}", "paths": paths})
    sheet = build(rows, headers, f"EDIT OUTPUTS  BEFORE AND AFTER {args.purifier}")
    save_image(sheet, args.out / f"edits_{args.purifier}.png")
    print(f"寫出 {args.out / f'edits_{args.purifier}.png'}  "
          f"{sheet.shape[2]}x{sheet.shape[1]}")

    # ---- 表二：防禦圖本身，判可見度 ----
    headers2 = ["ORIG"] + [f"{t} DEFENDED" for t in tags]
    rows2 = []
    for i, n in enumerate(names, 1):
        paths = [orig[n][0] if orig[n] else None] + [def_png(t, n) for t in tags]
        rows2.append({"label": f"#{i:02d} {n[:34]}", "paths": paths})
    sheet2 = build(rows2, headers2, "DEFENDED IMAGES  VISIBILITY")
    save_image(sheet2, args.out / "defended.png")
    print(f"寫出 {args.out / 'defended.png'}  {sheet2.shape[2]}x{sheet2.shape[1]}")


if __name__ == "__main__":
    main()
