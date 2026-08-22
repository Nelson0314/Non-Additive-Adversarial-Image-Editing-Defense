"""把「原圖 → 編輯」的成對影像排成對照張，供人眼逐張判定編輯有沒有成功。

為什麼需要它
────────────────────────────────────────────────────────────────────
本專案的主讀數是位移 `LPIPS(編輯(原圖), 編輯(防禦圖))`。這個量只有在
**未防禦的編輯確實執行了指令**時才有意義——若攻擊本來就失敗，兩張輸出之間
的差異量到的是擴散取樣的雜訊，不是防禦。

`RESULTS.md` 已記載服從率不能用 CLIP／SigLIP 驗收（OmniEdit 給的是**指令**
而 CLIP 量的是影像與**描述**的相符度，25 張上的語意增益 15/25 為正，近乎
隨機），並裁定「服從率改由人眼判定」。本檔提供人眼判定所需的版面。

版面
────────────────────────────────────────────────────────────────────
每一列一組，左為輸入、右為輸出，列與列之間留白，左上角標序號。序號用來把
判定寫回 CSV，不做任何自動判讀。

用法：
    python scripts/edit_contact_sheet.py --src <PNG 目錄> --condition phase_s0100 \
        --pair orig edit_orig --out runs/obedience_audit/sheet --rows 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from defense_purify_gallery import RESOLUTION, discover  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

TILE = 320          # 每格邊長
PAD = 10            # 格與格、列與列的間距
LABEL = 26          # 序號帶的高度
BG = 0.5            # 中性灰底，與比對頁同一階


def _digit_glyphs() -> dict:
    """3x5 點陣數字。刻意不引字型檔——對照張只需要序號可讀，
    而多一個字型相依就多一個在遠端跑不起來的理由。"""
    raw = {
        "0": "111101101101111", "1": "010110010010111",
        "2": "111001111100111", "3": "111001111001111",
        "4": "101101111001001", "5": "111100111001111",
        "6": "111100111101111", "7": "111001001001001",
        "8": "111101111101111", "9": "111101111001111",
        "#": "101111101111101", "/": "001001010100100",
    }
    return {k: [[int(c) for c in v[r * 3:(r + 1) * 3]] for r in range(5)]
            for k, v in raw.items()}


GLYPHS = _digit_glyphs()


def draw_text(canvas: torch.Tensor, text: str, top: int, left: int,
              scale: int = 4) -> None:
    """就地畫上白字。canvas 是 (3,H,W)。"""
    x = left
    for ch in text:
        g = GLYPHS.get(ch)
        if g is None:
            x += 4 * scale
            continue
        for r in range(5):
            for c in range(3):
                if g[r][c]:
                    y0, x0 = top + r * scale, x + c * scale
                    canvas[:, y0:y0 + scale, x0:x0 + scale] = 1.0
        x += 4 * scale


def tile_of(path: Path, tile: int = TILE) -> torch.Tensor:
    """讀一格。`tile == RESOLUTION` 時**不縮放**，逐位元就是原生 512。

    縮放會把接縫與鱗片重影平均掉，而那正是判「單純劣化還是不可辨」時要看的
    東西——`runs/obedience_audit` 的判定就是在原生解析度上做的。
    """
    x = load_image_tensor(path, torch.device("cpu"), size=RESOLUTION)
    if tile == RESOLUTION:
        return x[0]
    return F.interpolate(x, size=(tile, tile), mode="area")[0]


def build_sheet(rows: List[Tuple[int, List[Path]]],
                tile: int = TILE) -> torch.Tensor:
    """每一列一組，欄數由該列給幾張圖決定（至少兩欄）。

    三欄的用途是判「原圖還認得出來嗎」：原圖 → 未防禦的編輯 → 防禦後的編輯。
    只有兩欄時看不出未防禦的編輯本來畫成什麼樣，而那是判準的參照點。
    """
    if not rows:
        raise ValueError("沒有任何一列可排")
    cols = len(rows[0][1])
    if any(len(paths) != cols for _, paths in rows):
        raise ValueError("每一列的欄數必須相同")
    if cols < 2:
        raise ValueError(f"至少兩欄，收到 {cols}")
    n = len(rows)
    h = n * (tile + LABEL + PAD) + PAD
    w = cols * tile + (cols + 1) * PAD
    canvas = torch.full((3, h, w), BG)
    y = PAD
    for idx, paths in rows:
        draw_text(canvas, f"#{idx}", y + 4, PAD, scale=4)
        y += LABEL
        for c, path in enumerate(paths):
            x0 = (c + 1) * PAD + c * tile
            canvas[:, y:y + tile, x0:x0 + tile] = tile_of(path, tile)
        y += tile + PAD
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--condition", required=True,
                    help="批次子目錄名，如 phase_s0100")
    ap.add_argument("--right-condition", default=None,
                    help="右格改取另一個條件的同一張影像。用來在**同一個失真"
                         "水準**上把兩個方法的防禦圖並排——這是「擋下率高是不"
                         "是靠把照片弄爛換來的」唯一能判的方式")
    ap.add_argument("--pair", nargs="+", default=["orig", "edit_orig"],
                    help="每一欄各取哪一種：orig / def / edit_orig / edit_def。"
                         "兩欄是預設；三欄（orig edit_orig edit_def）才看得出"
                         "未防禦的編輯本來畫成什麼樣，而那是「原圖還認得出來"
                         "嗎」這個判準的參照點")
    ap.add_argument("--out", type=Path, required=True, help="輸出檔名前綴")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile", type=int, default=TILE,
                    help="每格邊長。給 512 即原生解析度不縮放——判"
                         "「單純劣化還是不可辨」要看接縫與鱗片重影，"
                         "縮放會把它們平均掉")
    ap.add_argument("--images", nargs="+", default=None,
                    help="只排這些影像。用來把對照張限制在「未防禦編輯確實"
                         "執行了指令」的子集上——攻擊本來就失敗的影像，兩張"
                         "輸出之間的差異量到的是取樣雜訊而不是防禦")
    ap.add_argument("--data", type=Path, default=None,
                    help="原圖的資料集根目錄（<name>/<name>.png）。`--pair` 的"
                         "左格取 orig 而 PNG 目錄裡沒有 __orig.png 時用它補上")
    args = ap.parse_args()

    if len(args.pair) < 2:
        raise SystemExit("--pair 至少要兩欄")
    found = discover(args.src)
    items: List[Tuple[str, List[Path]]] = []
    for image in sorted(found):
        if args.images and image not in args.images:
            continue
        by_cond = found[image]
        kinds = by_cond.get(args.condition, {})
        other = (by_cond.get(args.right_condition, {})
                 if args.right_condition else kinds)
        paths: List[Path] = []
        missing = None
        for col, kind in enumerate(args.pair):
            src = kinds if col == 0 else other
            path = src.get(kind)
            if path is None and kind == "orig":
                for c in by_cond.values():
                    if "orig" in c:
                        path = c["orig"]
                        break
                if path is None and args.data is not None:
                    cand = args.data / image / f"{image}.png"
                    path = cand if cand.exists() else None
            if path is None:
                missing = kind
                break
            paths.append(path)
        if missing is not None:
            print(f"  略過 {image}：缺 {missing}")
            continue
        items.append((image, paths))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    index_lines = []
    from PIL import Image
    for s in range(0, len(items), args.rows):
        chunk = items[s:s + args.rows]
        sheet = build_sheet(
            [(s + k + 1, paths) for k, (_, paths) in enumerate(chunk)],
            tile=args.tile)
        arr = (sheet.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
        out = args.out.with_name(f"{args.out.name}_{s // args.rows + 1:02d}.png")
        Image.fromarray(arr).save(out)
        for k, (image, _) in enumerate(chunk):
            index_lines.append(f"{s + k + 1}\t{image}\t{out.name}")
        print(f"  {out}  {len(chunk)} 組")
    (args.out.with_name(args.out.name + "_index.tsv")).write_text(
        "序號\t影像\t對照張\n" + "\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"索引：{args.out.with_name(args.out.name + '_index.tsv')}")


if __name__ == "__main__":
    main()
