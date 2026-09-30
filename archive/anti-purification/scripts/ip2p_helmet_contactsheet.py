"""把 helmet 掃描的輸出排成可逐格看圖的接觸印樣（contact sheet）。

掃描的判定是看圖，不是看數字：每一格要回答安全帽有沒有出現、有沒有多出
第二個人、臉有沒有被換掉、整體可不可用。四件事裡有三件只有把同一張影像的
原圖與編輯結果放在一起才看得出來，所以這支只做排版、不算任何指標。

兩種輸出
    grid  每個設定一張 4×2 的總覽（八張人像，順序固定為
          man_00..03、woman_00..03），另外附一張原圖的總覽當對照。
    pair  指定一個設定，逐張輸出「原圖｜編輯結果」的並排全解析度圖，
          用在總覽上看不清楚的格子（例如判斷臉有沒有被換掉）。

用法
    python scripts/ip2p_helmet_contactsheet.py \
        --sweep-dir runs/ip2p_helmet_sweep/guidance_grid \
        --out runs/ip2p_helmet_sweep/sheets
    python scripts/ip2p_helmet_contactsheet.py ... --pair-cell t75_i15
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PORTRAITS = [f"{sex}_{i:02d}" for sex in ("man", "woman") for i in range(4)]
COLUMNS = 4
LABEL_SCALE = 3
MARGIN = 6


def label_strip(text: str, width: int) -> Image.Image:
    """用內建點陣字型畫標籤再整條放大，避免相依任何 TTF 檔案。"""
    font = ImageFont.load_default()
    small = Image.new("RGB", (max(width // LABEL_SCALE, 1), 14), "black")
    ImageDraw.Draw(small).text((2, 1), text, fill="white", font=font)
    return small.resize((width, 14 * LABEL_SCALE), Image.NEAREST)


def stacked(image: Image.Image, text: str) -> Image.Image:
    strip = label_strip(text, image.width)
    tile = Image.new("RGB", (image.width, image.height + strip.height), "black")
    tile.paste(image, (0, 0))
    tile.paste(strip, (0, image.height))
    return tile


def grid(tiles: list[Image.Image]) -> Image.Image:
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    width, height = tiles[0].width, tiles[0].height
    sheet = Image.new(
        "RGB",
        (COLUMNS * width + (COLUMNS + 1) * MARGIN, rows * height + (rows + 1) * MARGIN),
        "black")
    for index, tile in enumerate(tiles):
        column, row = index % COLUMNS, index // COLUMNS
        sheet.paste(tile, (MARGIN + column * (width + MARGIN),
                           MARGIN + row * (height + MARGIN)))
    return sheet


def read_rows(sweep_dir: Path) -> list[dict]:
    path = sweep_dir / "sweep.csv"
    if not path.is_file():
        raise SystemExit(f"找不到 {path}，掃描還沒寫出任何格子")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def cell_images(rows: list[dict], cell: str, repo: Path) -> dict[str, Path]:
    """掃描 CSV 的 output_png 是相對 repo 根目錄的路徑，逐張存在才回傳。"""
    selected = {row["image"]: repo / row["output_png"] for row in rows if row["cell"] == cell}
    missing = [name for name in PORTRAITS if name not in selected]
    if missing:
        raise SystemExit(f"設定 {cell} 缺這幾張：{' '.join(missing)}")
    for path in selected.values():
        if not path.is_file():
            raise SystemExit(f"CSV 指到不存在的檔案：{path}")
    return selected


def originals(rows: list[dict], repo: Path) -> dict[str, Path]:
    """原圖每個臂都存了一份，取第一個臂的即可（同一份輸入、同一個前處理）。"""
    first = rows[0]
    directory = (repo / first["output_png"]).parent
    return {name: directory / f"{name}__orig.png" for name in PORTRAITS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-cell", default=None,
                        help="只輸出這個設定的逐張並排圖，不畫總覽")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    rows = read_rows(args.sweep_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    source = originals(rows, repo)

    if args.pair_cell is not None:
        edited = cell_images(rows, args.pair_cell, repo)
        for name in PORTRAITS:
            left = stacked(Image.open(source[name]), f"{name} orig")
            right = stacked(Image.open(edited[name]), f"{name} {args.pair_cell}")
            pair = Image.new("RGB", (left.width + right.width + 3 * MARGIN,
                                     left.height + 2 * MARGIN), "black")
            pair.paste(left, (MARGIN, MARGIN))
            pair.paste(right, (2 * MARGIN + left.width, MARGIN))
            pair.save(args.out / f"pair_{args.pair_cell}_{name}.png")
        print(f"寫出 {len(PORTRAITS)} 張並排圖到 {args.out}")
        return

    sheet = grid([stacked(Image.open(source[name]), name) for name in PORTRAITS])
    sheet.save(args.out / "sheet_original.png")
    written = ["sheet_original.png"]
    for cell in dict.fromkeys(row["cell"] for row in rows):
        edited = cell_images(rows, cell, repo)
        first = next(row for row in rows if row["cell"] == cell)
        title = (f"{cell}  s_t={first['s_t']} s_i={first['s_i']}"
                 f"  neg={'yes' if first['negative_prompt'] else 'no'}")
        tiles = [stacked(Image.open(edited[name]), f"{name}  {title}")
                 for name in PORTRAITS]
        grid(tiles).save(args.out / f"sheet_{cell}.png")
        written.append(f"sheet_{cell}.png")
    print(f"寫出 {len(written)} 張總覽到 {args.out}: {' '.join(written)}")


if __name__ == "__main__":
    main()
