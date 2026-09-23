"""把**已經跑好的防禦圖**併成臨時報告頁用的拼圖（無損 WebP）。

與 `build_report_assets.py` 的差別：那一支收完整批次（防禦＋編輯＋淨化後編輯，
896 格），這一支只收防禦圖與它的副圖，給「跑到一半先看一眼」用。
還沒跑完的臂會缺格，缺幾格逐列印出來，**不靜默補空白**。

用法
    python lab/scripts/peek_sheets.py --arms ab_warp style_affine ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TILE = 512
IMAGES = ["man_00", "man_01", "man_02", "man_03",
          "woman_00", "woman_01", "woman_02", "woman_03"]


def sheet(paths, out: Path, cols: int = 4) -> dict:
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * TILE, rows * TILE), (24, 24, 28))
    missing = 0
    for i, p in enumerate(paths):
        if p is None or not Path(p).is_file():
            missing += 1
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE), Image.LANCZOS)
        canvas.paste(im, ((i % cols) * TILE, (i // cols) * TILE))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "WEBP", lossless=True, quality=100, method=6)
    return {"file": out.name, "cols": cols, "rows": rows,
            "bytes": out.stat().st_size, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("lab/runs/defence"))
    ap.add_argument("--out", type=Path, default=Path("lab/report_peek/img"))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--aux", nargs="*", default=[],
                    help="`<臂>:<後綴>`，例如 ab_warp:warp、inpaint_bg:raw")
    args = ap.parse_args()

    manifest = {"tile": TILE, "images": IMAGES, "arms": args.arms, "sheets": {}}

    def add(key: str, paths, name: str) -> None:
        info = sheet(paths, args.out / name)
        manifest["sheets"][key] = info
        size = info["bytes"] / 1e6
        print(f"{key:34s} {name:34s} {size:6.2f} MB  缺 {info['missing']}",
              flush=True)

    add("original", [args.root / args.arms[0] / f"{n}__orig.png" for n in IMAGES],
        "def_original.webp")
    for arm in args.arms:
        add(arm, [args.root / arm / f"{n}__{arm}__def.png" for n in IMAGES],
            f"def_{arm}.webp")
    for spec in args.aux:
        arm, suffix = spec.split(":", 1)
        add(f"aux/{arm}/{suffix}",
            [args.root / arm / f"{n}__{suffix}.png" for n in IMAGES],
            f"aux_{arm}_{suffix}.webp")

    total = sum(v["bytes"] for v in manifest["sheets"].values())
    miss = sum(v["missing"] for v in manifest["sheets"].values())
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n拼圖 {len(manifest['sheets'])} 張，合計 {total/1e6:.1f} MB，"
          f"缺 {miss} 格 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
