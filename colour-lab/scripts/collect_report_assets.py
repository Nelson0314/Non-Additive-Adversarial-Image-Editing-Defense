"""把一批的防禦圖與編輯圖收成報告頁要用的檔案清單。

為什麼要挑
────────────────────────────────────────────────────────────────────
artifact 的硬限制是**整份 256 個檔、每版 64 MB**。防禦圖要逐像素判自然度，
所以用**無損** WebP；編輯圖只是看攻擊結果，用**有損** WebP（預設 q90），
這件事要寫在頁面上，不藏。

挑選規則
────────────────────────────────────────────────────────────────────
- 防禦圖：每個條件 × 每張影像，全收（自然度是這一頁的主要問題）。
- 編輯圖：每個條件 × 每張影像 × `--prompt-indices` 指定的指令，加上同樣那些
  格子的未防禦編輯當對照。指令不挑滿四句是為了把檔數壓在上限內。

用法
    python scripts/collect_report_assets.py \
        --solve-root runs/amplitude_ladder --edits-root runs/amplitude_ladder_edits \
        --preflight runs/colour_probe_preflight --data data/portraits \
        --conditions cap16 cap22 cap28 cap36 --prompt-indices 0 2 \
        --out runs/report/amplitude_ladder/img
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

LOSSLESS_QUALITY = 100


def save_webp(source: Path, target: Path, *, lossless: bool, quality: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        im = im.convert("RGB")
        if lossless:
            im.save(target, "WEBP", lossless=True, quality=LOSSLESS_QUALITY)
        else:
            im.save(target, "WEBP", quality=quality, method=6)
    return target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solve-root", type=Path, required=True)
    parser.add_argument("--edits-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/portraits"))
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--prompt-indices", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--scenarios", nargs="+", default=["ip2p"])
    parser.add_argument("--arm", default="ip2p_si18")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    images = args.images or sorted(
        p.stem for p in sorted(args.data.glob("*/*.png"))
        if p.parent.name in ("man", "woman"))
    manifest = {"images": images, "conditions": args.conditions,
                "prompt_indices": args.prompt_indices, "files": {}}
    total = 0
    missing = []

    for name in images:
        source = next((args.data / d / f"{name}.png"
                       for d in ("man", "woman")
                       if (args.data / d / f"{name}.png").is_file()), None)
        if source is None:
            missing.append(f"原圖 {name}")
            continue
        key = f"orig/{name}"
        total += save_webp(source, args.out / f"{key}.webp",
                           lossless=True, quality=LOSSLESS_QUALITY)
        manifest["files"][key] = f"img/{key}.webp"

    for index in args.prompt_indices:
        for name in images:
            src = args.preflight / args.arm / f"{name}__p{index}.png"
            if not src.is_file():
                missing.append(str(src))
                continue
            key = f"plain_edit/{name}__p{index}"
            total += save_webp(src, args.out / f"{key}.webp",
                               lossless=False, quality=args.quality)
            manifest["files"][key] = f"img/{key}.webp"

    for condition in args.conditions:
        for name in images:
            src = args.solve_root / "defended" / condition / f"{name}__def.png"
            if not src.is_file():
                hits = list(args.solve_root.glob(
                    f"*/{name}__{condition}__defended.png"))
                src = hits[0] if hits else src
            if not src.is_file():
                missing.append(str(src))
                continue
            key = f"defended/{condition}__{name}"
            total += save_webp(src, args.out / f"{key}.webp",
                               lossless=True, quality=LOSSLESS_QUALITY)
            manifest["files"][key] = f"img/{key}.webp"
            for index in args.prompt_indices:
                e = args.edits_root / condition / args.arm / f"{name}__p{index}.png"
                if not e.is_file():
                    missing.append(str(e))
                    continue
                ekey = f"edit/{condition}__{name}__p{index}"
                total += save_webp(e, args.out / f"{ekey}.webp",
                                   lossless=False, quality=args.quality)
                manifest["files"][ekey] = f"img/{ekey}.webp"

    manifest["missing"] = missing
    manifest["total_bytes"] = total
    manifest["file_count"] = len(manifest["files"])
    (args.out.parent / "assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{manifest['file_count']} 個檔、{total / 1e6:.1f} MB -> {args.out}")
    if missing:
        print(f"缺 {len(missing)} 個來源，前十個：")
        for m in missing[:10]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
