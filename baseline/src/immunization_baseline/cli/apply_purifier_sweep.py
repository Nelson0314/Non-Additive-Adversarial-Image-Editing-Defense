"""淨化強度掃描：以 core 的淨化算子，對防禦圖（或資料集原圖）套用指定的 (kind, strength) 清單。

輸出版面與 `apply_purifiers` 相同：`<輸出目錄>/<標籤>/<影像>__def.png` 與 `purified.csv`；
標籤沿用 `immunization_core.purifiers.protocol.label`（例如 `jpeg95`、`blur0.5`、`resample_roundtrip256`）。
縮放比例 r 以 `resample_roundtrip`（降到 round(512·r) 再升回 512）實作。

用法（遠端 baseline 專案根）：
    python -m immunization_baseline.cli.apply_purifier_sweep --defenses-dir <條件目錄> --output-dir <目錄> \\
        --purifiers jpeg:95 jpeg:90 blur:0.5 resample_roundtrip:384
    python -m immunization_baseline.cli.apply_purifier_sweep --data-root data/portraits --output-dir <目錄> ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from immunization_core.artifacts.images import save_image
from immunization_core.io import load_image_tensor, write_csv
from immunization_core.pipelines.purification import RESOLUTION, dataset_sources, defended_sources
from immunization_core.purifiers.operators import Purifier
from immunization_core.purifiers.protocol import label


def parse_spec(text: str) -> tuple:
    kind, _, strength = text.partition(":")
    if not strength:
        raise argparse.ArgumentTypeError(f"淨化設定須為 kind:strength，收到 {text}")
    return kind, float(strength)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--defenses-dir", dest="defended", type=Path)
    source.add_argument("--data-root", dest="data", type=Path)
    parser.add_argument("--output-dir", dest="out", type=Path, required=True)
    parser.add_argument("--purifiers", nargs="+", type=parse_spec, required=True)
    parser.add_argument("--condition", default=None)
    args = parser.parse_args(argv)

    if args.defended is not None:
        sources = defended_sources(args.defended)
        condition = args.condition or args.defended.name
    else:
        sources = dataset_sources(args.data)
        condition = args.condition or "undefended"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for kind, strength in args.purifiers:
        strength = int(strength) if kind in ("jpeg", "resample_roundtrip") else strength
        tag = label(kind, strength)
        purifier = Purifier(kind, strength)
        target = args.out / tag
        target.mkdir(parents=True, exist_ok=True)
        for name, path in sources.items():
            x01 = load_image_tensor(path, device, size=RESOLUTION)
            with torch.no_grad():
                y01 = purifier.evaluate(x01).clamp(0, 1)
            save_image(y01, target / f"{name}__def.png")
            rows.append({"condition": condition, "purifier": tag, "kind": kind, "strength": strength,
                         "image": name, "source_png": path.as_posix(),
                         "output_png": (target / f"{name}__def.png").as_posix()})
        print(f"[DONE] {condition:16s} {tag:24s} {len(sources)} 張", flush=True)
    write_csv(args.out / "purified.csv", rows)
    print(f"[ALLDONE] {args.out / 'purified.csv'}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
