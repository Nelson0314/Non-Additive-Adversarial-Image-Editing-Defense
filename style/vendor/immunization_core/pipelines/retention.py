"""淨化後的位移保留量。

    淨增益 = 位移( 編輯(原圖), 編輯(防禦圖) )
           − 位移( 編輯(淨化(原圖)), 編輯(淨化(防禦圖)) )

兩項位移的量法與 `pipelines.displacement` 相同（同一份 LPIPS 權重、同一遮罩極性）。
兩側皆經同一淨化，淨化後的分母取自 `<淨化根>/undefended/<算子>/`。
`retained` 為淨化後位移除以未淨化位移，未淨化位移為 0 時留空；`net_gain` 為兩者之差。
幾何算子（`crop_resize0.1`、`rotate15`）的遮罩經 `purified_mask()` 與影像同步變換。

用法
    python -m immunization_core.pipelines.retention --purified-edits-root <淨化根> \\
        --displacement-csv <位移 CSV> --data-root <資料集根> --output-csv <CSV>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from immunization_core.io import load_image_tensor, write_csv
from immunization_core.metrics.regional import RegionalLPIPS, split_displacement
from immunization_core.metrics.standard import SIGLIP_BLOCKED_THRESHOLD
from immunization_core.metrics.suite import MetricSuite
from immunization_core.pipelines.masks import GEOMETRIC, purified_mask, subject_mask

RESOLUTION = 512
UNDEFENDED = "undefended"


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def cell(root: Path, condition: str, purifier: str, scenario: str,
         name: str, index: str) -> Path:
    arm = f"{scenario}_{condition}_{purifier}"
    return root / condition / purifier / arm / f"{name}__p{index}.png"


def purified_conditions(root: Path, conditions=None) -> list:
    """淨化根下的防禦條件；未指定 `conditions` 時為全部非底線開頭且非未防禦的目錄。"""
    found = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("_")
                   and d.name != UNDEFENDED)
    if conditions:
        found = [c for c in found if c in set(conditions)]
    if not found:
        raise SystemExit(f"{root} 下沒有任何條件目錄")
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--purified-edits-root", dest="purified_root", type=Path, required=True)
    parser.add_argument("--displacement-csv", dest="displacement", type=Path, required=True)
    parser.add_argument("--data-root", dest="data", type=Path, required=True,
                        help="資料集根目錄，只讀取其中的 masks/")
    parser.add_argument("--output-csv", dest="out", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="只計算這些條件（預設為淨化根下全部）")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    base = {}
    for row in read_csv(args.displacement):
        key = (row["condition"], row["scenario"], row["image"], row["prompt_index"])
        base[key] = row

    conditions = purified_conditions(args.purified_root, args.conditions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)
    masks = {}

    rows = []
    for condition in conditions:
        for purifier_dir in sorted((args.purified_root / condition).iterdir()):
            if not purifier_dir.is_dir():
                continue
            purifier = purifier_dir.name
            for key, row in sorted(base.items()):
                if key[0] != condition:
                    continue
                _, scenario, name, index = key
                a = cell(args.purified_root, UNDEFENDED, purifier, scenario, name, index)
                b = cell(args.purified_root, condition, purifier, scenario, name, index)
                if not a.is_file() or not b.is_file():
                    raise SystemExit(
                        f"缺檔：{a if not a.is_file() else b}；淨化後的兩側必須齊全")
                if (name, purifier) not in masks:
                    repaint = load_image_tensor(args.data / "masks" / f"{name}.png",
                                                device, size=RESOLUTION)[:, :1]
                    masks[(name, purifier)] = purified_mask(
                        subject_mask(repaint), purifier)
                xa = load_image_tensor(a, device, size=RESOLUTION)
                xb = load_image_tensor(b, device, size=RESOLUTION)
                with torch.no_grad():
                    split = split_displacement(regional, xa, xb,
                                               masks[(name, purifier)])
                    similar = suite.image_similarity(xa, xb)
                plain = float(base[key]["disp_lpips_full"])
                purified = float(split["lpips_full"])
                siglip = float(similar["siglip"])
                rows.append({
                    "condition": condition, "purifier": purifier,
                    "geometric": purifier in GEOMETRIC,
                    "scenario": scenario, "image": name, "prompt_index": index,
                    "disp_plain": round(plain, 5),
                    "disp_purified": round(purified, 5),
                    "net_gain": round(plain - purified, 5),
                    "retained": round(purified / plain, 5) if plain else "",
                    "disp_purified_subject": round(float(split["lpips_subject"]), 5),
                    "disp_purified_background": round(float(split["lpips_background"]), 5),
                    "siglip_pair": round(siglip, 5),
                    "blocked": siglip < SIGLIP_BLOCKED_THRESHOLD,
                    "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
                })
            write_csv(args.out, rows)
            print(f"[DONE] {condition:18s} {purifier:16s} 累計 {len(rows)} 列", flush=True)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
