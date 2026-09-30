"""編輯位移：同一設定下，編輯(原圖) 與 編輯(防禦圖) 之間的距離。

`位移 = LPIPS( 編輯(原圖), 編輯(防禦圖) )`；另以主體遮罩分出主體內與主體外兩區
（`metrics.regional`，全 1 遮罩逐位元等於全圖 LPIPS）。`disp_*` 標準欄位的方向見
`metrics.standard`。`siglip_pair` 為兩張編輯結果的 SigLIP 影像餘弦，低於
`SIGLIP_BLOCKED_THRESHOLD` 記為 `blocked`，門檻逐列寫入 CSV。

用法
    python -m immunization_core.pipelines.displacement --defended-edits-root <條件根> \\
        --undefended-edits-root <未防禦編輯根> --data-root <資料集根> --output-csv <CSV>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from immunization_core.io import load_image_tensor, write_csv
from immunization_core.metrics.regional import RegionalLPIPS, split_displacement
from immunization_core.metrics.standard import SIGLIP_BLOCKED_THRESHOLD, standard_row
from immunization_core.metrics.suite import MetricSuite
from immunization_core.pipelines.masks import subject_mask

RESOLUTION = 512

#: 未防禦的對照 arm；設定須與防禦 arm 逐項相同。
DEFAULT_ARMS = {"ip2p": "ip2p_si18", "inpaint": "inpaint_undefended"}


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def undefended_png(preflight: Path, arm: str, name: str, index: str) -> Path:
    path = preflight / arm / f"{name}__p{index}.png"
    if not path.is_file():
        raise SystemExit(f"找不到未防禦的對照圖：{path}")
    return path


def defended_png(condition_dir: Path, arm: str, name: str, index: str) -> Path:
    path = condition_dir / arm / f"{name}__p{index}.png"
    if not path.is_file():
        raise SystemExit(f"找不到防禦後的編輯圖：{path}")
    return path


def condition_directories(root: Path, conditions=None) -> list:
    """帶 preflight.csv 的條件目錄；指定 `conditions` 時只保留這些條件。"""
    directories = sorted(d for d in root.iterdir()
                         if d.is_dir() and not d.name.startswith("_")
                         and (d / "preflight.csv").is_file())
    if conditions:
        keep = set(conditions)
        directories = [d for d in directories if d.name in keep]
    if not directories:
        raise SystemExit(f"{root} 下沒有任何帶 preflight.csv 的條件目錄")
    return directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defended-edits-root", dest="defended_root", type=Path, required=True)
    parser.add_argument("--undefended-edits-root", dest="preflight", type=Path, required=True)
    parser.add_argument("--data-root", dest="data", type=Path, required=True,
                        help="資料集根目錄，只讀取其中的 masks/")
    parser.add_argument("--output-csv", dest="out", type=Path, required=True)
    parser.add_argument("--ip2p-arm", default=DEFAULT_ARMS["ip2p"])
    parser.add_argument("--inpaint-arm", default=DEFAULT_ARMS["inpaint"])
    parser.add_argument("--conditions", nargs="+", default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    arms = {"ip2p": args.ip2p_arm, "inpaint": args.inpaint_arm}
    directories = condition_directories(args.defended_root, args.conditions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)

    rows = []
    for directory in directories:
        condition = directory.name
        for row in read_csv(directory / "preflight.csv"):
            scenario, name, index = row["scenario"], row["image"], row["prompt_index"]
            if scenario not in arms:
                raise SystemExit(f"場景 {scenario} 沒有指定未防禦的對照 arm")
            a = load_image_tensor(undefended_png(args.preflight, arms[scenario], name, index),
                                  device, size=RESOLUTION)
            b = load_image_tensor(defended_png(directory, row["arm"], name, index),
                                  device, size=RESOLUTION)
            mask_path = args.data / "masks" / f"{name}.png"
            mask = subject_mask(
                load_image_tensor(mask_path, device, size=RESOLUTION)[:, :1])
            with torch.no_grad():
                pair = suite.pairwise(a, b)
                split = split_displacement(regional, a, b, mask)
                similar = suite.image_similarity(a, b)
            siglip = float(similar["siglip"])
            rows.append({
                "condition": condition, "scenario": scenario, "arm": row["arm"],
                "image": name, "prompt_index": index, "prompt": row["prompt"],
                "undefended_arm": arms[scenario],
                **standard_row("disp_", pair),
                "disp_rms": round(float(pair["rms"]), 6),
                "disp_linf": round(float(pair["linf"]), 6),
                **{f"disp_{k}": round(float(v), 5) for k, v in split.items()},
                "clip_pair": round(float(similar["clip"]), 5),
                "siglip_pair": round(siglip, 5),
                "blocked": siglip < SIGLIP_BLOCKED_THRESHOLD,
                "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
                "defended_png": defended_png(directory, row["arm"], name, index).as_posix(),
                "undefended_png": undefended_png(args.preflight, arms[scenario],
                                                 name, index).as_posix(),
            })
            write_csv(args.out, rows)
        print(f"[DONE] {condition:16s} 累計 {len(rows)} 列", flush=True)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
