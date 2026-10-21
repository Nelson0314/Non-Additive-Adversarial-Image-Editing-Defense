"""FLUX 淨化後的位移保留量：與 `measure_purified_displacement`（`immunization_core.pipelines.retention`）同一組讀數，換一種目錄版面。

`run_flux_edits --purifier <算子>` 的輸出攤平成 `<淨化根>/<arm>/<算子>/<影像>__p<指令>.png`，沒有
`<場景>_<條件>_<算子>` 子目錄，`pipelines.retention` 的路徑假設對不上，故另立這一支。

    淨增益 = 位移( 編輯(原圖), 編輯(防禦圖) ) − 位移( 編輯(淨化(原圖)), 編輯(淨化(防禦圖)) )

未淨化位移讀 `measure_flux_displacement` 的 CSV（`disp_lpips_full`）；淨化後兩側皆經同一淨化，分母取自
`<--undefended-root>/undefended/<算子>/`（預設與淨化根相同）。遮罩、SigLIP 門檻與欄位與 `pipelines.retention` 相同，
幾何算子的遮罩經 `purified_mask()` 與影像同步變換；解析度為 FLUX 輸出的 1024。

用法（遠端 baseline 專案根，CPU 或 GPU 皆可）：
    python -m immunization_baseline.cli.measure_flux_retention --purified-edits-root artifacts/flux/purified_edits \\
        --displacement-csv results/flux/displacement.csv --output-csv results/flux/retention.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from immunization_baseline import layout
from immunization_core.io import load_image_tensor, write_csv
from immunization_core.metrics.regional import RegionalLPIPS, split_displacement
from immunization_core.metrics.standard import SIGLIP_BLOCKED_THRESHOLD
from immunization_core.metrics.suite import MetricSuite
from immunization_core.pipelines.masks import GEOMETRIC, purified_mask, subject_mask

RESOLUTION = 1024  # FluxKontextPipeline 的輸出解析度，與 measure_flux_displacement 相同
UNDEFENDED = "undefended"


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--purified-edits-root", dest="purified_root", type=Path, required=True)
    ap.add_argument("--undefended-root", type=Path, default=None,
                    help="含 `undefended/<算子>/` 的根目錄，預設與 --purified-edits-root 相同")
    ap.add_argument("--displacement-csv", dest="displacement", type=Path, required=True)
    ap.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS,
                    help="資料集根目錄，只讀取其中的 masks/")
    ap.add_argument("--output-csv", dest="out", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=None, help="預設為淨化根下全部非未防禦的條件目錄")
    ap.add_argument("--purifiers", nargs="+", default=None, help="預設為各條件目錄下的全部算子目錄")
    args = ap.parse_args()
    undefended_root = args.undefended_root or args.purified_root

    base = {(r["condition"], r["image"], r["prompt_index"]): r for r in read_csv(args.displacement)}
    conditions = args.conditions or sorted(
        d.name for d in args.purified_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name != UNDEFENDED)
    if not conditions:
        raise SystemExit(f"{args.purified_root} 下沒有任何條件目錄")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)
    masks = {}

    rows = []
    for condition in conditions:
        purifiers = args.purifiers or sorted(d.name for d in (args.purified_root / condition).iterdir() if d.is_dir())
        keys = sorted(k for k in base if k[0] == condition)
        if not keys:
            raise SystemExit(f"{args.displacement} 沒有 {condition} 的未淨化位移")
        for purifier in purifiers:
            for key in keys:
                _, name, index = key
                a = undefended_root / UNDEFENDED / purifier / f"{name}__p{index}.png"
                b = args.purified_root / condition / purifier / f"{name}__p{index}.png"
                if not a.is_file() or not b.is_file():
                    raise SystemExit(f"缺檔：{a if not a.is_file() else b}；淨化後的兩側必須齊全")
                if (name, purifier) not in masks:
                    repaint = load_image_tensor(args.data / "masks" / f"{name}.png",
                                                device, size=RESOLUTION)[:, :1]
                    masks[(name, purifier)] = purified_mask(subject_mask(repaint), purifier)
                xa = load_image_tensor(a, device, size=RESOLUTION)
                xb = load_image_tensor(b, device, size=RESOLUTION)
                with torch.no_grad():
                    split = split_displacement(regional, xa, xb, masks[(name, purifier)])
                    similar = suite.image_similarity(xa, xb)
                plain = float(base[key]["disp_lpips_full"])
                purified = float(split["lpips_full"])
                siglip = float(similar["siglip"])
                rows.append({
                    "condition": condition, "purifier": purifier,
                    "geometric": purifier in GEOMETRIC,
                    "scenario": "ip2p", "editor": "flux", "image": name, "prompt_index": index,
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
