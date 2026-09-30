"""FLUX 全表的位移：與 `measure_edit_displacement` 同一組讀數，換一種目錄版面。

FLUX 全表(`run_flux_edits --arm`)沒有 ip2p／inpaint 那種兩場景、
`preflight.csv`、`<場景>_<條件>` 子目錄的版面──只有一個場景(ip2p 指令)，
每個 arm 攤平成 `artifacts/flux_edits/<arm>/<影像>__p<指令>.png`，讀數在
`results/flux/edits_<arm>.csv`。`pipelines.displacement` 的目錄假設對不上，
故另立這一支，共用同一套指標模組（`RegionalLPIPS`、`standard_row`、
`MetricSuite`），輸出欄位與 `results/displacement.csv` 一致，可以直接併進
同一張報告表。

位移定義、主體遮罩極性、SigLIP 門檻都與 `pipelines.displacement` 相同，
理由見該檔案 docstring，這裡不重複。

用法（遠端，CPU 或 GPU 皆可，不需要大量顯存）
    python -m immunization_baseline.cli.measure_flux_displacement --out results/flux/displacement.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from immunization_baseline import layout  # noqa: E402

import torch  # noqa: E402

from immunization_core.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from immunization_core.metrics.standard import SIGLIP_BLOCKED_THRESHOLD, standard_row  # noqa: E402
from immunization_core.metrics.suite import MetricSuite  # noqa: E402
from immunization_core.io import load_image_tensor, write_csv  # noqa: E402
from immunization_core.pipelines.masks import subject_mask  # noqa: E402

RESOLUTION = 1024  # FluxKontextPipeline 強制的輸出解析度，見 run_flux_edits
CONDITIONS = ["dct_shield_y", "mist", "dct_shield", "photoguard_linf", "danp",
             "sifm", "dayn", "dia_pt", "dia_r", "photoguard_c",
             "colour_curve_ours", "diffvax"]


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=layout.RESULTS / "flux" / "displacement.csv")
    ap.add_argument("--data", type=Path, default=layout.PORTRAITS,
                    help="資料集根目錄，只讀取其中的 masks/")
    ap.add_argument("--edits-csv-root", type=Path, default=layout.RESULTS / "flux",
                    help="逐 arm 的 `edits_<arm>.csv` 所在目錄")
    ap.add_argument("--edits-root", type=Path, default=layout.FLUX_EDITS,
                    help="逐 arm 的編輯影像根目錄")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)

    undefended = {(r["image"], r["prompt_index"]): r
                  for r in read_csv(args.edits_csv_root / "edits_undefended.csv")}
    mdir = args.data / "masks"

    rows = []
    for condition in args.conditions:
        cond_rows = read_csv(args.edits_csv_root / f"edits_{condition}.csv")
        for row in cond_rows:
            key = (row["image"], row["prompt_index"])
            u = undefended.get(key)
            if u is None:
                raise SystemExit(f"分母缺這一格：{condition} {key}")
            a = load_image_tensor(args.edits_root / "undefended" /
                                  f"{row['image']}__p{row['prompt_index']}.png",
                                  device, size=RESOLUTION)
            b = load_image_tensor(args.edits_root / condition /
                                  f"{row['image']}__p{row['prompt_index']}.png",
                                  device, size=RESOLUTION)
            mask_path = mdir / f"{row['image']}.png"
            mask = subject_mask(
                load_image_tensor(mask_path, device, size=RESOLUTION)[:, :1])
            with torch.no_grad():
                pair = suite.pairwise(a, b)
                split = split_displacement(regional, a, b, mask)
                similar = suite.image_similarity(a, b)
            siglip = float(similar["siglip"])
            rows.append({
                "condition": condition, "image": row["image"],
                "prompt_index": row["prompt_index"], "prompt": row["prompt"],
                **standard_row("disp_", pair),
                "disp_rms": round(float(pair["rms"]), 6),
                "disp_linf": round(float(pair["linf"]), 6),
                **{f"disp_{k}": round(float(v), 5) for k, v in split.items()},
                "clip_pair": round(float(similar["clip"]), 5),
                "siglip_pair": round(siglip, 5),
                "blocked": siglip < SIGLIP_BLOCKED_THRESHOLD,
                "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
            })
            write_csv(args.out, rows)
        print(f"[DONE] {condition:20s} 累計 {len(rows)} 列", flush=True)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
