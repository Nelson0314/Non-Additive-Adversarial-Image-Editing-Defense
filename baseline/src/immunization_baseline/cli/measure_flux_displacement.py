"""FLUX 全表的位移：跟 `edit_displacement.py` 同一組讀數，換一種目錄版面。

FLUX 全表(`edit_flux_preview.py --arm`)沒有 ip2p／inpaint 那種兩場景、
`preflight.csv`、`<場景>_<條件>` 子目錄的版面──只有一個場景(ip2p 指令)，
每個 arm 攤平成 `images/flux_full/<arm>/<影像>__p<指令>.png`，讀數在
`results/flux_full_<arm>.csv`。`edit_displacement.py` 的目錄假設對不上，
故另立這一支，共用同一套指標模組（`RegionalLPIPS`、`standard_row`、
`MetricSuite`），輸出欄位與 `results/displacement.csv` 一致，可以直接併進
同一張報告表。

位移定義、主體遮罩極性、SigLIP 門檻都與 `edit_displacement.py` 相同，
理由見該檔案 docstring，這裡不重複。

用法（遠端，CPU 或 GPU 皆可，不需要大量顯存）
    python code/edit_displacement_flux.py --out results/displacement_flux.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from src.metrics.standard import SIGLIP_BLOCKED_THRESHOLD, standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 1024  # FluxKontextPipeline 強制的輸出解析度，見 edit_flux_preview.py
CONDITIONS = ["dct_shield_y", "mist", "dct_shield", "photoguard_linf", "danp",
             "sifm", "dayn", "dia_pt", "dia_r", "photoguard_c",
             "colour_curve_ours", "diffvax"]


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def subject_mask(repaint: torch.Tensor) -> torch.Tensor:
    """同 `edit_displacement.py`：`masks/` 白＝重繪＝主體以外，故要翻極性。"""
    return 1.0 - (repaint >= 0.5).float()


def masks_dir() -> Path:
    """`main_table/images/masks/` 是 gitignored、遠端沒同步過；
    `../data/portraits/masks/` 是逐位元相同的另一份，README 已記過這件事。"""
    for c in (paths.IMAGES / "masks", paths.PORTRAITS / "masks"):
        if c.is_dir():
            return c
    raise SystemExit("找不到遮罩目錄")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "displacement_flux.csv")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)

    undefended = {(r["image"], r["prompt_index"]): r
                  for r in read_csv(paths.RESULTS / "flux_full_undefended.csv")}
    mdir = masks_dir()

    rows = []
    for condition in args.conditions:
        cond_rows = read_csv(paths.RESULTS / f"flux_full_{condition}.csv")
        for row in cond_rows:
            key = (row["image"], row["prompt_index"])
            u = undefended.get(key)
            if u is None:
                raise SystemExit(f"分母缺這一格：{condition} {key}")
            a = load_image_tensor(paths.IMAGES / "flux_full" / "undefended" /
                                  f"{row['image']}__p{row['prompt_index']}.png",
                                  device, size=RESOLUTION)
            b = load_image_tensor(paths.IMAGES / "flux_full" / condition /
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
