"""`sweep_editor_parameters` 產出的編輯，在指定配件以外改動了多少。

讀數（原圖與編輯結果都縮到 512×512 後逐像素比）
────────────────────────────────────────────────────────────────────
- `de_bg`：背景區的平均 ΔE00。背景區 = `data/portraits/masks/<影像>.png` 的白色
  部分（該遮罩是主體的補集，inpaint 場景的重繪區）。四條 ip2p 指令都衝著人，
  背景區的改動都不是指令要求的。安全帽、配件超出主體輪廓的部分會落進背景區，
  所以這一欄對 p2（安全帽）偏高是預期的來源之一。
- `de_subject`：主體區的平均 ΔE00，含指令要求的配件與衣服。
- `lpips_full`：整張圖的 LPIPS（`piq.LPIPS`，與 `immunization_core/metrics/suite.py` 同一個）。

輸出逐格一列，鍵與輸入 CSV 相同。不需要 GPU。

用法（CSV 的 png 欄為相對路徑時，於其基準目錄執行）
    python -m immunization_baseline.cli.measure_off_target_changes \\
        --edits-csv results/sweeps/sdxl_ip2p/guidance/guidance_portraits.csv \\
        --output-csv results/sweeps/sdxl_ip2p/guidance/guidance_portraits_off_target.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from immunization_baseline import layout  # noqa: E402
from immunization_core.io import write_rows_atomic  # noqa: E402

import numpy as np  # noqa: E402
import piq  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.color import deltaE_ciede2000, rgb2lab  # noqa: E402

RES = 512
KEYS = ["variant", "image", "prompt_index", "guidance_scale", "image_guidance_scale", "strength"]


def load01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edits-csv", dest="edits", type=Path, required=True,
                        help="sweep_editor_parameters 寫出的 CSV")
    parser.add_argument("--output-csv", dest="out", type=Path, required=True)
    parser.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS,
                        help="資料集根目錄：原圖與 masks/")
    args = parser.parse_args()
    with args.edits.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lpips = piq.LPIPS(reduction="none")
    cache = {}
    out = []
    for r in rows:
        name = r["image"]
        if name not in cache:
            cls = name.split("_")[0]
            x = load01(args.data / cls / f"{name}.png")
            mask = np.asarray(Image.open(args.data / "masks" / f"{name}.png")
                              .convert("L").resize((RES, RES), Image.NEAREST)) > 127
            cache[name] = (x, rgb2lab(x), mask)
        x, lab_x, bg = cache[name]
        y = load01(Path(r["png"]))
        de = deltaE_ciede2000(lab_x, rgb2lab(y))
        tx = torch.from_numpy(x).permute(2, 0, 1)[None]
        ty = torch.from_numpy(y).permute(2, 0, 1)[None]
        with torch.no_grad():
            lp = float(lpips(tx, ty)[0])
        out.append({**{k: r.get(k) or ("verbatim" if k == "variant" else "") for k in KEYS}, "id_orig": r["id_orig"],
                    "de_bg": round(float(de[bg].mean()), 3),
                    "de_subject": round(float(de[~bg].mean()), 3),
                    "lpips_full": round(lp, 4)})
    out_csv = args.out
    write_rows_atomic(out_csv, list(out[0]), out)
    print(f"{len(out)} 格 -> {out_csv}")


if __name__ == "__main__":
    main()
