"""`edit_sd_family_preview.py` 產出的編輯，在指定配件以外改動了多少。

讀數（原圖與編輯結果都縮到 512×512 後逐像素比）
────────────────────────────────────────────────────────────────────
- `de_bg`：背景區的平均 ΔE00。背景區 = `data/portraits/masks/<影像>.png` 的白色
  部分（該遮罩是主體的補集，inpaint 場景的重繪區）。四條 ip2p 指令都衝著人，
  背景區的改動都不是指令要求的。安全帽、配件超出主體輪廓的部分會落進背景區，
  所以這一欄對 p2（安全帽）偏高是預期的來源之一。
- `de_subject`：主體區的平均 ΔE00，含指令要求的配件與衣服。
- `lpips_full`：整張圖的 LPIPS（`piq.LPIPS`，與 `src/metrics/suite.py` 同一個）。

輸出 `results/sd_family_offtarget_<批次>.csv`，逐格一列，鍵與原 CSV 相同。
不需要 GPU。

用法（遠端 repo 根目錄，CSV 的 png 欄是相對它的路徑）
    python main_table/code/sd_family_offtarget_readout.py sdxl_ip2p_all8
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

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
    batch = sys.argv[1]
    src_csv = paths.RESULTS / f"sd_family_{batch}.csv"
    rows = list(csv.DictReader(src_csv.open(encoding="utf-8", newline="")))
    lpips = piq.LPIPS(reduction="none")
    cache = {}
    out = []
    for r in rows:
        name = r["image"]
        if name not in cache:
            cls = name.split("_")[0]
            x = load01(paths.PORTRAITS / cls / f"{name}.png")
            mask = np.asarray(Image.open(paths.PORTRAITS / "masks" / f"{name}.png")
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
    out_csv = paths.RESULTS / f"sd_family_offtarget_{batch}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"{len(out)} 格 -> {out_csv}")


if __name__ == "__main__":
    main()
