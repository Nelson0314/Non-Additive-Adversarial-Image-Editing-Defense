"""以**批次實際使用的 seed** 量全資料集的 attention_box 涵蓋率，並存疊圖。

遮罩由一次加噪前向的注意力圖決定，故它依賴 seed。前兩次探測用的是 0 與 7，
與批次的 20260805 不同——dog_02 在 seed 0 與 7 下量到 0.4475 與 0.8750，
差一倍。選圖的判準必須在**批次會用的那個 seed** 上量。

輸出 `~/wacv_runs/ip2_maskprobe3/`。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/WACV"))

import torch

from src.data.masks import content_mask
from src.experiment.executors import load_lo_aligned, save_image, write_csv
from src.models.sd import SDInpaintWrapper

SEED = 20260805          # run_stage.py 的 --seed 預設值
OUT = Path(os.path.expanduser("~/wacv_runs/ip2_maskprobe3"))
OUT.mkdir(parents=True, exist_ok=True)

sd = SDInpaintWrapper("runwayml/stable-diffusion-inpainting",
                      dtype=torch.float32)
entries = load_lo_aligned(Path(os.path.expanduser("~/WACV/data/lo_aligned")),
                          512, sd.device)

rows = []
for e in entries:
    r = content_mask(sd, e.x01, e.content, mode="attention_box", tau=0.5,
                     timestep=500, seed=SEED)
    cov = float(r["coverage"])
    rows.append({"image_id": e.image_id, "group": e.group,
                 "coverage": round(cov, 4),
                 "in_window": 0.15 <= cov <= 0.45})
    if 0.15 <= cov <= 0.45:
        save_image((e.x01 * (1.0 - 0.6 * r["mask"])).clamp(0, 1),
                   OUT / f"{e.image_id}_overlay.png")
    print(rows[-1], flush=True)

write_csv(OUT / "coverage_seed20260805.csv", rows)
print("在窗內：", [r["image_id"] for r in rows if r["in_window"]], flush=True)
