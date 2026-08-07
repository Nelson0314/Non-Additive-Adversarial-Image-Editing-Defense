"""替代遮罩設定的疊圖與涵蓋率，供 ip2 的遮罩形狀裁決。

ip2 的 `attention` τ=0.5 遮罩是破碎斑塊而非物件剪影（dog_03 還蓋到背景），
`attention_box` τ=0.5 又在三張圖之一超過 0.6。此處量四組替代設定。

輸出 `~/wacv_runs/ip2_maskprobe2/`：逐設定的疊圖與 coverage.csv。
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/WACV"))

import torch

from src.data.masks import content_mask
from src.experiment.executors import load_lo_aligned, save_image, write_csv
from src.models.sd import SDInpaintWrapper

OUT = Path(os.path.expanduser("~/wacv_runs/ip2_maskprobe2"))
OUT.mkdir(parents=True, exist_ok=True)

SETTINGS = [("attention", 0.3), ("attention", 0.4),
            ("attention_box", 0.5), ("attention_box", 0.7)]
IDS = ["bird_03", "cat_02", "dog_03", "cat_01", "horse_01", "dog_02"]

sd = SDInpaintWrapper("runwayml/stable-diffusion-inpainting",
                      dtype=torch.float32)
entries = load_lo_aligned(Path(os.path.expanduser("~/WACV/data/lo_aligned")),
                          512, sd.device, ids=IDS)

rows = []
for e in entries:
    for mode, tau in SETTINGS:
        r = content_mask(sd, e.x01, e.content, mode=mode, tau=tau,
                         timestep=500, seed=7)
        tag = f"{e.image_id}__{mode}_{tau}"
        save_image((e.x01 * (1.0 - 0.6 * r["mask"])).clamp(0, 1),
                   OUT / f"{tag}_overlay.png")
        rows.append({"image_id": e.image_id, "mode": mode, "tau": tau,
                     "coverage": round(float(r["coverage"]), 4)})
        print(rows[-1], flush=True)

write_csv(OUT / "coverage.csv", rows)
print("寫出", OUT / "coverage.csv", flush=True)
