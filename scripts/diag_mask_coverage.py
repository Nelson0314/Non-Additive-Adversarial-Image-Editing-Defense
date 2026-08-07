"""逐影像量遮罩涵蓋率：`attention` 與 `attention_box` × 兩個 tau。

存在理由：ip1 段 0 量到 dog_03 的 attention_box 涵蓋率為 0.875，超過
`HANDOVER_2026-08-08` §3.2a 的 0.6 停止線。要決定換圖、換模式還是調 tau，
需要整個資料集在四種設定下的涵蓋率，而不是只看那三張。

輸出 `~/wacv_runs/ip1_maskprobe/coverage.csv`。
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/WACV"))

import torch

from src.data.masks import content_mask
from src.experiment.executors import load_lo_aligned
from src.models.sd import SDInpaintWrapper

OUT = Path(os.path.expanduser("~/wacv_runs/ip1_maskprobe"))
OUT.mkdir(parents=True, exist_ok=True)

sd = SDInpaintWrapper("runwayml/stable-diffusion-inpainting",
                      dtype=torch.float32)
entries = load_lo_aligned(Path(os.path.expanduser("~/WACV/data/lo_aligned")),
                          512, sd.device)
print(f"影像 {len(entries)} 張", flush=True)

rows = []
for e in entries:
    row = {"image_id": e.image_id, "group": e.group, "content": e.content}
    for mode in ("attention", "attention_box"):
        for tau in (0.5, 0.7):
            r = content_mask(sd, e.x01, e.content, mode=mode, tau=tau,
                             timestep=500)
            row[f"{mode}_{tau}"] = round(float(r["coverage"]), 4)
    rows.append(row)
    print(row, flush=True)

with open(OUT / "coverage.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("寫出", OUT / "coverage.csv", flush=True)
