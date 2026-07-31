"""對 E19 三格 x 各 arm 逐張計算銳利度保留率。由 repo 根目錄執行。"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.acutance import acutance

ARMS = ["roundtrip", "latent_opt", "asym_free", "latent_opt_asym"]
CELLS = ["car_00", "car_01", "dog_00", "dog_01", "person_00", "person_01"]
RUNS = ["e19_lam0.1", "e19_lam1", "e19_lam10"]
ROOT = Path(__file__).resolve().parent.parent / "runs"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


print(f"{'run':13s}" + "".join(f"{a:>18s}" for a in ARMS))
print("-" * 85)
for run in RUNS:
    acc = {a: [] for a in ARMS}
    for cell in CELLS:
        d = ROOT / run / cell
        if not (d / "orig.png").exists():
            continue
        o = load(d / "orig.png")
        for a in ARMS:
            p = d / f"{a}.png"
            if p.exists():
                acc[a].append(acutance(o, load(p))["acutance_ratio"])
    line = f"{run:13s}"
    for a in ARMS:
        line += f"{100 * np.mean(acc[a]):17.1f}%" if acc[a] else f"{'—':>18s}"
    print(line)

print()
print("逐張（λ=10，看鈍化是否與影像難度相關）")
print("-" * 85)
d0 = ROOT / "e19_lam10"
print(f"{'影像':11s}" + "".join(f"{a:>18s}" for a in ARMS))
for cell in CELLS:
    d = d0 / cell
    if not (d / "orig.png").exists():
        continue
    o = load(d / "orig.png")
    line = f"{cell:11s}"
    for a in ARMS:
        p = d / f"{a}.png"
        line += (f"{100 * acutance(o, load(p))['acutance_ratio']:17.1f}%"
                 if p.exists() else f"{'—':>18s}")
    print(line)
