"""用銳利度保留率重新判讀 E15 的 site S vs site P。

E15 的結論是「匹配 LPIPS 後 S 領先 P 約 1.15 倍」，但當時只用 LPIPS 對齊失真。
本腳本補上第四項指標，檢查那個「匹配」在銳利度上是否成立。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.acutance import acutance

ROOT = Path(__file__).resolve().parent.parent / "runs"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def run_acut(run: Path):
    vals = []
    for d in sorted(run.iterdir()):
        o, x = d / "orig.png", d / "defended.png"
        if d.is_dir() and o.exists() and x.exists():
            vals.append(acutance(load(o), load(x))["acutance_ratio"])
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


print(f"{'tau':>6s}  {'site S 銳利度':>16s}  {'site P 銳利度':>16s}  {'差距':>8s}")
print("-" * 56)
for tau in ["0.02", "0.05", "0.10"]:
    s = run_acut(ROOT / f"e15_S_tau{tau}")
    p = run_acut(ROOT / f"e15_P_tau{tau}")
    if s and p:
        print(f"{tau:>6s}  {100*s[0]:9.1f}% ±{100*s[1]:4.1f}  "
              f"{100*p[0]:9.1f}% ±{100*p[1]:4.1f}  {100*(p[0]-s[0]):7.1f} pp")

print()
print("E15 主張：tau=0.05 時 S 的 net 0.1050、P 內插到同一 LPIPS 為 0.0916，S 領先 1.15x")
print("該匹配只對齊 LPIPS。若銳利度差距大，兩者的失真並非同一種東西。")
