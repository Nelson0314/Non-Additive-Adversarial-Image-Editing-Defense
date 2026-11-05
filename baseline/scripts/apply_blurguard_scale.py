"""BlurGuard 的事後縮放：x_s = clip(x + s·(x̃ − x))，量化為 8 位元 PNG。
calibrate <防禦圖目錄> <目標 LPIPS>：在該目錄（<影像>__blurguard__def.png 與 <影像>__orig.png）上二分搜尋全域倍率 s，
    使 8 張的平均防禦圖 LPIPS（piq.LPIPS，交付 PNG 讀入，CPU）等於目標，印出 s。
apply <輸入目錄> <輸出目錄> <s>：對輸入目錄的每張防禦圖套用 s，複製 __orig.png，並把 results_*.csv 的 modification_note 附上縮放倍率。"""
import csv
import sys
from pathlib import Path

import numpy as np
import piq
import torch
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def scaled(x, d, s):
    return np.clip(np.rint(x + s * (d - x)), 0, 255).astype(np.uint8)


def tensor(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def main():
    mode = sys.argv[1]
    if mode == "calibrate":
        root, target = Path(sys.argv[2]), float(sys.argv[3])
        pairs = [(load(p.with_name(p.name.replace("__blurguard__def.png", "__orig.png"))), load(p)) for p in sorted(root.glob("*__blurguard__def.png"))]
        lp = piq.LPIPS()

        def mean_lpips(s):
            with torch.no_grad():
                return float(np.mean([lp(tensor(scaled(x, d, s)), tensor(x.astype(np.uint8))).item() for x, d in pairs]))
        lo, hi = 0.0, 1.0
        print(f"s=1 lpips={mean_lpips(1.0):.4f}（{len(pairs)} 張）", flush=True)
        for _ in range(14):
            mid = (lo + hi) / 2
            if mean_lpips(mid) < target:
                lo = mid
            else:
                hi = mid
        s = round((lo + hi) / 2, 4)
        print(f"[SCALE] s={s} lpips={mean_lpips(s):.4f} target={target}", flush=True)
    elif mode == "apply":
        src, dst, s = Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(src.glob("*__blurguard__def.png")):
            o = p.with_name(p.name.replace("__blurguard__def.png", "__orig.png"))
            x = load(o)
            Image.fromarray(scaled(x, load(p), s)).save(dst / p.name)
            Image.fromarray(x.astype(np.uint8)).save(dst / o.name)
            n += 1
        for c in src.glob("results_*.csv"):
            rows = list(csv.DictReader(open(c, encoding="utf-8")))
            for r in rows:
                r["modified_from_paper"] = "True"
                r["modification_note"] = r["modification_note"] + f"; perturbation scaled after solving by {s} (x + s*(x_def - x)) to the aligned LPIPS anchor"
            with open(dst / c.name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
        print(f"[APPLIED] s={s} {n} 張 → {dst}", flush=True)


if __name__ == "__main__":
    main()
