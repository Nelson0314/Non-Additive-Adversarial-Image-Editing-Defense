"""Edit-result LPIPS in RGB and in grayscale for every ISR cell of the ip2p crosseditor100 pilot.

For each defended cell (6 methods x 4 purifications x 400 image-instruction pairs) the defended edit is
compared with the undefended edit under the same purification, as in the edit-result LPIPS protocol
(piq.LPIPS, inputs in [0, 1], no resize: both images must already share one size).
Grayscale: both images are converted to luminance Y = 0.299 R + 0.587 G + 0.114 B (ITU-R BT.601) and
replicated to three channels before the same LPIPS network.

Usage: python measure_isr_lpips_grayscale.py --cells-csv <cells.csv> --repo-root <baseline root> --out <csv>
"""
import argparse
import csv
import os

import numpy as np
import piq
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cells-csv", required=True)
ap.add_argument("--repo-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
args = ap.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA is required (CUDA_VISIBLE_DEVICES must name a leased card)")
device = torch.device(args.device)
net = piq.LPIPS().to(device).eval()
weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)

rows = [r for r in csv.DictReader(open(args.cells_csv)) if r["condition"] != "undefended"]
print(f"cells to measure: {len(rows)}", flush=True)


def load(rel):
    path = os.path.join(args.repo_root, rel)
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def gray(x):
    return (x * weights).sum(1, keepdim=True).repeat(1, 3, 1, 1)


with open(args.out + ".partial", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sheet", "cell", "image", "prompt_id", "condition", "purification", "lpips_rgb", "lpips_gray"])
    with torch.no_grad():
        for s in range(0, len(rows), args.batch_size):
            chunk = rows[s:s + args.batch_size]
            x = [load(r["edit_png"]) for r in chunk]
            y = [load(r["undefended_same_purification_png"]) for r in chunk]
            for r, a, b in zip(chunk, x, y):
                if a.shape != b.shape:
                    raise SystemExit(f"size mismatch {r['sheet']} {r['cell']}: {tuple(a.shape)} vs {tuple(b.shape)}")
            x = torch.stack(x).to(device)
            y = torch.stack(y).to(device)
            rgb = [float(net(x[i:i + 1], y[i:i + 1])) for i in range(len(chunk))]
            gx, gy = gray(x), gray(y)
            gr = [float(net(gx[i:i + 1], gy[i:i + 1])) for i in range(len(chunk))]
            for r, a, b in zip(chunk, rgb, gr):
                w.writerow([r["sheet"].removesuffix(".jpg"), r["cell"], r["image"], r["prompt_id"], r["condition"],
                            r["purification"], f"{a:.5f}", f"{b:.5f}"])
            if s % 800 == 0:
                print(f"{s + len(chunk)}/{len(rows)}", flush=True)
os.replace(args.out + ".partial", args.out)
print("done", args.out)
