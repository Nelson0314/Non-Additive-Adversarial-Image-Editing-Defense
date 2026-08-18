"""擾動的徑向功率譜 — 「現有防護擾動多為高頻」這句動機的自家量測。

為什麼需要
────────────────────────────────────────────────────────────────────
DEC-025 把研究主軸改為「頻域／相位方法及其抗淨化能力」，動機是既有防護擾動
多為高頻擾動，會被 JPEG、resize、模糊抹平。這句話目前只有文獻依據
（arXiv:2505.01267 測到擴散淨化對幅度譜與相位譜的破壞都隨頻率單調遞增），
**本專案自己的條件從未量過**。沒有這張圖，動機是借來的。

量什麼
────────────────────────────────────────────────────────────────────
對每個條件的防禦圖 `x_def`，取擾動 `δ = x_def − x`，做二維 DFT，把功率
`|F(δ)|²` 依到零頻的距離分箱，得到徑向功率譜。再報三個摘要：

* `f50`：累積能量達 50% 的頻率（以 Nyquist 為 1）——擾動的「中位頻率」
* `hi_frac`：`f > 0.5·Nyquist` 的能量佔比
* `lo_frac`：`f < 0.125·Nyquist` 的能量佔比

半徑一律以**到零頻的歐氏距離除以 `size/2`** 表示，故 1.0 是水平／垂直方向的
Nyquist，角落可達 √2。分箱在該尺度上等寬。

不需要 GPU，也不載入 SD。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import statistics
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

BINS = 64


def radial_power(delta: torch.Tensor, bins: int = BINS) -> np.ndarray:
    """(1,C,H,W) 的擾動 → 長度 `bins` 的徑向功率（已正規化成總和為 1）。

    逐通道做 DFT 再把功率相加：擾動是實數三通道影像，通道之間沒有相位關係
    可言，先合成單一亮度會把通道間的差異抹掉。
    """
    if delta.dim() != 4:
        raise ValueError(f"需要 (N,C,H,W)，收到 {tuple(delta.shape)}")
    h, w = delta.shape[-2:]
    if h != w:
        raise ValueError(f"目前只支援正方形，收到 {h}×{w}")
    z = torch.fft.fft2(delta.double(), dim=(-2, -1))
    power = (z.abs() ** 2).sum(dim=(0, 1))                 # (H,W)

    u = torch.fft.fftfreq(h) * 2.0                          # −1 … 1，1 = Nyquist
    v = torch.fft.fftfreq(w) * 2.0
    r = (u.view(-1, 1) ** 2 + v.view(1, -1) ** 2).sqrt()    # 0 … √2

    idx = torch.clamp((r / r.max() * bins).long(), max=bins - 1)
    out = torch.zeros(bins, dtype=torch.float64)
    out.scatter_add_(0, idx.reshape(-1), power.reshape(-1))
    total = float(out.sum())
    return (out / total).numpy() if total > 0 else out.numpy()


def summarise(prof: np.ndarray, bins: int = BINS) -> dict:
    """由徑向功率譜算三個摘要。頻率以 Nyquist（=1）為單位。"""
    edges = np.arange(bins + 1) / bins * np.sqrt(2.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(prof)
    f50 = float(np.interp(0.5, cum, centres))
    return {"f50": f50,
            "hi_frac": float(prof[centres > 0.5].sum()),
            "lo_frac": float(prof[centres < 0.125].sum())}


TAG = re.compile(r"^(?P<image>.+?)__(?P<tag>.+)__def\.png$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True,
                    help="含 *__def.png 的目錄，可給多個")
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--out", type=Path, default=Path("runs/spectral/radial.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from src.utils.io import load_image_tensor

    originals = {p.stem: p for p in args.data.glob("*/*.png")}
    rows, prof_by = [], defaultdict(list)
    for run in args.run:
        for png in sorted(run.glob("*__def.png")):
            m = TAG.match(png.name)
            if not m or m["image"] not in originals:
                continue
            x = load_image_tensor(originals[m["image"]], "cpu", size=512).double()
            d = load_image_tensor(png, "cpu", size=512).double() - x
            prof = radial_power(d)
            s = summarise(prof)
            cond = m["tag"].replace("__human", "")
            prof_by[cond].append(prof)
            rows.append({"image": m["image"], "condition": cond,
                         "rms": round(float(d.pow(2).mean().sqrt()), 6),
                         "f50": round(s["f50"], 4),
                         "hi_frac": round(s["hi_frac"], 4),
                         "lo_frac": round(s["lo_frac"], 4)})

    if not rows:
        raise SystemExit("沒有找到任何 *__def.png")
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    print(f"{'條件':<16}{'n':>3} {'f50':>7} {'高頻佔比':>9} {'低頻佔比':>9} {'RMS':>9}")
    for c, g in sorted(by.items(), key=lambda kv: -statistics.fmean(
            r["f50"] for r in kv[1])):
        f = lambda k: statistics.fmean(r[k] for r in g)   # noqa: E731
        print(f"{c:<16}{len(g):>3} {f('f50'):>7.3f} {f('hi_frac'):>9.3f} "
              f"{f('lo_frac'):>9.3f} {f('rms'):>9.5f}")

    np.savez(args.out.with_suffix(".npz"),
             **{c: np.mean(np.stack(v), axis=0) for c, v in prof_by.items()})
    print(f"\n表：{args.out}；曲線：{args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
