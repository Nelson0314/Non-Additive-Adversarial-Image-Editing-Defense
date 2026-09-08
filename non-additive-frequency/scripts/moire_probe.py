"""週期圖樣過了重取樣之後，會不會在低頻長出差頻（摩爾紋）。

要判定的事
────────────────────────────────────────────────────────────────────
`docs/reference/SURVEY_NONADDITIVE_CARRIERS.md` §2.2 的候選是：把載體做成
週期圖樣，讓它與重取樣的格點產生**差頻**。差頻是兩個高頻的差，落在低頻，
而低頻是重取樣抹不掉的——若成立，那就是一個「用淨化本身產生效果」的載體。

**但差頻要產生，必須先有混疊。** 本專案的重取樣算子開著 `antialias=True`
（`src/purify/ops.py:201-202`），而抗混疊濾波正是在取樣**之前**把會混疊的
那一段濾掉。所以這條路的成立與否，取決於一個可以直接量的東西：

> 週期圖樣過了算子之後，**低頻帶裡多出了多少能量**？

這一支就量那個。**純 CPU。**

怎麼量
────────────────────────────────────────────────────────────────────
探針是頻率 `f` 的正弦光柵（差頻是兩個窄帶訊號的交互作用，用窄帶探針才問得
乾淨；寬頻探針的低頻成分會混進讀數）。過完算子之後：

    beat = 算子輸出落在 f < LOW_CUT 的能量 ÷ 探針的總能量

`LOW_CUT` 取 0.02 cyc/px（週期 50 像素以上）——`runs/purifier_transfer` 量到
`blur1.5` 在該帶留下 0.96 以上，也就是「低通抹不掉」的那一段。

**同時跑 `antialias=True` 與 `False` 兩組。** 只跑一組的話，「沒有差頻」分不出
是「這條路本來就不成立」還是「這個算子剛好擋掉」。兩組並列才判得出機制在不在。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = 0.5
AMP = 0.15
SIDE = 512
LOW_CUT = 0.02


def grating(f: float, theta: float, side: int = SIDE) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(side, dtype=torch.float32),
                          torch.arange(side, dtype=torch.float32), indexing="ij")
    phase = 2 * math.pi * f * (x * math.cos(theta) + y * math.sin(theta))
    return (AMP * torch.sin(phase))[None, None].expand(1, 3, side, side).contiguous()


def low_band_gain(d: torch.Tensor, probe: torch.Tensor,
                  cut: float = LOW_CUT) -> float:
    """算子輸出在低頻帶的能量，**除以探針的總能量**。

    **不可以除以「存活下來的能量」。** 第一版那樣寫，分母會隨著抗混疊一起
    縮小，於是 `antialias=True` 反而量出更高的低頻佔比——那是分母變小，
    不是差頻變多。實測：0.5x、週期 2.2 px 時，AA=on 的「佔比」是 0.058、
    AA=off 是 0.0006，看起來 AA 讓差頻變強，而真相相反。

    這與本專案已記過的數次「代理讀數與判讀相反」同型。
    """
    g = d.mean(1)[0]
    g = g - g.mean()
    e = torch.fft.fft2(g).abs() ** 2
    h, w = g.shape
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[None, :]
    r = (fy ** 2 + fx ** 2).sqrt()
    pg = probe.mean(1)[0]
    pg = pg - pg.mean()
    denom = float((torch.fft.fft2(pg).abs() ** 2).sum())
    return float(e[r < cut].sum() / denom) if denom > 0 else 0.0


def resize_round(x: torch.Tensor, inner: int, antialias: bool) -> torch.Tensor:
    """降到 `inner` 再升回原尺寸。與 `ops.resample_roundtrip` 同構，
    但 `antialias` 開放成參數——這一支的整個問題就在那個旗標上。"""
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(inner, inner), mode="bicubic",
                          antialias=antialias)
    return F.interpolate(small, size=(h, w), mode="bicubic",
                         antialias=antialias).clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--periods", nargs="+", type=float,
                    default=[2.2, 2.5, 3, 4, 5, 6, 8, 11, 16, 24, 32],
                    help="週期（像素）。差頻要出現，週期必須接近取樣格點")
    ap.add_argument("--inner", nargs="+", type=int, default=[410, 256, 128],
                    help="降取樣的中間尺寸。410≈0.8×（resize_only）、256=0.5×")
    ap.add_argument("--out", default="runs/moire_probe")
    args = ap.parse_args()

    base = torch.full((1, 3, SIDE, SIDE), BASE)
    thetas = (0.0, math.pi / 4)
    rows: List[dict] = []

    for inner in args.inner:
        print(f"\n── 降到 {inner}（{inner/SIDE:.3f}×）再升回 {SIDE} " + "─" * 30)
        print(f"{'週期(px)':>9}{'f':>9}"
              f"{'低頻增益 AA=on':>19}{'低頻增益 AA=off':>19}{'off/on':>9}")
        for T in args.periods:
            f = 1.0 / T
            vals = {}
            for aa in (True, False):
                fr = []
                for th in thetas:
                    d = grating(f, th)
                    a = resize_round(base.clone(), inner, aa)
                    b = resize_round((base + d).clamp(0, 1), inner, aa)
                    fr.append(low_band_gain(b - a, d))
                vals[aa] = sum(fr) / len(fr)
            ratio = vals[False] / vals[True] if vals[True] > 0 else float("inf")
            rows.append({"inner": inner, "scale": round(inner / SIDE, 4),
                         "period_px": T, "freq": round(f, 5),
                         "low_gain_aa_on": round(vals[True], 6),
                         "low_gain_aa_off": round(vals[False], 6),
                         "ratio_off_over_on": round(ratio, 2)
                         if ratio != float("inf") else ""})
            print(f"{T:>9.1f}{f:>9.4f}{vals[True]:>20.6f}{vals[False]:>20.6f}"
                  f"{ratio:>9.1f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
