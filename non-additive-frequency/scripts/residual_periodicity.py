"""擾動殘差的空間週期性。**不跑 GPU、不重算任何防禦。**

假設
────────────────────────────────────────────────────────────────────
同樣的擾動能量下，**結構化**的圖樣比**非結構化**的雜訊顯眼得多——規則圖樣
落在視覺系統敏感的頻帶上而且彼此不互相遮蔽。本方法的重疊相加固定在
hop=16 的格點上，人眼在原始解析度下看到的正是規則的區塊狀鱗片重影；
DCT-Shield 的擾動是 8x8 量化雜訊，看起來接近乾淨。

這支量的就是那件事：殘差 `x_def - x01` 的自相關若在 hop 的倍數上出現峰，
代表失真是格點鎖定的，那麼「打散格點」是有依據的改良方向；若沒有峰，
可見度的來源是別的東西，不該往那裡投機時。

作法
────────────────────────────────────────────────────────────────────
取殘差的亮度、扣掉均值，算歸一化自相關（走 FFT），再沿兩軸各取一維切片。
`peak_ratio` = 位移為 hop 倍數處的平均自相關 ÷ 其餘位移處的平均自相關。
比值明顯大於 1 即格點鎖定。**同時報 8 的倍數**，那是 DCT-Shield 的格點，
用來確認這個量本身抓得到已知的週期。
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from defense_purify_gallery import RESOLUTION, discover  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

MAX_LAG = 64


def autocorr_slices(residual: torch.Tensor) -> Dict[str, List[float]]:
    """殘差亮度的歸一化自相關，沿 x 與 y 各取一維切片。"""
    r = (0.299 * residual[:, 0] + 0.587 * residual[:, 1]
         + 0.114 * residual[:, 2])
    r = r - r.mean()
    f = torch.fft.rfft2(r)
    ac = torch.fft.irfft2(f * f.conj(), s=r.shape[-2:])
    ac = ac / ac[..., 0, 0].clamp_min(1e-12)
    return {"x": ac[0, 0, :MAX_LAG].tolist(),
            "y": ac[0, :MAX_LAG, 0].tolist()}


def peak_ratio(slices: Dict[str, List[float]], period: int,
               half_window: int = 3) -> float:
    """`period` 倍數處的自相關相對**鄰近位移**的比值。

    不與全體位移比：自相關本來就隨位移衰減，而 `period` 的倍數平均落在較大的
    位移上，直接比全體會把衰減讀成「沒有週期性」——第一版就是這樣，三個條件
    全部低於 1。格點鎖定要看的是**局部**凸起，故基準取同一個位移附近。
    """
    ratios = []
    for axis in ("x", "y"):
        v = slices[axis]
        for lag in range(period, len(v), period):
            lo = max(1, lag - half_window)
            hi = min(len(v), lag + half_window + 1)
            near = [abs(v[k]) for k in range(lo, hi) if k != lag]
            if near:
                ratios.append(abs(v[lag]) / max(st.fmean(near), 1e-9))
    return st.fmean(ratios) if ratios else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--periods", type=int, nargs="+", default=[8, 16, 32])
    args = ap.parse_args()

    found = discover(args.src)
    rows = []
    for image, by_cond in sorted(found.items()):
        orig = args.data / image / f"{image}.png"
        if not orig.exists():
            continue
        x01 = load_image_tensor(orig, torch.device("cpu"), size=RESOLUTION)
        for cond, kinds in sorted(by_cond.items()):
            if "def" not in kinds:
                continue
            x_def = load_image_tensor(kinds["def"], torch.device("cpu"),
                                      size=RESOLUTION)
            sl = autocorr_slices(x_def - x01)
            row = {"image": image, "condition": cond,
                   "rms": round(float((x_def - x01).pow(2).mean().sqrt()), 5)}
            for p in args.periods:
                row[f"peak_ratio_{p}"] = round(peak_ratio(sl, p), 3)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    by_cond: Dict[str, List[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    header = "".join(f"{'倍數 ' + str(p):>12s}" for p in args.periods)
    print(f"\n{'條件':22s}{'n':>4s}{'殘差 RMS':>11s}{header}")
    for cond, rs in sorted(by_cond.items()):
        cells = "".join(
            f"{st.fmean(float(r[f'peak_ratio_{p}']) for r in rs):12.3f}"
            for p in args.periods)
        print(f"{cond:22s}{len(rs):4d}"
              f"{st.fmean(float(r['rms']) for r in rs):11.5f}{cells}")


if __name__ == "__main__":
    main()
