"""淨化算子把殘差的哪一段吃掉。**不跑 GPU、不需要防禦圖。**

問題
────────────────────────────────────────────────────────────────────
抗淨化的頭對頭表只給出「淨增益差多少」，不說**為什麼**。本方法在
`crop_resize` 上一直是最弱的一格（`docs/RESULTS.md`：淨增益保留 13%），而
JPEG 那一格反而最強。要判斷該往哪裡改，得先知道每個算子吃掉的是哪一段頻率。

作法
────────────────────────────────────────────────────────────────────
把一個**帶限**的小擾動 `d` 加到真實影像上，量

    存活率(算子, 帶) = ‖ p(x + d) − p(x) ‖² / ‖ d ‖²

用 `p(x+d) − p(x)` 而不是 `p(d)`：JPEG 與夾取都是非線性的，離開真實影像的
工作點量出來的是另一回事。`d` 取得夠小（RMS 0.01）使這個差商接近線性響應，
但仍在真實的量化格點上。

`crop_resize` 另外要注意一件事：它**幾何上放大了 1.25 倍**，於是來源第 r 帶的
能量落到 r/1.25。所以「原帶存活率」不是對的量，本腳本量的是**總能量存活率**
（擾動還剩多少，不論落在哪一帶），並另外報出重心搬到哪裡。

用法：
    python scripts/purifier_band_transfer.py --out runs/.../band_transfer.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.purify import ops as purify_ops  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
PROBE_RMS = 0.01          # 夠小以貼近線性，夠大以跨過 JPEG 的量化格
BANDS = ((0.00, 0.12), (0.12, 0.25), (0.25, 0.40), (0.40, 0.55),
         (0.55, 0.70), (0.70, 0.85), (0.85, 1.05))


def radial_grid(n: int, device, dtype) -> torch.Tensor:
    """(n, n//2+1) 的歸一化半徑，1 即 Nyquist。與 `radial_gate` 同座標。"""
    fy = torch.fft.fftfreq(n, device=device, dtype=dtype) * 2.0
    fx = torch.fft.rfftfreq(n, device=device, dtype=dtype) * 2.0
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def band_limited_noise(shape, lo: float, hi: float, seed: int,
                       device, dtype) -> torch.Tensor:
    """能量只落在 [lo, hi) 的實值雜訊，RMS 正規化到 1。"""
    b, c, n, _ = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(b, c, n, n, generator=g).to(device=device, dtype=dtype)
    spec = torch.fft.rfft2(w)
    r = radial_grid(n, device, dtype)
    spec = spec * ((r >= lo) & (r < hi)).to(spec.dtype)
    out = torch.fft.irfft2(spec, s=(n, n))
    return out / out.pow(2).mean().sqrt().clamp_min(1e-12)


def spectral_centroid(d: torch.Tensor) -> float:
    """殘差亮度的能量加權平均半徑。用來看算子把能量搬到哪裡。"""
    lum = 0.299 * d[:, 0] + 0.587 * d[:, 1] + 0.114 * d[:, 2]
    p = torch.fft.rfft2(lum).abs() ** 2
    r = radial_grid(lum.shape[-1], d.device, d.dtype)
    total = p.sum()
    return float((p * r).sum() / total) if float(total) > 0 else float("nan")


def survival(purifier, x: torch.Tensor, d: torch.Tensor) -> dict:
    """能量存活率**與**方向存活率。

    只看能量會讀錯 JPEG：量化是階梯函數，小擾動不是被抹掉就是把係數推過一個
    量化界而產生整整一階的改變，平均下來能量比 1 還大。那不代表擾動還在——
    它可能已經變成與原擾動無關的雜訊。故一併報餘弦相似度：**能量高而餘弦低
    ＝ 擾動被打散成雜訊，不是被保留。**

    `crop_resize` 的餘弦另有一層意思：它幾何上放大 1.25 倍，殘差即使逐點都在
    也對不回原位，餘弦因此會低。這正是要分辨的事——能量還在、方向不在。
    """
    with torch.no_grad():
        left = purifier.evaluate(x + d)
        right = purifier.evaluate(x)
        # 控制組：把同一個擾動放在中性灰上跑同一個算子。灰底落在 [0,1] 正中央
        # 故夾取不會咬到，量出來的就是**算子自己對這個擾動做了什麼**，與影像
        # 內容無關。`crop_resize` 之下它等於「被搬過位置的同一個擾動」。
        warped = purifier.evaluate(torch.full_like(x, 0.5) + d) - 0.5
    surv = left - right

    def cos(a, b):
        return float((a.flatten() @ b.flatten())
                     / (a.flatten().norm() * b.flatten().norm()).clamp_min(1e-12))

    return {"energy_ratio": float(surv.pow(2).sum() / d.pow(2).sum()),
            "cosine": cos(surv, d),
            "cosine_vs_warped": cos(surv, warped),
            "warped_energy_ratio": float(warped.pow(2).sum() / d.pow(2).sum()),
            "centroid_in": spectral_centroid(d),
            "centroid_out": spectral_centroid(surv)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--verdict", type=Path,
                    default=Path("runs/obedience_audit/"
                                 "recognisability_verdict.csv"),
                    help="沒給 --images 時由這份判定檔取影像清單")
    ap.add_argument("--purifiers", nargs="+",
                    default=["blur1", "jpeg75", "jpeg30", "crop_resize0.1",
                             "jpeg_then_resize75"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names: Sequence[str] = args.images or sorted(
        {r["image"] for r in csv.DictReader(
            args.verdict.open(encoding="utf-8"))})
    dev = torch.device("cpu")
    made = {p_.kind if not p_.strength else f"{p_.kind}{p_.strength:g}": p_
            for p_ in (purify_ops.Purifier("blur", 1.0),
                       purify_ops.Purifier("jpeg", 75),
                       purify_ops.Purifier("jpeg", 30),
                       purify_ops.Purifier("crop_resize",
                                           purify_ops.CROP_FRACTION_DIA),
                       purify_ops.Purifier("jpeg_then_resize",
                                           purify_ops.CR_JPEG_QUALITY))}
    unknown = set(args.purifiers) - set(made)
    if unknown:
        raise SystemExit(f"未知的算子：{sorted(unknown)}；可用 {sorted(made)}")

    rows: List[dict] = []
    for name in names:
        x = load_image_tensor(args.data / name / f"{name}.png", dev,
                              size=RESOLUTION)
        for lo, hi in BANDS:
            d = band_limited_noise(x.shape, lo, hi, args.seed, dev,
                                   x.dtype) * PROBE_RMS
            for label in args.purifiers:
                s = survival(made[label], x, d)
                rows.append({"image": name, "purifier": label,
                             "band_lo": lo, "band_hi": hi,
                             "probe_rms": PROBE_RMS,
                             "energy_ratio": round(s["energy_ratio"], 5),
                             "cosine": round(s["cosine"], 5),
                             "cosine_vs_warped": round(s["cosine_vs_warped"], 5),
                             "warped_energy_ratio": round(
                                 s["warped_energy_ratio"], 5),
                             "centroid_in": round(s["centroid_in"], 4),
                             "centroid_out": round(s["centroid_out"], 4)})
        print(f"{name} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    purs = list(args.purifiers)
    for key, title in (("energy_ratio", "能量存活率"),
                       ("cosine", "方向存活率（對原擾動的餘弦）"),
                       ("cosine_vs_warped", "對照：算子自己搬過的同一擾動")):
        head = f"{'帶':>12s}" + "".join(f"{p:>19s}" for p in purs)
        print(f"\n{title}（{len(names)} 張平均）")
        print(head); print("-" * len(head))
        for lo, hi in BANDS:
            cells = ""
            for p in purs:
                sel = [r[key] for r in rows
                       if r["purifier"] == p and r["band_lo"] == lo]
                cells += f"{statistics.fmean(sel):19.3f}"
            print(f"{f'{lo:.2f}–{hi:.2f}':>12s}{cells}")

    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
