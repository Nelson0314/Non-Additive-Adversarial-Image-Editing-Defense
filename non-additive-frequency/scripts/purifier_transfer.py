"""十二個淨化算子的頻率響應：擾動放在哪一個空間頻率上活得下來。

為什麼要有這一支
────────────────────────────────────────────────────────────────────
`docs/DIRECTION.md` §6.4 說，若「效果 ↔ 抗低通」是內在取捨，本方向的貢獻就
改成**把那條取捨曲線畫出來**——載體的頻帶是自變數，各類淨化的存活率是應變數。

那條曲線的 x 軸此前只有三個點（逐像素自由 → 低頻懲罰 → 全域映射），而且是
用**跑完整批 GPU** 得到的。但 x 軸本身與擴散模型無關：它問的是
「一個放在頻率 f 上的擾動，過了算子 P 之後還剩多少」。那是算子的性質，
**純 CPU 就量得到**，而且可以量得很密。

這一支因此把 §6.4 的 x 軸一次補滿，並且回答三個現在還在猜的問題：

1. `--patch-res S` 的帶限應該設在哪裡才划算（S 決定截止在 1/S）。
2. `--patch-tile T` 的基頻 1/T 落在哪一段（平舖不等於帶限，但基頻仍然要選）。
3. 摩爾紋那條路成不成立——**若算子的低通有效，差頻就產生不出來**。

兩個讀數，缺一不可
────────────────────────────────────────────────────────────────────
    energy   ‖P(x+δ) − P(x)‖ / ‖δ‖        擾動還剩多少
    align    cos( P(x+δ) − P(x), δ )      它還在不在原來的位置

**分開量是必要的。** `runs/ip2p_residual_signature/band_transfer.csv` 已經記過
一個直接相關的事實：裁切之後殘差對**原格點**的餘弦是 0.000，但對**算子自己
搬過的那一份**是 0.995——擾動原封不動地通過了，只是沒有人替它對回去。
只看 energy 會把「被破壞」與「被搬走」讀成同一件事，而它們的對策完全不同。

探針：帶限雜訊，不是單頻光柵
────────────────────────────────────────────────────────────────────
第一版用正弦光柵，量出來 JPEG 在**每一個頻率**上都是 1.00——包含 0.45
cyc/px。那不是結果，是探針錯了：**單一正弦的能量全部集中在一個 DCT 係數
上**，那個係數遠大於任何品質的量化階，四捨五入推不動它。學出來的補丁是
寬頻的，能量攤在幾百個係數上、每一個都小，量化把它們整批歸零。

**JPEG 的傷害是逐係數的，取決於能量怎麼分布，不只是落在哪一帶。**
故探針改成**環帶內的白雜訊**（頻率落在 f/√2 到 f·√2 的環內、相位隨機），
那才是「一個頻帶被佔住」的正確模型。

振幅也是變因，不是常數：量化階固定，同一個頻帶上大振幅活得下來、小振幅
活不下來。故 RMS 是掃描軸之一。參考值——補丁族的 fid_psnr 約 13.5，
對應 RMS 約 0.20；相位族約 0.05；加性 baseline 約 0.01。

灰底取 0.5：夾取會把量到的東西從「算子的響應」換成「值域的響應」。

**這一支不量「防禦有沒有效」。** 它量的是算子，與載體裡長什麼樣子無關。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.purify import ops  # noqa: E402

#: 旋轉的種子。直接引 `purify_identity` 的常數而不是抄一份同樣的數字：
#: 這一支量的算子與抗淨化那一支量的必須是同一個，抄兩份會各自漂走。
sys.path.insert(0, str(ROOT / "scripts"))
from purify_identity import ROTATE_SEED_DEFAULT as ROTATE_SEED  # noqa: E402

BASE = 0.5          # 灰底。夾取會污染讀數，故不用 0 或 1 附近。
AMP = 0.15          # 振幅。0.5 ± 0.15 落在值域內，任何算子都不會夾到。
SIDE = 512


def band_noise(f: float, rms: float, seed: int, side: int = SIDE) -> torch.Tensor:
    """(1,3,S,S)，能量落在 f/√2 到 f·√2 環帶內的雜訊，RMS 正規化到 `rms`。

    在頻域造環帶遮罩、乘上隨機相位、逆變換取實部。三個通道各自獨立——
    補丁族學出來的內容也是逐通道的。
    """
    g = torch.Generator().manual_seed(seed)
    fy = torch.fft.fftfreq(side)[:, None]
    fx = torch.fft.fftfreq(side)[None, :]
    r = (fy ** 2 + fx ** 2).sqrt()
    lo, hi = f / math.sqrt(2.0), f * math.sqrt(2.0)
    mask = ((r >= lo) & (r <= hi)).to(torch.float32)
    if float(mask.sum()) == 0:
        raise ValueError(f"環帶 [{lo:.4f}, {hi:.4f}] 在 {side}x{side} 上是空的")
    chans = []
    for _ in range(3):
        ph = torch.rand((side, side), generator=g) * 2 * math.pi
        v = torch.fft.ifft2(mask * torch.exp(1j * ph)).real
        chans.append(v / v.pow(2).mean().sqrt() * rms)
    return torch.stack(chans)[None]


def grating(f: float, theta: float, side: int = SIDE) -> torch.Tensor:
    """(1,3,S,S) 的正弦光柵。**保留但不再使用**，理由見檔頭的探針一節。"""
    y, x = torch.meshgrid(torch.arange(side, dtype=torch.float32),
                          torch.arange(side, dtype=torch.float32), indexing="ij")
    phase = 2 * math.pi * f * (x * math.cos(theta) + y * math.sin(theta))
    return (AMP * torch.sin(phase))[None, None].expand(1, 3, side, side).contiguous()


def purifiers() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """`docs/DIRECTION.md` §3.1 的那一組，參數逐項照抄。

    取不到相依套件的算子**明確回報**，不靜默跳過——少一個算子的表看起來
    完全正常，而結論會少一整欄。
    """
    out: Dict[str, Callable] = {
        "blur1.5": lambda t: ops.gaussian_blur(t, 1.5),
        "blur1.0": lambda t: ops.gaussian_blur(t, 1.0),
        "quantize8": lambda t: ops.quantize_real(t, 8),
        "jpeg90": lambda t: ops.jpeg_real(t, 90),
        "jpeg75": lambda t: ops.jpeg_real(t, 75),
        "jpeg60": lambda t: ops.jpeg_real(t, 60),
        "jpeg30": lambda t: ops.jpeg_real(t, 30),
        "crop_resize0.1": lambda t: ops.crop_resize(t, 0.1),
        "resize_only": ops.resize_only,
        "jpeg_then_resize75": lambda t: ops.jpeg_then_resize(t, 75),
        # 種子不是 0：0 在 10° 上抽到 −0.075°，512 px 影像各處位移都在次像素
        # 量級，那一欄量的是恆等映射而不是旋轉。見 `ops.rotate_angle`。
        "rotate10": lambda t: ops.rotate_random(t, 10.0, seed=ROTATE_SEED),
        "shift_only": ops.shift_only,
        "gaussian_noise": lambda t: ops.gaussian_noise(t, 0.02, seed=0),
    }
    if ops.has_guided_filter():
        out["adverse_cleaner"] = ops.adverse_cleaner_real
    else:
        print("[缺] adverse_cleaner：取不到 guided filter 的相依套件", flush=True)
    return out


def measure(fn: Callable, base: torch.Tensor, delta: torch.Tensor
            ) -> Tuple[float, float]:
    """回傳 `(energy, align)`。"""
    with torch.no_grad():
        a = fn(base.clone())
        b = fn((base + delta).clamp(0, 1))
    r = (b - a).reshape(-1)
    d = delta.reshape(-1)
    nr, nd = float(r.norm()), float(d.norm())
    if nd == 0:
        raise ValueError("探針的能量為零")
    align = float(torch.dot(r, d) / (nr * nd)) if nr > 0 else 0.0
    return nr / nd, align


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freqs", nargs="+", type=float, default=None,
                    help="cycles/pixel。預設為 1/256 到 0.45 的對數格點")
    ap.add_argument("--rms", nargs="+", type=float, default=[0.20, 0.05, 0.01],
                    help="擾動的 RMS。0.20 約等於補丁族（PSNR 13.5）、"
                         "0.05 約等於相位族、0.01 是加性 baseline 的量級")
    ap.add_argument("--out", default="runs/purifier_transfer")
    args = ap.parse_args()

    freqs = args.freqs or [round(1.0 / t, 6) for t in
                           (256, 181, 128, 91, 64, 45, 32, 23, 16, 11, 8, 6, 4, 3, 2.5, 2.2)]
    freqs = sorted(f for f in freqs if 0 < f <= 0.5)
    base = torch.full((1, 3, SIDE, SIDE), BASE)
    purs = purifiers()
    seeds = (0, 1)
    print(f"{len(purs)} 個算子 x {len(freqs)} 個頻率 x {len(args.rms)} 個振幅"
          f" x {len(seeds)} 個種子", flush=True)

    rows: List[dict] = []
    for rms in args.rms:
        print(f"-- RMS = {rms}（PSNR ~ {-20 * math.log10(rms):.1f} dB）"
              + "-" * 46, flush=True)
        print(f"{'f (cyc/px)':>11}{'週期px':>9}"
              + "".join(f"{n[:11]:>12}" for n in purs), flush=True)
        for f in freqs:
            cells = {}
            for name, fn in purs.items():
                e = [measure(fn, base, band_noise(f, rms, s)) for s in seeds]
                energy = sum(v[0] for v in e) / len(e)
                align = sum(v[1] for v in e) / len(e)
                cells[name] = (energy, align)
                # `quant_floor` 是「算子的輸出量化階 ÷ 擾動振幅」，也就是
                # 擾動被完全移除時 `energy` 會停在的那個值。它隨振幅縮小而
                # **上升**，故小振幅上的高 `energy` 讀起來像存活、其實是地板。
                # 判別法是 `align`：地板上的殘差與擾動無關，`align` 必為 0。
                # 實例見 `runs/purifier_transfer/README.md` §七。
                rows.append({"freq": f, "period_px": round(1 / f, 2), "rms": rms,
                             "purifier": name, "energy": round(energy, 5),
                             "align": round(align, 5),
                             "quant_floor": round(1.0 / (rms * 255.0), 5),
                             "at_quant_floor": int(
                                 abs(energy - 1.0 / (rms * 255.0)) < 0.02
                                 and abs(align) < 0.05)})
            print(f"{f:>11.5f}{1/f:>9.1f}"
                  + "".join(f"{cells[n][0]:>12.3f}" for n in purs), flush=True)
        print("", flush=True)
    print("對齊 cos：能量還在但位置被搬走的算子，只有這一欄看得出來", flush=True)
    rms0 = args.rms[0]
    for f in (freqs[0], freqs[len(freqs) // 2], freqs[-1]):
        sel = [r for r in rows if r["freq"] == f and r["rms"] == rms0]
        print(f"  f={f:.5f}  " + "  ".join(f"{r['purifier'][:9]}={r['align']:+.2f}"
                                          for r in sel), flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["freq", "period_px", "rms", "purifier",
                                            "energy", "align",
                                            "quant_floor", "at_quant_floor"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
