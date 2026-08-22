"""擾動殘差的「指紋」。**不跑 GPU、不重算任何防禦，只讀已存的防禦圖。**

問題
────────────────────────────────────────────────────────────────────
含加性下限的主線設定（`--spectral-floor 0.04`）在頻譜上多了一個加法項，
而 DCT-Shield 本身就是「在頻域上加一個受量化表定價的加法項」。兩者因此在
**構造敘述**上靠得很近。新穎性主張要站得住，就得先量出兩者的擾動在多大
程度上真的不同——這支腳本提供的是那個量，不是主張。

四個量，每一個對應一條構造上的差異
────────────────────────────────────────────────────────────────────
`phase_share`  在加窗重疊區塊的頻譜上，把殘差拆成極座標的兩半：

                   幅度那半  Σ (|S'| − |S|)²
                   相位那半  Σ (|S| · Δφ)²        Δφ = arg(S'/S) 繞回 (−π, π]

               `phase_share` 是相位那半的佔比。純相位旋轉在重疊相加之前
               幅度那半恰為零；自由的加性擾動兩半各半。**這個量直接讀出
               「乘法那一半還剩多少」**。

`block_gini`   逐區塊殘差能量的 Gini 係數，以及前 10% 區塊的能量佔比。
               DCT-Shield 的預算是逐係數的 ±ε·Q，**跨區塊是均勻的**；
               本方法的乘法那半乘了紋理閘，能量集中在少數區塊。空間選擇性
               是對 DCT-Shield 的主要差異，這是它的讀數。

`hf_share`     整張圖的 FFT 上，半 Nyquist 以上的能量佔比（FND-060 的量）。

`peak_ratio_8` 殘差自相關在 8 的倍數上的局部凸起，即 DCT-Shield 的 8×8
               格點鎖定（`residual_periodicity.py` 的同名量，此處一併算出，
               免得兩支腳本要對兩次檔）。

分析用的區塊與 hop 由旗標給定，**預設與現行主線相同**（32 / 8）。它決定
`phase_share` 與 `block_gini` 在哪一組基底上量；換基底會換數值，故逐列寫出。
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from defense_purify_gallery import RESOLUTION, discover  # noqa: E402
from residual_periodicity import autocorr_slices, peak_ratio  # noqa: E402
from src.residual.texture_rephase import PhaseResidual  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402


def polar_split(spec: torch.Tensor, spec_def: torch.Tensor) -> Dict[str, float]:
    """區塊頻譜的極座標分解。回傳兩半的能量與相位那半的佔比。

    `Δφ` 由 `angle(S' · conj(S))` 取得而不是兩個 `angle()` 相減：後者在
    ±π 附近會差一整圈，而相位是週期量，那個跳躍會被誤讀成巨大的相位改動。
    """
    d_mag = spec_def.abs() - spec.abs()
    d_phi = torch.angle(spec_def * spec.conj())          # 已在 (−π, π]
    e_mag = float((d_mag ** 2).sum())
    e_pha = float(((spec.abs() * d_phi) ** 2).sum())
    total = e_mag + e_pha
    return {
        "energy_mag": e_mag,
        "energy_phase": e_pha,
        "phase_share": e_pha / total if total > 0 else float("nan"),
    }


def gini(values: List[float]) -> float:
    """非負序列的 Gini 係數。0 = 完全均勻，1 = 全部集中在一格。

    均勻的預算（DCT-Shield 的逐係數 ±ε·Q）在區塊層級應接近 0；乘了紋理閘的
    擾動應明顯大於 0。
    """
    v = sorted(float(x) for x in values)
    if not v or v[0] < 0:
        raise ValueError("Gini 只對非負序列有定義")
    n = len(v)
    total = sum(v)
    if total <= 0:
        return 0.0
    cum = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(v))
    return cum / (n * total)


def block_energy(residual: torch.Tensor, block: int, hop: int) -> torch.Tensor:
    """逐區塊的殘差能量 (L,)。區塊格與分析用的 STFT 格相同。"""
    r = (0.299 * residual[:, 0] + 0.587 * residual[:, 1]
         + 0.114 * residual[:, 2]).unsqueeze(1)
    patches = torch.nn.functional.unfold(r, kernel_size=block, stride=hop)
    return (patches ** 2).sum(dim=1).squeeze(0)


def hf_share(residual: torch.Tensor) -> float:
    """整張圖 FFT 上，歸一化半徑 > 0.5 的能量佔比（FND-060 的量）。"""
    r = (0.299 * residual[:, 0] + 0.587 * residual[:, 1]
         + 0.114 * residual[:, 2])
    f = torch.fft.rfft2(r - r.mean())
    n = r.shape[-1]
    fy = torch.fft.fftfreq(r.shape[-2]) * 2.0
    fx = torch.fft.rfftfreq(n) * 2.0
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    p = (f.abs() ** 2)[0]
    total = float(p.sum())
    return float(p[rad > 0.5].sum()) / total if total > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True,
                    help="含 <image>__<cond>__def.png 的目錄，可給多個")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--hop", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device("cpu")
    module = PhaseResidual(size=RESOLUTION, block=args.block, hop=args.hop)

    rows: List[dict] = []
    for src in args.src:
        found = discover(src)
        for image, by_cond in sorted(found.items()):
            orig = args.data / image / f"{image}.png"
            if not orig.exists():
                raise SystemExit(f"找不到原圖 {orig}")
            x01 = load_image_tensor(orig, dev, size=RESOLUTION)
            module.prepare_gates(x01)
            spec = module.analyze(x01)
            for cond, kinds in sorted(by_cond.items()):
                if "def" not in kinds:
                    continue
                x_def = load_image_tensor(kinds["def"], dev, size=RESOLUTION)
                res = x_def - x01
                pol = polar_split(spec, module.analyze(x_def))
                be = block_energy(res, args.block, args.hop)
                order = torch.sort(be, descending=True).values
                top = int(math.ceil(0.10 * len(order)))
                sl = autocorr_slices(res)
                rows.append({
                    "image": image, "condition": cond,
                    "src": str(kinds["def"].parent),
                    "analysis_block": args.block, "analysis_hop": args.hop,
                    "rms": round(float(res.pow(2).mean().sqrt()), 6),
                    "phase_share": round(pol["phase_share"], 4),
                    "energy_mag": round(pol["energy_mag"], 4),
                    "energy_phase": round(pol["energy_phase"], 4),
                    "block_gini": round(gini(be.tolist()), 4),
                    "top10_block_share": round(
                        float(order[:top].sum() / order.sum()), 4),
                    "hf_share": round(hf_share(res), 4),
                    "peak_ratio_8": round(peak_ratio(sl, 8), 3),
                    "peak_ratio_32": round(peak_ratio(sl, 32), 3),
                })
                print(f"{image:34s} {cond:16s} phase_share="
                      f"{rows[-1]['phase_share']:.3f} gini="
                      f"{rows[-1]['block_gini']:.3f} hf="
                      f"{rows[-1]['hf_share']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    by_cond: Dict[str, List[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    print(f"\n{'條件':20s}{'n':>4s}{'RMS':>9s}{'phase_share':>13s}"
          f"{'gini':>8s}{'top10':>8s}{'hf':>8s}{'peak8':>8s}")
    for cond, rs in sorted(by_cond.items()):
        f = lambda k: st.fmean(r[k] for r in rs)  # noqa: E731
        print(f"{cond:20s}{len(rs):4d}{f('rms'):9.4f}{f('phase_share'):13.3f}"
              f"{f('block_gini'):8.3f}{f('top10_block_share'):8.3f}"
              f"{f('hf_share'):8.3f}{f('peak_ratio_8'):8.3f}")
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
