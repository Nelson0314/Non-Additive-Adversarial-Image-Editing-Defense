"""`--coarsen` 的上機前檢查：粗網格到底有沒有減少寬頻外溢、有沒有更耐 JPEG。

**不需要 GPU、不需要擴散模型。** 量的是參數化本身的性質，與最佳化無關：
固定同一組隨機 theta（同種子、同分布），只改視窗網格的粗細，看

  1. **帶外能量比例**——殘差的功率譜落在徑向帶通 [r_min, r_max] 之外的比例。
     docstring 宣稱相鄰視窗獨立旋轉會在選定頻格之外攤出寬頻能量，這一欄就是
     那句話的直接讀數。
  2. **JPEG 保留率**——殘差經過 `jpeg_roundtrip` 之後，在原方向上的分量佔多少
     （`<d', d> / <d, d>`）。這是抗 JPEG 的直接代理，判準沿用
     `runs/ip2p_deliver_jpeg/README.md` 的 0.22 隨機基準。

**為了公平，殘差先被歸一化到同一個 L2。** 不歸一化的話粗網格因為自由度少、
隨機和的方差小，會自動比較弱，兩件事會混在一起。

用法：
    python scripts/coarsen_probe.py --image data/omniedit150/<name>/input.png
    python scripts/coarsen_probe.py --data data/omniedit150 --n 6
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.baselines.jpeg_codec import jpeg_roundtrip  # noqa: E402
from src.residual.texture_rephase import PhaseResidual  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

SIZE = 512


def radial_index(n: int, device, dtype) -> torch.Tensor:
    """每個 rfft2 頻格的歸一化半徑，與 `radial_gate` 用的是同一個尺度。"""
    fy = torch.fft.fftfreq(n, device=device, dtype=dtype).view(-1, 1)
    fx = torch.fft.rfftfreq(n, device=device, dtype=dtype).view(1, -1)
    return (fy ** 2 + fx ** 2).sqrt()


def band_split(resid: torch.Tensor, r_min: float, r_hi: float) -> dict:
    """殘差能量的徑向分布（整張影像的全域 FFT）。

    **本方法的徑向帶通是 `[r_min, inf)`，沒有上界**，所以「帶外」只有低頻那
    一側，量它回答不了「有沒有寬頻高頻外溢」——假說講的是後者。因此分成兩欄：
    `below` 是漏到 r < r_min 的比例（重疊相加與紋理閘造成的洩漏），`high` 是
    落在 r > r_hi 的比例，也就是 JPEG 亮度量化表壓得最重、最先丟掉的那一段。
    假說預測 `high` 會隨粗細倍率下降。
    """
    spec = torch.fft.rfft2(resid.mean(dim=1))
    power = spec.abs() ** 2
    # rfft2 除了第 0 與 Nyquist 行以外每一格代表兩個共軛頻率，這裡不補權重，
    # 因為分子分母用的是同一組格點，比例不受影響。
    r = radial_index(resid.shape[-1], resid.device, resid.dtype)
    total = float(power.sum())
    if total <= 0:
        return {"below": float("nan"), "high": float("nan")}
    return {"below": float(power[:, r < r_min].sum()) / total,
            "high": float(power[:, r > r_hi].sum()) / total}


def jpeg_retention(x01: torch.Tensor, resid: torch.Tensor,
                   quality: float, deliver: float = 0.0) -> float:
    """`<d', d> / <d, d>`，d' 是攻擊方壓縮後的殘差。1.0 表示完全保留。

    `deliver > 0` 時模擬**量化交付**：先把防禦圖與原圖各自壓到 QD（那是主線
    條件 `--deliver-jpeg` 實際交出去的東西），再讓攻擊方壓一次。參照方向 `d`
    也隨之改成交付後的殘差——問的是「交出去的那個擾動剩多少」，不是「最佳化
    當下的那個擾動剩多少」。
    """
    if deliver > 0:
        x_def = jpeg_roundtrip((x01 + resid).clamp(0.0, 1.0), deliver)
        x_ref = jpeg_roundtrip(x01, deliver)
    else:
        x_def, x_ref = (x01 + resid).clamp(0.0, 1.0), x01
    d = x_def - x_ref
    d2 = jpeg_roundtrip(x_def, quality) - jpeg_roundtrip(x_ref, quality)
    denom = float((d * d).sum())
    return float((d2 * d).sum()) / denom if denom > 0 else float("nan")


def probe_one(x01: torch.Tensor, coarsen: int, seed: int, args) -> dict:
    m = PhaseResidual(size=SIZE, block=args.block, hop=args.hop,
                      r_min=args.r_min, theta_max=math.pi,
                      spectral_floor=args.spectral_floor,
                      freq_weight=args.freq_weight,
                      freq_weight_power=args.freq_weight_power,
                      coarsen=coarsen).to(x01.dtype)
    m.prepare_gates(x01)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.theta.copy_(torch.randn(m.theta.shape, generator=gen) * args.theta)
        m.theta.clamp_(-args.radius, args.radius)
        if args.spectral_floor > 0:
            m.floor.copy_(torch.randn(m.floor.shape, generator=gen).clamp(-1, 1))
        resid = m.pixel_residual(x01).clamp(0, 1) - x01
    # **等 L2**：不歸一化的話粗網格自由度少、隨機和方差小，會自動比較弱。
    rms = float(resid.pow(2).mean().sqrt())
    resid = resid * (args.rms / rms)
    split = band_split(resid, args.r_min, args.r_hi)
    row = {"coarsen": coarsen, "params": int(m.theta.shape[1]),
           "rms_raw": rms,
           "below_band": split["below"], "high_frac": split["high"]}
    for q in args.qualities:
        row[f"keep_q{int(q * 100)}"] = jpeg_retention(
            x01, resid, q, deliver=args.deliver)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--coarsen", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--qualities", type=float, nargs="+",
                    default=[0.9, 0.75, 0.5, 0.3])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--hop", type=int, default=8)
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=1.0, help="隨機 theta 的標準差")
    ap.add_argument("--spectral-floor", type=float, default=0.04)
    ap.add_argument("--freq-weight", default="jpeg_luma")
    ap.add_argument("--freq-weight-power", type=float, default=0.25)
    ap.add_argument("--rms", type=float, default=0.02, help="等 L2 的目標 RMS")
    ap.add_argument("--deliver", type=float, default=0.0,
                    help="模擬量化交付的品質（主線用 0.85）。0 = 交付未壓縮的"
                         "連續值圖，即加 --deliver-jpeg 之前的行為")
    ap.add_argument("--r-hi", type=float, default=0.35,
                    help="算「高頻比例」的門檻，歸一化半徑。0.35 之後 JPEG "
                         "亮度量化表的階距已經很粗，是最先被丟掉的一段")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/ip2p_coarsen/probe.csv"))
    args = ap.parse_args()

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()][:args.n]
    dev = torch.device("cpu")
    rows = []
    for i, n in enumerate(names):
        hits = sorted((args.data / n).glob("*.png")) + \
            sorted((args.data / n).glob("*.jpg"))
        if not hits:
            raise SystemExit(f"{args.data / n} 下沒有影像")
        x01 = load_image_tensor(hits[0], dev).clamp(0, 1)
        if x01.shape[-1] != SIZE:
            raise SystemExit(f"{n} 不是 {SIZE}x{SIZE}，收到 {tuple(x01.shape)}")
        for c in args.coarsen:
            r = probe_one(x01, c, seed=1000 + i, args=args)
            r["image"] = n
            rows.append(r)
        print(f"  {i + 1}/{len(names)} {n}", flush=True)

    import csv
    from collections import defaultdict
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image", "coarsen", "params", "rms_raw", "below_band", "high_frac"] + \
        [f"keep_q{int(q * 100)}" for q in args.qualities]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    per = defaultdict(list)
    for r in rows:
        per[r["coarsen"]].append(r)
    print(f"\n表：{args.out}（{len(rows)} 列，{len(names)} 張，等 RMS {args.rms}）")
    print("coarsen  參數量   低頻洩漏  高頻比例  " + "  ".join(
        f"keep_q{int(q * 100)}" for q in args.qualities))
    for c in args.coarsen:
        rs = per[c]
        def m(k):
            return sum(r[k] for r in rs) / len(rs)
        cells = "  ".join(f"{m(f'keep_q{int(q * 100)}'):9.4f}"
                          for q in args.qualities)
        print(f"{c:7d}  {rs[0]['params']:6d}  {m('below_band'):8.4f}  "
              f"{m('high_frac'):8.4f}  {cells}")


if __name__ == "__main__":
    main()
