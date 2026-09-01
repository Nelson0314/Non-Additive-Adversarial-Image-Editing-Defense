"""模糊那一欄的可贏上界：每個頻帶的**存活率**與**知覺價錢**。**不跑 GPU。**

問題
────────────────────────────────────────────────────────────────────
高斯模糊是零相位濾波器，頻域乘的是實正數 `H(ω)`，所以它**在原理上動不了
相位**——`runs/phase_drift_diagnosis` 量到相位保留 ρ=0.967 而殘差能量只剩
8.9%。擾動是被**靜音**不是被破壞。

那麼把同一份擾動搬到 `H(ω)` 還沒壓掉的頻率上，能拿回多少？這個問題不需要
GPU：`H(ω)` 是已知的曲線，而「那個頻帶的知覺價錢」可以直接量。

作法
────────────────────────────────────────────────────────────────────
對每一個徑向帶，造一個**帶限**的擾動、RMS 固定，加到真實影像上，量兩件事：

    知覺價錢 = DISTS(x + d, x)            ← 這一帶的擾動有多明顯
    存活     = ‖blur(x+d) − blur(x)‖² / ‖d‖²   與方向餘弦

兩者相除得到 **每單位知覺代價換到的存活能量**。上界就是這條曲線的極大值。

`d` 取得夠小（RMS 與 `purifier_band_transfer` 同為 0.01）使差商接近線性響應，
但仍加在真實影像上——夾取與非線性都在工作點附近量。

為什麼這一支能回答「要不要繼續花機時」
────────────────────────────────────────────────────────────────────
若最好的那一帶相對現行工作帶只有小幅的提升，那麼把擾動搬過去換到的淨增益
上界就是「現行淨增益 × 那個倍數」。現行主線在 blur σ1 上的淨增益是 0.121，
而該欄的空白地板是 0.156、人眼判定三個方法皆未擋下——**倍數要很大才有意義**。

**判準（跑之前寫下）**：最佳帶對現行工作帶（0.60–0.80，本方法 60% 的殘差能量
在半 Nyquist 以上）的「存活／知覺價錢」比值若低於 3，模糊那一欄結案為
結構性負面結果，不再排 GPU。

用法：
    python scripts/blur_band_ceiling.py --out runs/fixedpoint_blur/ceiling.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from purifier_band_transfer import (  # noqa: E402
    band_limited_noise, radial_grid,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.purify import ops as purify_ops  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
PROBE_RMS = 0.01
# 低頻要切細：模糊的通帶就在那裡，粗帶會把上界整個蓋掉。
BANDS = ((0.00, 0.03), (0.03, 0.06), (0.06, 0.12), (0.12, 0.20),
         (0.20, 0.30), (0.30, 0.45), (0.45, 0.60), (0.60, 0.80),
         (0.80, 1.05))
# 現行工作帶的代表：本方法 60% 的殘差能量在半 Nyquist 以上（`RESULTS.md`）。
REFERENCE_BAND = (0.60, 0.80)


def gaussian_transfer(sigma: float, lo: float, hi: float, n: int = 512) -> float:
    """高斯模糊在 `[lo, hi)` 這一帶的平均 `|H|²`，**解析值**，用來對照實測。

    連續高斯的頻率響應是 `H(f) = exp(−2π²σ²f²)`，`f` 是每像素的週期數。本專案
    的徑向座標把 Nyquist 歸一到 1，故 `f = r/2`。
    """
    dev = torch.device("cpu")
    r = radial_grid(n, dev, torch.float32)
    m = (r >= lo) & (r < hi)
    if not bool(m.any()):
        return float("nan")
    f = r[m] / 2.0
    h = torch.exp(-2.0 * (torch.pi ** 2) * (sigma ** 2) * f ** 2)
    return float((h ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-response", type=Path,
                    default=Path("runs/encoder_frequency_response/"
                                 "latent_norm.csv"),
                    help="逐帶的編碼器敏感度。給了就一併算出乘積上界")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names: Sequence[str] = [ln.strip() for ln in
                            args.images.read_text(encoding="utf-8").splitlines()
                            if ln.strip()]
    dev = torch.device("cpu")
    suite = MetricSuite(device=dev)
    blur = purify_ops.Purifier("blur", args.sigma)

    rows: List[dict] = []
    for name in names:
        x = load_image_tensor(args.data / name / f"{name}.png", dev,
                              size=RESOLUTION)
        for lo, hi in BANDS:
            d = band_limited_noise(x.shape, lo, hi, args.seed, dev, x.dtype)
            d = d * PROBE_RMS
            xd = (x + d).clamp(0.0, 1.0)
            # **夾取之後的實際殘差**才是付出去的東西，量價錢要用它。
            d_eff = xd - x
            with torch.no_grad():
                left, right = blur.evaluate(xd), blur.evaluate(x)
            surv = left - right
            m = suite.pairwise(xd, x)
            e_in = float(d_eff.pow(2).sum())
            energy = float(surv.pow(2).sum()) / max(e_in, 1e-12)
            cos = float((surv.flatten() @ d_eff.flatten())
                        / (surv.flatten().norm() * d_eff.flatten().norm()
                           ).clamp_min(1e-12))
            rows.append({
                "image": name, "band_lo": lo, "band_hi": hi,
                "sigma": args.sigma,
                "dists": round(m["dists"], 6),
                "lpips": round(m["lpips"], 6),
                "psnr": round(m["psnr"], 4),
                "rms": round(float(d_eff.pow(2).mean().sqrt()), 6),
                "energy_ratio": round(energy, 6),
                "cosine": round(cos, 6),
                "h2_analytic": round(gaussian_transfer(args.sigma, lo, hi), 6),
                # 每單位知覺代價換到的存活能量。**這一欄就是要比的東西。**
                "surv_per_dists": round(energy / max(m["dists"], 1e-9), 4),
            })
        print(f"  {name} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    print(f"\n寫出 {args.out}（{len(rows)} 列）\n")
    print("帶".ljust(14), "DISTS".ljust(10), "存活能量".ljust(11),
          "餘弦".ljust(9), "解析 |H|^2".ljust(11), "存活/DISTS")
    summary = {}
    for lo, hi in BANDS:
        sel = [r for r in rows if (r["band_lo"], r["band_hi"]) == (lo, hi)]
        agg = {k: round(st.mean([r[k] for r in sel]), 6)
               for k in ("dists", "energy_ratio", "cosine", "h2_analytic",
                         "surv_per_dists")}
        summary[(lo, hi)] = agg
        print(f"{lo:.2f}–{hi:.2f}".ljust(14),
              f"{agg['dists']:.5f}".ljust(10),
              f"{agg['energy_ratio']:.5f}".ljust(11),
              f"{agg['cosine']:.4f}".ljust(9),
              f"{agg['h2_analytic']:.5f}".ljust(11),
              f"{agg['surv_per_dists']:.3f}")

    best = max(summary.items(), key=lambda kv: kv[1]["surv_per_dists"])
    ref = summary[REFERENCE_BAND]
    ratio = best[1]["surv_per_dists"] / max(ref["surv_per_dists"], 1e-12)
    print()
    print(f"最佳帶 {best[0][0]:.2f}-{best[0][1]:.2f}: "
          f"存活/DISTS = {best[1]['surv_per_dists']:.3f}")
    print(f"現行工作帶 {REFERENCE_BAND[0]:.2f}-{REFERENCE_BAND[1]:.2f}: "
          f"{ref['surv_per_dists']:.3f}")
    print(f"存活這一半的倍數 = {ratio:.2f}"
          "（**這只是乘積的一半，見下**）")

    if args.encoder_response and args.encoder_response.exists():
        combine(args)


def combine(args) -> None:
    """把存活率乘上**編碼器在該帶的敏感度**，才是淨增益的上界。

    只看存活率會得到「往低頻搬」的錯誤結論：低頻確實活得下來
    （r<0.03 的存活/DISTS 是現行工作帶的 13.5 倍），**但編碼器在那裡幾乎
    看不見**。`runs/encoder_frequency_response/` 量到跨帶的 `move_per_dists`
    由 278 升到 242507，差三個數量級，方向與存活率完全相反。

    要比的是乘積：上界(帶) = 編碼器敏感度(帶) × 存活振幅(帶)。

    **取振幅不取能量**：`latent_move` 對小擾動是一階量，隨振幅線性，
    故乘的是 `sqrt(|H|^2)`。能量模型一併報出，兩者結論相同時才算穩。

    `|H|^2` 取解析值——它與本檔實測的一致到小數第三位（0.99570 對
    0.99351、0.53196 對 0.53444），而編碼器那份用的帶界與本檔不同，
    解析式才對得齊。
    """
    rows = list(csv.DictReader(args.encoder_response.open(encoding="utf-8")))
    bands: List[tuple] = []
    for r in rows:
        k = (float(r["r_lo"]), float(r["r_hi"]))
        if k not in bands:
            bands.append(k)

    out_rows, tab = [], {}
    for k in bands:
        sel = [r for r in rows if (float(r["r_lo"]), float(r["r_hi"])) == k]
        mpd = st.mean([float(r["move_per_dists"]) for r in sel])
        h2 = gaussian_transfer(args.sigma, k[0], k[1])
        amp = h2 ** 0.5
        tab[k] = mpd * amp
        out_rows.append({"r_lo": k[0], "r_hi": k[1],
                         "move_per_dists": round(mpd, 3),
                         "h2_analytic": round(h2, 8),
                         "amp_survival": round(amp, 6),
                         "ceiling_amplitude": round(mpd * amp, 2),
                         "ceiling_energy": round(mpd * h2, 2),
                         "n_images": len({r["image"] for r in sel})})
    combined = args.out.with_name(args.out.stem + "_combined.csv")
    write_csv(combined, out_rows)

    best = max(tab.items(), key=lambda kv: kv[1])
    cur = [k for k in bands if k[0] >= 0.53 and k[1] <= 0.89]
    cur_val = st.mean([tab[k] for k in cur])
    print()
    print(f"寫出 {combined}")
    print("band".ljust(16), "move/DISTS".ljust(12), "amp surv".ljust(10),
          "ceiling")
    for k in bands:
        r = next(o for o in out_rows if (o["r_lo"], o["r_hi"]) == k)
        print(f"{k[0]:.3f}-{k[1]:.3f}".ljust(16),
              f"{r['move_per_dists']:.1f}".ljust(12),
              f"{r['amp_survival']:.5f}".ljust(10),
              f"{r['ceiling_amplitude']:.1f}")
    print()
    print(f"最佳帶 {best[0][0]:.3f}-{best[0][1]:.3f}: 上界 {best[1]:.1f}")
    print(f"現行配置代表帶平均: {cur_val:.1f}")
    print(f"**上界倍數 = {best[1] / max(cur_val, 1e-12):.2f}**"
          "（事前判準：低於 3 則模糊那一欄結案）")



if __name__ == "__main__":
    main()
