"""候選三 步驟 0／1：log-periodic 場對裁切放大的不變性與它的失真代價。

`docs/reference/SURVEY_ARCHITECTURE.md` 候選三：在對數極座標 `(log r, φ)` 上
定義擾動，並要求它在 `log r` 上以 `ln s` 為週期（`s = 1.2488` 是評測裁切算子
的放大倍率）。則繞中心放大 `s` 倍會把擾動映射到**它自己**——構造上的不動點。

推導很短：裁切縮放的輸出在半徑 `r` 的內容來自輸入的 `r / s`。若場滿足
`d(r, φ) = d(r / s, φ)`，輸出的場就逐點等於輸入的場。

第 6 點的兩個否證步驟，**都不需要 GPU、不需要最佳化**：

- **步驟 0**：(i) 量它對 `crop_resize(0.10)` 的**方向存活率**，要遠高於
  `band_transfer.csv` 裡任何一帶的 0.015；**沒有明顯高於 0.5 就是構造有錯，
  停手**。(ii) 量它在 13 張圖上的失真；**等 RMS 下高於現行殘差兩倍就停手**。
- **步驟 1**：掃 `s' ∈ [1.1, 1.5]` 的方向存活率曲線。**若只在 s = 1.2488 有一個
  窄峰**，「先驗」的辯護不成立，本候選退回為對單一算子的 co-adapt——那與
  已否決的「針對淨化最佳化」是同一種毛病。

三個實作陷阱照該節寫的處理：中心附近 `log r → −∞` 是奇異點（設 `r > r0` 的
內圈並在圈內置零）；角度方向必須環繞，否則 `φ = 0` 有接縫；環寬約 `(s−1)·r`
像素，在 r 小時只有幾個像素，取樣品質沿半徑劇烈變化。

用法：

    python scripts/log_periodic_probe.py --out runs/log_periodic_probe
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.purify.ops import crop_resize  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 評測算子的放大倍率：crop_resize(0.10) 每邊各裁 10%，保留中央 (1-2f) 的邊長。
EVAL_FRACTION = 0.10
EVAL_SCALE = 1.0 / (1.0 - 2 * EVAL_FRACTION)      # 1.2488
# 內圈半徑。log r 在中心發散，圈內置零；圈內即不變性失效區。
R0 = 8.0
# 現行工作點的殘差（`b_ph_rpi`：純相位＋下限 r=pi）。步驟 0 的失真判準用它。
CURRENT_RMS = 0.0560
CURRENT_DISTS = 0.1286


def log_periodic_field(seed: int, scale: float, n_logr: int = 8,
                       n_phi: int = 16, size: int = RESOLUTION,
                       device=None) -> torch.Tensor:
    """(1,1,size,size) 的 log-periodic 場，單位標準差，內圈置零。

    `scale` 是要不變的放大倍率；場在 `log r` 上的週期是 `ln(scale)`。
    係數格 `(n_logr, n_phi)` 用**環繞的**雙線性取樣貼回像素域：`log r` 方向
    環繞是週期性的定義本身，`φ` 方向環繞是為了避免 0 與 2π 之間出現接縫。
    """
    device = device or torch.device("cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randn(n_logr, n_phi, generator=g).to(device)

    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2,
        torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2,
        indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    phi = torch.atan2(yy, xx)

    ln_s = math.log(scale)
    # log r 除以週期取小數部分 → [0,1)，這一步就是「以 ln s 為週期」。
    u = torch.remainder(torch.log(r.clamp_min(1e-6)) / ln_s, 1.0)
    v = (phi + math.pi) / (2 * math.pi)                       # [0,1)

    # 環繞的雙線性取樣：兩個維度都對 n 取模。
    def sample(u, v):
        fu, fv = u * n_logr, v * n_phi
        i0, j0 = torch.floor(fu).long(), torch.floor(fv).long()
        du, dv = (fu - i0).to(c.dtype), (fv - j0).to(c.dtype)
        i0, i1 = i0 % n_logr, (i0 + 1) % n_logr
        j0, j1 = j0 % n_phi, (j0 + 1) % n_phi
        return (c[i0, j0] * (1 - du) * (1 - dv) + c[i1, j0] * du * (1 - dv)
                + c[i0, j1] * (1 - du) * dv + c[i1, j1] * du * dv)

    d = sample(u, v)
    # **正規化只在有效區上算，歸零放在最後**。反過來做的話，減均值會把內圈由
    # 0 變成 −mean，於是那個圓盤變成一塊常數偏移——不變性在圈內本來就失效，
    # 再加一塊直流只會讓失真白付。
    valid = r > R0
    dv = d[valid]
    d = (d - dv.mean()) / dv.std()
    d = torch.where(valid, d, torch.zeros_like(d))
    return d.to(torch.float32)[None, None]


def survival(x01: torch.Tensor, d: torch.Tensor, fraction: float) -> tuple:
    """`purifier_band_transfer.py` 的兩個量：能量存活率與方向存活率。

    方向是對**原網格**的餘弦——`band_transfer.csv` 裡所有頻帶都低於 0.02，
    正是「擾動被搬走」的證據。log-periodic 場若構造正確，這一項應該接近 1。
    """
    p0 = crop_resize(x01, fraction)
    p1 = crop_resize((x01 + d).clamp(0, 1), fraction)
    diff = (p1 - p0).flatten()
    dv = d.expand_as(x01).flatten()
    energy = float(diff.pow(2).sum() / dv.pow(2).sum())
    denom = diff.norm() * dv.norm()
    direction = float((diff * dv).sum() / denom) if float(denom) > 0 else float("nan")
    return energy, direction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-logr", type=int, default=8)
    ap.add_argument("--n-phi", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 步驟 0：不變性與失真 ----------------------------------------
    rows = []
    for name in names:
        x = load_image_tensor(dataset[name]["path"], device, size=RESOLUTION)
        for seed in range(args.seeds):
            f = log_periodic_field(seed, EVAL_SCALE, args.n_logr, args.n_phi,
                                   RESOLUTION, device)
            # 兩個尺度：band_transfer 用的小訊號 0.01，與現行工作點的 0.0560。
            for target_rms in (0.01, CURRENT_RMS):
                d = f * target_rms          # f 是單位標準差，故縮放即 RMS
                e, c = survival(x, d, EVAL_FRACTION)
                m = suite.pairwise(x, (x + d).clamp(0, 1))
                rows.append({
                    "image": name, "seed": seed, "field": "log_periodic",
                    "scale": round(EVAL_SCALE, 5), "target_rms": target_rms,
                    "energy_survival": round(e, 5),
                    "direction_survival": round(c, 5),
                    "dists": round(m["dists"], 6), "lpips": round(m["lpips"], 6),
                    "psnr": round(m["psnr"], 4), "rms": round(m["rms"], 6),
                    "linf": round(m["linf"], 6),
                })
            # 對照：同 RMS 的白雜訊，方向存活率應該塌到接近 0。
            g = torch.Generator(device="cpu").manual_seed(1000 + seed)
            w = torch.randn(1, 1, RESOLUTION, RESOLUTION, generator=g).to(device)
            w = (w - w.mean()) / w.std()
            for target_rms in (0.01, CURRENT_RMS):
                d = w * target_rms
                e, c = survival(x, d, EVAL_FRACTION)
                m = suite.pairwise(x, (x + d).clamp(0, 1))
                rows.append({
                    "image": name, "seed": seed, "field": "white_noise",
                    "scale": round(EVAL_SCALE, 5), "target_rms": target_rms,
                    "energy_survival": round(e, 5),
                    "direction_survival": round(c, 5),
                    "dists": round(m["dists"], 6), "lpips": round(m["lpips"], 6),
                    "psnr": round(m["psnr"], 4), "rms": round(m["rms"], 6),
                    "linf": round(m["linf"], 6),
                })
        write_csv(args.out / "step0.csv", rows)
        print(f"{name} 完成", flush=True)

    print()
    print("步驟 0（13 張 × 3 個場）")
    print(f"{'場':<14s} {'RMS':>7s} {'能量存活':>9s} {'方向存活':>9s} {'DISTS':>8s} {'PSNR':>7s}")
    for fld in ("log_periodic", "white_noise"):
        for t in (0.01, CURRENT_RMS):
            sel = [r for r in rows if r["field"] == fld and r["target_rms"] == t]
            print(f"{fld:<14s} {t:7.4f} "
                  f"{statistics.fmean(r['energy_survival'] for r in sel):9.4f} "
                  f"{statistics.fmean(r['direction_survival'] for r in sel):9.4f} "
                  f"{statistics.fmean(r['dists'] for r in sel):8.5f} "
                  f"{statistics.fmean(r['psnr'] for r in sel):7.2f}")

    lp = [r for r in rows if r["field"] == "log_periodic"
          and r["target_rms"] == CURRENT_RMS]
    dir_mean = statistics.fmean(r["direction_survival"] for r in lp)
    dists_mean = statistics.fmean(r["dists"] for r in lp)
    print()
    print("步驟 0 的兩個判準（SURVEY_ARCHITECTURE 候選三 第 6 點）：")
    print(f"  (i) 方向存活率要明顯高於 0.5：實測 {dir_mean:.4f} -> "
          + ("通過" if dir_mean > 0.5 else "**不通過，構造有錯，停手**"))
    print(f"  (ii) 等 RMS 下失真不得高於現行殘差兩倍（{CURRENT_DISTS} 的兩倍 "
          f"= {2 * CURRENT_DISTS:.4f}）：實測 {dists_mean:.4f} -> "
          + ("通過" if dists_mean <= 2 * CURRENT_DISTS
             else "**不通過，代價超出預算，停手**"))

    # ---- 步驟 1：不變性是窄峰還是先驗 --------------------------------
    print()
    print("步驟 1：掃攻擊方實際用的放大倍率 s'，看不變性是不是只在 1.2488 有峰")
    sweep = []
    subset = names[:5]
    print(f"{chr(115)+chr(39):>7s} {'裁切比例':>9s} {'方向存活':>9s} {'能量存活':>9s}")
    for s_prime in (1.05, 1.10, 1.15, 1.2488, 1.30, 1.40, 1.50):
        frac = (1.0 - 1.0 / s_prime) / 2.0
        vals_c, vals_e = [], []
        for name in subset:
            x = load_image_tensor(dataset[name]["path"], device, size=RESOLUTION)
            for seed in range(args.seeds):
                f = log_periodic_field(seed, EVAL_SCALE, args.n_logr,
                                       args.n_phi, RESOLUTION, device)
                e, c = survival(x, f * CURRENT_RMS, frac)
                vals_c.append(c)
                vals_e.append(e)
        sweep.append({"s_prime": round(s_prime, 5), "fraction": round(frac, 5),
                      "direction_survival": round(statistics.fmean(vals_c), 5),
                      "energy_survival": round(statistics.fmean(vals_e), 5),
                      "n": len(vals_c)})
        print(f"{s_prime:7.4f} {frac:9.4f} {sweep[-1]['direction_survival']:9.4f} "
              f"{sweep[-1]['energy_survival']:9.4f}")
    write_csv(args.out / "step1_scale_sweep.csv", sweep)

    peak = max(sweep, key=lambda r: r["direction_survival"])
    others = [r["direction_survival"] for r in sweep
              if abs(r["s_prime"] - EVAL_SCALE) > 1e-6]
    print()
    print(f"峰值在 s'={peak['s_prime']}，方向存活 {peak['direction_survival']:.4f}；"
          f"其餘倍率的平均是 {statistics.fmean(others):.4f}")
    print(f"表：{args.out}")


if __name__ == "__main__":
    main()
