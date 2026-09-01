"""整併版的動作要往哪裡轉：四種目標方向的天花板對照。**純 CPU，不最佳化。**

先講一個把設計簡化掉的結構事實
────────────────────────────────────────────────────────────────────
`runs/integration_design/README.md` 把「保長的整數位移」（設計丙）與「限制在
非零支撐上的旋轉」（設計甲）列為兩個候選。**在單一區塊上它們是同一族**：
保長映射把係數向量 `c` 送到等長的 `c'`，而 `c` 與 `c'` 必定張成一個二維平面，
在那個平面上它就是一個旋轉。所以「不用旋轉、改用一般的保長位移」買不到新的
可行集——**真正的自由度只有一個：目標方向 `e2` 怎麼選。**

於是這一支探針問的是：**在同樣的殘差大小下，哪一種 `e2` 最便宜？**

四種取法
────────────────────────────────────────────────────────────────────
| 取法 | `e2` 落在哪 | 對應的設計 |
|---|---|---|
| `full` | 整個通帶（含量化後為零的格子） | 現行整併版 |
| `support` | 只在 `α ≠ 0` 的格子上 | 設計甲 |
| `lowfreq` | 只在通帶內半徑最小的一半格子上 | 對照：往低頻搬便不便宜 |
| `priced` | 依 JPEG 量化表加權（步長大＝人眼不敏感＝便宜） | 本專案的知覺定價搬進 DCT 域 |

`e1` 一律對齊該區塊自己的係數向量 `c/‖c‖`——那是保長映射的天花板
（`‖Δc‖ = 2‖Pc‖·sin(θ/2)`，等號在 `e1 ∥ c` 時成立）。**不對齊量到的是隨機解
不是天花板**，第一版 `dct_nonadditive_ceiling` 在這裡低估過 18.6 倍。

量什麼
────────────────────────────────────────────────────────────────────
交付即參數：旋轉作用在 `jpeg_encode` 的整數係數上、取整、`jpeg_decode` 輸出。
恆等的對象因此是**壓縮圖**不是原圖，失真一律對壓縮圖量。

`off_support_frac` 是關鍵的診斷欄：位移的能量有多少落在**原本是零**的格子上。
現行整併版的 L∞ 精確等於 1.000，假說是能量被丟進空的高頻格子；若 `support`
把這一欄壓下去而失真跟著降，假說成立。

判準（跑之前寫下）
────────────────────────────────────────────────────────────────────
  A1  等殘差 RMS 下，`support` 的 DISTS 要低於 `full`。**降不下來就代表
      「能量被丟進空格子」這個歸因是錯的**，設計甲當場否決。
  A2  `support` 的 L∞ 要明顯低於 1.000。仍然飽和代表病因在別處。
  A3  `priced` 若優於 `support`，本專案的知覺定價在 DCT 域仍然有效，
      整併版應該走定價而不是走支撐。

用法：
    python scripts/dct_action_ceiling.py --out runs/integration_design/action_ceiling.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baselines.jpeg_codec import (  # noqa: E402
    CHANNEL_NAMES, jpeg_decode, jpeg_encode, normalize_quality, quant_table,
)
from src.defense.dct_nonadditive import band_indices  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
MODES = ("full", "support", "lowfreq", "priced")
THETAS = (0.2, 0.4, 0.8, 1.2, 1.8, 2.5)


def direction_weight(mode: str, alpha: torch.Tensor, idx, chroma: bool,
                     device, dtype) -> torch.Tensor:
    """每一格在 `e2` 裡被允許佔多少。形狀可廣播到 `alpha`。

    回傳的是**遮罩／權重**，不是方向本身；方向由隨機向量乘上它再正交化得到。
    """
    k = len(idx)
    if mode == "full":
        return torch.ones(k, device=device, dtype=dtype)
    if mode == "support":
        return (alpha != 0).to(dtype)
    if mode == "lowfreq":
        r = torch.tensor([(u * u + v * v) ** 0.5 / 8.0 for u, v in idx],
                         device=device, dtype=dtype)
        return (r <= float(r.median())).to(dtype)
    if mode == "priced":
        # 量化步長大＝該頻率人眼不敏感＝把能量放那裡便宜。用步長本身當權重，
        # 與 `--freq-weight jpeg_luma` 同一張表，只是這裡它是原生的、不必重採樣。
        tbl = quant_table(85, chroma=chroma, device=device, dtype=dtype)
        return torch.tensor([float(tbl[u, v]) for u, v in idx],
                            device=device, dtype=dtype)
    raise ValueError(f"未知的 mode {mode!r}")


@torch.no_grad()
def rotate_block(c: torch.Tensor, w: torch.Tensor, theta: float
                 ) -> torch.Tensor:
    """`e1` 對齊 `c`，`e2` 由 `w` 正交化而來，轉 `theta`。

    `e1 ∥ c` 時 `R(θ)c = cos θ · c + sin θ · ‖c‖ · e2`，不必展開一般式。
    `c` 為零向量或 `w` 與 `c` 平行時該區塊不動——**回傳原值而不是 NaN**，
    退化的比例由 `degenerate_frac` 報出來。
    """
    n = c.norm(dim=-1, keepdim=True)
    e1 = c / n.clamp_min(1e-12)
    w = w - (w * e1).sum(dim=-1, keepdim=True) * e1
    wn = w.norm(dim=-1, keepdim=True)
    e2 = w / wn.clamp_min(1e-12)
    live = ((n > 0) & (wn > 1e-12)).to(c.dtype)
    out = torch.cos(torch.tensor(theta)) * c + torch.sin(torch.tensor(theta)) * n * e2
    return c + (out - c) * live


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--qd", type=float, default=0.85)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--thetas", type=float, nargs="+", default=list(THETAS))
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in
             args.images.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dev = torch.device("cpu")
    suite = MetricSuite(device=dev)
    idx = band_indices(0.12)
    rows_i = [u for u, _ in idx]
    cols_i = [v for _, v in idx]
    q = normalize_quality(args.qd)

    out_rows: List[dict] = []
    for name in names:
        x = load_image_tensor(args.data / name / f"{name}.png", dev,
                              size=RESOLUTION)
        alpha = {k: v.detach() for k, v in jpeg_encode(x, args.qd).items()}
        ref = jpeg_decode(alpha, q)          # theta = 0 的輸出：壓縮圖
        for mode in args.modes:
            for th in args.thetas:
                g = torch.Generator().manual_seed(args.seed)
                coef: Dict[str, torch.Tensor] = {}
                off_num = off_den = 0.0
                degen = tot = 0
                for cname in CHANNEL_NAMES:
                    a = alpha[cname]
                    c = a[..., rows_i, cols_i]
                    wgt = direction_weight(mode, c, idx, cname != "Y",
                                           dev, a.dtype)
                    w = torch.randn(c.shape, generator=g).to(a.dtype) * wgt
                    new = rotate_block(c, w, th)
                    r = a.clone()
                    r[..., rows_i, cols_i] = new
                    r = torch.round(r)
                    coef[cname] = r
                    d = (r - a)[..., rows_i, cols_i]
                    zero = (c == 0)
                    off_num += float((d * zero).pow(2).sum())
                    off_den += float(d.pow(2).sum())
                    degen += int((c.norm(dim=-1) == 0).sum())
                    tot += int(c.norm(dim=-1).numel())
                xd = jpeg_decode(coef, q)
                m = suite.pairwise(xd, ref)
                resid = xd - ref
                out_rows.append({
                    "image": name, "mode": mode, "theta": th, "qd": args.qd,
                    "dists": round(m["dists"], 6),
                    "lpips": round(m["lpips"], 6),
                    "psnr": round(m["psnr"], 4),
                    "ssim": round(m["ssim"], 6),
                    "rms": round(float(resid.pow(2).mean().sqrt()), 6),
                    "linf": round(float(resid.abs().max()), 6),
                    # 位移的能量有多少落在**原本是零**的係數上
                    "off_support_frac": round(off_num / max(off_den, 1e-12), 6),
                    "degenerate_frac": round(degen / max(tot, 1), 6),
                })
        print(f"  {name} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, out_rows)
    print()
    print(f"寫出 {args.out}（{len(out_rows)} 列）")
    print()
    print("mode".ljust(10), "theta".ljust(7), "RMS".ljust(9), "DISTS".ljust(9),
          "PSNR".ljust(8), "Linf".ljust(8), "off_support")
    for mode in args.modes:
        for th in args.thetas:
            sel = [r for r in out_rows if r["mode"] == mode and r["theta"] == th]
            if not sel:
                continue
            a = {c: st.mean([r[c] for r in sel]) for c in
                 ("rms", "dists", "psnr", "linf", "off_support_frac")}
            print(mode.ljust(10), f"{th:.2f}".ljust(7),
                  f"{a['rms']:.5f}".ljust(9), f"{a['dists']:.5f}".ljust(9),
                  f"{a['psnr']:.3f}".ljust(8), f"{a['linf']:.4f}".ljust(8),
                  f"{a['off_support_frac']:.4f}")


if __name__ == "__main__":
    main()
