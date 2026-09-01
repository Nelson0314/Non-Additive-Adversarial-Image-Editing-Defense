"""用 BlurGuard 自己的判準量本專案與對照組的防禦圖。**不跑 GPU、不需要模型。**

BlurGuard（NeurIPS 2025，[arXiv:2511.00143](https://arxiv.org/abs/2511.00143)）
主張防護噪聲除了不可見還必須**不可還原**——在看不到原圖的前提下難以被辨識為
噪聲——並把它操作化成一個徑向功率譜的約束。這一支把那個約束搬過來，量的是
「我們交出去的圖，在它的判準上偏離原圖多少」。

移植的來源與差異
────────────────────────────────────────────────────────────────────
逐行對照公開原始碼 `BlurGuard/code/attacks.py`（`fft_fps`、
`compute_histogram_fft`、`pgd_freq` 裡的 `sigma_loss`）：

    fps(x)      = Σ_channel |FFT2(x)|²                      # fft_fps
    hist(f)[k]  = Σ_{round(r(u,v)) = k} f[u, v]             # compute_histogram_fft
    sigma_loss  = relu( max_k |log10 hist(fps(x)) − log10 hist(fps(x'))| − eps )

`eps` 是原始碼的 `eps_sigma`（預設 0.01）。**本支報未扣 `eps` 的原始最大值**
（也就是 `max_k |Δlog10|`），扣不扣只差一個常數，報原值才看得出方法之間的距離。

**原始碼有一處與論文敘述不符，兩種讀法都報。** `compute_histogram_fft` 由陣列
中心 `((H−1)/2, (W−1)/2)` 起算半徑，但 `fft_fps` 的 `torch.fft.fft2` **沒有做
fftshift**，直流分量在 `[0, 0]` 而不是中心。於是它的「半徑分箱」不是論文寫的
radially averaged PSD，而是對未平移頻譜的一個環狀分割。

- `unshifted` 逐位元照原始碼，可與該文的數字對照；
- `shifted` 補上 `fftshift`，才是論文敘述的那個量。

兩欄一起報，不代為裁定哪一個才算數。

用法：
    python scripts/blurguard_spectrum.py --pairs <def.png>=<orig.png> ...
    python scripts/blurguard_spectrum.py --scan runs/ip2p_gate_floor
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# 原始碼 `LF_PGD.__init__` 的預設值。列出來是為了讓「扣掉它之後是不是零」
# 這個問題有得查，本支本身報未扣的原值。
EPS_SIGMA_DEFAULT = 0.01


def load01(path: Path, size: int = 512) -> torch.Tensor:
    """讀成 `(3, H, W)`、值域 `[0, 1]` 的 float64。

    BlurGuard 的 `fft_fps` 收的是 `[-1, 1]` 並先做 `x/2 + 0.5`，等價於直接吃
    `[0, 1]`，故這裡直接用 `[0, 1]`，數值相同。
    """
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BICUBIC)
    a = np.asarray(img).copy()
    return torch.from_numpy(a).permute(2, 0, 1).to(torch.float64) / 255.0


def fps(x: torch.Tensor, shift: bool) -> torch.Tensor:
    """`Σ_channel |FFT2(x)|²`，形狀 `(H, W)`。`shift` 決定直流放在哪裡。"""
    f = torch.fft.fft2(x)
    if shift:
        f = torch.fft.fftshift(f, dim=(-2, -1))
    return (f.abs() ** 2).sum(dim=0)


def radial_bins(h: int, w: int) -> torch.Tensor:
    """`round(r)` 的整數分箱索引，半徑由陣列中心起算——與原始碼相同。"""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    return torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).round().long()


def hist(f: torch.Tensor, idx: torch.Tensor, n_bins: int) -> torch.Tensor:
    """逐環**求和**（不是平均）——原始碼是 `(masks * spectrum).sum()`。"""
    return torch.zeros(n_bins, dtype=f.dtype).scatter_add_(
        0, idx.reshape(-1), f.reshape(-1))


def sigma_gap(x_orig: torch.Tensor, x_def: torch.Tensor,
              shift: bool) -> float:
    """`max_k |log10 hist(fps(orig)) − log10 hist(fps(def))|`。"""
    h, w = x_orig.shape[-2:]
    idx = radial_bins(h, w)
    n_bins = int(idx.max()) + 1
    a = hist(fps(x_orig, shift), idx, n_bins)
    b = hist(fps(x_def, shift), idx, n_bins)
    d = (torch.log10(a + 1e-8) - torch.log10(b + 1e-8)).abs()
    return float(d.max())


def scan(root: Path) -> List[Tuple[str, str, Path, Path]]:
    """走訪 `<root>/<tag>/<image>__<cond>__def.png` 與同目錄的 `__orig.png`。

    缺 `__orig.png` 的一律拋錯而不是略過：靜默略過會讓出表時少一列，而沒有人
    會發現那一列本來應該在。
    """
    out = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        for defp in sorted(d.glob("*__def.png")):
            stem = defp.name[: -len("__def.png")]
            image, _, cond = stem.partition("__")
            origp = d / f"{image}__orig.png"
            if not origp.exists():
                raise SystemExit(f"{defp} 沒有對應的 {origp.name}")
            out.append((f"{d.name}/{cond}", image, defp, origp))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", type=Path, nargs="*", default=[],
                    help="遞迴走訪這些目錄底下的 `*__def.png`")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="`<def.png>=<orig.png>` 形式的額外配對")
    ap.add_argument("--out", type=Path, default=None, help="寫出 CSV")
    args = ap.parse_args()

    jobs: List[Tuple[str, str, Path, Path]] = []
    for root in args.scan:
        jobs += scan(root)
    for pair in args.pairs:
        dp, _, op = pair.partition("=")
        jobs.append((Path(dp).parent.name, Path(dp).stem, Path(dp), Path(op)))
    if not jobs:
        raise SystemExit("沒有指定 --scan 或 --pairs")

    rows = []
    for tag, image, defp, origp in jobs:
        xo, xd = load01(origp), load01(defp)
        rms = float((xd - xo).pow(2).mean().sqrt())
        rows.append({
            "tag": tag, "image": image,
            "rms": round(rms, 6),
            "linf": round(float((xd - xo).abs().max()), 6),
            "sigma_gap_unshifted": round(sigma_gap(xo, xd, shift=False), 6),
            "sigma_gap_shifted": round(sigma_gap(xo, xd, shift=True), 6),
        })

    by_tag: Dict[str, List[dict]] = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)

    print(f"BlurGuard 的 sigma_loss（未扣 eps_sigma = {EPS_SIGMA_DEFAULT}）")
    print("tag".ljust(30) + "  n" + "rms".rjust(10)
          + "unshifted".rjust(12) + "shifted".rjust(12))
    for tag in sorted(by_tag):
        g = by_tag[tag]
        m = lambda k: sum(r[k] for r in g) / len(g)  # noqa: E731
        print(tag.ljust(30) + f" {len(g):2d}" + f"{m('rms'):10.4f}"
              + f"{m('sigma_gap_unshifted'):12.4f}"
              + f"{m('sigma_gap_shifted'):12.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
