"""逐區塊的相位差熱力圖：防禦圖與淨化後的防禦圖，各自對**原圖**量。

要看的是什麼
────────────────────────────────────────────────────────────────────
把原圖的相位當成基準（相當於「原相位都是 0 度」），量防禦圖每一個區塊相對
它偏了多少；再用同一條尺量 `p(防禦圖)` 相對**原圖**偏了多少。並排之後，
一眼看得出淨化算子把那個偏移改成什麼樣子。

**這與 `phase_drift_diagnosis.py` 量的不是同一件事，兩者不可混用。**

    本檔            angle(STFT(p(x_def))) − angle(STFT(x))
    diagnosis       angle(STFT(p(x_def))) − angle(STFT(p(x)))

diagnosis 用「同一個算子也過一遍的原圖」當參照，扣掉了算子自己造成的相位
變化，量的是**裝上去的偏移還剩多少**。本檔不扣，量的是**攻擊方拿到的那張圖
相對真正的原圖偏了多少**——所以它把「防禦造成的」與「算子造成的」混在一起，
那正是這張圖要呈現的東西，但引用數字時必須講明。

量在哪裡
────────────────────────────────────────────────────────────────────
本方法自己的分析域：32×32 區塊、hop 8、Hann 窗的加窗 STFT
（`texture_rephase.analyze`），**不是全域 FFT**。每個區塊在通帶內以 `|S(原圖)|`
為權重取 `|wrap(Δφ)|` 的平均——權重不可省，相位在幅度接近零的地方由捨入雜訊
決定，不加權會讓平坦區的雜訊主導整張圖。

色階固定在 0–π 且全圖共用，否則列與列、欄與欄之間的亮度不可比。

用法：
    python scripts/phase_shift_heatmap.py \
        --defended <取回的防禦圖目錄> --out report_phase_shift_heatmap.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                          "PMingLiU", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.purify import ops as purify_ops              # noqa: E402
from src.residual.texture_rephase import PhaseResidual  # noqa: E402
from src.utils.io import load_image_tensor            # noqa: E402

RESOLUTION = 512
BLOCK, HOP = 32, 8
BAND = (0.12, 1.05)          # 通帶：與 r_min 一致，排除低頻與 DC

COLUMNS: Sequence[Tuple[str, purify_ops.Purifier]] = (
    ("防禦圖（未淨化）", purify_ops.Purifier("identity")),
    ("模糊 σ1", purify_ops.Purifier("blur", 1.0)),
    ("模糊 σ2", purify_ops.Purifier("blur", 2.0)),
    ("裁切 10%", purify_ops.Purifier("crop_resize", 0.10)),
    ("JPEG 30", purify_ops.Purifier("jpeg", 30)),
)


def wrap(a: torch.Tensor) -> torch.Tensor:
    """折回 (−π, π]。相位是週期量，直接相減再平均會被 ±π 的折返騙到。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def radial(block: int) -> torch.Tensor:
    fy = torch.fft.fftfreq(block) * 2.0
    fx = torch.fft.rfftfreq(block) * 2.0
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def shift_map(analyzer: PhaseResidual, ref: torch.Tensor, cur: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """(side, side) 的逐區塊 |Δφ|，以 `|S(ref)|` 加權，值域 [0, π]。"""
    s_ref, s_cur = analyzer.analyze(ref), analyzer.analyze(cur)
    d = wrap(torch.angle(s_cur) - torch.angle(s_ref)).abs()
    w = s_ref.abs() * mask
    num = (w * d).sum(dim=(-2, -1))
    den = w.sum(dim=(-2, -1)).clamp_min(1e-12)
    per = (num / den).mean(dim=1)[0]                 # 通道平均 → (L,)
    side = int(round(per.numel() ** 0.5))
    return per.reshape(side, side)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defended", type=Path, required=True)
    ap.add_argument("--tag", default="ours_ph_q")
    ap.add_argument("--condition", default="phase")
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    dev = torch.device("cpu")
    analyzer = PhaseResidual(size=RESOLUTION, block=BLOCK, hop=HOP).to(dev)
    r = radial(BLOCK)
    mask = ((r >= BAND[0]) & (r < BAND[1])).to(torch.float32)

    nrow, ncol = len(names), len(COLUMNS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.05 * nrow))
    im = None
    for i, name in enumerate(names):
        x = load_image_tensor(args.data / name / f"{name}.png", dev, size=RESOLUTION)
        x_def = load_image_tensor(
            args.defended / args.tag / f"{name}__{args.condition}__def.png",
            dev, size=RESOLUTION)
        analyzer.prepare_gates(x)                    # Hann 窗在這裡才填進 buffer
        for j, (label, pur) in enumerate(COLUMNS):
            m = shift_map(analyzer, x, pur.evaluate(x_def), mask)
            ax = axes[i][j]
            im = ax.imshow(m.numpy(), cmap=args.cmap, vmin=0.0, vmax=math.pi)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{float(m.mean()):.2f}", fontsize=7.5, labelpad=1.5)
            if i == 0:
                ax.set_title(label, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(name.replace("task_", "").replace("_", " ")[:20],
                              fontsize=6.5)
        print(f"  {name} 完成", flush=True)

    fig.suptitle(
        "逐區塊相位差：以原圖為基準（相位視為 0），越亮偏得越多\n"
        f"{args.tag}／十張／32×32 hop 8 STFT／通帶 r 由 {BAND[0]} 起／色階固定 0–π"
        "　　每格下方數字是該格的平均值（弧度）",
        fontsize=11)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.012)
    cbar.set_ticks([0, math.pi / 2, math.pi])
    cbar.set_ticklabels(["0", "π/2", "π"])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"→ {args.out}（{args.out.stat().st_size / 1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
