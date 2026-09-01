"""把「相位有沒有跑掉」畫成看得見的一頁。**不跑 GPU、影像不入版控。**

`scripts/phase_drift_diagnosis.py` 給的是數字；這一支把同一件事在一張真實
影像上攤開，讓三種失效方式各自的樣子直接對照：

    模糊    載體被抽掉——相位角原封不動，但它乘上去的那個係數幅度沒了
    JPEG    角度被打散——幅度還在（甚至更大），角度與裝上去的那一個脫鉤
    裁切    整份被搬走——殘差還在、能量還在，只是不在原來的區塊格點上

版面（每個算子一列）：

    原圖局部 | 防禦圖局部 | 裝上去的殘差 | 淨化後存活的殘差 | 逐區塊的相位差

殘差一律用**同一個放大倍率**顯示，否則四列之間的深淺不可比；倍率印在圖上。
最後一欄是把 `Δφ`（在指定頻帶上以 |S| 加權平均）攤回區塊的空間位置，
色階固定在 ±π，兩張圖看起來一不一樣就是「跑掉了沒有」。

用法：
    python scripts/phase_drift_figure.py \
        --defended <取回的防禦圖目錄> --image task_attr_mod_color_11699 \
        --out <輸出目錄>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

# 預設字型沒有中日韓字，標題會整排變成空方框而且**不會報錯**。
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                          "PMingLiU", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.purify import ops as purify_ops            # noqa: E402
from src.residual.texture_rephase import PhaseResidual  # noqa: E402
from src.utils.io import load_image_tensor          # noqa: E402

RESOLUTION = 512
BLOCK, HOP = 32, 8
AMP = 12.0                       # 殘差顯示放大倍率，四列共用
PHASE_BAND = (0.40, 0.70)        # 畫相位圖用的頻帶：帶內、且三個算子差異最大
CROP = (128, 128, 256)           # 局部放大的 (top, left, size)

ALL_ROWS: Dict[str, Tuple[str, purify_ops.Purifier]] = {
    "identity": ("未淨化", purify_ops.Purifier("identity")),
    "blur1": ("模糊 σ1", purify_ops.Purifier("blur", 1.0)),
    "blur2": ("模糊 σ2", purify_ops.Purifier("blur", 2.0)),
    "jpeg75": ("JPEG 75", purify_ops.Purifier("jpeg", 75)),
    "jpeg30": ("JPEG 30", purify_ops.Purifier("jpeg", 30)),
    "crop_resize0.1": ("裁切 10%", purify_ops.Purifier("crop_resize", 0.10)),
    "crop_resize0.15": ("裁切 15%", purify_ops.Purifier("crop_resize", 0.15)),
}
DEFAULT_ROWS = ("identity", "blur1", "jpeg30", "crop_resize0.1")


def wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def radial_index(block: int) -> torch.Tensor:
    fy = torch.fft.fftfreq(block) * 2.0
    fx = torch.fft.rfftfreq(block) * 2.0
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def installed_shift(analyzer: PhaseResidual, a: torch.Tensor,
                    b: torch.Tensor) -> torch.Tensor:
    """`b` 相對 `a` 的逐（區塊, 頻格）相位偏移，形狀同 `analyze` 的輸出。"""
    sa, sb = analyzer.analyze(a), analyzer.analyze(b)
    return wrap(torch.angle(sb) - torch.angle(sa)), sa.abs()


def agreement_map(d_inst: torch.Tensor, d_surv: torch.Tensor,
                  w: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
    """(side, side) 的逐區塊相位一致度 rho_b，值域 [0, 1]。

    每個區塊在指定頻帶上取 `|Σ_ω w·exp(i(Δφ_surv − Δφ_inst))| / Σ_ω w`，
    與 `phase_drift_diagnosis.py` 的 `rho` 是同一個量，只是那邊對整張圖取、
    這邊留在區塊上以便看**空間分布**。1 = 相位原封不動，0 = 與裝上去的無關。
    直接把角度相減再平均是錯的（角度有週期性），故走合成向量。
    """
    m = ((radial >= PHASE_BAND[0]) & (radial < PHASE_BAND[1])).to(torch.float32)
    ww = w * m
    diff = d_surv - d_inst
    re = (ww * torch.cos(diff)).sum(dim=(-2, -1))
    im = (ww * torch.sin(diff)).sum(dim=(-2, -1))
    tot = ww.sum(dim=(-2, -1)).clamp_min(1e-12)
    rho = (torch.sqrt(re ** 2 + im ** 2) / tot).mean(dim=1)[0]   # 通道平均 → (L,)
    side = int(round(rho.numel() ** 0.5))
    return rho.reshape(side, side)


def show(ax, img: torch.Tensor, title: str = "") -> None:
    t, l, s = CROP
    a = img[0, :, t:t + s, l:l + s].permute(1, 2, 0).clamp(0, 1).numpy()
    ax.imshow(a)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def show_residual(ax, d: torch.Tensor, title: str = "") -> None:
    t, l, s = CROP
    a = (d[0, :, t:t + s, l:l + s] * AMP + 0.5).permute(1, 2, 0).clamp(0, 1).numpy()
    ax.imshow(a)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defended", type=Path, required=True)
    ap.add_argument("--image", default="task_attr_mod_color_11699")
    ap.add_argument("--tag", default="ours_ph_q")
    ap.add_argument("--condition", default="phase")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--purifiers", nargs="+", default=list(DEFAULT_ROWS),
                    help="要畫哪幾列，名稱同 phase_drift_diagnosis 的算子名")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    device = torch.device("cpu")
    analyzer = PhaseResidual(size=RESOLUTION, block=BLOCK, hop=HOP).to(device)
    radial = radial_index(BLOCK)

    x = load_image_tensor(args.data / args.image / f"{args.image}.png", device,
                          size=RESOLUTION)
    x_def = load_image_tensor(
        args.defended / args.tag / f"{args.image}__{args.condition}__def.png",
        device, size=RESOLUTION)
    analyzer.prepare_gates(x)
    d_pix = x_def - x
    shift_inst, w_inst = installed_shift(analyzer, x, x_def)

    rows = []
    for name in args.purifiers:
        if name not in ALL_ROWS:
            raise SystemExit(f"未知的算子 {name!r}；可用的是 {list(ALL_ROWS)}")
        rows.append(ALL_ROWS[name])

    fig, axes = plt.subplots(len(rows), 5, figsize=(14.5, 3.05 * len(rows)))
    for r, (label, pur) in enumerate(rows):
        px, px_def = pur.evaluate(x), pur.evaluate(x_def)
        d_surv = px_def - px
        shift_surv, _ = installed_shift(analyzer, px, px_def)
        rho_map = agreement_map(shift_inst, shift_surv, w_inst, radial)

        show(axes[r][0], px, "原圖經算子" if r == 0 else "")
        show(axes[r][1], px_def, "防禦圖經算子" if r == 0 else "")
        show_residual(axes[r][2], d_pix, f"裝上去的殘差 ×{AMP:g}" if r == 0 else "")
        show_residual(axes[r][3], d_surv, f"存活的殘差 ×{AMP:g}" if r == 0 else "")

        im = axes[r][4].imshow(rho_map.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        axes[r][4].set_xticks([]); axes[r][4].set_yticks([])
        if r == 0:
            axes[r][4].set_title(
                f"逐區塊相位一致度 ρ\n(帶 {PHASE_BAND[0]}–{PHASE_BAND[1]}，"
                "亮 = 相位原封不動)", fontsize=9)
        axes[r][0].set_ylabel(label, fontsize=11)

        e0, e1 = float(d_pix.pow(2).sum()), float(d_surv.pow(2).sum())
        cos = float((d_pix * d_surv).sum() /
                    (d_pix.norm() * d_surv.norm()).clamp_min(1e-12))
        extra = ""
        if pur.kind == "crop_resize":
            # 只把殘差送進同一個幾何變換（不夾取值域，見 diagnosis 的說明）。
            import torch.nn.functional as F
            h, w = d_pix.shape[-2:]
            dh = int(round(h * pur.strength)); dw = int(round(w * pur.strength))
            warped = F.interpolate(d_pix[..., dh:h - dh, dw:w - dw], size=(h, w),
                                   mode=purify_ops.CROP_INTERPOLATION,
                                   antialias=purify_ops.CROP_ANTIALIAS)
            cw = float((warped * d_surv).sum() /
                       (warped.norm() * d_surv.norm()).clamp_min(1e-12))
            extra = f"\n對「搬過的同一份」cos {cw:.4f}"
        axes[r][3].set_xlabel(f"能量 {e1 / e0:.2f}×   對原格點 cos {cos:.3f}{extra}",
                              fontsize=8)
        # **與 `phase_drift_diagnosis.py` 的 `rho` 不是同一個平均**：那邊是
        # 全圖以 |S| 加權的單一合成向量，這裡是逐區塊算完再取未加權平均，
        # 平坦區塊（|S| 小、角度由捨入決定）在這裡佔一樣的份量，故偏低。
        axes[r][4].set_xlabel(f"區塊平均 ρ {float(rho_map.mean()):.3f}", fontsize=8)

    fig.colorbar(im, ax=axes[:, 4].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle(f"{args.tag} / {args.image}：淨化之後相位還在不在", fontsize=13)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"phase_drift_{args.tag}_{args.image}_{len(rows)}rows.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
