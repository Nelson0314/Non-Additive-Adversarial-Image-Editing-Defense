"""淨化算子把相位改動了多少：逐頻格的紋理閘加權平均 `|Δφ|`，畫成頻譜形狀的熱圖。

**不跑 GPU、不需要模型。** 只讀 `__orig.png`、`__def.png` 與淨化後的圖。

量的是什麼
────────────────────────────────────────────────────────────────────
分析基底與方法本身同一組：512² 切 32×32 加窗區塊、hop 8、Hann 窗、`rfft2`
（`PhaseResidual.analyze`）。對第 `b` 個區塊、第 `ω` 個頻格：

    Δφ_b(ω) = wrap( ∠S'_b(ω) − ∠S_b(ω) )        包回 (−π, π]
    熱圖(ω) = Σ_b g_b·|Δφ_b(ω)| / Σ_b g_b        單位：弧度

`g_b` 是**紋理閘係數**（`texture_gate`，逐區塊一個純量，由原圖算出、不參與
最佳化）。三個通道各自算完再平均。

顯示成完整的 32×32 平面而不是 `rfft2` 的半平面：`|Δφ|` 在共軛對稱下是偶函數
（`φ(−ω) = −φ(ω)` ⇒ `Δφ(−ω) = −Δφ(ω)` ⇒ `|Δφ|` 相同），所以鏡射是**精確**的
不是近似。再 `fftshift` 把直流放到中心，半徑即頻率、外圈是 Nyquist。

三件讀圖之前必須知道的事
────────────────────────────────────────────────────────────────────
1. **相位在幅度接近零的地方沒有意義。** 自然影像頻譜約 1/f，高頻的 `|S|` 很小，
   那裡的 `∠S` 主要由數值噪聲決定，`|Δφ|` 會趨向「兩個獨立均勻角度之差的絕對
   值」的期望 **π/2 ≈ 1.571**。所以外圈接近 1.571 讀成「本來就沒有訊號」，
   不是「相位被改了很多」。圖上以虛線標出這條無資訊水平線。
   權數用紋理閘是使用者指定的（逐區塊），它不解決逐頻格的這個問題。
2. **`crop_resize` 之後區塊格點與原圖對不上。** 該算子是繞中心的純放大
   1.2488×，同一個區塊索引在兩張圖上不是同一塊內容，逐區塊相減沒有意義。
   該欄仍照畫並標註，不可與其餘欄並列解讀。
3. 高斯模糊是**零相位濾波器**（頻域乘實正數），原理上不改相位；它拿走的是
   幅度。若模糊欄的圖接近 0（低頻）而外圈接近 π/2，那正是「低頻相位活著、
   高頻本來就沒有訊號」的樣子，不是模糊改了相位。

用法：
    python scripts/phase_shift_heatmap.py --src runs/ip2p_ig_loss \\
        --cond ig_f08_eot --out phase_shift_ig_f08_eot.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from PIL import Image                    # noqa: E402

from src.residual.texture_rephase import PhaseResidual, texture_gate  # noqa: E402

BLOCK, HOP = 32, 8
OPS = ["identity", "jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OP_LABEL = {
    "identity": "未淨化", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
    "blur1": "模糊 σ=1", "blur2": "模糊 σ=2",
    "crop_resize0.1": "裁切縮放 10%（格點對不上）",
}
# 兩個獨立均勻角度之差的絕對值的期望值。外圈趨近它 = 該頻格本來就沒有訊號。
NO_INFO = math.pi / 2


def load01(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), resample=Image.BICUBIC)
    a = np.asarray(img).copy()
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(torch.float64) / 255.0


def gate_weighted_phase_shift(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """回傳 `(BLOCK, BLOCK)` 的 fftshift 後熱圖，單位是弧度。"""
    mod = PhaseResidual(size=512, block=BLOCK, hop=HOP).to(torch.float64)
    # `analyze` 用 `self.window`，而窗是在 `prepare_gates` 裡才填的——不呼叫它
    # 窗是全零，`analyze` 會**安靜地**回傳全零頻譜（不拋錯），熱圖看起來就是
    # 一片乾淨的 0。必須先呼叫，且要用**原圖**（閘一律由原圖算出）。
    mod.prepare_gates(x)
    sx, sy = mod.analyze(x), mod.analyze(y)               # (1,3,L,32,17) 複數
    d = torch.angle(sy) - torch.angle(sx)
    # 包回 (−π, π]：相位是週期量，直接相減會把 −π+ε 與 π−ε 讀成差 2π。
    d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
    g = texture_gate(x, BLOCK, HOP, energy_quantile=0.0, edge_power=1.0)  # (1,L)
    w = g[0].to(d.dtype)
    num = (d.abs() * w[None, None, :, None, None]).sum(dim=2)   # (1,3,32,17)
    half = (num / w.sum().clamp_min(1e-12)).mean(dim=1)[0]      # (32,17)

    # 半平面補成整平面。`|Δφ|` 在共軛對稱下相同，故鏡射是精確的。
    full = torch.zeros(BLOCK, BLOCK, dtype=half.dtype)
    full[:, : BLOCK // 2 + 1] = half
    for u in range(BLOCK):
        for v in range(BLOCK // 2 + 1, BLOCK):
            full[u, v] = half[(-u) % BLOCK, BLOCK - v]
    return torch.fft.fftshift(full).numpy()


def panel(ax, m: np.ndarray, title: str, vmax: float, note: str = ""):
    im = ax.imshow(m, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.06, f"閘加權均值 {m.mean():.3f} rad{note}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            color="#444")
    return im


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("runs/ip2p_ig_loss"))
    ap.add_argument("--cond", default="ig_f08_eot")
    ap.add_argument("--images", nargs="+",
                    default=["task_attr_mod_color_11699",
                             "task_attr_mod_color_6205"])
    ap.add_argument("--out", type=Path, default=Path("phase_shift.png"))
    args = ap.parse_args()

    gal = args.src / "purify" / f"gallery_{args.cond}"
    rows = []
    for img in args.images:
        orig = args.src / args.cond / f"{img}__orig.png"
        defended = args.src / args.cond / f"{img}__phase_gain__def.png"
        if not (orig.exists() and defended.exists()):
            raise SystemExit(f"缺 {orig} 或 {defended}")
        x = load01(orig)
        maps = [("防禦圖 對 原圖\n（注入了多少相位改動）",
                 gate_weighted_phase_shift(x, load01(defended)))]
        for op in OPS:
            p = gal / f"{img}__phase_gain__{op}__pur.png"
            if not p.exists():
                raise SystemExit(f"缺 {p}")
            maps.append((f"淨化({OP_LABEL[op]}) 對 原圖",
                         gate_weighted_phase_shift(x, load01(p))))
        rows.append((img, maps))

    vmax = max(m.max() for _, ms in rows for _, m in ms)
    ncol = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(2.05 * ncol, 2.55 * len(rows) + 1.4))
    axes = np.atleast_2d(axes)
    im = None
    for r, (img, maps) in enumerate(rows):
        for c, (title, m) in enumerate(maps):
            note = "" if c == 0 else ""
            im = panel(axes[r][c], m, title if r == 0 else "", vmax, note)
        axes[r][0].set_ylabel(img.replace("task_attr_mod_color_", "影像 "),
                              fontsize=8)

    fig.suptitle(
        f"逐頻格的紋理閘加權平均 |Δφ|（弧度）　條件 {args.cond}\n"
        f"直流在中心、外圈為 Nyquist；色階上限 {vmax:.2f}　"
        f"π/2 = {NO_INFO:.3f} 是「該頻格本來就沒有訊號」的水平",
        fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("|Δφ|（弧度）", fontsize=8)
    cbar.ax.axhline(NO_INFO, color="#00e5ff", lw=1.2, ls="--")
    cbar.ax.text(1.6, NO_INFO, " π/2", va="center", fontsize=7,
                 color="#00b8d4", transform=cbar.ax.get_yaxis_transform())

    fig.text(0.01, 0.012,
             "相位在幅度接近零處沒有意義：自然影像頻譜約 1/f，高頻的 |S| 很小，"
             "∠S 由數值噪聲決定，|Δφ| 趨向 π/2≈1.571。外圈接近 π/2 讀成"
             "「本來就沒有訊號」，不是「相位被改了很多」。\n"
             "裁切縮放是繞中心的純放大 1.2488×，區塊格點與原圖對不上，"
             "該欄不可與其餘欄並列解讀。高斯模糊是零相位濾波器，原理上不改相位。",
             fontsize=7.5, color="#444", va="bottom")
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
