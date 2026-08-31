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
from matplotlib import font_manager as _fm  # noqa: E402

# 標題與註解是中文。缺中文字型時 matplotlib **不會拋錯**，它印一行 warning
# 然後把每個字畫成豆腐方塊——圖產得出來、字全部看不懂。故在此逐一試裝在機器上
# 的候選，一個都沒有時明確拋錯而不是產出一張讀不了的圖。
_CJK = ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC",
        "Noto Sans TC", "PingFang TC", "Source Han Sans TW", "SimHei",
        "PMingLiU", "WenQuanYi Zen Hei")
_have = {f.name for f in _fm.fontManager.ttflist}
_pick = [n for n in _CJK if n in _have]
if not _pick:
    raise SystemExit(
        "找不到任何中文字型，圖上的中文會變成豆腐方塊。"
        f"試過：{_CJK}。請在有中文字型的機器上跑（本專案的本機 Windows 有 "
        "Microsoft JhengHei），或先安裝一個。")
plt.rcParams["font.sans-serif"] = _pick + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from PIL import Image                    # noqa: E402

from src.residual.texture_rephase import PhaseResidual, texture_gate  # noqa: E402

BLOCK, HOP = 32, 8
# `identity` 不畫成一欄：它的「淨化圖」就是防禦圖本身，會與第一欄逐位元重複。
# 仍然算一次當守門——兩者若不相等，代表 gallery 的 `__pur.png` 不是我以為的
# 那一張，而那種錯不會有任何症狀。
OPS = ["jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OP_LABEL = {
    "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
    "blur1": "模糊 σ=1", "blur2": "模糊 σ=2",
    "crop_resize0.1": "裁切縮放 10%\n（格點對不上）",
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


def radial_profile(m: np.ndarray) -> tuple:
    """把熱圖依半徑分箱取平均。回傳 `(歸一化半徑, 均值)`，1 = 軸向 Nyquist。"""
    n = m.shape[0]
    c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    idx = np.round(r).astype(int)
    n_bins = idx.max() + 1
    tot = np.bincount(idx.ravel(), weights=m.ravel(), minlength=n_bins)
    cnt = np.bincount(idx.ravel(), minlength=n_bins).clip(min=1)
    return np.arange(n_bins) / c, tot / cnt


def panel(ax, m: np.ndarray, title: str, vmax: float):
    """色階上限固定取 π/2（無資訊水平），不取資料的最大值。

    取最大值會讓整張圖被噪聲地板佔滿色階、低頻那一小塊結構全部擠成一色——
    而低頻那一塊正是唯一有資訊的地方。**上限以上一律飽和成同一色**，讀圖時
    「亮＝與噪聲地板無異」，「暗＝相位真的被保住了」。
    """
    im = ax.imshow(m, cmap="magma", vmin=0.0, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.05, f"閘加權均值 {m.mean():.3f} rad",
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

    vmax = NO_INFO
    ncol = len(rows[0][1])
    nrow = len(rows)
    fig, axes = plt.subplots(nrow + 1, ncol,
                             figsize=(2.15 * ncol, 2.7 * nrow + 3.6),
                             gridspec_kw={"height_ratios": [1] * nrow + [0.9]})
    axes = np.atleast_2d(axes)
    im = None
    for r, (img, maps) in enumerate(rows):
        for c, (title, m) in enumerate(maps):
            im = panel(axes[r][c], m, title if r == 0 else "", vmax)
        axes[r][0].set_ylabel(img.replace("task_attr_mod_color_", "影像 "),
                              fontsize=8.5)

    # 最後一列：徑向剖面。熱圖上唯一有資訊的是低頻那一小塊，剖面把它拉直了看。
    for c in range(ncol):
        ax = axes[nrow][c]
        for r, (img, maps) in enumerate(rows):
            rr, pp = radial_profile(maps[c][1])
            ax.plot(rr, pp, lw=1.4,
                    label=img.replace("task_attr_mod_color_", "影像 "))
        ax.axhline(NO_INFO, color="#00b8d4", lw=1.0, ls="--")
        ax.set_ylim(0, 1.75); ax.set_xlim(0, 1.0)
        ax.set_xlabel("歸一化半徑（1 = 軸向 Nyquist）", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("徑向平均 |Δφ|", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"逐頻格的紋理閘加權平均 |Δφ|（弧度）　條件 {args.cond}\n"
        f"直流在中心、外圈為 Nyquist；色階上限 {vmax:.2f}　"
        f"π/2 = {NO_INFO:.3f} 是「該頻格本來就沒有訊號」的水平",
        fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("|Δφ|（弧度），上限 π/2", fontsize=8)
    

    fig.text(0.01, 0.012,
             "相位在幅度接近零處沒有意義：自然影像頻譜約 1/f，高頻的 |S| 很小，"
             "∠S 由數值噪聲決定，|Δφ| 趨向 π/2 = 1.571。外圈接近 π/2 讀成"
             "「本來就沒有訊號」，不是「相位被改了很多」。\n"
             "裁切縮放是繞中心的純放大 1.2488×，區塊格點與原圖對不上，"
             "該欄不可與其餘欄並列解讀。高斯模糊是零相位濾波器，原理上不改相位。",
             fontsize=7.5, color="#444", va="bottom")
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
