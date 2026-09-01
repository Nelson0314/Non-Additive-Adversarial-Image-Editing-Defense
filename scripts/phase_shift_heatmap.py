"""淨化算子把**擾動的相位**轉了多少：殘差頻譜的幅度加權 `|Δφ|`，畫成頻譜熱圖。

**不跑 GPU、不需要模型。** 只讀 `__orig.png`、`__def.png` 與淨化後的圖。

為什麼分析殘差而不是影像
────────────────────────────────────────────────────────────────────
先前的版本比的是「淨化後的防禦圖」對「原圖」的相位。那量到的是**整張影像**
的相位譜，而 `|S|` 主要是原圖內容、擾動只是疊在上面的一小份——所以「JPEG 幾乎
沒動熱圖」只證明了「JPEG 沒怎麼改原圖的相位」，**沒有**證明「JPEG 沒破壞擾動
的相位」。`runs/ip2p_residual_signature/band_transfer.csv` 量到 JPEG 確實把擾動
的方向存活率由 0.68 打散到 0.05，兩份量測不矛盾，量的是不同的東西。

本版把擾動單獨隔出來。**兩側都要扣掉各自的乾淨底**，否則扣掉的是原圖而不是
淨化對原圖做的事：

    注入的擾動   δ  = 防禦圖 - 原圖
    存活的擾動   δ' = T(防禦圖) - T(原圖)          T 是淨化算子

    R(ω), R'(ω) = δ 與 δ' 的加窗區塊頻譜（32×32、hop 8、Hann、rfft2）
    Δφ_b(ω)     = wrap( ∠R'_b(ω) - ∠R_b(ω) )      包回 (-π, π]
    相位熱圖(ω)  = Σ_b |R_b(ω)|·|Δφ_b(ω)| / Σ_b |R_b(ω)|      單位：弧度
    能量存活(ω)  = Σ_b |R'_b(ω)| / Σ_b |R_b(ω)|               無單位

權數取**注入的擾動自己的幅度**：`|R| ≈ 0` 的頻格上 `∠R` 由數值噪聲決定，
加權等於只在真的放了擾動的地方取平均。

兩個量必須並排讀，因為三個算子壞掉的方式不同（`band_transfer.csv`）：

- **模糊拿走能量**：方向存活率每一帶 0.88–1.00，但高頻能量剩 0.3% 甚至 0。
  高斯模糊在頻域乘的是實正數 `exp(-2π²σ²f²)`，**構造上不可能改相位**；
  它抽掉的是載體。所以模糊欄應該是「相位低、能量存活低」。
- **JPEG 打散方向**：能量比 1 還大（1.36–1.80，量化把係數推過整整一階），
  方向卻由 0.68 塌到 0.05。所以 JPEG 欄應該是「能量存活高、相位高」。
- **裁切把擾動搬走**：能量留 51–99%，對原格點的方向 0.000，對算子自己搬過的
  同一擾動 0.995。它是繞中心的純放大 1.2488×，區塊格點與原圖對不上，
  **該欄不可與其餘欄並列解讀**，照畫並標註。

`identity` 不畫成一欄：`T = 恆等` 時 `δ' ≡ δ`，相位熱圖恆為 0、能量存活恆為 1。
仍然算一次當守門——不成立代表讀到的檔案不是我以為的那一張，那種錯沒有症狀。

π/2 = 1.571 是「兩個獨立均勻角度之差的絕對值」的期望，即無資訊水平；
色階上限固定取它，**亮＝與噪聲無異，暗＝相位真的被保住了**。

用法：
    python scripts/phase_shift_heatmap.py --src runs/ip2p_ig_loss \
        --cond ig_f08_eot --out phase_shift_amp.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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

from src.residual.texture_rephase import PhaseResidual  # noqa: E402

BLOCK, HOP = 32, 8
OPS = ["jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OP_LABEL = {
    "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
    "blur1": "模糊 σ=1", "blur2": "模糊 σ=2",
    "crop_resize0.1": "裁切縮放 10%\n（格點對不上）",
}
# 兩個獨立均勻角度之差的絕對值的期望值，即無資訊水平。
NO_INFO = math.pi / 2


def load01(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), resample=Image.BICUBIC)
    a = np.asarray(img).copy()
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(torch.float64) / 255.0


def block_spectrum(d: torch.Tensor) -> torch.Tensor:
    """殘差 → 加窗區塊頻譜 `(1,3,L,32,17)`。分析基底與方法本身同一組。"""
    mod = PhaseResidual(size=512, block=BLOCK, hop=HOP).to(torch.float64)
    # `analyze` 用 `self.window`，而窗是在 `prepare_gates` 裡才填的——不呼叫它
    # 窗是全零，`analyze` 會**安靜地**回傳全零頻譜（不拋錯），熱圖看起來就是
    # 一片乾淨的 0。傳什麼進去只影響閘（本支不用閘），窗與區塊格點不受影響。
    mod.prepare_gates(d.abs())
    return mod.analyze(d)


def to_full_plane(half: torch.Tensor) -> np.ndarray:
    """`rfft2` 半平面 → fftshift 後的完整 32×32 平面。

    `|Δφ|` 與 `|R|` 在共軛對稱下都是偶函數（`φ(-ω) = -φ(ω)`），故鏡射是
    **精確**的不是近似。
    """
    full = torch.zeros(BLOCK, BLOCK, dtype=half.dtype)
    full[:, : BLOCK // 2 + 1] = half
    for u in range(BLOCK):
        for v in range(BLOCK // 2 + 1, BLOCK):
            full[u, v] = half[(-u) % BLOCK, BLOCK - v]
    return torch.fft.fftshift(full).numpy()


def residual_maps(delta: torch.Tensor, delta_pur: torch.Tensor) -> tuple:
    """回傳 `(相位熱圖, 能量存活圖)`，各 `(32, 32)`。

        Δφ_b(ω)    = wrap( ∠R'_b(ω) - ∠R_b(ω) )
        相位熱圖(ω) = Σ_b |R_b(ω)|·|Δφ_b(ω)| / Σ_b |R_b(ω)|      弧度
        能量存活(ω) = Σ_b |R'_b(ω)| / Σ_b |R_b(ω)|               無單位

    權數取**注入的擾動自己的幅度** `|R|`：`|R| ≈ 0` 的頻格上 `∠R` 由數值噪聲
    決定，加權等於只在真的放了擾動的地方取平均。
    """
    r, rp = block_spectrum(delta), block_spectrum(delta_pur)
    d = torch.angle(rp) - torch.angle(r)
    # 包回 (-π, π]：相位是週期量，直接相減會把 -π+ε 與 π-ε 讀成差 2π。
    d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
    w = r.abs()
    den = w.sum(dim=2).clamp_min(1e-12)
    phase = ((d.abs() * w).sum(dim=2) / den).mean(dim=1)[0]
    surv = (rp.abs().sum(dim=2) / den).mean(dim=1)[0]
    return to_full_plane(phase), to_full_plane(surv)


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


def panel(ax, m: np.ndarray, title: str):
    """色階上限固定取 π/2（無資訊水平），不取資料的最大值。

    取最大值會讓噪聲地板佔滿色階、有資訊的那一塊全部擠成一色。上限以上一律
    飽和成同一色，讀圖時「亮＝與噪聲無異」「暗＝相位真的被保住了」。
    """
    im = ax.imshow(m, cmap="magma", vmin=0.0, vmax=NO_INFO,
                   interpolation="nearest")
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.05, f"幅度加權均值 {m.mean():.3f} rad",
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
    # 空白地板是「原圖被同一個算子淨化過」，**只與算子有關、與條件無關**，
    # 所以同兩張影像同六個算子的批次之間可以互用。互用之前呼叫端要自己確認
    # 原圖逐位元相同（本專案的兩張是同一份檔案，md5 相同）。
    ap.add_argument("--floor-src", type=Path, default=None,
                    help="地板圖庫所在的批次目錄；預設與 --src 同一個")
    ap.add_argument("--out", type=Path, default=Path("phase_shift.png"))
    args = ap.parse_args()

    gal = args.src / "purify" / f"gallery_{args.cond}"
    floor = (args.floor_src or args.src) / "purify" / "gallery_floor"

    # 逐算子累出兩個量，最後在影像之間取平均——兩張圖答案一致時分開畫沒有
    # 資訊，只是把圖撐成兩倍高。
    acc = {op: {"ph": [], "sv": []} for op in OPS}
    for img in args.images:
        orig = args.src / args.cond / f"{img}__orig.png"
        defended = args.src / args.cond / f"{img}__phase_gain__def.png"
        for q in (orig, defended):
            if not q.exists():
                raise SystemExit(f"缺 {q}")
        delta = load01(defended) - load01(orig)          # 注入的擾動

        # 守門：`identity` 下 `δ' ≡ δ`，相位恆為 0、能量存活恆為 1。不成立
        # 代表讀到的檔案不是我以為的那一張，那種錯沒有症狀。
        pid_d = gal / f"{img}__phase_gain__identity__pur.png"
        pid_o = floor / f"{img}__None__identity__pur.png"
        if pid_d.exists() and pid_o.exists():
            ph, sv = residual_maps(delta, load01(pid_d) - load01(pid_o))
            if float(np.abs(ph).max()) > 1e-9 or abs(float(sv.mean()) - 1) > 1e-9:
                raise SystemExit(
                    f"identity 守門失敗（相位最大 {np.abs(ph).max():.3e}、"
                    f"能量存活均值 {sv.mean():.6f}，理應是 0 與 1）——"
                    "讀到的檔案不是防禦圖／原圖本身")

        for op in OPS:
            pd = gal / f"{img}__phase_gain__{op}__pur.png"
            po = floor / f"{img}__None__{op}__pur.png"
            for q in (pd, po):
                if not q.exists():
                    raise SystemExit(f"缺 {q}")
            # **兩側都扣掉各自的乾淨底**：`T(防禦圖) - T(原圖)`，不是
            # `T(防禦圖) - 原圖`。後者會把「淨化對原圖做的事」算進擾動裡。
            ph, sv = residual_maps(delta, load01(pd) - load01(po))
            acc[op]["ph"].append(ph)
            acc[op]["sv"].append(sv)

    phase = {op: np.mean(acc[op]["ph"], axis=0) for op in OPS}
    surv = {op: np.mean(acc[op]["sv"], axis=0) for op in OPS}
    # 加權平均要用**注入的擾動的幅度**當權數，而 `residual_maps` 已經在
    # 頻格內做過了；跨頻格再取一次平均是等權的，兩者不同，故照實標為
    # 「頻格上的等權平均」。
    ph_mean = {op: float(phase[op].mean()) for op in OPS}
    sv_mean = {op: float(surv[op].mean()) for op in OPS}

    ncol = len(OPS)
    fig = plt.figure(figsize=(2.65 * ncol, 9.0))
    # 三列：熱圖條、橫向色階、兩張長條圖。色階自己佔一列而不是掛在熱圖旁邊
    # ——掛旁邊時 matplotlib 會把它拉成整列的高度，熱圖被擠成一條。
    gs = fig.add_gridspec(3, 2 * ncol, height_ratios=[1.0, 0.05, 0.95],
                          hspace=0.34, wspace=0.9)

    im = None
    for c, op in enumerate(OPS):
        ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
        im = ax.imshow(phase[op], cmap="magma", vmin=0.0, vmax=NO_INFO,
                       interpolation="nearest")
        ax.set_title(OP_LABEL[op], fontsize=10.5, pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    cax = fig.add_subplot(gs[1, ncol - 3:ncol + 3])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("|Δφ|（弧度），色階上限固定取 π/2 = "
                   f"{NO_INFO:.3f}　　亮＝與噪聲無異，暗＝相位還在",
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    xs = np.arange(ncol)
    labs = [OP_LABEL[o].splitlines()[0] for o in OPS]

    ax = fig.add_subplot(gs[2, :ncol])
    b = ax.bar(xs, [ph_mean[o] for o in OPS], 0.62, color="#7b3294")
    ax.bar_label(b, fmt="%.3f", fontsize=9.5, padding=2)
    ax.axhline(NO_INFO, color="#00b8d4", lw=1.5, ls="--")
    ax.text(-0.42, NO_INFO * 1.02, f"π/2 = {NO_INFO:.3f}　無資訊水平",
            fontsize=9, color="#0097a7", va="bottom", ha="left")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=9.5)
    ax.set_ylim(0, NO_INFO * 1.28)
    ax.set_ylabel("擾動的相位被轉了多少　|Δφ|（弧度）", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[2, ncol:])
    b = ax2.bar(xs, [sv_mean[o] for o in OPS], 0.62, color="#008837")
    ax2.bar_label(b, fmt="%.3f", fontsize=9.5, padding=2)
    ax2.axhline(1.0, color="#888", lw=1.5, ls="--")
    ax2.text(ncol - 0.55, 1.02, "1.0 ＝ 能量原封不動", fontsize=9,
             color="#666", va="bottom", ha="right")
    ax2.set_xticks(xs); ax2.set_xticklabels(labs, fontsize=9.5)
    ax2.set_ylim(0, 1.28)
    ax2.set_ylabel("擾動還剩多少能量　能量存活率", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"淨化把「我們施加的相位偏移」轉掉了多少　條件 {args.cond}　"
        f"（{len(args.images)} 張影像平均）" + "\n"
        + f"δ = 防禦圖 - 原圖,   δ' = T(防禦圖) - T(原圖)　　"
        f"熱圖：直流在中心、外圈 Nyquist",
        fontsize=11.5)
    fig.text(0.5, 0.02,
             "裁切縮放是繞中心的放大 1.2488×，區塊格點與原圖對不上，"
             "該欄量到的是對位破壞不是相位破壞，不可與其餘欄並列。",
             fontsize=8.5, color="#666", ha="center")
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1e6:.2f} MB）")
    for op in OPS:
        print(f"  {op:<16s} |dphi| {ph_mean[op]:.4f} rad   "
              f"energy {sv_mean[op]:.4f}")


if __name__ == "__main__":
    main()
