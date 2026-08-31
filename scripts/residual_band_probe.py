"""殘差的能量落在哪些頻帶，以及裁切把它搬到哪裡。**純 CPU，不佔卡。**

兩個問題，共用一份徑向能譜
────────────────────────────────────────────────────────────────────

**問題一：模糊 σ2 之後活下來的那 13% 是什麼？**

高斯模糊的頻率響應是 `H_σ(ω) = exp(−σ²ω²/2)`，以 r = 1 對應 Nyquist
（ω = π）代入：`H_{σ=2}(0.95π) ≈ 1.8×10⁻⁸`。本方法的旋鈕帶是
`r ∈ [0.884, 1.061]`，所以**帶內的東西過了 σ2 應該是零**。存活門檻換算：

    H_{σ=2}(r) ≥ 0.5  →  r ≤ 0.187
    H_{σ=2}(r) ≥ 0.1  →  r ≤ 0.342
    H_{σ=2}(r) ≥ 0.01 →  r ≤ 0.485

但實測整張圖的殘差能量 σ2 之後還剩 0.130。**那 0.130 只可能來自旋鈕帶之外。**
候選：8-bit 量化的捨入、重疊相加的重建誤差、逐窗變化的增益產生的低頻包絡。

若成立，含意是：模糊 σ2 那一欄現在拿到的淨增益，是被一個**從未被最佳化過的
副產物**守住的——那個自由度是 0 訓練預算，不是「試過而失敗」。

**問題二：裁切是把擾動搬到座標上，還是搬到頻帶上？**

`crop_resize(0.1)` 是 1.2488× 放大，頻率乘 `1/1.2488 = 0.8009`，把旋鈕帶搬到
`[0.708, 0.850]`。既有的歸因是「對原格點餘弦 0.005 ⇒ 座標對不上」，但**那個
0.005 同時包含座標搬移與頻帶搬移兩件事**。

本檔把殘差**單獨**送進同一個幾何變換，量變換前後的徑向能譜；再由
`runs/encoder_frequency_response/` 既有的逐帶 `latent_move` 讀出兩個帶的敏感度
比。若頻帶錯位就足以解釋落差，病因是尺度而非對位——**而尺度可以用參數化
處理，對位不行。**

**殘差不可夾取值域。** `purify.ops.crop_resize` 最後一行 `.clamp(0,1)` 是給
影像用的；殘差是有號的小量，夾取會把每一個負值推成 0（踩過一次，`cos_vs_warped`
被架高成 0.71）。這裡用 `warp_residual` 那條不夾取的路徑。

用法：
    python scripts/residual_band_probe.py \\
        --defense runs/ip2p_ig_converge/ig_d25 --condition phase_gain \\
        --out runs/residual_band/ig_d25.csv
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLUTION = 512
# 與 `runs/encoder_frequency_response/` 同一組邊界，好逐帶對照（角落到 sqrt2）。
EDGES = [0.0, 0.1768, 0.3536, 0.5303, 0.7071, 0.8839, 1.0607, 1.2374, 1.4142]
SIGMAS = (1.0, 2.0)
CROP_FRACS = (0.10, 0.15)


def radial_energy(d, edges):
    """全域 rfft2 的徑向能量分佈。回傳每一帶的能量佔比。

    用全域 FFT 而不是逐區塊 STFT：這裡問的是「能量落在哪個頻率」，與區塊之間
    的相對位置無關；逐區塊會把兩者混在一起。
    """
    import torch
    h, w = d.shape[-2:]
    F = torch.fft.rfft2(d.mean(dim=1))            # 通道平均，(B, h, w//2+1)
    fy = torch.fft.fftfreq(h, d=1.0, device=d.device).view(-1, 1) * 2
    fx = torch.fft.rfftfreq(w, d=1.0, device=d.device).view(1, -1) * 2
    r = torch.sqrt(fy ** 2 + fx ** 2)
    e = (F.abs() ** 2)[0]
    total = max(float(e.sum()), 1e-20)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        out.append(float(e[m].sum()) / total)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defense", type=Path, required=True)
    ap.add_argument("--condition", default="phase_gain")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sensitivity", type=Path,
                    default=Path("runs/encoder_frequency_response/latent_norm.csv"),
                    help="既有的逐帶 VAE 敏感度，用來讀 S(0.78)/S(0.97)")
    args = ap.parse_args()

    import csv
    import torch
    from src.purify.ops import gaussian_blur
    from src.utils.io import load_image_tensor, write_csv

    dev = torch.device("cpu")
    paths = sorted(args.defense.glob(f"*__{args.condition}__def.png"))
    if not paths:
        raise SystemExit(f"{args.defense} 裡沒有 __{args.condition}__def.png")

    rows = []
    for p in paths:
        name = p.name[: -len(f"__{args.condition}__def.png")]
        x = load_image_tensor(args.data / name / f"{name}.png", dev, size=RESOLUTION)
        xd = load_image_tensor(p, dev, size=RESOLUTION)
        d = xd - x
        base = {"image": name}

        frac = radial_energy(d, EDGES)
        for (lo, hi), f in zip(zip(EDGES[:-1], EDGES[1:]), frac):
            rows.append({**base, "kind": "residual", "param": "",
                         "r_lo": lo, "r_hi": hi, "energy_frac": round(f, 6)})

        # 模糊：整張殘差的能量存活，以及存活的能量落在哪一帶。
        for sg in SIGMAS:
            ds = gaussian_blur(xd, sg) - gaussian_blur(x, sg)
            keep = float(ds.pow(2).sum() / d.pow(2).sum().clamp_min(1e-20))
            fr = radial_energy(ds, EDGES)
            for (lo, hi), f in zip(zip(EDGES[:-1], EDGES[1:]), fr):
                rows.append({**base, "kind": "blur_survived", "param": sg,
                             "r_lo": lo, "r_hi": hi, "energy_frac": round(f, 6),
                             "energy_keep": round(keep, 6)})

        # 裁切：**殘差單獨**送進同一個幾何變換，不夾取值域。
        for cf in CROP_FRACS:
            dw = warp_residual_nc(d, cf)
            fr = radial_energy(dw, EDGES)
            for (lo, hi), f in zip(zip(EDGES[:-1], EDGES[1:]), fr):
                rows.append({**base, "kind": "crop_warped", "param": cf,
                             "r_lo": lo, "r_hi": hi, "energy_frac": round(f, 6)})
        print(f"  {name} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    # ---- 摘要 ----
    def mean_frac(kind, param=""):
        out = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            v = [float(r["energy_frac"]) for r in rows
                 if r["kind"] == kind and str(r["param"]) == str(param)
                 and float(r["r_lo"]) == lo]
            out.append(st.fmean(v) if v else float("nan"))
        return out

    print(f"\n{args.out}：{len(rows)} 列，{len(paths)} 張\n")
    hdr = "".join(f"{lo:.2f}–{hi:.2f}".rjust(12)
                  for lo, hi in zip(EDGES[:-1], EDGES[1:]))
    print(f"{'':16}{hdr}")
    print(f"{'殘差':16}" + "".join(f"{v:12.4f}" for v in mean_frac("residual")))
    for sg in SIGMAS:
        print(f"{'模糊σ%s 後' % sg:16}"
              + "".join(f"{v:12.4f}" for v in mean_frac("blur_survived", sg)))
    for cf in CROP_FRACS:
        print(f"{'裁切%s 搬過' % cf:16}"
              + "".join(f"{v:12.4f}" for v in mean_frac("crop_warped", cf)))

    keep = {sg: st.fmean(float(r["energy_keep"]) for r in rows
                         if r["kind"] == "blur_survived" and str(r["param"]) == str(sg)
                         and float(r["r_lo"]) == 0.0) for sg in SIGMAS}
    print("\n模糊之後整張殘差的能量存活率：" +
          "  ".join(f"σ{sg} {v:.4f}" for sg, v in keep.items()))
    print("H_{σ=2}(r) ≥ 0.01 需要 r ≤ 0.485，即前三帶。"
          "存活的能量若集中在那裡，就是旋鈕帶之外的副產物。")

    # ---- 既有敏感度曲線上的兩個帶 ----
    if args.sensitivity.exists():
        sens = {}
        for r in csv.DictReader(open(args.sensitivity, encoding="utf-8")):
            sens.setdefault(float(r["r_lo"]), []).append(float(r["latent_move"]))
        s = {k: st.fmean(v) for k, v in sens.items()}
        b_work = 0.8839      # 旋鈕帶所在
        b_zoom = 0.7071      # 1.2488× 之後落到的帶
        if b_work in s and b_zoom in s:
            print(f"\n既有 VAE 敏感度（latent_move）："
                  f"r∈[0.884,1.061] = {s[b_work]:.4f}，"
                  f"r∈[0.707,0.884] = {s[b_zoom]:.4f}，"
                  f"比值 {s[b_zoom] / s[b_work]:.3f}")
            print("**這個比值就是「頻帶錯位」單獨能解釋的落差。**"
                  "與裁切欄的實測落差相比，剩下的才是座標對位造成的。")


def warp_residual_nc(d, frac):
    """殘差過 crop_resize 的幾何，**不夾取值域**。

    `purify.ops.crop_resize` 最後一行 `.clamp(0,1)` 是給影像用的值域維護。殘差
    是有號的小量，直接套會把每一個負值推成 0——踩過一次，`cos_vs_warped` 因此
    被架高成 0.71–0.73，看起來像「只搬走了七成」，真值是 0.9996–0.9997。
    """
    import torch
    from src.purify.ops import CROP_ANTIALIAS, CROP_INTERPOLATION
    h, w = d.shape[-2:]
    dh, dw = int(round(h * frac)), int(round(w * frac))
    c = d[..., dh:h - dh, dw:w - dw]
    return torch.nn.functional.interpolate(
        c, size=(h, w), mode=CROP_INTERPOLATION, antialias=CROP_ANTIALIAS)


if __name__ == "__main__":
    main()
