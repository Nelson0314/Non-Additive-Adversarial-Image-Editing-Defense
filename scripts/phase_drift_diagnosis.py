"""淨化之後，最佳化裝上去的相位偏移還在不在。**不跑 GPU。**

要回答什麼
────────────────────────────────────────────────────────────────────
本方法動的是相位：在 32×32、hop 8 的加窗 STFT 上，把每一個（區塊, 頻格）的
係數乘 `exp(i·g_b·m_ω·θ_b)`。JPEG 那一欄有防禦，blur 與 crop 沒有。要問的是
**那個裝上去的相位偏移，在算子過完之後跑掉了沒有**，以及若跑掉，是怎麼跑的。

`runs/ip2p_residual_signature/band_transfer.csv` 問過相鄰的問題，但量的是
**隨機帶限探針**（RMS 0.01）的存活率，不是最佳化出來的殘差，也不是相位本身
——它量的是殘差向量的餘弦。本檔量的是相位角。

三個量（全部逐頻帶、以 |S| 為權重）
────────────────────────────────────────────────────────────────────
設 `S = STFT(x)`、`S' = STFT(x_def)`，裝上去的偏移

    Δφ_inst(b,ω) = angle(S'(b,ω)) − angle(S(b,ω))

算子 `p` 過完之後，用**同一個算子也過一遍的原圖**當參照（不是原圖本身，
否則量到的是算子自己造成的相位變化）：

    Δφ_surv(b,ω) = angle(STFT(p(x_def))) − angle(STFT(p(x)))

1. **相位保留 `rho`**：`|Σ w·exp(i(Δφ_surv − Δφ_inst))| / Σ w`。
   1 表示偏移原封不動地活著，0 表示與裝上去的那一個毫無關係。這是圓統計的
   平均合成長度，**不會被角度的週期性騙到**（直接相減再平均會）。
2. **偏移的大小 `inst_mag` 與 `surv_mag`**：`Σ w·|wrap(Δφ)| / Σ w`。
   兩者一起讀才分得開兩種死法——**被抹平**（`surv_mag → 0`）與**被打散**
   （`surv_mag` 不變但 `rho → 0`）。
3. **殘差的能量與方向存活率**：沿用 `band_transfer` 的定義，讓兩份量測可以
   對照。crop 另外報 `cos_vs_warped`：把殘差**單獨**送進同一個幾何變換再比，
   用來分辨「擾動被消滅」與「擾動只是被搬走」。

權重 `w = |S(b,ω)|`：相位在幅度接近零的地方沒有意義（角度由捨入雜訊決定），
不加權會讓平坦區的雜訊主導整個統計。

crop 的座標問題（**不可略過**）
────────────────────────────────────────────────────────────────────
`crop_resize(0.1)` 每邊各裁 10% 再放大回 512，幾何上是以中心為不動點的
1.2488 倍放大。於是 `p(x)` 的 STFT 區塊格點與 `x` 的**不是同一組區塊**，
`Δφ_inst(b,ω)` 與 `Δφ_surv(b,ω)` 的 `(b,ω)` 索引指的是不同的東西。逐格相比
在 crop 上量到的是「對不對得上」，不是「相位有沒有變」——這正是要看的現象，
但結論必須這樣寫，不可以寫成「相位被破壞」。`cos_vs_warped` 就是為了把這兩
件事分開而設。

用法：
    python scripts/phase_drift_diagnosis.py \
        --defended <取回的防禦圖目錄> \
        --out runs/phase_drift_diagnosis
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.purify import ops as purify_ops  # noqa: E402
from src.residual.texture_rephase import PhaseResidual  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
BLOCK, HOP = 32, 8          # 主線定案值（docs/METHOD.md）
BANDS = ((0.00, 0.12), (0.12, 0.25), (0.25, 0.40), (0.40, 0.55),
         (0.55, 0.70), (0.70, 0.85), (0.85, 1.05))

# 條件 → 防禦圖檔名裡的 condition 欄。ip2p_run.py 存檔用 `<img>__<cond>__def.png`。
CONDITIONS: Dict[str, str] = {
    "ours_ph_q": "phase",        # 純相位 r0.9 ＋ 量化交付 0.85（七軸更低的那一點）
    "ours_ph_n": "phase",        # 同半徑但不量化，用來拆開量化交付的貢獻
    "ours_pg_q20": "phase_gain",  # 最強點
    "dct_aj85": "dct_shield_y",  # 對手的論文設定
}

PURIFIERS: Sequence[Tuple[str, purify_ops.Purifier]] = (
    ("identity", purify_ops.Purifier("identity")),
    ("jpeg75", purify_ops.Purifier("jpeg", 75)),
    ("jpeg50", purify_ops.Purifier("jpeg", 50)),
    ("jpeg30", purify_ops.Purifier("jpeg", 30)),
    ("blur1", purify_ops.Purifier("blur", 1.0)),
    ("blur2", purify_ops.Purifier("blur", 2.0)),
    ("crop_resize0.1", purify_ops.Purifier("crop_resize", 0.10)),
    ("crop_resize0.15", purify_ops.Purifier("crop_resize", 0.15)),
)

# 只有幾何的算子才有「搬走」這個可能，其餘的 warped 控制組沒有意義。
GEOMETRIC = {"crop_resize0.1": 0.10, "crop_resize0.15": 0.15}


def warp_residual(d: torch.Tensor, fraction: float) -> torch.Tensor:
    """把**殘差**送進 `crop_resize` 的幾何那一半，`[0,1]` 的夾取要拿掉。

    `purify.ops.crop_resize` 最後一行是 `.clamp(0, 1)`，那是給影像用的值域
    維護。殘差是有號的小量，直接套進去會把**每一個負值都推成 0**，控制組
    於是只剩一半的訊號，餘弦被架高到 0.7 上下而看起來像「搬走了七成」。
    這裡逐行照抄 `crop_resize` 的切片與 `F.interpolate`（同樣的 bicubic ＋
    antialias），只省掉最後的夾取。
    """
    import torch.nn.functional as F

    h, w = d.shape[-2:]
    dh, dw = int(round(h * fraction)), int(round(w * fraction))
    cropped = d[..., dh:h - dh, dw:w - dw]
    return F.interpolate(cropped, size=(h, w),
                         mode=purify_ops.CROP_INTERPOLATION,
                         antialias=purify_ops.CROP_ANTIALIAS)


def wrap(a: torch.Tensor) -> torch.Tensor:
    """把角度折回 (−π, π]。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def radial_index(block: int, device, dtype) -> torch.Tensor:
    """(block, block//2+1) 的歸一化半徑，1 即 Nyquist。與 `radial_gate` 同座標。"""
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def band_stats(analyzer: PhaseResidual,
               x: torch.Tensor, x_def: torch.Tensor,
               px: torch.Tensor, px_def: torch.Tensor,
               radial: torch.Tensor) -> List[Dict[str, float]]:
    """逐頻帶的三個量。輸入四張圖皆 (1,3,H,W)。"""
    # `analyze` 用的 Hann 窗是在 `prepare_gates` 裡才填進 buffer 的，未呼叫時
    # 它是全零，頻譜也就全零——那會讓底下每一帶的權重和變成 0、統計整批消失。
    # 呼叫端負責先 prepare；這一道守門是為了不讓它變成靜默的空表。
    if float(analyzer.window.abs().sum()) <= 0:
        raise RuntimeError("analyzer.window 全零：呼叫 band_stats 前必須先 "
                           "prepare_gates(x)，否則 analyze 回傳的頻譜恆為零")
    s = analyzer.analyze(x)
    s_def = analyzer.analyze(x_def)
    s_p = analyzer.analyze(px)
    s_p_def = analyzer.analyze(px_def)

    d_inst = wrap(torch.angle(s_def) - torch.angle(s))
    d_surv = wrap(torch.angle(s_p_def) - torch.angle(s_p))
    # 權重取原圖的幅度：相位在 |S| ≈ 0 處由捨入雜訊決定。
    w = s.abs()

    out = []
    for lo, hi in BANDS:
        m = ((radial >= lo) & (radial < hi)).to(w.dtype)
        ww = w * m
        tot = ww.sum()
        if float(tot) <= 0:
            continue
        # 平均合成向量拆成實部／虛部算，不建複數張量：長度是保留度，
        # 幅角是「整體被系統性地多轉了多少」。
        diff = d_surv - d_inst
        re = (ww * torch.cos(diff)).sum()
        im = (ww * torch.sin(diff)).sum()
        rho = float(torch.sqrt(re ** 2 + im ** 2) / tot)
        bias = float(torch.atan2(im, re))
        out.append({
            "band_lo": lo, "band_hi": hi,
            "rho": round(rho, 5),
            "systematic_bias": round(bias, 5),
            "inst_mag": round(float((ww * d_inst.abs()).sum() / tot), 5),
            "surv_mag": round(float((ww * d_surv.abs()).sum() / tot), 5),
        })
    return out


def residual_stats(d: torch.Tensor, d_surv: torch.Tensor,
                   d_warped: torch.Tensor | None) -> Dict[str, float]:
    """殘差的能量與方向存活率，定義與 `purifier_band_transfer.py` 一致。"""
    e0 = float(d.pow(2).sum())
    e1 = float(d_surv.pow(2).sum())

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        na, nb = a.norm(), b.norm()
        if float(na) <= 0 or float(nb) <= 0:
            return float("nan")
        return float((a * b).sum() / (na * nb))

    row = {"energy_ratio": round(e1 / e0, 5) if e0 > 0 else float("nan"),
           "cosine": round(cos(d, d_surv), 5)}
    row["cos_vs_warped"] = (round(cos(d_surv, d_warped), 5)
                            if d_warped is not None else "")
    return row


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defended", type=Path, required=True,
                    help="含 <tag>/<img>__<cond>__def.png 的目錄")
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, default=Path("runs/phase_drift_diagnosis"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    device = torch.device(args.device)
    analyzer = PhaseResidual(size=RESOLUTION, block=BLOCK, hop=HOP).to(device)
    radial = radial_index(BLOCK, device, torch.float32)

    phase_rows: List[Dict] = []
    resid_rows: List[Dict] = []

    for tag, cond in CONDITIONS.items():
        for name in names:
            dpath = args.defended / tag / f"{name}__{cond}__def.png"
            if not dpath.exists():
                raise FileNotFoundError(
                    f"缺防禦圖 {dpath}。影像不入版控，要先由遠端取回；"
                    f"路徑見 runs/ip2p_mainline/README.md 第八節")
            x = load_image_tensor(args.data / name / f"{name}.png", device,
                                  size=RESOLUTION)
            x_def = load_image_tensor(dpath, device, size=RESOLUTION)
            d = x_def - x
            # 閘由**原圖**算（與前向同一條路），同時把 Hann 窗填進 buffer。
            analyzer.prepare_gates(x)

            for pname, pur in PURIFIERS:
                px = pur.evaluate(x)
                px_def = pur.evaluate(x_def)
                for row in band_stats(analyzer, x, x_def, px, px_def, radial):
                    phase_rows.append({"image": name, "condition": tag,
                                       "purifier": pname, **row})

                d_warped = None
                if pname in GEOMETRIC:
                    # 只把殘差送進同一個幾何變換：分辨「消滅」與「搬走」。
                    d_warped = warp_residual(d, GEOMETRIC[pname])
                resid_rows.append({"image": name, "condition": tag,
                                   "purifier": pname,
                                   **residual_stats(d, px_def - px, d_warped)})
        print(f"[{tag}] 完成 {len(names)} 張", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "phase_retention_by_band.csv", phase_rows)
    write_csv(args.out / "residual_survival.csv", resid_rows)

    # ---- 摘要：全帶（以 |S| 加權）與殘差 ----
    summary = []
    for tag in CONDITIONS:
        for pname, _ in PURIFIERS:
            ph = [r for r in phase_rows
                  if r["condition"] == tag and r["purifier"] == pname]
            rs = [r for r in resid_rows
                  if r["condition"] == tag and r["purifier"] == pname]
            summary.append({
                "condition": tag, "purifier": pname,
                "rho_mean": round(statistics.mean(r["rho"] for r in ph), 5),
                "inst_mag": round(statistics.mean(r["inst_mag"] for r in ph), 5),
                "surv_mag": round(statistics.mean(r["surv_mag"] for r in ph), 5),
                "energy_ratio": round(statistics.mean(r["energy_ratio"] for r in rs), 5),
                "cosine": round(statistics.mean(r["cosine"] for r in rs), 5),
                "cos_vs_warped": (round(statistics.mean(
                    r["cos_vs_warped"] for r in rs), 5)
                    if rs and rs[0]["cos_vs_warped"] != "" else ""),
            })
    write_csv(args.out / "summary.csv", summary)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
