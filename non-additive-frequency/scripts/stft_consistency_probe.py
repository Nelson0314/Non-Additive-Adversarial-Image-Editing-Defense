"""要求的頻譜與重疊相加**給得出來的**頻譜差多少。**純 CPU，不佔卡。**

要回答什麼
────────────────────────────────────────────────────────────────────
防禦圖上那層浮雕般的刻紋不是雜訊，是規則的干涉圖案。它有一個具體的來源：
block 32、hop 8 表示每個像素是 **16 個重疊視窗**的加總，而 `theta` 在每個
（視窗, 頻格）上是**自由**的——相鄰視窗可以帶著互相矛盾的相位。一組互相
矛盾的短時傅立葉係數**一般不對應任何真實影像**，重疊相加只能給出它的投影。
刻紋就是那個投影誤差。

這是 FND-049 記過的事（重疊相加是有損投影），但**從未在這些解上量過**。

量的是什麼
────────────────────────────────────────────────────────────────────
    dev = ‖ |analyze(synthesize(rot))| − |rot| ‖ / ‖ |rot| ‖

`rot` 是模組實際要求的頻譜（相位轉過、增益乘過、下限加過之後）。分母用它
自己，所以這是**相對**偏差，跨影像與跨工作點可比。

**不能直接用模組的 `amplitude_deviation`。** 那一支拿 `analyze(x01)` 當
基準，而本專案的批次一律 `--gain-ratio 1.0`，幅度是被**刻意**改動的——那個
量在這裡是「改了多少」，不是「差多少」，兩者意思相反。此處改用 `|rot|` 當
基準，與增益開不開無關。

怎麼取得 `rot`
────────────────────────────────────────────────────────────────────
`PhaseResidual.requested_spectrum` 就是投影**之前**的那個張量，而
`_rephase` 走的是同一支——**只有一份實作，不可能分岔**。
`tests/test_stft_consistency_probe.py` 釘住 `synthesize(requested_spectrum(x))`
與 `_rephase(x)` 逐位相同。

參數由 `--weights` 指的 `*__w.pt` 載入（`ip2p_run.py --save-weights` 存的），
構造旗標必須與當初那一批相同，否則張量形狀不合會拋錯而不是靜默跑錯。

用法：
    python scripts/stft_consistency_probe.py \\
        --weights runs/ip2p_ig_converge/ig_d25 --condition phase_gain \\
        --gain-ratio 1.0 --quantile 0 --freq-weight jpeg_luma \\
        --freq-weight-power 0.25 --hop 8 --out runs/stft_consistency/ig_d25.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLUTION = 512


def consistency_deviation(module, x01: torch.Tensor) -> dict:
    """回傳相對偏差，以及讓它可以被解讀的三個並列量。"""
    with torch.no_grad():
        rot = module.requested_spectrum(x01)
        x_def = module.synthesize(rot)
        got = module.analyze(x_def)
        num = (got.abs() - rot.abs()).norm()
        den = rot.abs().norm().clamp_min(1e-12)
        # 相位那一半分開報：幅度對得上不代表相位對得上，而我方動的正是相位。
        dphi = torch.angle(got) - torch.angle(rot)
        w = rot.abs()
        rho = ((w * torch.exp(1j * dphi)).sum().abs() / w.sum().clamp_min(1e-12))
        return {
            "amp_dev": round(float(num / den), 6),
            "phase_rho": round(float(rho), 6),
            "resid_rms": round(float((x_def - x01).pow(2).mean().sqrt()), 6),
            "resid_linf": round(float((x_def - x01).abs().max()), 6),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, nargs="+", required=True,
                    help="含 <影像>__<cond>__w.pt 的目錄，可給多個")
    ap.add_argument("--condition", default="phase_gain")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    # 構造旗標：必須與存權重的那一批相同，形狀不合會拋錯。
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--hop", type=int, default=8)
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--r-max", type=float, default=float("inf"))
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--gain-ratio", type=float, default=0.0)
    ap.add_argument("--freq-weight", default="binary")
    ap.add_argument("--freq-weight-power", type=float, default=1.0)
    ap.add_argument("--pixel-gate-sigma", type=float, default=0.0)
    ap.add_argument("--theta-budget", type=float, default=0.0)
    ap.add_argument("--radius", type=float, required=True)
    args = ap.parse_args()

    from phase_ablation import build
    from src.utils.io import load_image_tensor, write_csv

    device = torch.device("cpu")
    rows = []
    for root in args.weights:
        tag = root.name
        for wp in sorted(root.glob(f"*__{args.condition}__w.pt")):
            name = wp.name[: -len(f"__{args.condition}__w.pt")]
            img = args.data / name / f"{name}.png"
            if not img.exists():
                raise SystemExit(f"缺原圖 {img}")
            x01 = load_image_tensor(img, device, size=RESOLUTION)
            param, _, _ = build(
                args.condition, 0, block=args.block, hop=args.hop,
                r_min=args.r_min, r_max=args.r_max, quantile=args.quantile,
                gain_ratio=args.gain_ratio, freq_weight=args.freq_weight,
                freq_weight_power=args.freq_weight_power,
                pixel_gate_sigma=args.pixel_gate_sigma,
                theta_budget=args.theta_budget)
            param.set_radius(args.radius)
            param.reset(x01, 0)
            saved = torch.load(wp, map_location="cpu")
            ps = param.params()
            if len(saved) != len(ps):
                raise SystemExit(
                    f"{wp} 有 {len(saved)} 個張量，本次構造有 {len(ps)} 個"
                    "——構造不同，不可解讀")
            with torch.no_grad():
                for t, v in zip(ps, saved):
                    if tuple(t.shape) != tuple(v.shape):
                        raise SystemExit(
                            f"{wp} 的張量形狀 {tuple(v.shape)} 與本次的 "
                            f"{tuple(t.shape)} 不符——構造不同，不可解讀")
                    t.copy_(v.to(dtype=t.dtype))
            param.project()
            m = param.module
            row = {"tag": tag, "image": name, **consistency_deviation(m, x01)}
            rows.append(row)
            print(f"  {tag}／{name}  amp_dev={row['amp_dev']:.5f} "
                  f"phase_rho={row['phase_rho']:.5f}", flush=True)

    if not rows:
        raise SystemExit("沒有讀到任何權重檔——檢查 --weights 與 --condition")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    import statistics as st
    print(f"\n{args.out}：{len(rows)} 筆")
    print(f"  amp_dev   平均 {st.fmean(r['amp_dev'] for r in rows):.5f}")
    print(f"  phase_rho 平均 {st.fmean(r['phase_rho'] for r in rows):.5f}")


if __name__ == "__main__":
    main()
