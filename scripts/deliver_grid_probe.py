"""交付自壓的兩個前提，逐張量在真實影像上（**不跑 GPU**）。

要擋掉的兩個「靜默失效」
────────────────────────────────────────────────────────────────────
1. **codec 不一致。** 交付走的是本專案的浮點 JPEG（`src/baselines/jpeg_codec.py`，
   與最佳化迴圈的 STE 共用同一份量化表與色彩矩陣），而抗淨化評測的 `jpeg75`／
   `jpeg30` 走的是 PIL／libjpeg（`src/purify/ops.jpeg_real`）。兩者的 IDCT
   實作不同（libjpeg 是整數近似 islow），往返不逐位元相等。若我們把擾動釘在
   自己 codec 的格點上、而攻擊方用的是另一組格點，整個假說就不成立——**而它
   不會拋錯，只會讓數字看起來像是方法沒用**。
2. **八位元存檔。** 防禦圖以 PNG 存下再由 `phase_retention.py` 讀回，中間過
   一次 uint8 四捨五入。若那一步就把圖推離格點，交付這一步在流程上等於沒做。

量的東西
────────────────────────────────────────────────────────────────────
擾動取**隨機**且 RMS = 0.056（本方法工作點的殘差量級）。隨機是刻意的：這一支
問的是「格點這個機制成不成立」，不是「最佳化學不學得到」；後者要由
`runs/ip2p_deliver_jpeg/` 的 `deliver_retention` 欄回答。

    方向存活   攻擊方壓完之後的殘差 與 交付時殘差 的餘弦
    重壓變動   攻擊方壓完之後與交付圖的 RMS 差（0 即恆等）

用法：
    python scripts/deliver_grid_probe.py --out runs/ip2p_deliver_jpeg/codec_alignment.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.baselines.jpeg_codec import jpeg_roundtrip  # noqa: E402
from src.purify.ops import jpeg_real  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

WORKING_POINT_RMS = 0.056     # `runs/ip2p_axis_necessity/b_pg_r20` 的殘差 RMS
RESOLUTION = 512


def as_png(x: torch.Tensor) -> torch.Tensor:
    """存成 PNG 再讀回來會發生的事：夾到 [0,1]、量化成八位元。"""
    return torch.round(x.clamp(0, 1) * 255) / 255


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--deliver", type=int, nargs="+", default=[85, 75])
    ap.add_argument("--attack", type=int, nargs="+", default=[95, 75, 30])
    ap.add_argument("--rms", type=float, default=WORKING_POINT_RMS)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    paths = sorted(args.data.glob("*/*.png"))
    if args.images:
        keep = set(args.images)
        paths = [p for p in paths if p.parent.name in keep]
    paths = paths[: args.limit] if args.limit else paths
    if not paths:
        raise SystemExit(f"{args.data} 底下沒有影像")

    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for path in paths:
        x = load_image_tensor(path, torch.device("cpu"), size=RESOLUTION)
        d = torch.randn(x.shape, generator=gen)
        d = d / d.pow(2).mean().sqrt() * args.rms
        plain = as_png(x + d)
        for qd in [0] + list(args.deliver):
            # qd = 0 是對照組：直接交付未壓縮的圖，即現行作法。
            deliv = plain if qd == 0 else as_png(jpeg_roundtrip((x + d).clamp(0, 1), qd))
            ref = deliv - x
            for qa in args.attack:
                out = jpeg_real(deliv, qa)
                res = out - x
                rows.append({
                    "image": path.parent.name,
                    "deliver_quality": qd,
                    "attack_quality": qa,
                    "direction_survival": round(float(
                        res.flatten() @ ref.flatten()
                        / (res.norm() * ref.norm())), 5),
                    "recompress_rms": round(float(
                        (out - deliv).pow(2).mean().sqrt()), 6),
                    "delivered_rms": round(float(ref.pow(2).mean().sqrt()), 6),
                    "perturbation_rms": args.rms,
                    "deliver_codec": "jpeg_codec.jpeg_roundtrip（浮點）",
                    "attack_codec": "purify.ops.jpeg_real（PIL/libjpeg）",
                })
    write_csv(args.out, rows)

    print(f"{len(paths)} 張、擾動 RMS {args.rms}")
    print(f"{'交付':>6} {'攻擊':>6} {'方向存活':>10} {'重壓變動 RMS':>14}")
    for qd in [0] + list(args.deliver):
        for qa in args.attack:
            sel = [r for r in rows
                   if r["deliver_quality"] == qd and r["attack_quality"] == qa]
            label = "不壓" if qd == 0 else f"q{qd}"
            print(f"{label:>6} {'q' + str(qa):>6} "
                  f"{statistics.fmean(r['direction_survival'] for r in sel):>10.3f} "
                  f"{statistics.fmean(r['recompress_rms'] for r in sel):>14.5f}")
    print(f"寫入 {args.out}")


if __name__ == "__main__":
    main()
