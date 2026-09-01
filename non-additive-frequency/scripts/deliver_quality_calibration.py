"""把交付品質 `QD` 放粗要付多少失真，以及該配哪個半徑。**不跑 GPU。**

要回答什麼
────────────────────────────────────────────────────────────────────
`--deliver-jpeg QD` 交付的是壓縮後的圖，於是**交付本身**要付一筆失真。
`runs/ip2p_deliver_jpeg/README.md` 實測那一筆是**加性**而非比例的常數
（QD 0.85 約 +0.053、QD 0.75 約 +0.060 DISTS，與半徑幾乎無關）——因為 JPEG
對影像自己的重建誤差不隨擾動強度縮放。

把 `QD` 往下放粗到 0.6／0.45／0.35 時那個常數會變多大，決定了要配哪個
`--radius` 才落得進失真帶。猜錯半徑，`matched_distortion_table.py` 會因為
曲線沒跨過錨點而拒絕內插，整批 GPU 時間白花。

作法與它的偏差（**先寫下來**）
────────────────────────────────────────────────────────────────────
取已存的**未量化**防禦圖（`ours_ph_n`／`ours_pg_q20` 的同半徑未量化對照），
直接套 `jpeg_roundtrip(x_def, QD)` 量失真。這是**事後壓縮**，真正的
`--deliver-jpeg` 是把 `jpeg_roundtrip_ste` 放進最佳化前向、讓擾動長在量化
格點上，兩者不同：最佳化過的那一版會把係數推到格點邊界，失真通常**更高**。

偏差由一個已在 GPU 上量過的點當場校驗：`ours_ph_n`（r 0.9，未量化，DISTS
0.0497）套 QD 0.85 之後應該接近 `ours_ph_q` 的 **0.0928**。

用法：
    python scripts/deliver_quality_calibration.py \
        --defended <取回的防禦圖目錄> \
        --out runs/ip2p_mainline/deliver_quality_calibration.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.jpeg_codec import jpeg_decode, jpeg_encode  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
QDS = (0.85, 0.75, 0.60, 0.45, 0.35)

# tag → 防禦圖檔名的 condition 欄。取兩個未量化的點把半徑軸夾住。
SOURCES: Dict[str, str] = {
    "ours_ph_n": "phase",        # r 0.9，未量化，DISTS 0.0497
    "ours_pg_q20": "phase_gain",  # r 2.0，已量化 0.85（上界參照，非未量化）
}

# GPU 上已量過的真值，用來報代理的偏差。
KNOWN: Tuple[Tuple[str, float, float], ...] = (
    ("ours_ph_n", 0.85, 0.0928),   # → ours_ph_q
)

BAND_LO, BAND_HI = 0.1286, 0.1447


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defended", type=Path, required=True)
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path,
                    default=Path("runs/ip2p_mainline/deliver_quality_calibration.csv"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    device = torch.device(args.device)
    suite = MetricSuite(device=device)

    rows = []
    for tag, cond in SOURCES.items():
        for name in names:
            x = load_image_tensor(args.data / name / f"{name}.png", device,
                                  size=RESOLUTION).double()
            x_def = load_image_tensor(
                args.defended / tag / f"{name}__{cond}__def.png", device,
                size=RESOLUTION).double()
            base = suite.pairwise(x.float(), x_def.float())
            rows.append({"image": name, "source": tag, "qd": "none",
                         "fid_dists": round(base["dists"], 6),
                         "fid_psnr": round(base["psnr"], 4),
                         "delta_dists": 0.0})
            for qd in QDS:
                y = jpeg_decode(jpeg_encode(x_def, qd), qd).clamp(0, 1)
                m = suite.pairwise(x.float(), y.float())
                rows.append({"image": name, "source": tag, "qd": qd,
                             "fid_dists": round(m["dists"], 6),
                             "fid_psnr": round(m["psnr"], 4),
                             "delta_dists": round(m["dists"] - base["dists"], 6)})

    write_csv(args.out, rows)

    print(f"{len(names)} 張 → {args.out}")
    print(f"失真帶 DISTS {BAND_LO}–{BAND_HI}\n")
    for tag in SOURCES:
        sub = [r for r in rows if r["source"] == tag]
        base = statistics.mean(r["fid_dists"] for r in sub if r["qd"] == "none")
        print(f"{tag}  未交付 DISTS {base:.4f}")
        for qd in QDS:
            s = [r for r in sub if r["qd"] == qd]
            d = statistics.mean(r["fid_dists"] for r in s)
            dd = statistics.mean(r["delta_dists"] for r in s)
            mark = "  ← 帶內" if BAND_LO <= d <= BAND_HI else ""
            print(f"    QD {qd:.2f}  DISTS {d:.4f}  （交付加價 +{dd:.4f}）"
                  f"  PSNR {statistics.mean(r['fid_psnr'] for r in s):.2f}{mark}")
        print()

    print("校驗：代理 vs GPU 上已量過的真值")
    for tag, qd, truth in KNOWN:
        got = statistics.mean(r["fid_dists"] for r in rows
                              if r["source"] == tag and r["qd"] == qd)
        print(f"  {tag} + QD {qd:g}：代理 {got:.4f}  真值 {truth:.4f}  "
              f"比值 {got / truth:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
