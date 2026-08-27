"""DCT-Shield 在 `δ = 0` 時付的失真地板，逐 `Q_alg`。**不跑 GPU。**

要回答什麼
────────────────────────────────────────────────────────────────────
`Q_alg` 是 DCT-Shield **防禦方自選**的參數。既有的頭對頭全部固定在 0.85
（§6.3 圖 6 的 Y-only 設定），而該設定的抗 JPEG 保證是單向的（補充材料
D.4）——它是為了活過品質 ≥ 85 的重壓而設。若攻擊方會壓到品質 30，對手照
論文用法應該把 `Q_alg` 一起調低。要不要花 GPU 去掃 `Q_alg` 0.5／0.3，取決
於一件純算術的事：

    δ = 0 時 DCT-Shield 的輸出就是 `Q_alg` 品質的 JPEG 壓縮圖。

也就是它在最佳化任何東西**之前**就已經付掉一筆失真。若那筆地板本身已經
超出本專案的失真帶（DISTS 0.1286–0.1447），那個 `Q_alg` 在帶內**不存在**
可比的工作點，掃它是白跑。

`runs/ip2p_residual_signature/dct_shield_zero_floor.csv` 量過同一件事，但
只有 `Q_alg` 0.95／0.85 兩格、且是 13 張的清單。本檔覆蓋主線的十張與
0.95 → 0.30 的整條，供 `RUN_QUEUE` 第二優先派工前的判斷。

作法
────────────────────────────────────────────────────────────────────
`jpeg_decode(jpeg_encode(x, q), q)` 走本專案的可微 JPEG（`jpeg_codec`），
與 `run_dct_shield` 在 δ = 0 時的路徑逐行相同——不是用 PIL 另存一張，
那會多一次 uint8 四捨五入而量到別的東西。

用法：
    python scripts/dct_shield_qalg_floor.py \
        --images runs/ip2p_fair_comparison/images10.txt \
        --out runs/ip2p_mainline/dct_shield_qalg_floor.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.jpeg_codec import jpeg_decode, jpeg_encode  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
Q_ALGS = (0.95, 0.85, 0.75, 0.60, 0.50, 0.40, 0.30)

# 失真帶（`docs/EVALUATION.md` 的工作點對齊）：本方法主線工作點的 DISTS。
BAND_LO, BAND_HI = 0.1286, 0.1447


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path,
                    default=Path("runs/ip2p_mainline/dct_shield_qalg_floor.csv"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    device = torch.device(args.device)
    suite = MetricSuite(device=device)

    rows = []
    for name in names:
        path = args.data / name / f"{name}.png"
        x = load_image_tensor(path, device, size=RESOLUTION).double()
        for q in Q_ALGS:
            coef = jpeg_encode(x, q)
            y = jpeg_decode(coef, q).clamp(0, 1)
            m = suite.pairwise(x.float(), y.float())
            rows.append({
                "image": name,
                "q_alg": q,
                "fid_dists": round(m["dists"], 6),
                "fid_lpips": round(m["lpips"], 6),
                "fid_psnr": round(m["psnr"], 4),
                "fid_ssim": round(m["ssim"], 6),
                "fid_vif_p": round(m["vif_p"], 6),
                "fid_linf": round(m["linf"], 6),
                "rms": round(m["rms"], 6),
            })

    write_csv(args.out, rows)

    print(f"{len(names)} 張 × {len(Q_ALGS)} 個 Q_alg → {args.out}")
    print(f"失真帶 DISTS {BAND_LO}–{BAND_HI}\n")
    head = f"{'Q_alg':>6} {'DISTS':>9} {'LPIPS':>8} {'PSNR':>7} {'SSIM':>7} {'RMS':>7}  帶內還剩多少預算"
    print(head)
    print("-" * len(head))
    for q in Q_ALGS:
        sub = [r for r in rows if r["q_alg"] == q]
        d = statistics.mean(r["fid_dists"] for r in sub)
        note = ("**地板已超出帶上緣**" if d > BAND_HI
                else f"δ 可用 {BAND_HI - d:.4f}（佔帶寬 {(BAND_HI - d) / BAND_HI:.0%}）")
        print(f"{q:>6.2f} {d:>9.4f} "
              f"{statistics.mean(r['fid_lpips'] for r in sub):>8.4f} "
              f"{statistics.mean(r['fid_psnr'] for r in sub):>7.2f} "
              f"{statistics.mean(r['fid_ssim'] for r in sub):>7.4f} "
              f"{statistics.mean(r['rms'] for r in sub):>7.4f}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
