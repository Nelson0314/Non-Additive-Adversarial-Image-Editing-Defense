"""每個 `Q_alg` 要用哪些 `eps` 才跨得過失真錨點。**不跑 GPU。**

要回答什麼
────────────────────────────────────────────────────────────────────
`eps` 的單位是**量化後的整數係數**，而一格係數換算回像素的幅度等於該頻率
的量化表值，量化表又隨 `Q_alg` 反向縮放（libjpeg：`Q=85` 的表是 base 的
0.30 倍，`Q=30` 是 1.67 倍）。**同一個 `eps` 在不同 `Q_alg` 上不是同一個
失真預算**，差到 5.6 倍。派工前不先量，`matched_distortion_table.py` 會
因為曲線沒跨過錨點而拒絕內插，整批白跑（`warp_triad.sh` 已經踩過一次，
那次改用 `warp_radius_calibration.py` 先在本機量半徑）。

作法與它的偏差（**先寫下來，不是看到數字才補**）
────────────────────────────────────────────────────────────────────
本檔不最佳化。它取 `δ = eps · s`，`s` 是逐係數的隨機正負號，只加在 Y 通道
（§6.3 的 Y-only 設定），然後走與 `run_dct_shield` 逐行相同的解碼路徑量失真。

這是**飽和代理**，不是預測值。它與真正的 δ 差在兩件事，方向相反：

1. 真正的 δ **沒有完全飽和**——`DCTShieldSpec` 的 docstring 記錄 Y 通道
   `|δ|` 中位數 0.903、50.3% 超過 0.9（eps = 1）。代理把全部推到 1.0，
   **高估**失真。
2. 真正的 δ 的正負號來自梯度，**空間上相關**；隨機正負號在感知指標上
   （DISTS 對結構敏感）通常較便宜，**低估**失真。

兩者不保證抵消，所以本檔的輸出**只用來選 `eps` 的括弧**，不進任何結論表。
偏差的大小由三個已在 GPU 上量過的點當場校驗（見輸出的「校驗」段）：

    Q_alg 0.85 / eps 1.0 → DISTS 0.1118      （runs/ip2p_mainline/tables）
    Q_alg 0.85 / eps 1.5 → DISTS 0.1539
    Q_alg 0.75 / eps 0.6 → DISTS 0.1145

用法：
    python scripts/dct_shield_eps_calibration.py \
        --out runs/ip2p_mainline/dct_shield_eps_calibration.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.jpeg_codec import jpeg_decode, jpeg_encode  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
SEED = 20260812

# 逐 `Q_alg` 的 eps 括弧。低 `Q_alg` 的量化階大，同一個 eps 貴得多，故往下收。
GRID: Dict[float, Sequence[float]] = {
    0.85: (0.6, 1.0, 1.5, 2.2),
    0.75: (0.4, 0.6, 1.0, 1.5),
    0.50: (0.15, 0.25, 0.40, 0.60),
    0.30: (0.08, 0.15, 0.25, 0.40),
}

# 要跨過的錨點：對手的論文設定、本方法主線工作點、本方法最強點。
ANCHORS = (("dct_aj85 論文設定", 0.1118),
           ("主線工作點", 0.1447),
           ("ours_pg_q20 最強點", 0.1947))

# GPU 上已量過的真值，用來報代理的偏差。
KNOWN: Tuple[Tuple[float, float, float], ...] = (
    (0.85, 1.0, 0.1118),
    (0.85, 1.5, 0.1539),
    (0.75, 0.6, 0.1145),
)


def saturated_delta(alpha_y: torch.Tensor, eps: float, seed: int) -> torch.Tensor:
    """`δ = eps · s`，`s` 逐係數 ±1。與 `run_dct_shield` 一樣不排除 DC。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, 2, alpha_y.shape, generator=g).to(
        device=alpha_y.device, dtype=alpha_y.dtype) * 2.0 - 1.0
    return s * eps


def interp_eps(points: Sequence[Tuple[float, float]], target: float) -> float | None:
    """在 (eps, dists) 折線上求 dists = target 的 eps。不外插，超界回 None。"""
    pts = sorted(points)
    if target < pts[0][1] or target > pts[-1][1]:
        return None
    for (e0, d0), (e1, d1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            if d1 == d0:
                return e0
            return e0 + (e1 - e0) * (target - d0) / (d1 - d0)
    return None


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path,
                    default=Path("runs/ip2p_mainline/dct_shield_eps_calibration.csv"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    device = torch.device(args.device)
    suite = MetricSuite(device=device)

    rows = []
    for name in names:
        x = load_image_tensor(args.data / name / f"{name}.png", device,
                              size=RESOLUTION).double()
        for q, eps_list in GRID.items():
            alpha = {k: v.detach() for k, v in jpeg_encode(x, q).items()}
            for eps in eps_list:
                coef = dict(alpha)
                coef["Y"] = alpha["Y"] + saturated_delta(alpha["Y"], eps, SEED)
                y = jpeg_decode(coef, q).clamp(0, 1)
                m = suite.pairwise(x.float(), y.float())
                rows.append({
                    "image": name, "q_alg": q, "eps": eps,
                    "proxy": "saturated_random_sign",
                    "fid_dists": round(m["dists"], 6),
                    "fid_lpips": round(m["lpips"], 6),
                    "fid_psnr": round(m["psnr"], 4),
                    "rms": round(m["rms"], 6),
                })

    write_csv(args.out, rows)

    curves: Dict[float, List[Tuple[float, float]]] = {}
    for q, eps_list in GRID.items():
        curves[q] = [(e, statistics.mean(r["fid_dists"] for r in rows
                                         if r["q_alg"] == q and r["eps"] == e))
                     for e in eps_list]

    print(f"{len(names)} 張 → {args.out}\n")
    print("代理曲線（十張平均 DISTS）")
    for q in GRID:
        pretty = "  ".join(f"eps {e:g}: {d:.4f}" for e, d in curves[q])
        print(f"  Q_alg {q:.2f}   {pretty}")

    print("\n校驗：代理 vs GPU 上已量過的真值")
    for q, eps, truth in KNOWN:
        pred = interp_eps(curves[q], truth)
        got = dict(curves[q]).get(eps)
        line = f"  Q_alg {q:.2f} eps {eps:g}  真值 {truth:.4f}"
        if got is not None:
            line += f"  代理 {got:.4f}  比值 {got / truth:.3f}"
        if pred is not None:
            line += f"  （代理需要 eps {pred:.3f} 才到真值）"
        print(line)

    print("\n建議的 eps（代理內插，**要照校驗的比值修正後再用**）")
    for q in GRID:
        parts = []
        for label, a in ANCHORS:
            e = interp_eps(curves[q], a)
            parts.append(f"{label} {a:.4f} → " + (f"eps {e:.3f}" if e else "超出括弧"))
        print(f"  Q_alg {q:.2f}")
        for p in parts:
            print(f"      {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
