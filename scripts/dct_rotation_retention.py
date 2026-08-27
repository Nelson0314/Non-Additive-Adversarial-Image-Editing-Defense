"""8×8 DCT 保長旋轉的交付存活率：攻擊方重壓一次之後，擾動還剩多少。

要回答什麼
────────────────────────────────────────────────────────────────────
提案宣稱的主要新性質是**交付即參數**：擾動定義在量化後的整數係數上，交出去的
就是那組係數解碼出來的影像，攻擊方以相同或更高品質重壓時四捨五入不會把它推走
（DCT-Shield 抗 JPEG 的全部機制，`src/baselines/dct_shield.py` §4.2）。

現行的「量化交付」（`runs/ip2p_deliver_jpeg`）是**在像素域做完擾動再壓一次**，
量到的自壓保留率是 0.76–0.94，隨機擾動只有 0.22。宣稱要成立，DCT 原生的作法
必須明顯高於 0.9431（`qd85` 那一列），否則它宣稱的性質是現行作法已經拿到的，
提案沒有新東西。

判讀規則（跑之前就寫下）
────────────────────────────────────────────────────────────────────
R1  攻擊方品質 = 交付品質時，`post_int` 的保留率**沒有明顯高於 0.9431**
    → 「交付即參數」不是一個現行作法拿不到的性質，提案的主要理由消失。
R2  `post_int` 的保留率**沒有明顯高於同一格上的 `pre_deliver`**
    → 旋轉發生在量化的哪一側不重要，整個設計問題退化成「要不要自壓」，
    那已經測過了。
R3  攻擊方品質**低於**交付品質時（jpeg50／jpeg30）保留率若與現行作法同樣塌掉，
    則此設計沒有解決 `runs/ip2p_deliver_jpeg` 第五節第 4 點記下的那個弱點。

`identity` 那一列是自檢：交付之後不做任何事，保留率必須是 1.0。

用法：
    python scripts/dct_rotation_retention.py --out runs/dct_phase_design/retention.csv

**純 CPU、不跑 GPU、不做最佳化。**
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.baselines.jpeg_codec import (  # noqa: E402
    dct_matrix, jpeg_roundtrip,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402
from dct_rotation_ceiling import (  # noqa: E402
    build_pairs, gates_for, sign_pattern, variant_post_int, variant_pre_deliver,
)
from src.baselines.jpeg_codec import jpeg_encode  # noqa: E402

RESOLUTION = 512
# `runs/ip2p_deliver_jpeg/README.md` 第一節：qd85 的自壓保留率與隨機基準。
DELIVER_JPEG_QD85 = 0.9431
RANDOM_BASELINE = 0.22


def retention(x_ref: torch.Tensor, x_def: torch.Tensor, quality) -> float:
    """`<d', d> / <d, d>`。d 是交付後的殘差，d' 是攻擊方壓過之後的殘差。

    參照方向取**交付後**的殘差，問的是「交出去的那個擾動剩多少」，與
    `scripts/coarsen_probe.py` 的 `jpeg_retention` 同一個定義。
    """
    d = x_def - x_ref
    if quality is None:
        d2 = d
    else:
        d2 = jpeg_roundtrip(x_def, quality) - jpeg_roundtrip(x_ref, quality)
    den = float((d * d).sum())
    return float((d2 * d).sum()) / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--qd", type=float, default=0.85)
    ap.add_argument("--attack", type=float, nargs="+",
                    default=[0.95, 0.85, 0.75, 0.5, 0.3])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.3, 1.2, 3.14159265])
    ap.add_argument("--pairing", default="transpose")
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()][:args.n]
    device = torch.device("cpu")
    d = dct_matrix(device, torch.float32)
    pairs = build_pairs(args.pairing, args.r_min)
    rows = []
    for idx, name in enumerate(names):
        hits = sorted((args.data / name).glob("*.png")) + \
            sorted((args.data / name).glob("*.jpg"))
        if not hits:
            raise SystemExit(f"{args.data / name} 下沒有影像")
        x = load_image_tensor(hits[0], device, size=RESOLUTION).clamp(0, 1)
        alpha = jpeg_encode(x, args.qd)
        gates = gates_for(x, alpha, "band", 2.0, 1.0)
        sign = {k: sign_pattern((v.shape[0], v.shape[1], v.shape[2]),
                                "random", 1000 + idx, device, torch.float32)
                for k, v in alpha.items()}
        # 交付後的參照：兩個變體交出去的都是 QD 品質的 JPEG，故參照是
        # `jpeg_roundtrip(x, QD)` 而不是原圖——問的是擾動存活，不是失真。
        x_ref = jpeg_roundtrip(x, args.qd)
        for vname, fn in (("post_int", variant_post_int),
                          ("pre_deliver", variant_pre_deliver)):
            for th in args.thetas:
                t = torch.tensor(th, dtype=torch.float32)
                with torch.no_grad():
                    x_def, _ = fn(x, pairs, t, gates, sign, args.qd, d)
                    row = {"image": name, "variant": vname,
                           "theta": round(th, 4),
                           "rms": round(float((x_def - x_ref).pow(2).mean().sqrt()), 6),
                           "identity": round(retention(x_ref, x_def, None), 6)}
                    for q in args.attack:
                        row[f"keep_q{int(q * 100)}"] = round(
                            retention(x_ref, x_def, q), 6)
                rows.append(row)
        write_csv(args.out, rows)
        print(f"  {idx + 1}/{len(names)} {name}", flush=True)

    print()
    print(f"十張平均（交付品質 QD = {args.qd}）。對照：現行量化交付 qd85 是 "
          f"{DELIVER_JPEG_QD85}，隨機擾動基準是 {RANDOM_BASELINE}。")
    hdr = "  ".join(f"q{int(q * 100):>3d}" for q in args.attack)
    print(f"{'變體':<13s}{'theta':>7s}{'RMS':>9s}{'identity':>10s}  {hdr}")
    for vname in ("post_int", "pre_deliver"):
        for th in args.thetas:
            sel = [r for r in rows if r["variant"] == vname
                   and abs(r["theta"] - th) < 1e-3]
            if not sel:
                continue
            def f(k, s=sel):
                return statistics.fmean(r[k] for r in s)
            cells = "  ".join(f"{f(f'keep_q{int(q * 100)}'):6.4f}"
                              for q in args.attack)
            print(f"{vname:<13s}{th:7.3f}{f('rms'):9.5f}{f('identity'):10.4f}  {cells}")
    print(f"表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
