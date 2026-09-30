"""顏色曲線的位移，有多少可以只用「編輯結果被同一條曲線調色」解釋。

問題
────────────────────────────────────────────────────────────────────
主讀數是 `位移 = LPIPS(編輯(原圖), 編輯(防禦圖))`。顏色曲線這一條的防禦圖
整張都被重新映射過顏色，於是它的編輯結果也會帶著那個色偏。那麼位移裡有
多少是「編輯真的被推開」，多少只是「同一個畫面換了顏色」？

本檔在同一格、同一對 PNG 上量三個數：

    A  位移        `LPIPS(編輯(原圖), 編輯(防禦圖))`        現行主讀數
    B  純色彩搬移  `LPIPS(編輯(原圖), 曲線(編輯(原圖)))`    對照
    C  扣掉色彩    `LPIPS(曲線(編輯(原圖)), 編輯(防禦圖))`  殘量

B 的作法是把**同一張影像求出來的那條曲線**直接套在未防禦的編輯結果上：
畫面內容一個像素都沒有被重畫，只有顏色被搬到防禦圖的那個位置。A 與 B
放在一起，就把位移分成「顏色」與「其餘」兩塊。**三個數都照報，本檔不下
判準**（`CLAUDE.md`「訓練方法的實驗：不設判準」）。

曲線從哪裡來
────────────────────────────────────────────────────────────────────
不讀求解端的參數檔，直接**從交付的 PNG 反查**：以原圖某通道的像素值為鍵、
防禦圖同位置同通道的值為值。tone curve 是逐通道的全域映射（沒有空間
相依），所以反查出來的 256→256 查找表就是那條曲線本身，而且量到的是
**實際交付的 8-bit 產物**，不是求解端的浮點參數。

反查的前提逐張驗證並寫進 CSV：

    `lut_max_spread`  同一個輸入色階對到的輸出值全距。**不是 0 就代表
                      那張圖的映射不是全域的**，該張的 B 不可解讀。
    `lut_missing`     原圖裡沒有出現過的色階數（三個通道合計）。這些色階
                      只能由相鄰色階內插，而編輯結果裡可能出現它們。

用法
    python scripts/colour_shift_control.py \\
        --displacement runs/edit_defended/displacement.csv \\
        --defended runs/defence_portraits/colour_curve_ours \\
        --condition colour_curve_ours --out runs/colour_shift_control
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

CHANNELS = 3
LEVELS = 256


def build_lut(orig_path: Path, defended_path: Path) -> tuple:
    """從一對 PNG 反查逐通道的 256→256 查找表。

    回傳 `(lut (3,256) uint8, max_spread, missing)`。`max_spread` 是同一個
    輸入色階對到的輸出值全距的最大值——tone curve 是全域映射，這個數應該
    是 0；不是 0 就代表反查的前提不成立，呼叫端要把它報出來。
    """
    a = np.asarray(Image.open(orig_path).convert("RGB"))
    b = np.asarray(Image.open(defended_path).convert("RGB"))
    if a.shape != b.shape:
        raise SystemExit(f"{orig_path} 與 {defended_path} 的形狀不同："
                         f"{a.shape} 對 {b.shape}")

    lut = np.zeros((CHANNELS, LEVELS), np.uint8)
    max_spread = 0
    missing = 0
    for c in range(CHANNELS):
        src = a[..., c].ravel().astype(np.int64)
        dst = b[..., c].ravel().astype(np.int64)
        counts = np.bincount(src, minlength=LEVELS)
        seen = counts > 0
        low = np.full(LEVELS, LEVELS - 1, np.int64)
        high = np.zeros(LEVELS, np.int64)
        np.minimum.at(low, src, dst)
        np.maximum.at(high, src, dst)
        max_spread = max(max_spread, int((high[seen] - low[seen]).max()))
        mean = np.zeros(LEVELS, np.float64)
        mean[seen] = np.bincount(src, weights=dst, minlength=LEVELS)[seen] / counts[seen]
        index = np.flatnonzero(seen)
        missing += LEVELS - int(index.size)
        # 沒有出現過的色階用相鄰色階內插。編輯結果的色階分布與原圖不同，
        # 所以這些格子會被用到；留空會在套用時索引到 0。
        filled = np.interp(np.arange(LEVELS), index, mean[index])
        lut[c] = np.clip(np.rint(filled), 0, LEVELS - 1).astype(np.uint8)
    return lut, max_spread, missing


def apply_lut(path: Path, lut: np.ndarray) -> np.ndarray:
    """把查找表套在一張 PNG 上，回傳 `(H,W,3)` uint8。"""
    a = np.asarray(Image.open(path).convert("RGB"))
    return np.stack([lut[c][a[..., c]] for c in range(CHANNELS)], axis=-1)


def to_tensor(array: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(array)).float().div(255)
    return t.permute(2, 0, 1)[None].to(device)


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def median(values) -> float:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--displacement", type=Path, required=True)
    ap.add_argument("--defended", type=Path, required=True,
                    help="防禦圖目錄（`defence_run.py`／`immunise_as_condition.py` 的輸出）")
    ap.add_argument("--condition", default="colour_curve_ours")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="CSV 裡那兩欄 PNG 路徑的基準目錄")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    help="預設 cpu：這一支只算指標、不跑擴散模型，不必佔卡")
    args = ap.parse_args()

    rows_in = [r for r in read_csv(args.displacement)
               if r["condition"] == args.condition]
    if not rows_in:
        raise SystemExit(f"{args.displacement} 裡沒有 condition={args.condition} 的列")

    device = torch.device(args.device)
    suite = MetricSuite(device=device)
    args.out.mkdir(parents=True, exist_ok=True)

    luts = {}
    rows = []
    csv_path = args.out / "colour_shift_control.csv"
    for row in rows_in:
        image = row["image"]
        if image not in luts:
            orig = args.defended / f"{image}__orig.png"
            defended = args.defended / f"{image}__{args.condition}__def.png"
            for path in (orig, defended):
                if not path.is_file():
                    raise SystemExit(f"找不到 {path}")
            luts[image] = build_lut(orig, defended)
        lut, spread, missing = luts[image]

        undefended_png = args.root / row["undefended_png"]
        defended_png = args.root / row["defended_png"]
        curved = apply_lut(undefended_png, lut)
        curved_png = (args.out / row["scenario"] /
                      f"{image}__p{row['prompt_index']}__curved.png")
        curved_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(curved).save(curved_png)

        u = to_tensor(np.asarray(Image.open(undefended_png).convert("RGB")), device)
        d = to_tensor(np.asarray(Image.open(defended_png).convert("RGB")), device)
        cur = to_tensor(curved, device)

        a_disp = suite.pairwise(u, d)
        b_shift = suite.pairwise(u, cur)
        c_resid = suite.pairwise(cur, d)
        rows.append({
            "condition": args.condition,
            "scenario": row["scenario"],
            "image": image,
            "prompt_index": row["prompt_index"],
            "prompt": row["prompt"],
            "a_disp_lpips": round(a_disp["lpips"], 5),
            "b_shift_lpips": round(b_shift["lpips"], 5),
            "c_residual_lpips": round(c_resid["lpips"], 5),
            "a_disp_deltaE00": round(a_disp["deltaE00"], 4),
            "b_shift_deltaE00": round(b_shift["deltaE00"], 4),
            "c_residual_deltaE00": round(c_resid["deltaE00"], 4),
            "a_disp_dists": round(a_disp["dists"], 5),
            "b_shift_dists": round(b_shift["dists"], 5),
            "c_residual_dists": round(c_resid["dists"], 5),
            "a_disp_rms": round(a_disp["rms"], 6),
            "b_shift_rms": round(b_shift["rms"], 6),
            "c_residual_rms": round(c_resid["rms"], 6),
            "a_disp_ssim": round(a_disp["ssim"], 5),
            "b_shift_ssim": round(b_shift["ssim"], 5),
            "c_residual_ssim": round(c_resid["ssim"], 5),
            # CSV 裡已經有的那一格主讀數，抄過來供核對：兩者應該一致，
            # 不一致代表配對錯了（靜默失效的典型形狀）。
            "csv_disp_lpips_full": row.get("disp_lpips_full", ""),
            "lut_max_spread": spread,
            "lut_missing": missing,
            "undefended_png": row["undefended_png"],
            "defended_png": row["defended_png"],
            "curved_png": str(curved_png).replace("\\", "/"),
        })
        write_csv(csv_path, rows)
        print(f"[DONE] {row['scenario']:8s} {image:10s} p{row['prompt_index']} "
              f"A={a_disp['lpips']:.4f} B={b_shift['lpips']:.4f} "
              f"C={c_resid['lpips']:.4f} (spread={spread})", flush=True)

    print(f"\n[ALLDONE] {csv_path}（{len(rows)} 列）", flush=True)
    for scenario in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == scenario]
        print(f"{scenario:8s} n={len(sub):3d}  "
              f"A={median([r['a_disp_lpips'] for r in sub])}  "
              f"B={median([r['b_shift_lpips'] for r in sub])}  "
              f"C={median([r['c_residual_lpips'] for r in sub])}", flush=True)
    spreads = {r["lut_max_spread"] for r in rows}
    print(f"lut_max_spread 的相異值：{sorted(spreads)}", flush=True)


if __name__ == "__main__":
    main()
