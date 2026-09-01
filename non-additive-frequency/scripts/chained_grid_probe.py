"""交付到 JPEG 格點的圖，經過重取樣之後還在不在格點上。**純 CPU，不跑 GPU。**

要分開的兩個解釋
────────────────────────────────────────────────────────────────────
不動點框架（`runs/fixedpoint_framework/README.md` §3.5）對串接算子的預測是：
重取樣會把 JPEG 的格點打散，交集很小，**所以交付到格點在那一格幫助有限**。
實測相反——量化交付在 C&R 串接上是 0.1832、不交付是 0.1271，**交付仍然幫了
1.44 倍**（`runs/ip2p_deliver_jpeg`）。

兩個解釋，本檔要分開它們：

1. **格點部分存活**：重取樣的比例小（每邊 10%、放大 1.2488），8×8 的格點在
   重取樣後沒有被完全打散，交集因此比預期大。
2. **與不動點無關**：交付的好處有一部分只是「擾動被限制在較粗的格點上，
   因此對任何小擾動都比較穩」——那是一個邊際效果，不是落在不動點集合上。

怎麼分
────────────────────────────────────────────────────────────────────
量一張圖離 JPEG(QD) 的格點有多遠：

    格點距離(y) = RMS( jpeg_roundtrip(y, QD) − y )

交付圖 `d = jpeg_roundtrip(x, QD)` 的格點距離**由構造接近零**（JPEG 幾乎冪等）。
問題是**過了算子之後**：

    ratio = 格點距離( T(d) ) / 格點距離( T(x) )

`T(x)` 是**沒有交付過**的圖過同一個算子——它是「一般影像離格點多遠」的參照。

- `ratio` 明顯小於 1 → 交付圖過了算子之後**仍然比一般影像靠近格點**
  → 解釋 1（格點部分存活）。
- `ratio` 接近 1 → 重取樣把格點打散得跟沒交付一樣
  → 解釋 2（好處與不動點無關）。

**擾動也要在場**：真正交出去的圖帶著防禦擾動，而擾動本身會影響離格點的距離。
本檔用固定種子的隨機擾動，RMS 對齊工作點（0.043–0.061，取 0.05），
在**交付之前**加進去——與 `--deliver-jpeg` 的順序一致。

判準（跑之前寫下）
────────────────────────────────────────────────────────────────────
  C1  `crop_resize` 與 `jpeg_then_resize` 兩格的 `ratio` 若都 **> 0.9**，
      解釋 2 成立：框架 §3.5 的預測沒有錯，錯的是把交付的全部好處都歸給
      不動點集合——**框架要加一條「粗格點本身有與不動點無關的穩定性」**。
  C2  `ratio` 若 **< 0.6**，解釋 1 成立：格點在小幅重取樣下部分存活，
      §3.5 的預測要改寫成「交集隨重取樣比例衰減」而不是「交集很小」。
  C3  兩者之間（0.6–0.9）就是分不開，照實寫成分不開。

**控制組是 `jpeg95`／`jpeg85`，不是 `jpeg75`。** 交付品質是 85，而 75 比它
**更低**——攻擊方壓得更狠，兩張圖都會被推到 75 的格點上，離 85 的格點都很遠。
真正的近似恆等是攻擊方壓得**比交付更輕**（95）或**相同**（85），那時交付圖應該
仍然貼在格點上，`ratio` 必須明顯小於 1，否則就是本檔的量法有問題。
（第一版把 75 當控制組，量到 1.52 —— **那是控制組設錯，不是量法有問題**。）

用法：
    python scripts/chained_grid_probe.py --out runs/fixedpoint_framework/chained_grid.csv
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baselines.jpeg_codec import jpeg_roundtrip  # noqa: E402
from src.purify import ops as purify_ops  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
PERTURB_RMS = 0.05           # 工作點的殘差 RMS 是 0.043–0.061
OPS = ("jpeg95", "jpeg85", "jpeg75", "crop_resize0.1",
       "jpeg_then_resize75")


def build_op(name: str):
    if name.startswith("jpeg") and name[4:].isdigit():
        return purify_ops.Purifier("jpeg", int(name[4:]))
    if name == "crop_resize0.1":
        return purify_ops.Purifier("crop_resize", 0.10)
    if name == "jpeg_then_resize75":
        return purify_ops.Purifier("jpeg_then_resize", 75)
    raise ValueError(f"未知的算子 {name!r}")


def grid_gap(y: torch.Tensor, qd: float) -> float:
    """離 JPEG(QD) 格點的距離，以 RMS 計。交付圖上它由構造接近零。"""
    return float((jpeg_roundtrip(y, qd) - y).pow(2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--qd", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in
             args.images.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dev = torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    rows: List[dict] = []
    for name in names:
        x = load_image_tensor(args.data / name / f"{name}.png", dev,
                              size=RESOLUTION)
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        noise = noise / noise.pow(2).mean().sqrt() * PERTURB_RMS
        # 交付的順序與 `--deliver-jpeg` 一致：先加擾動、再壓到 QD 的格點。
        delivered = jpeg_roundtrip((x + noise).clamp(0, 1), args.qd)
        plain = (x + noise).clamp(0, 1)      # 同樣的擾動，但沒有交付
        for op_name in OPS:
            op = build_op(op_name)
            with torch.no_grad():
                td, tp = op.evaluate(delivered), op.evaluate(plain)
            g_before = grid_gap(delivered, args.qd)
            g_after, g_plain = grid_gap(td, args.qd), grid_gap(tp, args.qd)
            rows.append({
                "image": name, "op": op_name, "qd": args.qd,
                "grid_gap_delivered": round(g_before, 6),
                "grid_gap_after": round(g_after, 6),
                "grid_gap_plain_after": round(g_plain, 6),
                "ratio": round(g_after / max(g_plain, 1e-12), 5),
            })
        print(f"  {name} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    print()
    print(f"寫出 {args.out}（{len(rows)} 列）")
    print()
    print("算子".ljust(22), "交付圖離格點".ljust(14), "過算子後".ljust(12),
          "沒交付過算子後".ljust(16), "ratio")
    for op_name in OPS:
        sel = [r for r in rows if r["op"] == op_name]
        a = {k: st.mean([r[k] for r in sel]) for k in
             ("grid_gap_delivered", "grid_gap_after", "grid_gap_plain_after",
              "ratio")}
        print(op_name.ljust(22), f"{a['grid_gap_delivered']:.6f}".ljust(14),
              f"{a['grid_gap_after']:.6f}".ljust(12),
              f"{a['grid_gap_plain_after']:.6f}".ljust(16),
              f"{a['ratio']:.4f}")


if __name__ == "__main__":
    main()
