"""防禦把影像 latent 推了多遠。**只跑 VAE 編碼器，不跑擴散取樣。**

要回答什麼
────────────────────────────────────────────────────────────────────
使用者確認的判定上，`acutance_ratio`（防禦後編輯的梯度能量 ÷ 未防禦編輯的
梯度能量）把兩類完全分開：擋下 0.704、失敗 0.119。**失敗的格子就是被糊掉
的格子**，而糊掉不算擋下。

`latent_norm` 損失壓的正是 `‖E(x')‖₂`。機制上的說法是：壓到夠低時 UNet 的
影像條件失去資訊，IP2P 退化成純文生圖，輸出是把 prompt 畫出來（重畫）；
壓不夠時模型仍跟著一個劣化的 latent 走，輸出是同一個場景變糊。

若逐圖的範數比值能預測擋下，它就是比 DISTS 好的**逐圖停止準則**。
逐圖對齊 DISTS 已經測過並失敗（同一批 7 張，平均 DISTS 幾乎不變而擋下率由
6/7 掉到 3/7），原因正是對齊了錯的量。

量什麼
────────────────────────────────────────────────────────────────────
    norm_orig   ‖E(x)‖₂
    norm_def    ‖E(x')‖₂
    norm_ratio  norm_def / norm_orig      損失實際壓下去多少
    move_ratio  ‖E(x') − E(x)‖₂ / ‖E(x)‖₂  latent 被推開多遠（方向無關）

兩個都報：範數可以壓下去而方向不變，也可以方向大變而範數不動，兩者對
UNet 的影響不同。

用法：
    python scripts/latent_norm_ratio.py --src <擺著 __def.png 的目錄> \
        --out runs/<批次>/latent_norm_ratio.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def ratios(z_orig: torch.Tensor, z_def: torch.Tensor) -> Dict[str, float]:
    """兩個比值。原圖 latent 範數為零時**拋錯**，不回傳 inf。

    那是編碼器壞掉才會有的情況；回傳 inf 會讓那一列看起來只是個極端值，
    而整批的統計會被一個無聲的錯誤污染。
    """
    n0 = float(z_orig.flatten().norm())
    if n0 == 0.0:
        raise ValueError("原圖 latent 的範數為零，編碼器有問題")
    return {
        "norm_orig": n0,
        "norm_def": float(z_def.flatten().norm()),
        "norm_ratio": float(z_def.flatten().norm()) / n0,
        "move_ratio": float((z_def - z_orig).flatten().norm()) / n0,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="擺著 <image>__<cond>__def.png 的目錄，會遞迴尋找")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", nargs="+", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    from src.models.ip2p import IP2PWrapper

    ip2p = IP2PWrapper(dtype=torch.float32)
    rows: List[Dict] = []
    keep = set(args.images) if args.images else None
    for path in sorted(args.src.rglob("*__def.png")):
        m = re.match(r"^(.*)__([^_].*?)__def$", path.stem)
        if not m:
            continue
        image, cond = m.group(1), m.group(2)
        if keep and image not in keep:
            continue
        orig = args.data / image / f"{image}.png"
        if not orig.exists():
            continue
        with torch.no_grad():
            z0 = ip2p.encode_image(
                load_image_tensor(orig, ip2p.device, size=RESOLUTION))
            z1 = ip2p.encode_image(
                load_image_tensor(path, ip2p.device, size=RESOLUTION))
        r = ratios(z0, z1)
        rows.append({
            "image": image,
            "condition": path.parent.name,
            "norm_orig": round(r["norm_orig"], 4),
            "norm_def": round(r["norm_def"], 4),
            "norm_ratio": round(r["norm_ratio"], 5),
            "move_ratio": round(r["move_ratio"], 5),
        })
        print(f"  {image[:34]:34s} {path.parent.name:14s} "
              f"norm {r['norm_ratio']:.4f}  move {r['move_ratio']:.4f}",
              flush=True)
    if not rows:
        raise SystemExit(f"{args.src} 底下沒有找到 __def.png")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    print(f"\n表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
