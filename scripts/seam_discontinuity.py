#!/usr/bin/env python
"""接縫不連續度：inpainting 特有的讀出量，量生成內容與脈絡接不接得起來。

    python scripts/seam_discontinuity.py --batch runs/ip3_merged \
        --images bird_02 cat_01 horse_02 --mask-dir runs/ip3_bird_02/masks \
        --out runs/ip3_seam.csv

只讀既有的 PNG，不跑任何模型，**不佔 GPU**。

## 為什麼需要這個量

`HANDOVER_METRICS_2026-08-08` §6.2：inpainting 的結構優勢在於「生成內容必須
與未遮罩的脈絡一致」，其症狀是**遮罩邊界的不連續**與**遮罩內容與脈絡矛盾**，
不是整張圖的對齊度下降。前一項可以直接量，本腳本量的就是它。

三層判準（§4）裡這一項屬於第 1 層（位移）的補充：它與 PSNR／LPIPS 一樣不看
語意，但它只看邊界，而那正是 img2img 沒有的地方。

## 量的定義

    ring   := dilate(mask, k) ∧ ¬erode(mask, k)     寬 2k 的邊界帶
    g      := |∇I| 的逐像素大小（灰階，前向差分）
    seam   := mean(g[ring]) / mean(g[¬ring])

分母是**同一張圖**遮罩帶以外的平均梯度，用來吸收「這張圖本來就比較銳利」
這個逐影像差異；比值因此可跨影像比較。接得起來時邊界帶與其餘部分沒有系統性
差異，比值接近 1；接不起來時邊界上出現一圈階梯，比值明顯大於 1。

**判讀方向**：防禦有效 → 攻擊方產生的內容與脈絡矛盾 → `seam` **升高**。
故要看的是「防禦側減對照側」，不是絕對值。單獨的絕對值不可解讀——遮罩邊界
本來就常落在物件輪廓上，那裡的梯度天生就高。

`k` 取 2（邊界帶寬 4 px）。取太寬會把物件內部的紋理算進來而稀釋訊號，取 1
則對遮罩下採樣造成的鋸齒過度敏感（`mask_latents` 用最近鄰）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment.executors import load_image_tensor  # noqa: E402

# `edit_pngs` 與 `meta_for` 直接借用 `class_margin`，不另寫一份：那兩個函式
# 承載的是 (τ, seed) 的配對規則，而該規則正是 2026-08-09 修掉的缺陷所在
# （四個 τ 塌成一個鍵）。同一條規則有兩份實作就會有一份先腐爛。
_SPEC = importlib.util.spec_from_file_location(
    "class_margin", ROOT / "scripts" / "class_margin.py")
_CM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CM)

FIELDS = ["batch", "image_id", "condition", "purify_dir", "seed", "tau",
          "png", "ring_px", "g_ring", "g_rest", "seam"]

RING_K = 2


def boundary_ring(mask: torch.Tensor, k: int = RING_K) -> torch.Tensor:
    """(1,1,H,W) 的布林邊界帶：膨脹減侵蝕。

    以 `max_pool2d` 做膨脹、以 `-max_pool2d(-x)` 做侵蝕，不引入新相依。
    """
    m = mask.float()
    ker = 2 * k + 1
    dil = F.max_pool2d(m, ker, stride=1, padding=k)
    ero = -F.max_pool2d(-m, ker, stride=1, padding=k)
    return (dil - ero) > 0.5


def grad_mag(x01: torch.Tensor) -> torch.Tensor:
    """(1,1,H,W) 的梯度大小。灰階前向差分，邊界補零使形狀不變。"""
    g = x01.mean(dim=1, keepdim=True)
    dx = torch.zeros_like(g)
    dy = torch.zeros_like(g)
    dx[..., :, :-1] = g[..., :, 1:] - g[..., :, :-1]
    dy[..., :-1, :] = g[..., 1:, :] - g[..., :-1, :]
    return (dx.pow(2) + dy.pow(2)).sqrt()


def seam_ratio(x01: torch.Tensor, ring: torch.Tensor) -> Dict[str, float]:
    g = grad_mag(x01)
    inside = g[ring]
    outside = g[~ring]
    a, b = float(inside.mean()), float(outside.mean())
    return {"g_ring": a, "g_rest": b,
            "seam": (a / b if b > 0 else float("nan"))}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--mask-dir", type=Path, nargs="+", required=True,
                    help="遮罩目錄，可給多個（逐分片各有自己那張）")
    ap.add_argument("--conditions", nargs="+",
                    default=["control", "N1", "N2", "N3", "R",
                             "photoguard_c", "mist", "dia_r"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cpu")
    rows: List[Dict] = []

    for image_id in args.images:
        mp = next((d / f"{image_id}_mask.png" for d in args.mask_dir
                   if (d / f"{image_id}_mask.png").exists()), None)
        if mp is None:
            raise SystemExit(
                f"{image_id}：在 {[str(d) for d in args.mask_dir]} 裡找不到遮罩。"
                "不猜一個矩形——接縫的位置完全由遮罩決定")
        mask = (load_image_tensor(mp, dev)[:, :1] > 0.5)
        ring = boundary_ring(mask)
        n_ring = int(ring.sum())
        print(f"{image_id}: 邊界帶 {n_ring} px "
              f"（{100 * n_ring / ring.numel():.1f}%）", flush=True)

        for cond in args.conditions:
            root = args.batch / cond / image_id / "purify"
            if not root.is_dir():
                continue
            for cdir in sorted(root.glob("*")):
                for tau, seed, png in _CM.edit_pngs(cdir):
                    m = _CM.meta_for(cdir, tau, seed)
                    x = load_image_tensor(png, dev)
                    rows.append({
                        "batch": args.batch.name, "image_id": image_id,
                        "condition": cond, "purify_dir": cdir.name,
                        "seed": seed,
                        "tau": tau if tau is not None else m.get("tau", ""),
                        "png": png.relative_to(args.batch).as_posix(),
                        "ring_px": n_ring,
                        **seam_ratio(x, ring),
                    })
            print(f"  {cond}: 累計 {len(rows)} 列", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n寫入 {args.out}（{len(rows)} 列）")

    # 摘要：identity 上「防禦側 − 對照側」的接縫比，逐 τ。絕對值不可解讀。
    base = {}
    for r in rows:
        if r["condition"] == "control" and r["purify_dir"] == _CM.IDENTITY_DIR:
            base.setdefault(r["image_id"], []).append(r["seam"])
    base = {k: sum(v) / len(v) for k, v in base.items() if v}
    if not base:
        print("\n（沒有對照側，接縫比的絕對值不可解讀，不出摘要）")
        return

    agg: Dict = {}
    for r in rows:
        if r["condition"] == "control" or r["purify_dir"] != _CM.IDENTITY_DIR:
            continue
        agg.setdefault((str(r["tau"]), r["condition"]), []).append(
            r["seam"] - base.get(r["image_id"], float("nan")))
    print("\nidentity 上的接縫比相對對照側（正值＝更接不起來）")
    for (tau, cond), v in sorted(agg.items()):
        print(f"  τ={tau:<6} {cond:<14} {sum(v) / len(v):+.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
