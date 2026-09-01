"""AdvDrop 原生威脅模型用的 ImageNet 驗證影像。

論文（arXiv:2108.09034 §4.1）的設定是「randomly selected 2000 **correctly
classified** images from ImageNet」，目標模型 ResNet-50。**「已被正確分類」
這個過濾不是細節**——攻擊成功率的定義是 `f(x) ≠ f(x')`，本來就分錯的影像會
讓分母失去意義，故此處照做並把過濾前後的張數都寫進 provenance。

資料來源：Hugging Face 的 `mrm8488/ImageNet1K-val`（ungated，parquet）。
**這不是官方 ILSVRC 下載點**，官方點需要帳號授權；本專案不把任何 token 寫進
入庫檔案，故取用鏡像。鏡像的內容是否逐位元等於官方 val split 未經查證，
報表上要寫明用的是鏡像。

前處理照 torchvision 的 ImageNet 慣例：Resize(256) → CenterCrop(224)。
224 是 8 的倍數，AdvDrop 的 8×8 分塊直接可用。
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image

REPO = "mrm8488/ImageNet1K-val"
SHARD = "data/train-00000-of-00014.parquet"   # 檔名寫 train，內容是 val split
RESIZE, CROP = 256, 224


def _resnet50(device):
    from torchvision.models import ResNet50_Weights, resnet50

    w = ResNet50_Weights.IMAGENET1K_V1
    m = resnet50(weights=w).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, w.meta["categories"]


def _preprocess(img: Image.Image) -> torch.Tensor:
    """Resize(256) → CenterCrop(224) → [0,1] 張量。**不做 normalize**——
    AdvDrop 在 [0,1]（內部 [0,255]）上工作，normalize 屬於分類器那一側。"""
    w, h = img.size
    s = RESIZE / min(w, h)
    img = img.convert("RGB").resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    img = img.crop((l, t, l + CROP, t + CROP))
    import numpy as np
    a = np.asarray(img, dtype="float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/imagenet_advdrop"))
    ap.add_argument("--n", type=int, default=200,
                    help="要保留的『已被正確分類』張數。論文用 2000")
    ap.add_argument("--scan", type=int, default=1200, help="最多掃幾列")
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, SHARD, repo_type="dataset")
    tbl = pq.read_table(path)
    cols = tbl.column_names
    print(f"parquet 欄位：{cols}，共 {tbl.num_rows} 列")
    icol = next((c for c in cols if "image" in c.lower() or c == "img"), None)
    lcol = next((c for c in cols if "label" in c.lower()), None)
    if icol is None or lcol is None:
        raise SystemExit(f"找不到影像／標籤欄位，實際欄位是 {cols}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, categories = _resnet50(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(tbl.num_rows, generator=g)[:args.scan].tolist()

    args.out.mkdir(parents=True, exist_ok=True)
    kept, seen = [], 0
    buf_x, buf_y, buf_i = [], [], []

    def flush():
        nonlocal buf_x, buf_y, buf_i
        if not buf_x:
            return
        x = torch.stack(buf_x).to(device)
        with torch.no_grad():
            pred = model((x - mean) / std).argmax(1).cpu()
        for xi, yi, ri, pi in zip(buf_x, buf_y, buf_i, pred.tolist()):
            if pi != yi or len(kept) >= args.n:
                continue
            name = f"in{len(kept):04d}_r{ri}"
            Image.fromarray(
                (xi.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            ).save(args.out / f"{name}.png")
            kept.append({"name": name, "row": ri, "label": yi,
                         "category": categories[yi]})
        buf_x, buf_y, buf_i = [], [], []

    for ri in order:
        if len(kept) >= args.n:
            break
        rec = tbl.slice(ri, 1).to_pylist()[0]
        raw = rec[icol]
        data = raw["bytes"] if isinstance(raw, dict) else raw
        try:
            img = Image.open(io.BytesIO(data))
            buf_x.append(_preprocess(img))
        except Exception as e:                # 壞檔就跳過，並記在 provenance
            print(f"  第 {ri} 列解不開：{e}")
            continue
        buf_y.append(int(rec[lcol]))
        buf_i.append(int(ri))
        seen += 1
        if len(buf_x) >= 32:
            flush()
    flush()

    prov = {"repo": REPO, "shard": SHARD, "note": "非官方 ILSVRC 下載點，鏡像",
            "preprocess": f"Resize({RESIZE}) → CenterCrop({CROP})",
            "model": "torchvision resnet50 IMAGENET1K_V1",
            "filter": "只保留已被正確分類者（論文 §4.1）",
            "scanned": seen, "kept": len(kept), "seed": args.seed,
            "images": kept}
    (args.out / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    acc = len(kept) / seen if seen else 0.0
    print(f"掃了 {seen} 張、保留 {len(kept)} 張（乾淨正確率約 {acc:.3f}）→ {args.out}")


if __name__ == "__main__":
    main()
