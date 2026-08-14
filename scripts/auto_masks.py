"""用實例分割自動產生 inpainting 遮罩，取代手工繪製。

AdvPaint（ICLR 2025）用 **Grounded SAM**：Grounding DINO 由文字偵測物件、
SAM 產分割遮罩，再取**放大的外接框**（ρ = 1.2）。本腳本做同一件事，但偵測
那一半不需要——每張影像要保護的類別已由 `prompts.yaml` 的 `content` 給定，
故直接用 torchvision 的 Mask R-CNN（COCO 預訓練，涵蓋 person／dog／cat／
horse／bird）取該類別分數最高的實例。**不引入新相依。**

產出兩種遮罩，對應文獻的兩種 inpainting 場景：

    subject      主體本身。前景重畫用：攻擊方把主體換掉
    background   主體之外（含 ρ 保護帶）。背景重畫用：主體不動、其餘重畫

`1` 一律表示**攻擊方要重畫的區域**，與 `SDWrapper.mask_latents` 同一約定。

保護帶 ρ 的意思與 AdvPaint 相同：把主體遮罩擴張後才當成「不可重畫」，
使主體邊緣不落在重畫區內——邊緣若被重畫，主體看起來就被動過了。

用法：
    python scripts/auto_masks.py --data data/lo_aligned --out data/lo_masks_auto
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torchvision.models.detection import (  # noqa: E402
    MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2,
)

from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

# `content` 是防禦方選擇要保護的詞（prompts.yaml），COCO 的類別名不完全同名。
CLASS_ALIAS = {"man": "person", "woman": "person", "dog": "dog",
               "cat": "cat", "horse": "horse", "bird": "bird"}
RESOLUTION = 512


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """以最大池化做形態學膨脹。`px` 為半徑（像素）。"""
    if px <= 0:
        return mask
    k = 2 * px + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=px)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--rho", type=float, default=1.2,
                    help="主體遮罩的放大倍率，AdvPaint 的 ρ。1.2 表示保護帶為"
                         "等效半徑的 20%%")
    ap.add_argument("--score-min", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load((args.data / "prompts.yaml").read_text(encoding="utf-8"))
    items = [{"name": p.stem, "content": spec[c]["content"], "path": p}
             for c in sorted(spec) for p in sorted((args.data / c).glob("*.png"))]
    if args.images:
        keep = set(args.images)
        items = [i for i in items if i["name"] in keep]

    w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    cats = w.meta["categories"]
    model = maskrcnn_resnet50_fpn_v2(weights=w).eval().to(args.device)

    rows = []
    for it in items:
        want = CLASS_ALIAS[it["content"]]
        if want not in cats:
            raise KeyError(f"COCO 沒有類別 {want!r}（來自 content={it['content']!r}）")
        target = cats.index(want)
        x = load_image_tensor(it["path"], args.device, size=RESOLUTION)
        with torch.no_grad():
            out = model(x)[0]

        sel = [(float(s), m) for s, lab, m in
               zip(out["scores"], out["labels"], out["masks"])
               if int(lab) == target and float(s) >= args.score_min]
        if not sel:
            best = max((float(s) for s, lab in zip(out["scores"], out["labels"])
                        if int(lab) == target), default=0.0)
            raise RuntimeError(
                f"{it['name']}：找不到分數 ≥ {args.score_min} 的 {want}"
                f"（最高 {best:.3f}）。此時不可退而求其次選別的類別——"
                f"遮罩定義的是威脅模型，選錯就換了一個實驗")
        # 同類別的多個實例全部併入：畫面裡有兩匹馬時，兩匹都是要保護的主體。
        # `out["masks"]` 是 (N,1,H,W)，逐張取出後是 (1,H,W)，補回 batch 軸。
        subject = torch.zeros_like(sel[0][1])[None]
        for _, m in sel:
            subject = torch.maximum(subject, m[None])
        subject = (subject > 0.5).to(x.dtype)

        area = float(subject.mean())
        # ρ 定義在等效半徑上：面積 A 的等效半徑 r = sqrt(A/π)，膨脹 (ρ−1)·r。
        r_eq = (area * RESOLUTION * RESOLUTION / 3.141592653589793) ** 0.5
        pad = int(round((args.rho - 1.0) * r_eq))
        guarded = (dilate(subject, pad) > 0.5).to(x.dtype)
        background = 1.0 - guarded

        save_image(subject.expand(-1, 3, -1, -1), args.out / f"{it['name']}__subject.png")
        save_image(background.expand(-1, 3, -1, -1),
                   args.out / f"{it['name']}__background.png")
        # 疊圖供人眼複驗：紅＝主體、藍＝保護帶
        ov = x.clone()
        ov[:, 0] = torch.maximum(ov[:, 0], subject[:, 0])
        ov[:, 2] = torch.maximum(ov[:, 2], (guarded - subject)[:, 0])
        save_image(ov, args.out / f"{it['name']}__overlay.png")

        row = {"image": it["name"], "content": it["content"], "coco_class": want,
               "n_instances": len(sel), "score_max": round(max(s for s, _ in sel), 4),
               "subject_area": round(area, 4),
               "guard_px": pad, "rho": args.rho,
               "repaint_area_background": round(float(background.mean()), 4),
               "repaint_area_subject": round(area, 4)}
        rows.append(row)
        print(row, flush=True)

    with (args.out / "masks.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "provenance.json").write_text(json.dumps({
        "method": "torchvision maskrcnn_resnet50_fpn_v2, COCO weights",
        "weights": str(w),
        "rho": args.rho, "score_min": args.score_min,
        "note": "AdvPaint 用 Grounded SAM（Grounding DINO 偵測 + SAM 分割，ρ=1.2）；"
                "此處類別由 prompts.yaml 的 content 給定，故偵測改為直接取該類別"
                "分數最高的實例，分割仍由實例分割模型產出",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} 張 → {args.out}")


if __name__ == "__main__":
    main()
