"""用實例分割自動產生 inpainting 遮罩，取代手工繪製。

AdvPaint（ICLR 2025）用 **Grounded SAM**：Grounding DINO 由文字偵測物件、
SAM 產分割遮罩，再取**放大的外接框**（ρ = 1.2）。本腳本做同一件事，但偵測
那一半不需要——每張影像要保護的類別已由 `prompts.yaml` 的 `content` 給定，
故直接用 torchvision 的 Mask R-CNN（COCO 預訓練，涵蓋 person／dog／cat／
horse／bird）取該類別的實例。**不引入新相依。**

四種遮罩，`1` 一律表示**攻擊方要重畫的區域**（與 `mask_latents` 同一約定）：

    subject      主體本身
    background   主體之外（含 ρ 保護帶）
    head         頭部**方框**，向上延伸留出戴帽子的空間
    torso        軀幹方框（僅 person），用於換衣服

`head` 與 `torso` 是**方框而非貼合輪廓**：要在主體上加東西（帽子、外套），
重畫區必須涵蓋主體之外的空間，貼著輪廓切就沒地方長出來。

person 的頭與軀幹由 Keypoint R-CNN 的 COCO 17 點決定（眼耳鼻定頭、肩髖定
軀幹）；動物沒有對應的關鍵點模型，頭部改取實例外接框的上緣一段。

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
    KeypointRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights,
    keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2,
)

from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

CLASS_ALIAS = {"man": "person", "woman": "person", "dog": "dog",
               "cat": "cat", "horse": "horse", "bird": "bird"}
RESOLUTION = 512
KP = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.meta["keypoint_names"]
# 頭框向上延伸的倍率（以頭框高為單位）。戴帽子需要頭頂之上的空間。
HEAD_UP = 1.1
# 動物的頭部取實例外接框上緣這個比例。
ANIMAL_HEAD_FRAC = 0.45


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return mask
    k = 2 * px + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=px)


def box_mask(like: torch.Tensor, y0, y1, x0, x1) -> torch.Tensor:
    m = torch.zeros_like(like)
    n = like.shape[-1]
    y0, y1 = max(0, int(y0)), min(n, int(y1))
    x0, x1 = max(0, int(x0)), min(n, int(x1))
    m[..., y0:y1, x0:x1] = 1.0
    return m


def bbox_of(mask: torch.Tensor):
    ys = torch.nonzero(mask[0, 0].sum(1) > 0).flatten()
    xs = torch.nonzero(mask[0, 0].sum(0) > 0).flatten()
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--rho", type=float, default=1.2)
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
    seg = maskrcnn_resnet50_fpn_v2(weights=w).eval().to(args.device)
    kpm = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(args.device)

    rows = []
    for it in items:
        want = CLASS_ALIAS[it["content"]]
        target = cats.index(want)
        x = load_image_tensor(it["path"], args.device, size=RESOLUTION)
        with torch.no_grad():
            out = seg(x)[0]
        sel = [(float(s), m) for s, lab, m in
               zip(out["scores"], out["labels"], out["masks"])
               if int(lab) == target and float(s) >= args.score_min]
        if not sel:
            raise RuntimeError(
                f"{it['name']}：找不到分數 ≥ {args.score_min} 的 {want}。"
                "遮罩定義的是威脅模型，選錯類別就換了一個實驗")
        subject = torch.zeros_like(sel[0][1])[None]
        for _, m in sel:
            subject = torch.maximum(subject, m[None])
        subject = (subject > 0.5).to(x.dtype)

        area = float(subject.mean())
        r_eq = (area * RESOLUTION * RESOLUTION / 3.141592653589793) ** 0.5
        pad = int(round((args.rho - 1.0) * r_eq))
        guarded = (dilate(subject, pad) > 0.5).to(x.dtype)
        y0, y1, x0, x1 = bbox_of(subject)

        head = torso = None
        if want == "person":
            with torch.no_grad():
                kp = kpm(x)[0]
            if len(kp["scores"]) and float(kp["scores"][0]) >= args.score_min:
                p = kp["keypoints"][0]          # (17, 3)
                idx = {n: i for i, n in enumerate(KP)}
                face = [p[idx[n]] for n in
                        ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
                fx = torch.stack([q[0] for q in face])
                fy = torch.stack([q[1] for q in face])
                hw = float(fx.max() - fx.min()) or 40.0
                hh = float(fy.max() - fy.min()) or 30.0
                cx = float(fx.mean())
                # 臉部關鍵點只涵蓋眼耳鼻，下巴與頭頂都在外，故縱向放寬。
                head = box_mask(subject, fy.min() - HEAD_UP * max(hh, hw),
                                fy.max() + 1.2 * max(hh, hw),
                                cx - 1.1 * max(hw, hh), cx + 1.1 * max(hw, hh))
                sh = [p[idx[n]] for n in ("left_shoulder", "right_shoulder")]
                hp = [p[idx[n]] for n in ("left_hip", "right_hip")]
                sy = float(min(q[1] for q in sh))
                hy = float(max(q[1] for q in hp))
                sx = [float(q[0]) for q in sh + hp]
                wd = max(sx) - min(sx)
                if hy <= sy:                     # 髖部在框外（半身像）
                    hy = min(y1, sy + 2.2 * wd)
                torso = box_mask(subject, sy - 0.12 * wd, hy + 0.10 * wd,
                                 min(sx) - 0.35 * wd, max(sx) + 0.35 * wd)
        if head is None:
            h = y1 - y0
            head = box_mask(subject, y0 - HEAD_UP * ANIMAL_HEAD_FRAC * h,
                            y0 + ANIMAL_HEAD_FRAC * h, x0, x1)

        out_masks = {"subject": subject, "background": 1.0 - guarded, "head": head}
        if torso is not None:
            out_masks["torso"] = torso
        for k, mk in out_masks.items():
            save_image(mk.expand(-1, 3, -1, -1), args.out / f"{it['name']}__{k}.png")

        ov = x.clone()
        ov[:, 0] = torch.maximum(ov[:, 0], subject[:, 0] * 0.75)
        ov[:, 1] = torch.maximum(ov[:, 1], head[:, 0])
        if torso is not None:
            ov[:, 2] = torch.maximum(ov[:, 2], torso[:, 0])
        save_image(ov, args.out / f"{it['name']}__overlay.png")

        row = {"image": it["name"], "content": it["content"], "coco_class": want,
               "n_instances": len(sel), "score_max": round(max(s for s, _ in sel), 4),
               "subject_area": round(area, 4), "guard_px": pad, "rho": args.rho,
               "area_background": round(float((1.0 - guarded).mean()), 4),
               "area_head": round(float(head.mean()), 4),
               "area_torso": round(float(torso.mean()), 4) if torso is not None else ""}
        rows.append(row)
        print(row, flush=True)

    with (args.out / "masks.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "provenance.json").write_text(json.dumps({
        "seg": "torchvision maskrcnn_resnet50_fpn_v2, COCO",
        "keypoints": "torchvision keypointrcnn_resnet50_fpn, COCO 17 點（person）",
        "rho": args.rho, "score_min": args.score_min,
        "head_up": HEAD_UP, "animal_head_frac": ANIMAL_HEAD_FRAC,
        "note": "head/torso 是方框而非輪廓：要在主體上加東西，重畫區必須涵蓋"
                "主體之外的空間",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} 張 → {args.out}")


if __name__ == "__main__":
    main()
