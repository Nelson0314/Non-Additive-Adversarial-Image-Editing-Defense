"""為資料集產生 inpainting／主體分區用的遮罩：`<data>/masks/<影像>.png`、`overview/<類別>.png`、`provenance.json`。

遮罩定義見 `immunization_baseline.subject_masks`。`overview/` 每列三欄為原圖、主體著色疊圖、二值遮罩，
供逐張目視檢查。CLIPSeg 為 rd64，以 CPU 執行。

用法
    python -m immunization_baseline.cli.generate_subject_masks --data-root data/pexels_portraits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml

from immunization_baseline.subject_masks import CLIPSEG_REPO, DILATE, THRESHOLD, subject_mask
from immunization_core.io import load_image_tensor

RESOLUTION = 512
TINT = (1.0, 0.25, 0.25)
TINT_ALPHA = 0.45


def load_subjects(root: Path) -> list:
    """只讀各類別的 `content`，不讀編輯指令。"""
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        entry = spec[cls]
        if not isinstance(entry, dict) or "content" not in entry:
            continue
        for img in sorted((root / cls).glob("*.png")):
            out.append({"name": img.stem, "class": cls, "path": img, "content": entry["content"]})
    if not out:
        raise SystemExit(f"{root} 底下找不到任何影像，或 prompts.yaml 缺 content")
    return out


def overview_row(x01, subject, mask):
    tint = torch.tensor(TINT, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    alpha = subject * TINT_ALPHA
    return torch.cat([x01, x01 * (1.0 - alpha) + tint * alpha, mask.repeat(1, 3, 1, 1)], dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", dest="data", type=Path, required=True)
    ap.add_argument("--overview-side", type=int, default=256)
    args = ap.parse_args()

    out = args.data / "masks"
    (out / "overview").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    rows, records = {}, []
    for item in load_subjects(args.data):
        x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
        subject = subject_mask(x01, item["content"], device=device)
        mask = 1.0 - subject
        path = out / f"{item['name']}.png"
        vutils.save_image(mask.clamp(0, 1), path)
        records.append({
            "name": item["name"], "image": path.parent.parent.joinpath(
                item["class"], f"{item['name']}.png").as_posix(),
            "mask": path.as_posix(), "text": item["content"], "model": CLIPSEG_REPO,
            "threshold": THRESHOLD, "dilate": DILATE,
            "subject_area": round(float(subject.mean()), 5),
            "repaint_area": round(float(mask.mean()), 5),
        })
        side = args.overview_side
        rows.setdefault(item["class"], []).append(F.interpolate(
            overview_row(x01, subject, mask), size=(side, 3 * side),
            mode="bilinear", align_corners=False))
        print(f"{item['name']}: 主體 {subject.mean():.3f}", flush=True)
    for cls, rs in rows.items():
        vutils.save_image(torch.cat(rs, dim=-2).clamp(0, 1), out / "overview" / f"{cls}.png")
    (out / "provenance.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
