"""以文字指出主體的二值遮罩（CLIPSeg），供資料集的 `masks/` 使用。

移植自 `archive/anti-purification/src/defense/subject_mask.py` 與
`archive/anti-purification/scripts/make_masks.py`，只保留產生 `data/portraits/masks/` 時實際使用的路徑：
`feather = 0`、`dilate = 4`、`threshold = 0.30`。數值與 `data/portraits/masks/provenance.json` 逐欄相同。

- CLIPSeg：`CIDAS/clipseg-rd64-refined`（arXiv:2112.10003），文字取 `prompts.yaml` 各類別的 `content`。
- 門檻後以最大池化做圓形膨脹 `dilate` 像素，作為主體與重繪區之間的安全邊。
- 主體面積落在 `[MIN_AREA, MAX_AREA]` 之外時拋錯。
- 輸出採 Stable Diffusion inpainting 的極性：白 = 重繪 = 主體以外，黑 = 保留 = 主體。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CLIPSEG_REPO = "CIDAS/clipseg-rd64-refined"
THRESHOLD = 0.30
DILATE = 4
MIN_AREA = 0.01
MAX_AREA = 0.80

_MODEL = None
_PROC = None


def _load(device):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        _PROC = CLIPSegProcessor.from_pretrained(CLIPSEG_REPO)
        _MODEL = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_REPO).to(device).eval()
    return _MODEL, _PROC


def _dilate(m: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return m
    return F.max_pool2d(m, kernel_size=2 * radius + 1, stride=1, padding=radius)


@torch.no_grad()
def _probability(x01: torch.Tensor, text: str, device) -> torch.Tensor:
    from PIL import Image
    model, proc = _load(device)
    h, w = x01.shape[-2:]
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    inputs = proc(text=[text], images=[Image.fromarray(arr.astype("uint8"))],
                  padding=True, return_tensors="pt")
    logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
    if logits.dim() == 2:
        logits = logits[None, None]
    elif logits.dim() == 3:
        logits = logits[:, None]
    return F.interpolate(torch.sigmoid(logits.float()), size=(h, w),
                         mode="bilinear", align_corners=False)


@torch.no_grad()
def subject_mask(x01: torch.Tensor, text: str, threshold: float = THRESHOLD,
                 dilate: int = DILATE, device=None) -> torch.Tensor:
    """(1,3,H,W) → (1,1,H,W) 的二值主體遮罩，1 = 主體。"""
    dev = device or x01.device
    hard = (_probability(x01, text, dev) >= threshold).to(x01.dtype)
    hard = _dilate(hard, dilate)
    area = float(hard.mean())
    if not MIN_AREA <= area <= MAX_AREA:
        raise ValueError(f"主體遮罩面積 {area:.3f} 落在 [{MIN_AREA}, {MAX_AREA}] 之外（文字 {text!r}）")
    return hard.to(device=x01.device, dtype=x01.dtype)
