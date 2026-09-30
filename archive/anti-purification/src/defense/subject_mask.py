"""用**文字**指出受保護的主體，得到一個跟著內容走的遮罩。

本模組換掉那個矩形：以 CLIPSeg（`CIDAS/clipseg-rd64-refined`，
[arXiv:2112.10003](https://arxiv.org/abs/2112.10003)）用一句文字指出主體，
得到跟著內容走的軟遮罩。文字就是 `data/decoy_catalogue.yaml` 的 `objects`
登記的主體名稱——**威脅模型的前提是防護對象已知**，攻擊指令不是。

四個本專案指定的參數（論文無出處，故是 CSV 欄位不是註解）
────────────────────────────────────────────────────────────────────
CLIPSeg 的輸出是 352² 的 logit，要變成「保留哪裡」還需要四個決定：

- `threshold`：sigmoid 的門檻。**刻意取低**（0.30）——這個遮罩的用途是
  「不要動到主體」，漏掉主體的一部分比多保留一點背景嚴重得多。
- `dilate`：門檻之後往外膨脹幾個像素。分割的邊緣會切在物體輪廓上，
  而編輯模型會沿著輪廓外側改動。
- `feather`：膨脹之後再往外羽化幾個像素。**羽化只往外**，故膨脹範圍內恆為 1，
  主體逐位元保留。
- `min_area` / `max_area`：面積的可用區間。太小表示文字沒有指到東西，
  太大表示整張圖都被判成主體、誘餌無處可放。**兩者都拋錯而不是靜默接受**
  ——一個空遮罩會讓「合成」退化成「整張換掉」，而那正是要防的失效模式。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

CLIPSEG_REPO = "CIDAS/clipseg-rd64-refined"

# 本專案指定的預設值。改動要進 CSV。
THRESHOLD = 0.30
DILATE = 16
FEATHER = 24
MIN_AREA = 0.01
MAX_AREA = 0.80

_MODEL = None
_PROC = None


def _load(device):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        _PROC = CLIPSegProcessor.from_pretrained(CLIPSEG_REPO)
        _MODEL = CLIPSegForImageSegmentation.from_pretrained(
            CLIPSEG_REPO).to(device).eval()
    return _MODEL, _PROC


def _dilate(m: torch.Tensor, radius: int) -> torch.Tensor:
    """圓形結構元素的膨脹，用最大池化實作。`radius = 0` 時是恆等。"""
    if radius <= 0:
        return m
    k = 2 * radius + 1
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)


def _feather(m: torch.Tensor, width: int) -> torch.Tensor:
    """往外羽化。`m` 為 0/1；回傳 [0,1]，**原本為 1 的地方仍是 1**。

    作法是對硬遮罩做 `width` 次半徑 1 的膨脹並累加，等價於一個線性衰減的
    距離場。用膨脹而不是高斯模糊，是因為模糊會把邊界內側也拉下來，
    那會破壞「框內逐位元保留」。
    """
    if width <= 0:
        return m
    acc = m.clone()
    cur = m
    for _ in range(width):
        cur = _dilate(cur, 1)
        acc = acc + cur
    return (acc / (width + 1)).clamp(0.0, 1.0)


@torch.no_grad()
def _probability(x01: torch.Tensor, text: str, device) -> torch.Tensor:
    """單一句文字的逐像素機率，已放大回輸入尺寸。"""
    from PIL import Image

    model, proc = _load(device)
    h, w = x01.shape[-2:]
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    pil = Image.fromarray(arr.astype("uint8"))
    inputs = proc(text=[text], images=[pil], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    if logits.dim() == 2:
        logits = logits[None, None]
    elif logits.dim() == 3:
        logits = logits[:, None]
    prob = torch.sigmoid(logits.float())
    return F.interpolate(prob, size=(h, w), mode="bilinear",
                         align_corners=False)


@torch.no_grad()
def subject_mask(x01: torch.Tensor, text,
                 threshold: float = THRESHOLD,
                 dilate: int = DILATE,
                 feather: int = FEATHER,
                 min_area: float = MIN_AREA,
                 max_area: float = MAX_AREA,
                 device=None) -> torch.Tensor:
    """(1,3,H,W) ＋ 一句或多句文字 → (1,1,H,W)，1 = 主體（要保留）。

    面積落在 `[min_area, max_area]` 之外時**拋錯**：空遮罩會讓合成退化成
    「整張換掉」，滿遮罩會讓誘餌無處可放，兩者都不該靜默通過。
    """
    dev = device or x01.device
    texts = [text] if isinstance(text, str) else list(text)
    if not texts:
        raise ValueError("至少要給一句描述主體的文字")
    hard = None
    for t in texts:
        h1 = (_probability(x01, t, dev) >= threshold).to(x01.dtype)
        hard = h1 if hard is None else torch.maximum(hard, h1)
    hard = _dilate(hard, dilate)
    area = float(hard.mean())
    if not (min_area <= area <= max_area):
        raise ValueError(
            f"主體遮罩的面積 {area:.3f} 落在 [{min_area}, {max_area}] 之外"
            f"（文字：{text!r}）。太小表示文字沒有指到東西，太大表示整張圖都被"
            f"判成主體、誘餌無處可放。**不靜默接受**。")
    return _feather(hard, feather).to(device=x01.device, dtype=x01.dtype)


def mask_stats(m: torch.Tensor) -> dict:
    """供 CSV 逐列記下的三個數：硬核面積、羽化後的平均、邊界寬度佔比。"""
    core = float((m >= 1.0).to(m.dtype).mean())
    return {
        "mask_core_area": round(core, 5),
        "mask_mean": round(float(m.mean()), 5),
        "mask_soft_band": round(float(m.mean()) - core, 5),
    }
