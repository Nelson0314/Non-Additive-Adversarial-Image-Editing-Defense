"""輕量淨化 — SPEC.md §5.2（Pixel is a Barrier 附錄 D；DiffusionGuard 附錄 F）。

- jpeg_compress(x, quality=65)：標準設定 quality 65
- gaussian_blur(x, sigma)：sigma ∈ {0.5, 1.0, 1.5}
- crop_resize(x, ratio=0.2)：中心裁切（去除 ratio 比例邊緣）後放大回原尺寸

輸入輸出一律 (1,3,H,W)、值域 [0,1]。全部為決定性操作。
"""

import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) [0,1] → (H,W,3) uint8。"""
    arr = (x[0].detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    return arr.permute(1, 2, 0).numpy()


def _from_uint8_hwc(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """(H,W,3) uint8 → (1,3,H,W) [0,1]，裝置與 dtype 同 like。"""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=like.device, dtype=like.dtype)


def jpeg_compress(x: torch.Tensor, quality: int = 65) -> torch.Tensor:
    pil = Image.fromarray(_to_uint8_hwc(x))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"))
    return _from_uint8_hwc(out, x)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    arr = _to_uint8_hwc(x).astype(np.float32)
    # ksize=(0,0)：由 sigma 自動決定核大小（cv2 慣例）
    out = cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return _from_uint8_hwc(np.clip(out, 0, 255).round().astype(np.uint8), x)


def crop_resize(x: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    h, w = x.shape[-2:]
    ch, cw = round(h * (1.0 - ratio)), round(w * (1.0 - ratio))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = x[..., top : top + ch, left : left + cw]
    out = F.interpolate(cropped, size=(h, w), mode="bicubic", align_corners=False)
    return out.clamp(0, 1)
