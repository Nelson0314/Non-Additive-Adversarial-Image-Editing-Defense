"""AdverseCleaner（= BF+GF）— SPEC.md §5.1（lllyasviel/AdverseCleaner）。

cv2 bilateral filter（多次迭代）+ cv2.ximgproc.guidedFilter，
以受保護影像本身作 guidance。作者自述此設計不安全（guidance 已含
對抗噪聲，guided filter 可能把噪聲帶回），故實作兩變體：
- "bf_only"：僅 bilateral filter
- "bf_gf"：完整 BF+GF
參數尺度依官方（0–255 像素域）：d=5、sigma_color=8、sigma_space=8、
gf_radius=4、gf_eps=16（configs/purify.yaml）。
"""

import numpy as np
import torch

import cv2

DEFAULTS = {
    "bf_iterations": 3,
    "bf_d": 5,
    "bf_sigma_color": 8,
    "bf_sigma_space": 8,
    "gf_radius": 4,
    "gf_eps": 16,
}


def adverse_clean(x: torch.Tensor, variant: str = "bf_gf", config: dict = None) -> torch.Tensor:
    """(1,3,H,W) [0,1] → 淨化結果，同形狀值域。variant ∈ {"bf_only","bf_gf"}。"""
    if variant not in ("bf_only", "bf_gf"):
        raise ValueError(f"未知 AdverseCleaner 變體: {variant}")
    cfg = {**DEFAULTS, **(config or {})}

    # 官方於 0–255 像素域運作（guided filter 之 eps=16 為該尺度）
    img = (x[0].detach().cpu().clamp(0, 1) * 255.0).permute(1, 2, 0).numpy().astype(np.float32)
    y = img.copy()
    for _ in range(cfg["bf_iterations"]):
        y = cv2.bilateralFilter(y, cfg["bf_d"], cfg["bf_sigma_color"], cfg["bf_sigma_space"])
    if variant == "bf_gf":
        y = cv2.ximgproc.guidedFilter(guide=img, src=y, radius=cfg["gf_radius"], eps=cfg["gf_eps"])

    out = torch.from_numpy(np.clip(y, 0, 255) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return out.to(device=x.device, dtype=x.dtype)
