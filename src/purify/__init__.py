"""淨化方法統一入口 — STRUCTURE.md §2.4。流程為「淨化 → 編輯 → 測劣化」。

purify() 之 strength 依方法而異（configs/purify.yaml）：
- "jpeg": quality（int）
- "blur": sigma（float）
- "crop_resize": ratio（float）
- "advclean_bf" / "advclean_bfgf": 無強度參數（strength 忽略），濾波參數由 config 提供
- "gridpure": {"pure_steps": int, "iterations": int}，並需以 purifier 參數注入
  淨化器（原介面未含模型握把，為必要補充，同 edit_image 之 sd）
"""

import torch

from src.purify.adverse_cleaner import adverse_clean
from src.purify.gridpure import grid_pure
from src.purify.lightweight import crop_resize, gaussian_blur, jpeg_compress


def purify(
    image: torch.Tensor,
    method: str,
    strength=None,
    *,
    config: dict = None,
    purifier=None,
) -> torch.Tensor:
    """(1,3,H,W) [0,1] → 淨化結果，同形狀值域。"""
    if method == "jpeg":
        return jpeg_compress(image, quality=strength if strength is not None else 65)
    if method == "blur":
        if strength is None:
            raise ValueError("blur 需提供 sigma（strength）")
        return gaussian_blur(image, sigma=strength)
    if method == "crop_resize":
        return crop_resize(image, ratio=strength if strength is not None else 0.2)
    if method == "advclean_bf":
        return adverse_clean(image, variant="bf_only", config=config)
    if method == "advclean_bfgf":
        return adverse_clean(image, variant="bf_gf", config=config)
    if method == "gridpure":
        if purifier is None:
            raise ValueError("gridpure 需以 purifier 參數注入淨化器（見 gridpure.py）")
        s = strength or {}
        if "pure_steps" in s and hasattr(purifier, "set_pure_steps"):
            purifier.set_pure_steps(s["pure_steps"])
        cfg = (config or {}).get("gridpure", {})
        return grid_pure(
            image,
            purifier,
            grid_size=cfg.get("grid_size", 256),
            stride=cfg.get("stride", 128),
            gamma=cfg.get("gamma", 0.1),
            iterations=s.get("iterations", 20),
        )
    raise ValueError(f"未知淨化方法: {method}")
