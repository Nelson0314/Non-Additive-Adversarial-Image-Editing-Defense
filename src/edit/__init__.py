"""編輯（攻擊）模組：SDEdit（img2img）、inpainting、seed 協定。

統一入口 edit_image()（STRUCTURE.md §2.2；另加 sd 參數傳入 SDWrapper，
原介面未含模型握把，為必要補充）。
"""

import torch

from src.edit.inpaint import inpaint_edit
from src.edit.sdedit import sdedit_edit


def edit_image(
    image: torch.Tensor,        # (1,3,H,W)，[0,1]
    prompt: str,
    seed: int,
    method: str,                # "sdedit" | "inpaint"
    mask: torch.Tensor = None,  # inpaint 時必要，(1,1,H,W)，[0,1]
    config: dict = None,
    *,
    sd=None,                    # SDWrapper
) -> torch.Tensor:
    """對影像執行編輯，回傳 (1,3,H,W)，[0,1]。

    生成超參數依 SPEC §2.2（config["generation"]）：
        guidance_scale=7.5, num_inference_steps=100, eta=1
    SDEdit 的 strength 由 config["edit"]["sdedit_strength"] 提供（SPEC §2.8 待確認）。
    """
    if method == "sdedit":
        return sdedit_edit(sd, image, prompt, seed, config)
    if method == "inpaint":
        if mask is None:
            raise ValueError("inpaint 需提供 mask")
        return inpaint_edit(sd, image, mask, prompt, seed, config)
    raise ValueError(f"未知編輯方法: {method}")
