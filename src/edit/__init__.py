"""編輯（攻擊）模組：SDEdit（img2img）、inpainting、seed 協定。

統一入口 edit_image()（STRUCTURE.md §2.2；另加 sd 參數傳入 SDWrapper，
原介面未含模型握把，為必要補充）。

v5 新增 edit_image_batch()：純推論無梯度，依 runtime.edit_batch_size 分塊批次
執行。每樣本各自 generator+seed，與 batch=1 逐位元一致（等價性由
tests/test_edit_batching.py 驗證；TWCC 首次執行須以真實模型重跑該測試，
不通過則 edit_batch_size 改回 1）。
"""

import torch

from src.edit.inpaint import inpaint_edit, inpaint_edit_batch
from src.edit.sdedit import sdedit_edit, sdedit_edit_batch


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


def edit_image_batch(
    images: list,               # list of (1,3,H,W)，[0,1]
    prompts: list,              # 同長度
    seeds: list,                # 同長度；每樣本各自 generator 以此 seed 初始化
    method: str,                # "sdedit" | "inpaint"
    masks: list = None,         # inpaint 時必要，list of (1,1,H,W)
    config: dict = None,
    *,
    sd=None,
) -> list:
    """批次編輯：依 runtime.edit_batch_size 分塊，回傳與輸入同長度之
    list of (1,3,H,W) [0,1]，順序對應。"""
    if not (len(images) == len(prompts) == len(seeds)):
        raise ValueError("images / prompts / seeds 長度須一致")
    bs = max(1, int(config["runtime"].get("edit_batch_size", 1)))
    outs = []
    for i in range(0, len(images), bs):
        j = i + bs
        if method == "sdedit":
            outs += sdedit_edit_batch(sd, images[i:j], prompts[i:j], seeds[i:j], config)
        elif method == "inpaint":
            if masks is None:
                raise ValueError("inpaint 需提供 masks")
            outs += inpaint_edit_batch(
                sd, images[i:j], masks[i:j], prompts[i:j], seeds[i:j], config)
        else:
            raise ValueError(f"未知編輯方法: {method}")
    return outs
