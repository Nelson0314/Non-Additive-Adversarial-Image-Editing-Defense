"""SDEdit 編輯（img2img）— SPEC.md §2.8、STRUCTURE.md §2.2。

使用 diffusers 之 img2img pipeline（即 SDEdit 之實作）。
"""

import torch
import torchvision.transforms as T

from src.utils.device import get_device


def sdedit_edit(sd, image: torch.Tensor, prompt: str, seed: int, config: dict) -> torch.Tensor:
    """(1,3,H,W) [0,1] → 編輯結果 (1,3,H,W) [0,1]。"""
    gen_cfg = config["generation"]
    strength = config["edit"]["sdedit_strength"]
    if strength is None:
        raise ValueError(
            "edit.sdedit_strength 未設定（SPEC §2.8 [待確認]，須向 DAYN 作者索取）；"
            "請於 config 明確提供"
        )
    generator = torch.Generator(device=str(get_device())).manual_seed(seed)
    pil = T.ToPILImage()(image[0].clamp(0, 1).cpu())
    out = sd.img2img(
        prompt=prompt,
        image=pil,
        strength=strength,
        guidance_scale=gen_cfg["guidance_scale"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        eta=gen_cfg["eta"],
        generator=generator,
        output_type="pt",
    )
    return out.images.detach()
