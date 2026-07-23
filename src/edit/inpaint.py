"""Inpainting 編輯 — SPEC.md §2.8。

給定 mask，僅重繪 mask 區域（mask=1 處重繪）。生成超參數同 SPEC §2.2。
使用與編輯相同之基礎模型（4-channel UNet，diffusers 以逐步 latent 混合處理）。
"""

import torch
import torchvision.transforms as T

from src.utils.device import get_device


def inpaint_edit(
    sd, image: torch.Tensor, mask: torch.Tensor, prompt: str, seed: int, config: dict
) -> torch.Tensor:
    """(1,3,H,W) [0,1] + mask (1,1,H,W) [0,1] → 編輯結果 (1,3,H,W) [0,1]。"""
    gen_cfg = config["generation"]
    generator = torch.Generator(device=str(get_device())).manual_seed(seed)
    to_pil = T.ToPILImage()
    pil = to_pil(image[0].clamp(0, 1).cpu())
    mask_pil = to_pil(mask[0].clamp(0, 1).cpu()).convert("L")
    out = sd.inpaint(
        prompt=prompt,
        image=pil,
        mask_image=mask_pil,
        guidance_scale=gen_cfg["guidance_scale"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        eta=gen_cfg["eta"],
        generator=generator,
        output_type="pt",
    )
    return out.images.detach()
