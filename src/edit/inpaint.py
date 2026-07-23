"""Inpainting 編輯 — SPEC.md §2.8。

給定 mask，僅重繪 mask 區域（mask=1 處重繪）。生成超參數同 SPEC §2.2。
使用與編輯相同之基礎模型（4-channel UNet，diffusers 以逐步 latent 混合處理）。

v5 批次化：同 sdedit——每樣本各自 generator+seed，與 batch=1 逐位元一致
（tests/test_edit_batching.py 驗證）。
"""

import torch
import torchvision.transforms as T

from src.utils.device import get_device


def inpaint_edit_batch(
    sd, images: list, masks: list, prompts: list, seeds: list, config: dict
) -> list:
    """批次 inpaint：list of (1,3,H,W) [0,1] + list of (1,1,H,W) [0,1] →
    同長度 list of (1,3,H,W) [0,1]。每樣本各自 generator+seed。"""
    gen_cfg = config["generation"]
    device = str(get_device())
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    to_pil = T.ToPILImage()
    pils = [to_pil(img[0].clamp(0, 1).cpu()) for img in images]
    mask_pils = [to_pil(m[0].clamp(0, 1).cpu()).convert("L") for m in masks]
    out = sd.inpaint(
        prompt=list(prompts),
        image=pils,
        mask_image=mask_pils,
        guidance_scale=gen_cfg["guidance_scale"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        eta=gen_cfg["eta"],
        generator=generators,
        output_type="pt",
    )
    return [im.unsqueeze(0).detach() for im in out.images]


def inpaint_edit(
    sd, image: torch.Tensor, mask: torch.Tensor, prompt: str, seed: int, config: dict
) -> torch.Tensor:
    """(1,3,H,W) [0,1] + mask (1,1,H,W) [0,1] → 編輯結果 (1,3,H,W) [0,1]。"""
    return inpaint_edit_batch(sd, [image], [mask], [prompt], [seed], config)[0]
