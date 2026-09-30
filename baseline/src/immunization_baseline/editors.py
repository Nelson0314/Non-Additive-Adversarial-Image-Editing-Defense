"""主表探索用的 SD 系列編輯器 pipeline 設定與影像轉換。"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

EDITORS = {
    "sdxl-ip2p": dict(repo="diffusers/sdxl-instructpix2pix-768", kind="ip2p", res=768),
    "sd3-ultraedit": dict(repo="BleachNick/SD3_UltraEdit_freeform", kind="ip2p", res=512,
                          call_kw=dict(negative_prompt="")),
    "sdxl-img2img": dict(repo="stabilityai/stable-diffusion-xl-base-1.0", kind="img2img", res=1024),
    "sd3-img2img": dict(repo="stabilityai/stable-diffusion-3-medium-diffusers", kind="img2img", res=1024),
}



def to_pil(x01: torch.Tensor) -> Image.Image:
    arr = (x01[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def to_tensor(img: Image.Image, device) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_pipeline(editor: str):
    spec = EDITORS[editor]
    if editor == "sdxl-ip2p":
        from diffusers import StableDiffusionXLInstructPix2PixPipeline as Pipe
    elif editor == "sd3-ultraedit":
        from immunization_baseline.third_party.ultraedit.pipeline import StableDiffusion3InstructPix2PixPipeline as Pipe
    elif editor == "sdxl-img2img":
        from diffusers import StableDiffusionXLImg2ImgPipeline as Pipe
    else:
        from diffusers import StableDiffusion3Img2ImgPipeline as Pipe
    pipe = Pipe.from_pretrained(spec["repo"], torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")
