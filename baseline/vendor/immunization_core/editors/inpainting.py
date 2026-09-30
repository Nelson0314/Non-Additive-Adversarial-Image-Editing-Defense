"""九通道 Stable Diffusion inpainting 權重 adapter；共用 SD 取樣與遮罩契約。"""

import torch

from .stable_diffusion import SDWrapper


class SDInpaintWrapper(SDWrapper):
    """SD 的 inpainting 權重（9 通道 UNet）。

    存在理由是威脅模型的形態改變，不是為了多支援一個模型：本專案的五篇
    baseline 中有三篇（PhotoGuard-c、AdvPaint、PromptFlare）**原作就是
    inpainting**，目前表上全部標著 `modified_from_paper=True`，而改動的性質
    是「梯度從哪條路進入計算圖」——AdvPaint 原作的梯度只走
    `masked_image_latents`（`advpaint.py:43`），img2img 沒有那條路。載入
    inpainting 權重之後那三篇回到原生形式。

    `runwayml/stable-diffusion-inpainting` 是那三篇共同指定的權重
    （`advpaint.py:345`、`promptflare.py:462` 的 `model` 欄）。

    只覆寫 pipeline 類別：9 通道的處置在 `SDWrapper.inpaint` 內，與權重
    來源無關，故不在此重複。
    """

    @staticmethod
    def _load_pipeline(model_name: str, dtype: torch.dtype):
        from diffusers import StableDiffusionInpaintPipeline

        return StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=dtype,
        )
