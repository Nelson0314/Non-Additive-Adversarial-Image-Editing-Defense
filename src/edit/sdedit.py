"""SDEdit 編輯（img2img）— SPEC.md §2.8、STRUCTURE.md §2.2。

使用 diffusers 之 img2img pipeline（即 SDEdit 之實作）。

v5 批次化：編輯為純推論、無梯度，可批次執行。批次中每個樣本使用各自的
torch.Generator 並以其應得之 seed 初始化——diffusers 對 generator list 逐樣本
取樣（VAE encode 逐樣本、初始噪聲與 eta>0 之變異噪聲逐樣本），故與 batch=1
逐位元一致（tests/test_edit_batching.py 驗證；seed 協定 SPEC §2.3 之前提）。
"""

import torch
import torchvision.transforms as T

from src.utils.device import get_device


def _require_strength(config: dict) -> float:
    strength = config["edit"]["sdedit_strength"]
    if strength is None:
        raise ValueError(
            "edit.sdedit_strength 未設定（SPEC §2.8 [待確認]，須向 DAYN 作者索取；"
            "無法取得時依 v5 掃描策略於 T2 以 {0.8, 0.7, 0.5, 0.3} 依序掃描）；"
            "請於 config 明確提供"
        )
    return strength


def sdedit_edit_batch(
    sd, images: list, prompts: list, seeds: list, config: dict
) -> list:
    """批次 SDEdit：list of (1,3,H,W) [0,1] → 同長度 list of (1,3,H,W) [0,1]。

    每樣本各自 generator+seed（見模組 docstring）。
    """
    gen_cfg = config["generation"]
    strength = _require_strength(config)
    device = str(get_device())
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    pils = [T.ToPILImage()(img[0].clamp(0, 1).cpu()) for img in images]
    out = sd.img2img(
        prompt=list(prompts),
        image=pils,
        strength=strength,
        guidance_scale=gen_cfg["guidance_scale"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        eta=gen_cfg["eta"],
        generator=generators,
        output_type="pt",
    )
    return [im.unsqueeze(0).detach() for im in out.images]


def sdedit_edit(sd, image: torch.Tensor, prompt: str, seed: int, config: dict) -> torch.Tensor:
    """(1,3,H,W) [0,1] → 編輯結果 (1,3,H,W) [0,1]。單樣本＝批次長度 1。"""
    return sdedit_edit_batch(sd, [image], [prompt], [seed], config)[0]
