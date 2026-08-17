"""像素臂的兩個替代損失：untargeted latent 與 untargeted CLIP。

現行的共用損失是 targeted 的——`‖E(x_def) − E(gray)‖²`，把 latent 推向一張
無內容的目標（`encoder_target.py`）。本檔提供兩個 untargeted 的形式，改成
「推離原圖自己」而不是「推向某個目標」：

    latent   maximise ‖E(x_def) − E(x)‖²        （在 VAE latent 上遠離原圖）
    clip     minimise cos(CLIP(x_def), CLIP(x)) （在 CLIP 影像嵌入上遠離原圖）

兩者都回傳「要最小化」的純量，與 `run_param_pgd` 的介面一致，故三個損失可以
在同一個迴圈、同樣的步數與種子下互換，**唯一變因是損失**。

為什麼要測這兩個：現行損失只看 VAE 編碼器，而編輯是由 UNet 在 latent 上做的、
語意由 CLIP 文字條件驅動。若擾動在 CLIP 嵌入上也能把影像推開，抗編輯的機制就
不只是「騙過 VAE」。2026-08-18 使用者指定。

**與既有否決結論的關係**：FND-023…030 否決過 latent 與 CLIP 作為**弱 baseline
階段二的 reward**（走完整條去噪軌跡、量解碼後的像素差）。這裡是不同的東西——
像素臂的 PGD 損失，不經過 UNet。結論不能互相引用。
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

# openai/clip-vit-base-patch32 的前處理常數。取自該 repo 的
# preprocessor_config.json；此處用張量重寫一份是為了可微——
# `AutoProcessor` 走 PIL，梯度在那裡就斷了。
CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def make_latent_untargeted_loss(sd, x_ref01: torch.Tensor) -> Callable:
    """回傳 `loss(x_def01)`，最小化它等於**最大化**與原圖 latent 的距離。

    參考 latent 只編碼一次並 detach——它是常數，重編碼除了浪費之外還會讓
    損失曲線帶上 VAE 的隨機性。

    取負號而不是取倒數：倒數在距離接近 0 時發散，負號的梯度在整個定義域上
    都是良好的，且與 targeted 版的尺度可比（同樣是 latent 的均方差）。
    """
    with torch.no_grad():
        z_ref = sd.encode_image(x_ref01).detach()

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        return -(sd.encode_image(x_def01) - z_ref).pow(2).mean()

    return loss


class _ClipEmbed:
    """可微的 CLIP 影像嵌入。模型只載入一次。"""

    def __init__(self, device, dtype=torch.float32):
        from transformers import AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained(CLIP_REPO).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)

    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        # 輸入已是正方形，故只需縮放不需裁切。bicubic 與 CLIP 前處理一致；
        # antialias=True 對縮小是必要的，關掉會讓 512 -> 224 出現混疊，
        # 那個混疊會被當成訊號一起最佳化。
        x = F.interpolate(x01, size=(CLIP_SIZE, CLIP_SIZE), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self.mean) / self.std
        # 本版 transformers 的 `get_image_features` 回傳的是
        # `BaseModelOutputWithPooling`，投影後的影像特徵在 `.pooler_output`
        # 而不是 `.image_embeds`（`CLIPModel.get_image_features` 原始碼：
        # `vision_outputs.pooler_output = self.visual_projection(pooled_output)`）。
        # `src/metrics/aesthetic.py::clip_image_similarity` 走的是同一條路，
        # 兩處必須一致，否則損失與指標會量到不同的東西。
        f = self.model.get_image_features(pixel_values=x).pooler_output
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def make_clip_untargeted_loss(device, x_ref01: torch.Tensor) -> Callable:
    """回傳 `loss(x_def01)`，最小化它等於把 CLIP 影像嵌入推離原圖。

    損失就是餘弦相似度本身（已正規化，故是內積）。範圍 [-1, 1]，
    與 latent 損失的尺度不同——兩者的絕對值不可互相比較，只能各自看
    「同一個損失下加性對相位誰贏」。
    """
    embed = _ClipEmbed(device, dtype=x_ref01.dtype)
    with torch.no_grad():
        f_ref = embed(x_ref01).detach()

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        return (embed(x_def01) * f_ref).sum(dim=-1).mean()

    return loss
