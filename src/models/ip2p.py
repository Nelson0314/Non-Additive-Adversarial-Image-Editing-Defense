"""InstructPix2Pix 攻擊器（DEC-031 起的主線威脅模型）。

**為什麼換掉 SDEdit**：DCT-Shield（arXiv:2504.17894）§5.3 的編輯模型就是
InstructPix2Pix，主線資料集也已改為 OmniEdit（DEC-030），其
`edited_prompt_list` 是**指令式**的句子。指令餵給 SDEdit 會被 text encoder
當成一句「要被畫出來的描述」，服從率不明，而 DEC-022 要求未防禦的編輯必須
真的成功。與其自行改寫指令（等於捏造資料），使用者 2026-08-19 裁定**主線
換成 IP2P**，SDEdit 那條線保留但凍結。

兩者在運算上不是介面差異
────────────────────────────────────────────────────────────────────
    SDEdit    z_t = √ᾱ_t·E(x) + √(1−ᾱ_t)·ε，由 t 開始去噪。
              原圖只以「被噪聲稀釋的殘影」進入（strength 0.7 時 √ᾱ = 0.2873）。
    IP2P      UNet 第一層卷積多開 4 個輸入通道，把**未加噪的** E(x) 直接
              拼在噪聲 latent 旁；生成由**純噪聲**起步。

FND-055 量到的機制（相位擾動貼著紋理分布、用得上那份殘存訊號，故 strength
由 0.8 降到 0.7 時相位升 3% 而加性掉 18%）**在 IP2P 上不成立**——那條通道被
換掉了。本方法在 IP2P 下是強是弱屬於待測，不得沿用 SDEdit 的結論。

去噪迴圈直接用 diffusers 的官方管線
────────────────────────────────────────────────────────────────────
`StableDiffusionInstructPix2PixPipeline`（diffusers 0.39.0）。**不自行重寫**：
它是參考實作，攻擊方那一側越忠實越好，而本專案不需要對編輯過程取梯度
（防禦的損失只經過 VAE 編碼器，見 `src/baselines/encoder_target.py`）。

逐行對照過的三個關鍵細節（`pipeline_stable_diffusion_instruct_pix2pix.py`）：

1. **三份批次，順序是 [text, image, uncond]**（第 443 行
   `noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)`），
   對應的文字嵌入是 `[prompt, negative, negative]`、影像 latent 是
   `[img, img, zeros]`（第 897 行）。
2. **導引式有兩個尺度**（第 445–447 行）：

       ε̃ = ε_uncond + s_T·(ε_text − ε_image) + s_I·(ε_image − ε_uncond)

3. **拼進去的影像 latent 不乘 scaling_factor**——第 877 行是
   `retrieve_latents(vae.encode(image), sample_mode="argmax")`，沒有再乘。
   噪聲 latent 那一側才在解碼時除以 scaling_factor（第 475 行）。兩側因此
   落在不同的尺度上，這是 IP2P 原本就有的不對稱，不是 bug。本封裝的
   `image_latents()` 明寫這件事，避免呼叫端拿 `encode_image()`（有乘）去拼。
   `sample_mode="argmax"` 取的是後驗的眾數，對角高斯下即平均，與本專案
   `SDWrapper.encode_image` 取 `latent_dist.mean` 同義。

推論參數是本專案指定的，不是論文的
────────────────────────────────────────────────────────────────────
**DCT-Shield §5.3 只寫「we utilize InstructPix2Pix (IP2P), a widely used
diffusion-based editing model」，沒有給步數、兩個導引尺度、排程器或 seed。**
故下面三個常數是**本專案指定**（取 diffusers 的預設值），任何用到它們的
報表都必須把值寫進 CSV 並標明出處缺口。**不要靜默改動**——改了就與既有批次
不可比。

白盒的界線
────────────────────────────────────────────────────────────────────
防禦的損失是 `‖E(x_def) − E(y_target)‖²`，**只經過攻擊方的 VAE 編碼器**。
換到 IP2P 之後，`E` 應該是 **IP2P 自己的 VAE**（白盒假設是攻擊方的模型已知），
故本封裝提供與 `SDWrapper` 同名同語意的 `encode_image`／`decode_latent`，
`make_encoder_target_loss(ip2p, y)` 不必改就能用。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.utils.checkpoint as ckpt

from src.utils.device import get_device, resolve_precision

MODEL_NAME = "timbrooks/instruct-pix2pix"

# **本專案指定**（diffusers 的預設值）。論文未載，見模組 docstring。
IP2P_STEPS = 100
IP2P_TEXT_GUIDANCE = 7.5      # s_T
IP2P_IMAGE_GUIDANCE = 1.5     # s_I
IP2P_SEED = 20260812          # 與 SDEdit 線的 EDIT_SEED 相同，方便逐圖對照


class IP2PWrapper:
    """InstructPix2Pix 的封裝。介面刻意與 `SDWrapper` 的子集同名同語意。

    `pipe` 供測試注入，避免為了驗結構而下載 4 GB 權重；給定時 `model_name`
    只作為記錄用的標籤。
    """

    def __init__(self, model_name: str = MODEL_NAME,
                 dtype: torch.dtype = torch.float32, pipe=None):
        self.model_name = model_name
        self.device = get_device()
        self.compute_dtype = dtype
        self.backbone_dtype, self.vae_dtype = resolve_precision(dtype)
        self.pipe = (pipe if pipe is not None
                     else self._load_pipeline(model_name, self.backbone_dtype))
        self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        self.vae.to(self.vae_dtype)
        # 攻擊方的模型全部凍結：本專案不訓練它，只當作已知的黑箱前向。
        for m in (self.unet, self.vae, self.text_encoder):
            m.requires_grad_(False)
            m.eval()
        self._check_channels()

    @staticmethod
    def _load_pipeline(model_name: str, dtype: torch.dtype):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        return StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_name, safety_checker=None, requires_safety_checker=False,
            torch_dtype=dtype)

    def _check_channels(self) -> None:
        """IP2P 的 UNet 必須是 8 通道（4 噪聲 ＋ 4 影像）。

        載錯 checkpoint（例如載成一般的 SD 1.5）會是 4 通道，而後面每一步都
        還是跑得動——影像條件那一半靜默消失，編輯結果變成純文生圖。
        那種失敗不會拋錯，只會讓整批數字無聲地失去意義，故在此擋掉。
        """
        want = 8
        got = int(self.unet.config.in_channels)
        if got != want:
            raise RuntimeError(
                f"{self.model_name} 的 UNet in_channels = {got}，不是 IP2P 的 "
                f"{want}（4 噪聲 ＋ 4 影像）。載錯 checkpoint 時影像條件會靜默"
                "失效，編輯退化成純文生圖而不報錯，故此處拒絕繼續")

    # ---- 與 SDWrapper 同名的三個入口（防禦端只用得到這些）----

    @property
    def unet(self):
        return self.pipe.unet

    @property
    def vae(self):
        return self.pipe.vae

    @property
    def text_encoder(self):
        return self.pipe.text_encoder

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    def encode_image(self, x01: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        """(N,3,H,W) [0,1] → **已乘 scaling_factor** 的 latent，取 mean 保持決定性。

        與 `SDWrapper.encode_image` 逐字同義，故
        `make_encoder_target_loss(ip2p, y)` 不必改。**要拼進 UNet 的影像條件
        不能用這個**，見 `image_latents`。
        """
        x = (x01.to(self.device) * 2.0 - 1.0).to(self.vae.dtype)
        if use_ckpt:
            return ckpt.checkpoint(
                lambda a: self.vae.encode(a).latent_dist.mean * self.scaling_factor,
                x, use_reentrant=False)
        return self.vae.encode(x).latent_dist.mean * self.scaling_factor

    def decode_latent(self, z: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        z = z.to(self.vae.dtype)
        if use_ckpt:
            x = ckpt.checkpoint(
                lambda a: self.vae.decode(a / self.scaling_factor).sample,
                z, use_reentrant=False)
        else:
            x = self.vae.decode(z / self.scaling_factor).sample
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def image_latents(self, x01: torch.Tensor) -> torch.Tensor:
        """要拼進 UNet 前 4 個新通道的影像條件：**不乘 scaling_factor**。

        存在的理由只有一個——把管線第 877 行那個容易看漏的細節寫成程式碼。
        `encode_image` 乘了、這裡沒乘，差一個 0.18 的倍率，補錯不會拋錯，
        只會讓影像條件的強度整個跑掉。
        """
        return self.encode_image(x01) / self.scaling_factor

    # ---- 攻擊 ----

    @torch.no_grad()
    def edit(self, x01: torch.Tensor, instruction: str, seed: int = IP2P_SEED,
             steps: int = IP2P_STEPS, s_t: float = IP2P_TEXT_GUIDANCE,
             s_i: float = IP2P_IMAGE_GUIDANCE,
             negative_prompt: Optional[str] = None) -> torch.Tensor:
        """依指令編輯，回傳 (N,3,H,W) [0,1]。同 `seed` 必得同一張輸出。

        走官方管線（理由見模組 docstring）。`output_type="pt"` 讓輸出留在
        張量域，不繞 PIL——繞一趟 PIL 會經過 uint8 量化，量測 LPIPS 這種
        小差異時那是可見的損失。
        """
        if x01.dim() != 4:
            raise ValueError(f"x01 必須是 (N,3,H,W)，收到 {tuple(x01.shape)}")
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(
            prompt=instruction,
            image=(x01.to(self.device).clamp(0, 1) * 2.0 - 1.0).to(self.vae.dtype),
            num_inference_steps=steps,
            guidance_scale=s_t,
            image_guidance_scale=s_i,
            negative_prompt=negative_prompt,
            generator=gen,
            output_type="pt",
        )
        return out.images.to(x01.dtype).clamp(0, 1)
