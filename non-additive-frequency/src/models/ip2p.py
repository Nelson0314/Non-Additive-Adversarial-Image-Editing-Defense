"""兩者在運算上不是介面差異

────────────────────────────────────────────────────────────────────
    SDEdit    z_t = √ᾱ_t·E(x) + √(1−ᾱ_t)·ε，由 t 開始去噪。
              原圖只以「被噪聲稀釋的殘影」進入（strength 0.7 時 √ᾱ = 0.2873）。
    IP2P      UNet 第一層卷積多開 4 個輸入通道，把**未加噪的** E(x) 直接
              拼在噪聲 latent 旁；生成由**純噪聲**起步。

去噪迴圈直接用 diffusers 的官方管線
────────────────────────────────────────────────────────────────────
`StableDiffusionInstructPix2PixPipeline`（diffusers 0.39.0）。**不自行重寫**：
它是參考實作，推論入口 `edit()` 維持使用官方管線。任務損失另外使用
`edit_differentiable()`：依同一套三分支與 scheduler 展開，僅截斷反向圖；
舊的 VAE 損失呼叫端不受影響。

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
        for m in (self.unet, self.vae, self.text_encoder):
            m.requires_grad_(False)
            m.eval()
        self._check_channels()

    @staticmethod
    def _load_pipeline(model_name: str, dtype: torch.dtype):
        from pathlib import Path
        from diffusers import StableDiffusionInstructPix2PixPipeline
        from huggingface_hub import constants, hf_hub_download

        # Offline diagnostics need the inference components, not every file in
        # the Hub repository (including alternative multi-GB checkpoints).
        # Resolve the cached index explicitly; component loading still raises
        # for any missing required weight/config. No download or silent fallback.
        if constants.HF_HUB_OFFLINE and not Path(model_name).is_dir():
            index = hf_hub_download(model_name, 'model_index.json', local_files_only=True)
            model_name = str(Path(index).parent)

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
        return self.posterior_mean(x01, use_ckpt=use_ckpt) * self.scaling_factor

    def posterior_mean(self, x01: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        """未乘 scaling_factor 的後驗平均，與官方管線第 877 行同義。

        `image_latents` 先前是 `encode_image() / scaling_factor`，在 bf16 下那一
        乘一除的捨入不會完全抵銷，拼進 UNet 的影像條件因此與官方推論有微小差異。
        兩個入口改成共用這裡，`encode_image` 自己乘。
        """
        x = (x01.to(self.device) * 2.0 - 1.0).to(self.vae.dtype)
        if use_ckpt:
            return ckpt.checkpoint(
                lambda a: self.vae.encode(a).latent_dist.mean,
                x, use_reentrant=False)
        return self.vae.encode(x).latent_dist.mean

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
        只會讓影像條件的強度整個跑掉。**不要寫成 `encode_image() / scaling`**：
        bf16 下那一乘一除不會完全抵銷。
        """
        return self.posterior_mean(x01)

    # ---- 攻擊 ----

    def edit_differentiable(
        self, x01: torch.Tensor, instruction: str, seed: int = IP2P_SEED,
        steps: int = IP2P_STEPS, grad_steps: int = 2,
        s_t: float = IP2P_TEXT_GUIDANCE, s_i: float = IP2P_IMAGE_GUIDANCE,
        negative_prompt: Optional[str] = None, use_ckpt: bool = True,
        sequential_cfg: bool = False,
    ) -> torch.Tensor:
        """沿實際 IP2P 軌跡生成，只對末端 ``grad_steps`` 次 UNet 呼叫反傳。"""
        from copy import deepcopy

        if x01.ndim != 4 or x01.shape[:2] != (1, 3):
            raise ValueError('可微編輯需要單張 (1,3,H,W) RGB 影像')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('instruction 必須是非空的真實編輯指令')
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps 必須為正整數')
        if not isinstance(grad_steps, int) or grad_steps < 1:
            raise ValueError('grad_steps 必須為正整數')
        if s_t <= 1 or s_i < 1:
            raise ValueError('三分支 CFG 需要 s_t > 1 且 s_i >= 1')
        factor = self.pipe.vae_scale_factor
        if any(side % factor for side in x01.shape[-2:]):
            raise ValueError('影像邊長必須可被 VAE 縮放倍率整除')
        scheduler = deepcopy(self.pipe.scheduler)
        scheduler.set_timesteps(steps, device=self.device)
        if grad_steps > len(scheduler.timesteps):
            raise ValueError('grad_steps 不得超過實際時間表長度')
        with torch.no_grad():
            embeds = self.pipe._encode_prompt(
                instruction, self.device, 1, True, negative_prompt)
        cond = (self.encode_image(x01.clamp(0, 1), use_ckpt=use_ckpt)
                / self.scaling_factor).to(embeds.dtype)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        z = torch.randn(cond.shape, device=self.device, dtype=embeds.dtype,
                        generator=gen) * scheduler.init_noise_sigma
        extra = self.pipe.prepare_extra_step_kwargs(gen, 0.0)
        cut = len(scheduler.timesteps) - grad_steps
        outer_grad = torch.is_grad_enabled()

        def predict(a, t, text):
            return self.unet(a, t, encoder_hidden_states=text, return_dict=False)[0]

        for i, t in enumerate(scheduler.timesteps):
            active = outer_grad and i >= cut
            with torch.set_grad_enabled(active):
                c = cond if active else cond.detach()
                image_batch = torch.cat([c, c, torch.zeros_like(c)])
                noise_batch = scheduler.scale_model_input(torch.cat([z] * 3), t)
                model_input = torch.cat([noise_batch, image_batch], dim=1)
                if sequential_cfg:
                    # 分支各自 checkpoint，反傳重算時 UNet 的 batch 維度維持 1。
                    branches = []
                    for branch_input, branch_text in zip(model_input.chunk(3), embeds.chunk(3)):
                        branches.append(ckpt.checkpoint(predict, branch_input, t, branch_text,
                                                        use_reentrant=False)
                                        if active and use_ckpt else
                                        predict(branch_input, t, branch_text))
                    text, image, uncond = branches
                else:
                    eps = (ckpt.checkpoint(predict, model_input, t, embeds,
                                           use_reentrant=False)
                           if active and use_ckpt else predict(model_input, t, embeds))
                    text, image, uncond = eps.chunk(3)
                guided = uncond + s_t * (text - image) + s_i * (image - uncond)
                z = scheduler.step(guided, t, z, **extra, return_dict=False)[0]
        return self.decode_latent(z, use_ckpt=use_ckpt).to(x01.dtype)

    @torch.no_grad()
    def edit_batch(self, images, instructions, seeds, steps: int = IP2P_STEPS,
                   s_t: float = IP2P_TEXT_GUIDANCE, s_i: float = IP2P_IMAGE_GUIDANCE,
                   negative_prompt: Optional[str] = None) -> torch.Tensor:
        """一次編輯一批，回傳 (B,3,H,W) [0,1]，第 i 列等同 `edit(images[i], ...)`。

        為什麼要批次
        ────────────────────────────────────────────────────────────────
        批次為 1 時 512² 的 UNet 遠遠餵不飽一張 3090：實測顯存只用到 24 GB 的
        29%，平行單元大半在等。整批的成本幾乎與單張相同，所以吞吐量直接隨批次
        大小上升。實驗設定完全不變——這是把卡餵飽，不是換實驗。

        **每張圖各自一個 generator，不是整批共用一個。**
        `diffusers` 拿到單一 generator 時會用同一條隨機序列依序抽整批的噪聲，
        於是第 i 張拿到的噪聲取決於批次裡有幾張、排第幾個——同樣的
        `(圖, 指令, 種子)` 在不同的批次組合下會得到不同的輸出，而報表上看不
        出來。傳一串 generator 則讓第 i 張吃 `seeds[i]` 自己的序列，與逐張跑
        逐位元相同。`tests/test_ip2p_batch.py` 釘住這一條。
        """
        import torch as _t
        images = list(images)
        instructions = list(instructions)
        seeds = [int(v) for v in seeds]
        if not images:
            raise ValueError("空的批次：沒有要編輯的影像")
        if not (len(images) == len(instructions) == len(seeds)):
            raise ValueError(
                f"三個清單長度必須相同，收到 影像 {len(images)}、"
                f"指令 {len(instructions)}、種子 {len(seeds)}")
        ref = images[0]
        batch = _t.cat([y.to(self.device).clamp(0, 1) for y in images], dim=0)
        gens = [_t.Generator(device=self.device).manual_seed(v) for v in seeds]
        out = self.pipe(
            prompt=instructions,
            image=(batch * 2.0 - 1.0).to(self.vae.dtype),
            num_inference_steps=steps,
            guidance_scale=s_t,
            image_guidance_scale=s_i,
            negative_prompt=negative_prompt,
            generator=gens,
            output_type="pt",
        )
        return out.images.to(ref.dtype).clamp(0, 1)

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
