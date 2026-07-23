"""Stable Diffusion 封裝（STRUCTURE.md §1）。

- 模型名稱由 config 提供，不硬編碼：本地開發用
  hf-internal-testing/tiny-stable-diffusion-pipe，TWCC 換真實 SD 即可。
- 暴露 vae / unet / text_encoder；img2img 與 inpaint pipeline 以同一組
  components 延遲建構。
- img2img_differentiable()：可微分 SDEdit（手動 DDIM，eta=0），供
  PhotoGuard diffusion attack（SPEC §3.3）與非加性方法（SPEC §4）反傳。
- capture_cross_attention()：擷取 UNet cross-attention 機率圖，供
  注意力抑制 reward（SPEC §4.2 方案一）。
- 裝置一律經 src/utils/device.py，禁用 .cuda()；不依賴 xformers。
"""

from contextlib import contextmanager

import torch

from src.utils.device import get_device


class SDWrapper:
    """單一 SD 模型之封裝。所有影像張量介面於 [-1,1] 空間（SD VAE 值域）。"""

    def __init__(self, model_name: str):
        from diffusers import StableDiffusionPipeline

        self.model_name = model_name
        self.device = get_device()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, safety_checker=None, requires_safety_checker=False
        ).to(self.device)
        self._img2img = None
        self._inpaint = None

    # ---- 元件 ----

    @property
    def vae(self):
        return self.pipe.vae

    @property
    def unet(self):
        return self.pipe.unet

    @property
    def text_encoder(self):
        return self.pipe.text_encoder

    @property
    def tokenizer(self):
        return self.pipe.tokenizer

    @property
    def scheduler(self):
        return self.pipe.scheduler

    @property
    def scaling_factor(self) -> float:
        return self.vae.config.scaling_factor

    @property
    def img2img(self):
        """diffusers img2img pipeline（= SDEdit，SPEC §2.8），同組件延遲建構。"""
        if self._img2img is None:
            from diffusers import StableDiffusionImg2ImgPipeline

            self._img2img = StableDiffusionImg2ImgPipeline(
                **self.pipe.components, requires_safety_checker=False
            )
        return self._img2img

    @property
    def inpaint(self):
        """diffusers inpaint pipeline。4-channel UNet（非 inpaint 專用權重）
        時 diffusers 以逐步 latent 混合處理（SPEC §2.8 Inpainting）。"""
        if self._inpaint is None:
            from diffusers import StableDiffusionInpaintPipeline

            self._inpaint = StableDiffusionInpaintPipeline(
                **self.pipe.components, requires_safety_checker=False
            )
        return self._inpaint

    # ---- 編解碼 ----

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """(1,3,H,W) [-1,1] → latent（已乘 scaling factor）。取 mean（決定性）。"""
        return self.vae.encode(x).latent_dist.mean * self.scaling_factor

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """latent → (1,3,H,W) [-1,1]（未 clamp，保留計算圖）。"""
        return self.vae.decode(z / self.scaling_factor).sample

    def encode_text(self, prompt: str) -> torch.Tensor:
        """prompt → text embeddings (1, L, dim)。"""
        tok = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    # ---- 可微分 img2img（SDEdit） ----

    def img2img_differentiable(
        self,
        x: torch.Tensor,
        prompt: str = "",
        num_steps: int = 10,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """可微分 SDEdit：加噪至 strength·T 後以 DDIM（eta=0）去噪 num_steps 步。

        x: (1,3,H,W)，[-1,1]，可含梯度；回傳同形狀 [-1,1]，計算圖保留。
        噪聲為隨機（EOT 之隨機性來源，SPEC §3.3）。
        prompt 為空字串時不做 CFG（uncond 與 cond 相同，結果不變、省一半記憶體）。
        """
        abar = self.scheduler.alphas_cumprod.to(self.device)
        t0 = min(int(len(abar) * strength), len(abar) - 1)

        z = self.encode_image(x)
        noise = torch.randn_like(z)
        z = abar[t0].sqrt() * z + (1 - abar[t0]).sqrt() * noise

        use_cfg = guidance_scale > 1.0 and prompt != ""
        emb = self.encode_text(prompt).detach()
        emb_uncond = self.encode_text("").detach() if use_cfg else None

        # t0 → 0 均分 num_steps 步；最後一步之 abar_prev 取 abar[0]（≈1）
        ts = torch.linspace(t0, 0, num_steps + 1).round().long()
        for i in range(num_steps):
            t, t_prev = ts[i], ts[i + 1]
            if use_cfg:
                eps_both = self.unet(
                    torch.cat([z, z]),
                    t,
                    encoder_hidden_states=torch.cat([emb_uncond, emb]),
                ).sample
                eps_u, eps_c = eps_both.chunk(2)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = self.unet(z, t, encoder_hidden_states=emb).sample

            abar_t, abar_prev = abar[t], abar[t_prev]
            pred_x0 = (z - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
            z = abar_prev.sqrt() * pred_x0 + (1 - abar_prev).sqrt() * eps

        return self.decode_latents(z)

    # ---- DDIM 時間格點與 inversion ----

    def ddim_timesteps(self, num_steps: int, t_max: int = None) -> torch.Tensor:
        """0 → t_max 均分之 num_steps+1 個 timestep（升冪）。

        inversion 依升冪走訪；sampling 依降冪走訪同一格點。
        """
        abar = self.scheduler.alphas_cumprod
        t_max = t_max if t_max is not None else len(abar) - 1
        return torch.linspace(0, t_max, num_steps + 1).round().long()

    def ddim_inversion(
        self, z0: torch.Tensor, prompt_emb: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """確定性 DDIM inversion（SPEC §4.1、APA 式 4）：z_0 → z_T。

        z_t = √ᾱ_t·(z_{t-1} − √(1−ᾱ_{t-1})·ε_θ)/√ᾱ_{t-1} + √(1−ᾱ_t)·ε_θ
        ε_θ 於當前狀態之 timestep（t-1）評估。保存原圖資訊、去噪可近似重建；
        與 DiffPure 的隨機加噪（遺忘原圖）方向相反。
        """
        abar = self.scheduler.alphas_cumprod.to(self.device)
        ts = self.ddim_timesteps(num_steps)
        z = z0
        for i in range(num_steps):
            t_prev, t = ts[i], ts[i + 1]
            eps = self.unet(z, t_prev, encoder_hidden_states=prompt_emb).sample
            pred_x0 = (z - (1 - abar[t_prev]).sqrt() * eps) / abar[t_prev].sqrt()
            z = abar[t].sqrt() * pred_x0 + (1 - abar[t]).sqrt() * eps
        return z

    # ---- cross-attention 擷取 ----

    @contextmanager
    def capture_cross_attention(self):
        """擷取 UNet 所有 cross-attention（attn2）之注意力機率圖。

        用法：
            with sd.capture_cross_attention() as maps:
                sd.unet(z, t, encoder_hidden_states=emb)
            # maps: list of {"name": 層名, "probs": (batch*heads, q_len, text_len)}

        後續之跨層聚合（bicubic 上採樣 + 逐像素相加，DAYN 式 3）由
        保護方法自行處理（SPEC §4.2）。
        """
        maps: list[dict] = []
        original = dict(self.unet.attn_processors)
        new_procs = {
            name: _CaptureAttnProcessor(name, maps)
            if name.endswith("attn2.processor")
            else proc
            for name, proc in original.items()
        }
        self.unet.set_attn_processor(new_procs)
        try:
            yield maps
        finally:
            self.unet.set_attn_processor(original)


class _CaptureAttnProcessor:
    """計算標準 attention 並將機率圖存入 store（依 diffusers AttnProcessor 實作）。"""

    def __init__(self, name: str, store: list):
        self.name = name
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, seq_len, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.store.append({"name": self.name, "probs": attention_probs})

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
