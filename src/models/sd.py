"""Stable Diffusion 封裝 — spec §4.3。

提供三條路徑：

1. `ddim_inversion`：z₀ → z_{k}，確定性、無梯度（防禦生成的前段，殘差模塊關閉）
2. `denoise`：z_k → z₀，可注入逐步殘差（防禦生成的後段，殘差模塊開啟）
3. `sdedit`：攻擊者的編輯管線，可微分（殘差模塊關閉）

殘差注入以 callback 形式傳入，SDWrapper 不知道殘差如何產生 —— 這使
site L 與其他 site 共用同一條去噪迴圈。

所有影像張量介面為 [0,1]、(1,3,H,W)；內部轉為 SD VAE 的 [-1,1] 值域。
"""

from typing import Callable, Optional

import torch
import torch.utils.checkpoint as ckpt

from src.utils.device import get_device

# eps_hook(eps, step_idx, t) -> eps'，回傳修改後的噪聲預測
EpsHook = Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]


class SDWrapper:
    """單一 SD 模型。防禦與編輯共用同一組權重，差別只在殘差模塊開關。"""

    def __init__(self, model_name: str, dtype: torch.dtype = torch.float32):
        from diffusers import StableDiffusionPipeline

        self.model_name = model_name
        self.device = get_device()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=dtype,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        # SD 全部凍結：φ 是唯一可訓練參數（spec §5.3）
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

    # ---- 元件 ----

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
    def tokenizer(self):
        return self.pipe.tokenizer

    @property
    def scheduler(self):
        return self.pipe.scheduler

    @property
    def scaling_factor(self) -> float:
        return self.vae.config.scaling_factor

    @property
    def num_train_timesteps(self) -> int:
        return len(self.scheduler.alphas_cumprod)

    def alphas_cumprod(self, device=None) -> torch.Tensor:
        return self.scheduler.alphas_cumprod.to(device or self.device)

    # ---- 編解碼 ----

    def encode_image(self, x01: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        """(1,3,H,W) [0,1] → latent。取 mean 以保持決定性。

        `use_ckpt` 把整個 encode 當作一個 checkpoint 區塊。理由見
        `decode_latent`。diffusers 的 `enable_gradient_checkpointing()`
        只在 module.training 為真時生效，而本封裝的 VAE 固定為 eval，
        故不能沿用，必須自行包。
        """
        x = (x01.to(self.device) * 2.0 - 1.0).to(self.vae.dtype)
        if use_ckpt:
            return ckpt.checkpoint(
                lambda a: self.vae.encode(a).latent_dist.mean * self.scaling_factor,
                x,
                use_reentrant=False,
            )
        return self.vae.encode(x).latent_dist.mean * self.scaling_factor

    def decode_latent(self, z: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        """latent → (1,3,H,W) [0,1]，保留計算圖。

        `use_ckpt` 的效果來自「同一張計算圖上有多次 VAE 呼叫」：不做
        checkpoint 時，每次呼叫的中間激活都必須同時留存到反向傳播，peak
        是各次的**總和**；做了 checkpoint 之後，反向時一次只重算一個區塊，
        peak 降為各次的**最大值**。單獨一次呼叫並不會因此變省。
        """
        if use_ckpt:
            x = ckpt.checkpoint(
                lambda a: self.vae.decode(a / self.scaling_factor).sample,
                z,
                use_reentrant=False,
            )
        else:
            x = self.vae.decode(z / self.scaling_factor).sample
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def encode_text(self, prompt: str) -> torch.Tensor:
        tok = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    # ---- 時間格點 ----

    def timesteps(self, num_steps: int, t_max: Optional[int] = None) -> torch.Tensor:
        """0 → t_max 的均分格點，升冪，長度 num_steps+1。

        inversion 依升冪走訪，denoise 依降冪走訪同一格點，兩者一致。
        """
        top = self.num_train_timesteps - 1 if t_max is None else t_max
        return torch.linspace(0, top, num_steps + 1).round().long()

    # ---- UNet 前向 ----

    def _eps(self, z, t, emb, use_ckpt: bool = False) -> torch.Tensor:
        if use_ckpt:
            return ckpt.checkpoint(
                lambda a, b, c: self.unet(a, b, encoder_hidden_states=c).sample,
                z,
                t,
                emb,
                use_reentrant=False,
            )
        return self.unet(z, t, encoder_hidden_states=emb).sample

    # ---- 1. DDIM inversion（無梯度，殘差關閉）----

    @torch.no_grad()
    def ddim_inversion(
        self, z0: torch.Tensor, emb: torch.Tensor, ts: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """在格點 ts 上走前 `steps` 步 inversion：z₀ → z_{ts[steps]}。

        確定性 DDIM，ε 於當前狀態的 timestep 評估。此段不依賴 φ，
        故結果可於優化開始前快取（spec §4.3 效率設計）。
        """
        abar = self.alphas_cumprod(z0.device)
        z = z0
        for i in range(steps):
            t_cur, t_next = ts[i], ts[i + 1]
            eps = self._eps(z, t_cur, emb)
            pred_x0 = (z - (1 - abar[t_cur]).sqrt() * eps) / abar[t_cur].sqrt()
            z = abar[t_next].sqrt() * pred_x0 + (1 - abar[t_next]).sqrt() * eps
        return z

    # ---- 2. 去噪（可注入殘差，殘差開啟）----

    def denoise(
        self,
        z_start: torch.Tensor,
        emb: torch.Tensor,
        ts: torch.Tensor,
        steps: int,
        eps_hook: Optional[EpsHook] = None,
        use_ckpt: bool = False,
        collect_x0: bool = False,
    ):
        """由 ts[steps] 沿格點降冪去噪至 ts[0]。

        `eps_hook(eps, step_idx, t)` 於每步的 ε 預測後呼叫，回傳修改後的 ε。
        `step_idx` 由 0 起算，對應 U/V 的第一個維度。
        `collect_x0=True` 時額外回傳每步的 x̂₀ 估計（spec §8.3 中間圖留存）。

        回傳 (z_final, x0_list)。collect_x0=False 時 x0_list 為空 list。
        """
        abar = self.alphas_cumprod(z_start.device)
        z = z_start
        x0_list = []

        for step_idx, i in enumerate(reversed(range(steps))):
            t, t_prev = ts[i + 1], ts[i]
            eps = self._eps(z, t, emb, use_ckpt=use_ckpt)
            if eps_hook is not None:
                eps = eps_hook(eps, step_idx, t)

            sqrt_1mabar = (1 - abar[t]).sqrt()
            pred_x0 = (z - sqrt_1mabar * eps) / abar[t].sqrt()
            if collect_x0:
                x0_list.append(pred_x0)
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps

        return z, x0_list

    # ---- 3. 編輯管線（攻擊者，殘差關閉，可微分）----

    def sdedit(
        self,
        x01: torch.Tensor,
        emb: torch.Tensor,
        noise: torch.Tensor,
        num_steps: int,
        strength: float = 0.5,
        use_ckpt: bool = False,
        vae_ckpt: bool = False,
    ) -> torch.Tensor:
        """可微分 SDEdit。

        `noise` 由呼叫端提供且**必須**在比較的兩條分支間共用（spec §5.1）。
        不在此處抽樣，是為了讓「兩分支共用同一個 ε」成為介面上的硬性要求，
        而非仰賴呼叫端自律。

        `use_ckpt` 控制 UNet、`vae_ckpt` 控制 VAE，兩者分開是為了讓 E0 能
        分別歸因記憶體，不是為了提供選項。

        回傳 (1,3,H,W) [0,1]，計算圖保留。
        """
        abar = self.alphas_cumprod(x01.device)
        t0 = min(int(self.num_train_timesteps * strength), self.num_train_timesteps - 1)

        z = self.encode_image(x01, use_ckpt=vae_ckpt)
        z = abar[t0].sqrt() * z + (1 - abar[t0]).sqrt() * noise

        ts = torch.linspace(t0, 0, num_steps + 1).round().long()
        for i in range(num_steps):
            t, t_prev = ts[i], ts[i + 1]
            eps = self._eps(z, t, emb, use_ckpt=use_ckpt)
            pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps

        return self.decode_latent(z, use_ckpt=vae_ckpt)

    def sample_edit_noise(self, z_like: torch.Tensor, seed: int) -> torch.Tensor:
        """以固定 seed 產生編輯噪聲，供兩條分支共用。"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            z_like.shape, generator=g, dtype=z_like.dtype
        ).to(z_like.device)

    def latent_shape(self, height: int, width: int):
        f = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return (1, self.unet.config.in_channels, height // f, width // f)
