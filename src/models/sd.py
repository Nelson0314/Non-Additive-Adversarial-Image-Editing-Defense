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

    def _eps_cfg(
        self,
        z,
        t,
        emb,
        guidance_scale: float,
        emb_uncond: Optional[torch.Tensor] = None,
        use_ckpt: bool = False,
    ) -> torch.Tensor:
        """帶 classifier-free guidance 的 ε 預測。

            ε = ε(z, t, ∅) + w · [ ε(z, t, c) − ε(z, t, ∅) ]

        **這個函式是 2026-08-01（E26）新增的，補的是一個使整個威脅模型失效的
        缺漏。** 修訂前全專案沒有任何 CFG：`_eps` 只以條件嵌入呼叫一次 UNet，
        等同 w = 1。Stable Diffusion v1.x 是**在 CFG 下訓練也在 CFG 下使用**
        的，w = 1 時 prompt 對輸出的影響極弱，SDEdit 退化成「加噪再去噪」。
        實測後果見 `docs/RESULTS_E26_guidance.md`：CLIP(原圖) 0.2030 →
        CLIP(所謂的編輯結果) 0.2132，只升 0.0101 而標準差 0.0169，即編輯
        根本沒有發生；使用者對 `runs/p5_semantic_axis/compare.html` 的判讀是
        「連原始圖片被文字編輯都沒有成功」。

        也就是說 E2–E23 全部是在防禦一個**不存在的攻擊**，量到的 `net_lpips`
        是兩次隨機去噪之間的漂移。

        **w = 1.0 時不走這條路徑**，直接回到單次前向：`_eps` 的行為必須逐位元
        不變，否則既有 53 個 run 的可重現性會斷掉。

        兩次前向**分開做而非合批**：合批把激活加倍，而 E0 已量出 512² 下記憶體
        才是綁定的資源（開 checkpoint 是必要條件）。代價是時間乘二。
        """
        if guidance_scale == 1.0:
            return self._eps(z, t, emb, use_ckpt=use_ckpt)
        if emb_uncond is None:
            raise ValueError(
                "guidance_scale != 1.0 需要無條件嵌入 emb_uncond。"
                "缺少時不可退回單分支：那會靜默把 w 變回 1，"
                "而那正是 E26 找到的缺陷"
            )
        eps_u = self._eps(z, t, emb_uncond, use_ckpt=use_ckpt)
        eps_c = self._eps(z, t, emb, use_ckpt=use_ckpt)
        return eps_u + guidance_scale * (eps_c - eps_u)

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

    # ---- 1b. BDIA 精確反演 ----

    def _ddim_step(self, z, eps, t_a, t_b, abar) -> torch.Tensor:
        """由 t_a 的狀態 z 與其 ε 預測，走一步 DDIM 到 t_b。t_b 可大於或小於 t_a。"""
        pred_x0 = (z - (1 - abar[t_a]).sqrt() * eps) / abar[t_a].sqrt()
        return abar[t_b].sqrt() * pred_x0 + (1 - abar[t_b]).sqrt() * eps

    @torch.no_grad()
    def bdia_inversion(
        self, z0: torch.Tensor, emb: torch.Tensor, ts: torch.Tensor,
        steps: int, gamma: float = 1.0,
    ) -> tuple:
        """BDIA 反演：z₀ → (z_K, z_{K−1})。回傳**一對**狀態，不是單一張量。

        BDIA（Zhang et al., "Exact Diffusion Inversion via Bi-directional
        Integration Approximation", arXiv 2307.10829, ECCV 2024）把 DDIM 的
        單步遞迴改成跨兩步的遞迴：

            z_{i+1} = γ·z_{i−1} − γ·DDIM(z_i, t_i→t_{i−1}) + DDIM(z_i, t_i→t_{i+1})

        兩個 DDIM 步都只用到在 z_i 處的同一次 ε 預測，故給定 (z_{i−1}, z_i)
        可解出 z_{i+1}，給定 (z_i, z_{i+1}) 也可解出 z_{i−1}——**兩個方向都是
        代數上的精確反解，不是近似**。γ=1 為標準選擇。

        既有的 `ddim_inversion` 不是精確的：反演時 ε 在 z_i 評估、去噪時在
        z_{i+1} 評估，兩者不同，誤差逐步累積。實測 t_max=500、k_inv=20 下
        `G(x; φ=0)` 與原圖相差 LPIPS 0.194 / PSNR 26.56 dB。

        **必須回傳一對狀態。** 遞迴的狀態是相鄰兩點；只交出 z_K，去噪端就得
        自己補一個近似的起手步，精確性當場失去。第 0 步（z₀→z₁）沒有前一點
        可用，以普通 DDIM 走，去噪端也不需要反解它——去噪只跑 i=K−1…1，
        正好是反演用到 BDIA 遞迴的那些步反過來，故整條來回是精確的。

        **精確的只有擴散這一段。** `G(x;0)` 仍要經過 VAE 的編碼與解碼，其
        來回誤差實測為 PSNR 27.51 dB / LPIPS 0.143，BDIA 不改變這一項。
        故本方法把重建地板由 LPIPS 0.194 降到 0.143 為止，仍高於像素側加性
        位置實際運作的 0.063。採用與否應以此為準，不應期待地板歸零。
        """
        if gamma == 0:
            raise ValueError("gamma=0 使 BDIA 退化為不可反解的 DDIM，無意義")
        abar = self.alphas_cumprod(z0.device)
        z_prev = z0                                        # z_{i−1}
        eps0 = self._eps(z0, ts[0], emb)
        z_cur = self._ddim_step(z0, eps0, ts[0], ts[1], abar)   # z_1，普通 DDIM

        for i in range(1, steps):
            eps = self._eps(z_cur, ts[i], emb)
            a_minus = self._ddim_step(z_cur, eps, ts[i], ts[i - 1], abar)
            a_plus = self._ddim_step(z_cur, eps, ts[i], ts[i + 1], abar)
            z_next = gamma * z_prev - gamma * a_minus + a_plus
            z_prev, z_cur = z_cur, z_next

        return z_cur, z_prev            # (z_K, z_{K−1})

    def bdia_denoise(
        self,
        z_pair: tuple,
        emb: torch.Tensor,
        ts: torch.Tensor,
        steps: int,
        eps_hook: Optional[EpsHook] = None,
        gamma: float = 1.0,
        use_ckpt: bool = False,
        collect_x0: bool = False,
    ):
        """BDIA 去噪：(z_K, z_{K−1}) → z₀，為 `bdia_inversion` 的精確反解。

        由上式解出下行遞迴：

            z_{i−1} = (1/γ)·(z_{i+1} − DDIM(z_i, t_i→t_{i+1})) + DDIM(z_i, t_i→t_{i−1})

        **注入點比 DDIM 少一個。** 迴圈跑 i=K−1…1 共 K−1 步，而 `denoise`
        跑 K 步，因為 BDIA 不需要反解反演的第 0 步。故 `eps_hook` 收到的
        `step_idx` 為 0…K−2，site L 那類以 steps 為第一維的模塊會有一格
        用不到。留著那一格而非改動模塊的形狀：模塊的參數量因此在兩種反演
        之間保持一致，比較才不會多一個變因。
        """
        if gamma == 0:
            raise ValueError("gamma=0 使 BDIA 退化為不可反解的 DDIM，無意義")
        abar = self.alphas_cumprod(z_pair[0].device)
        z_next, z_cur = z_pair          # z_{i+1}, z_i，起始 i=K−1
        x0_list = []

        for step_idx, i in enumerate(range(steps - 1, 0, -1)):
            eps = self._eps(z_cur, ts[i], emb, use_ckpt=use_ckpt)
            if eps_hook is not None:
                eps = eps_hook(eps, step_idx, ts[i])
            a_plus = self._ddim_step(z_cur, eps, ts[i], ts[i + 1], abar)
            a_minus = self._ddim_step(z_cur, eps, ts[i], ts[i - 1], abar)
            if collect_x0:
                x0_list.append(
                    (z_cur - (1 - abar[ts[i]]).sqrt() * eps) / abar[ts[i]].sqrt()
                )
            z_prev = (z_next - a_plus) / gamma + a_minus
            z_next, z_cur = z_cur, z_prev

        return z_cur, x0_list

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
        guidance_scale: float = 1.0,
        emb_uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """可微分 SDEdit。

        `noise` 由呼叫端提供且**必須**在比較的兩條分支間共用（spec §5.1）。
        不在此處抽樣，是為了讓「兩分支共用同一個 ε」成為介面上的硬性要求，
        而非仰賴呼叫端自律。

        `use_ckpt` 控制 UNet、`vae_ckpt` 控制 VAE，兩者分開是為了讓 E0 能
        分別歸因記憶體，不是為了提供選項。

        `guidance_scale` 為 classifier-free guidance 的權重 w，見 `_eps_cfg`。
        **預設維持 1.0（即無 CFG）**，這樣既有 53 個 run 的數值可重現；但
        1.0 正是 E26 找到的缺陷所在——攻擊方實際上會用 7.5 左右，w=1 時
        prompt 幾乎不起作用，SDEdit 退化成加噪再去噪。新的實驗必須明確指定。

        回傳 (1,3,H,W) [0,1]，計算圖保留。
        """
        abar = self.alphas_cumprod(x01.device)
        t0 = min(int(self.num_train_timesteps * strength), self.num_train_timesteps - 1)

        z = self.encode_image(x01, use_ckpt=vae_ckpt)
        z = abar[t0].sqrt() * z + (1 - abar[t0]).sqrt() * noise

        ts = torch.linspace(t0, 0, num_steps + 1).round().long()
        for i in range(num_steps):
            t, t_prev = ts[i], ts[i + 1]
            eps = self._eps_cfg(z, t, emb, guidance_scale, emb_uncond,
                                use_ckpt=use_ckpt)
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
