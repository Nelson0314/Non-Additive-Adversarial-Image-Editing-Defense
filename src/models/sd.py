"""Stable Diffusion 封裝 — spec §4.3。

提供三條路徑：

1. `ddim_inversion`：z₀ → z_{k}，確定性、無梯度（防禦生成的前段，殘差模塊關閉）
2. `denoise`：z_k → z₀，可注入逐步殘差（防禦生成的後段，殘差模塊開啟）
3. `sdedit`：攻擊者的編輯管線，可微分（殘差模塊關閉）

殘差注入以 callback 形式傳入，SDWrapper 不知道殘差如何產生 —— 這使
site L 與其他 site 共用同一條去噪迴圈。

所有影像張量介面為 [0,1]、(1,3,H,W)；內部轉為 SD VAE 的 [-1,1] 值域。

模型版本

`SDWrapper` 為 SD v1.x，`SDXLWrapper` 為 SDXL 1.0 base。兩者共用全部的
擴散迴圈，差異被收斂到三個覆寫點：`_load_pipeline`、`encode_text`、
`_unet_call`（含 `_cond_tensors`）。SDXL 的文字條件是 `SDXLPrompt`
而非裸張量，理由見該類別的 docstring。

**BDIA 在 SDXL 上尚未驗證。** 代數本身與模型無關，且 SDXL base 的排程
參數與 SD v1.x 相同（beta_start 0.00085、beta_end 0.012、scaled_linear、
1000 步），故 `alphas_cumprod` 逐元素相同、`_ddim_step` 不需任何改動。
但「來回誤差比 DDIM 小 2 個數量級以上」是一個數值性質，取決於 ε 預測的
條件數，必須在 GPU 上以真實 SDXL 權重重跑
`tests/test_pipeline.py::test_BDIA來回誤差遠小於DDIM` 的等價檢查才算數。
半精度會直接衝擊這一項：`_eps` 已把狀態張量保持在 fp32，但 ε 本身在
fp16 下只有 10 bit 尾數，該測試必須在 fp16 與 fp32 各跑一次並比較。
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.utils.checkpoint as ckpt

from src.utils.device import bf16_supported, get_device, resolve_precision

# eps_hook(eps, step_idx, t) -> eps'，回傳修改後的噪聲預測
EpsHook = Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]


def cat_cond(items: Sequence):
    """把數個文字條件沿批次維串接，裸張量與 `SDXLPrompt` 都能處理。

    存在的理由：`src/baselines/` 的 AdvPaint 與 PromptFlare 需要自建兩列
    批次。在 SD v1.x 上那就是 `torch.cat([...])`，但 SDXL 的條件是
    `SDXLPrompt`（序列嵌入 + pooled 嵌入兩件），`torch.cat` 會直接拋
    `TypeError`，而 `.repeat()` 這種張量方法在 `SDXLPrompt` 上根本不存在。

    寫成函式而非要求呼叫端 `isinstance` 判斷，是因為那種判斷會漏掉
    pooled 那一半——症狀是 UNet 的 additive embedding 只拿到一列條件，
    兩列共用同一組 pooled，看起來像「CFG 效果怪怪的」而沒有錯誤訊息。
    """
    items = list(items)
    if not items:
        raise ValueError("cat_cond 收到空序列")
    first = items[0]
    if isinstance(first, SDXLPrompt):
        return SDXLPrompt(
            torch.cat([p.embeds for p in items]),
            torch.cat([p.pooled for p in items]),
        )
    return torch.cat(items)


def expand_cond(emb, batch: int):
    """把單列的文字條件擴成 `batch` 列。裸張量與 `SDXLPrompt` 都能處理。

    對應 SD v1.x 的 `emb.expand(batch, -1, -1)`。`SDXLPrompt` 沒有 `expand`，
    且 pooled 是二維（B, 1280）與序列嵌入的三維不同，兩者要分別處理。
    """
    if isinstance(emb, SDXLPrompt):
        return SDXLPrompt(
            emb.embeds.expand(batch, -1, -1), emb.pooled.expand(batch, -1)
        )
    return emb.expand(batch, -1, -1)


class SDWrapper:
    """單一 SD 模型。防禦與編輯共用同一組權重，差別只在殘差模塊開關。

    `dtype` 是**計算精度**，不是全部子模組的 dtype。VAE 的 dtype 由
    `src.utils.device.resolve_precision` 決定：fp16 時 VAE 必須留在 fp32，
    否則 SDXL 的 VAE 會溢位成全黑圖。預設 fp32 時三者相同，故 E15–E23 的
    既有數字逐位元不變。

    `pipe` 供測試注入已建好的 pipeline，避免為了驗結構而下載權重。給定
    `pipe` 時 `model_name` 只作為記錄用的標籤。
    """

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.float32,
        pipe=None,
    ):
        self.model_name = model_name
        self.device = get_device()
        self.compute_dtype = dtype
        self.backbone_dtype, self.vae_dtype = resolve_precision(dtype)
        if dtype == torch.bfloat16 and not bf16_supported(self.device):
            raise RuntimeError(
                f"裝置 {self.device} 不支援 bf16（V100 是 sm_70，沒有 bf16 硬體）。"
                "請改用 float16——該精度下 VAE 會自動留在 fp32"
            )

        self.pipe = (pipe if pipe is not None
                     else self._load_pipeline(model_name, self.backbone_dtype))
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self._apply_precision()
        # SD 全部凍結：φ 是唯一可訓練參數（spec §5.3）
        for m in self._frozen_modules():
            m.requires_grad_(False)
            m.eval()

    @staticmethod
    def _load_pipeline(model_name: str, dtype: torch.dtype):
        from diffusers import StableDiffusionPipeline

        return StableDiffusionPipeline.from_pretrained(
            model_name,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=dtype,
        )

    def _frozen_modules(self) -> list:
        return [self.unet, self.vae, self.text_encoder]

    def _apply_precision(self) -> None:
        """把 §resolve_precision 的規則實際施加到子模組上。

        只搬 dtype，不換任何權重檔。`self.vae.to(self.vae_dtype)` 與
        「載入另一份 VAE」是兩件完全不同的事，後者在本專案是被禁止的
        （威脅模型要求 stock SDXL），程式中不存在該路徑。
        """
        self.unet.to(self.backbone_dtype)
        self.vae.to(self.vae_dtype)
        for enc in self._text_encoders():
            enc.to(self.backbone_dtype)

    def _text_encoders(self) -> list:
        return [self.text_encoder]

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
        是各次的總和；做了 checkpoint 之後，反向時一次只重算一個區塊，
        peak 降為各次的最大值。單獨一次呼叫並不會因此變省。
        """
        # 半精度時 z 由 UNet 產出（bf16／fp16），而 VAE 可能被 resolve_precision
        # 留在 fp32，兩者相接處必須明確轉換。不轉的話 diffusers 會以
        # "expected scalar type" 中止；用 autocast 蓋掉則會讓 VAE 實際跑在
        # 哪個精度變得看不出來。fp32 全程時 `.to` 回傳原張量，數值不變。
        z = z.to(self.vae.dtype)
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

    def uncond_prompt(self, batch: int = 1) -> torch.Tensor:
        """CFG 的無條件分支。

        SD v1.x 沒有 `force_zeros_for_empty_prompt`——其無條件嵌入就是空
        prompt 的 CLIP 編碼（`[BOS][EOS][PAD]×75`，非零）。故此處等同
        `encode_text("")`。

        存在的理由是**介面一致**：`optimize.py` 對兩種 wrapper 走同一段程式，
        由各自的 `uncond_prompt()` 決定該回傳什麼。SDXL 覆寫它以回傳零張量
        （其 `force_zeros_for_empty_prompt = true`）。呼叫端因此不需要
        判斷自己拿到的是哪一種模型——那種判斷正是會寫錯而且沒有症狀的地方。

        `batch` 供介面對齊；SD v1.x 的嵌入不依 batch 改變，故忽略。
        """
        return self.encode_text("")

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

        這個函式是 2026-08-01（E26）新增的，補的是一個使整個威脅模型失效的
        缺漏。修訂前全專案沒有任何 CFG：`_eps` 只以條件嵌入呼叫一次 UNet，
        等同 w = 1。Stable Diffusion v1.x 是在 CFG 下訓練也在 CFG 下使用
        的，w = 1 時 prompt 對輸出的影響極弱，SDEdit 退化成「加噪再去噪」。
        實測後果見 `docs/RESULTS_E25-E31.md`：CLIP(原圖) 0.2030 →
        CLIP(所謂的編輯結果) 0.2132，只升 0.0101 而標準差 0.0169，即編輯
        根本沒有發生；使用者對 `runs/p5_semantic_axis/compare.html` 的判讀是
        「連原始圖片被文字編輯都沒有成功」。

        也就是說 E2–E23 全部是在防禦一個不存在的攻擊，量到的 `net_lpips`
        是兩次隨機去噪之間的漂移。

        w = 1.0 時不走這條路徑，直接回到單次前向：`_eps` 的行為必須逐位元
        不變，否則既有 53 個 run 的可重現性會斷掉。

        兩次前向分開做而非合批：合批把激活加倍，而 E0 已量出 512² 下記憶體
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

    def _cond_tensors(self, emb) -> Tuple[torch.Tensor, ...]:
        """把條件拆成一串**張量**，供 `_unet_call` 與 checkpoint 使用。

        必須是張量而非容器：`torch.utils.checkpoint` 只認得引數串中的
        張量，藏在 dataclass 或 dict 裡的張量在反向重算時拿不到梯度，
        而且不會報錯——症狀是 φ 的梯度悄悄變成零。SDXL 的 pooled 嵌入
        正是這種會被藏起來的張量，故此處把它攤平成獨立引數。
        """
        return (emb,)

    def _unet_call(self, z, t, *cond, **unet_kwargs) -> torch.Tensor:
        return self.unet(
            z, t, encoder_hidden_states=cond[0], **unet_kwargs
        ).sample

    def unet_forward(self, z, t, emb, **unet_kwargs) -> torch.Tensor:
        """一次 UNet 前向，條件的內部結構由 `_cond_tensors`／`_unet_call` 吸收。

        供**必須自建批次**的呼叫端使用。`_eps_cfg` 做的是兩次分開的前向，
        產不出「同一批次內兩列的條件不同」這種輸入，而 AdvPaint（`[uncond; cond]`
        做 CFG）與 PromptFlare（`[完整注意力; 只看 BOS]`）都需要在**同一次**
        前向裡取得兩列並 `chunk(2)`——它們要 hook 的注意力就發生在那一次前向。

        直接呼叫 `self.unet(...)` 是不行的：SDXL 的條件是 `SDXLPrompt`，
        UNet 另需 `added_cond_kwargs`（pooled 嵌入 + time_ids），少傳會直接
        報錯；而在 SD v1.x 上同樣的程式碼卻能跑。經由本方法即與模型無關。

        `unet_kwargs` 原樣轉交 UNet，例如 PromptFlare 的 `encoder_attention_mask`。
        該遮罩作用在 **token 軸（77）**，SDXL 串接兩個 encoder 改變的是最後
        一維（768+1280=2048），token 軸不變，故語意在兩種模型上一致。
        """
        zc = z.to(self.unet.dtype)
        cond = tuple(c.to(self.unet.dtype) for c in self._cond_tensors(emb))
        return self._unet_call(zc, t, *cond, **unet_kwargs).to(z.dtype)

    def _eps(self, z, t, emb, use_ckpt: bool = False) -> torch.Tensor:
        """ε 預測。半精度時輸入轉成骨幹 dtype，輸出轉回 z 的 dtype。

        輸出轉回去是刻意的：DDIM 與 BDIA 的遞迴在 `alphas_cumprod`（fp32）
        上做加減，狀態張量留在 fp32 才不會逐步累積半精度的捨入誤差，而
        BDIA 的整個存在理由就是數值精確性。fp32 全程時兩次 `.to` 都回傳
        原張量，故 E15–E23 的既有數字逐位元不變。
        """
        zc = z.to(self.unet.dtype)
        cond = tuple(c.to(self.unet.dtype) for c in self._cond_tensors(emb))
        if use_ckpt:
            out = ckpt.checkpoint(
                self._unet_call, zc, t, *cond, use_reentrant=False
            )
        else:
            out = self._unet_call(zc, t, *cond)
        return out.to(z.dtype)

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
        """BDIA 反演：z₀ → (z_K, z_{K−1})。回傳一對狀態，不是單一張量。

        BDIA（Zhang et al., "Exact Diffusion Inversion via Bi-directional
        Integration Approximation", arXiv 2307.10829, ECCV 2024）把 DDIM 的
        單步遞迴改成跨兩步的遞迴：

            z_{i+1} = γ·z_{i−1} − γ·DDIM(z_i, t_i→t_{i−1}) + DDIM(z_i, t_i→t_{i+1})

        兩個 DDIM 步都只用到在 z_i 處的同一次 ε 預測，故給定 (z_{i−1}, z_i)
        可解出 z_{i+1}，給定 (z_i, z_{i+1}) 也可解出 z_{i−1}——兩個方向都是
        代數上的精確反解，不是近似。γ=1 為標準選擇。

        既有的 `ddim_inversion` 不是精確的：反演時 ε 在 z_i 評估、去噪時在
        z_{i+1} 評估，兩者不同，誤差逐步累積。實測 t_max=500、k_inv=20 下
        `G(x; φ=0)` 與原圖相差 LPIPS 0.194 / PSNR 26.56 dB。

        必須回傳一對狀態。遞迴的狀態是相鄰兩點；只交出 z_K，去噪端就得
        自己補一個近似的起手步，精確性當場失去。第 0 步（z₀→z₁）沒有前一點
        可用，以普通 DDIM 走，去噪端也不需要反解它——去噪只跑 i=K−1…1，
        正好是反演用到 BDIA 遞迴的那些步反過來，故整條來回是精確的。

        精確的只有擴散這一段。`G(x;0)` 仍要經過 VAE 的編碼與解碼，其
        來回誤差實測為 PSNR 27.51 dB / LPIPS 0.143，BDIA 不改變這一項。
        故本方法把重建誤差下限由 LPIPS 0.194 降到 0.143 為止，仍高於像素側加性
        位置實際運作的 0.063。採用與否應以此為準，不應期待下限歸零。
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

        注入點比 DDIM 少一個。迴圈跑 i=K−1…1 共 K−1 步，而 `denoise`
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
        step_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """可微分 SDEdit。

        `noise` 由呼叫端提供且必須在比較的兩條分支間共用（spec §5.1）。
        不在此處抽樣，是為了讓「兩分支共用同一個 ε」成為介面上的硬性要求，
        而非仰賴呼叫端自律。

        `use_ckpt` 控制 UNet、`vae_ckpt` 控制 VAE，兩者分開是為了讓 E0 能
        分別歸因記憶體，不是為了提供選項。

        `guidance_scale` 為 classifier-free guidance 的權重 w，見 `_eps_cfg`。
        預設維持 1.0（即無 CFG），這樣既有 53 個 run 的數值可重現；但
        1.0 正是 E26 找到的缺陷所在——攻擊方實際上會用 7.5 左右，w=1 時
        prompt 幾乎不起作用，SDEdit 退化成加噪再去噪。新的實驗必須明確指定。

        `step_hook(i, t, pred_x0)` 於每一步算出 x̂₀ 之後呼叫，供呼叫端取
        中間圖與 attention map（`CODE` §4.1、§4.2 要求兩者都要留存）。
        它**不能**改變 `z`：回傳值被忽略，簽名上就不提供修改的途徑。
        需要改 ε 的是防禦端的 `eps_hook`，那是另一件事、在另一條路徑上。

        回傳 (1,3,H,W) [0,1]，計算圖保留。
        """
        abar = self.alphas_cumprod(x01.device)
        t0 = min(int(self.num_train_timesteps * strength), self.num_train_timesteps - 1)

        z = self.encode_image(x01, use_ckpt=vae_ckpt)
        z = abar[t0].sqrt() * z + (1 - abar[t0]).sqrt() * noise

        ts = torch.linspace(t0, 0, num_steps + 1).round().long()
        for i in range(num_steps):
            t, t_prev = ts[i], ts[i + 1]
            if step_hook is not None:
                # 取樣旗標必須在前向**之前**設好：注意力是在 UNet 前向當中
                # 擷取的，前向跑完再問「這一步要不要記」已經來不及。
                step_hook(i, t, None)
            eps = self._eps_cfg(z, t, emb, guidance_scale, emb_uncond,
                                use_ckpt=use_ckpt)
            pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
            if step_hook is not None:
                step_hook(i, t, pred_x0)
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


# ===========================================================================
# SDXL
# ===========================================================================


@dataclass(frozen=True)
class SDXLPrompt:
    """SDXL 的文字條件：一條序列嵌入加上一個 pooled 嵌入。

    SDXL 有兩個 text encoder。CLIP-L 的倒數第二層隱狀態（768 維）與
    OpenCLIP-bigG 的倒數第二層隱狀態（1280 維）在最後一維串接成
    (B, 77, 2048)，即 `unet.config.cross_attention_dim`；bigG 的投影輸出
    另外給出一個 (B, 1280) 的 pooled 嵌入，經 `added_cond_kwargs` 與
    micro-conditioning 的 time_ids 一起進入 UNet 的 additive embedding，
    **不進 cross-attention**。

    兩者必須綁在一起傳遞，理由是 CFG：`_eps_cfg` 對條件與無條件各做一次
    前向，兩次的 pooled 不同。若 pooled 改由 wrapper 的狀態提供，無條件那
    一支會拿到條件的 pooled，而症狀只是「CFG 的效果怪怪的」，沒有任何錯誤
    訊息。綁成一個物件之後，走錯配對在型別上就不可能發生。

    `emb_residual`（site E）的作用對象**只有 `embeds`**——那是 cross-attention
    讀的序列，也是文字定位發生的地方。`__add__` 因此只加在 `embeds` 上，
    使 `generator.py` 的 `emb = emb + d_emb` 不必修改即有正確語意。
    """

    embeds: torch.Tensor      # (B, 77, 2048)
    pooled: torch.Tensor      # (B, 1280)

    @property
    def shape(self):
        """對外表現為序列嵌入的形狀，讓 `EmbeddingResidual` 的建構不需改動。"""
        return self.embeds.shape

    @property
    def dtype(self):
        return self.embeds.dtype

    @property
    def device(self):
        return self.embeds.device

    def detach(self) -> "SDXLPrompt":
        return SDXLPrompt(self.embeds.detach(), self.pooled.detach())

    def to(self, *a, **kw) -> "SDXLPrompt":
        return SDXLPrompt(self.embeds.to(*a, **kw), self.pooled.to(*a, **kw))

    def __add__(self, delta: torch.Tensor) -> "SDXLPrompt":
        if not isinstance(delta, torch.Tensor):
            raise TypeError(
                f"SDXLPrompt 只能加上序列嵌入的殘差張量，收到 {type(delta).__name__}"
            )
        if delta.shape[-1] != self.embeds.shape[-1]:
            raise ValueError(
                f"殘差的最後一維 {delta.shape[-1]} 與序列嵌入的 "
                f"{self.embeds.shape[-1]} 不符。site E 的作用對象是那條 2048 維"
                "的序列，不是 1280 維的 pooled 嵌入"
            )
        return SDXLPrompt(self.embeds + delta, self.pooled)


class SDXLWrapper(SDWrapper):
    """SDXL 1.0 base。與 `SDWrapper` 共用全部的擴散迴圈，只換三件事：

    1. pipeline 類別（`StableDiffusionXLImg2ImgPipeline`，無 safety checker 參數）
    2. 文字編碼（兩個 encoder，回傳 `SDXLPrompt`）
    3. UNet 的呼叫（多一組 `added_cond_kwargs`）

    `ddim_inversion`／`denoise`／`bdia_*`／`sdedit` 一行都不用改：它們只把
    `emb` 原樣轉交給 `_eps`，而條件的內部結構由 `_cond_tensors` 與
    `_unet_call` 這兩個覆寫點吸收。

    排程與 SD v1.x 相同（beta_start 0.00085、beta_end 0.012、scaled_linear、
    1000 步），故 `alphas_cumprod` 逐元素相同，DDIM 與 BDIA 的代數不變。
    SDXL 預設的 `EulerDiscreteScheduler` 仍持有 `alphas_cumprod`，本封裝
    只讀該陣列、不呼叫 scheduler 的 step，故排程器種類不影響結果。
    """

    @staticmethod
    def _load_pipeline(model_name: str, dtype: torch.dtype,
                       variant: Optional[str] = "fp16"):
        """載入 stock SDXL。`variant` 選官方 repo 內的權重檔變體。

        **`variant` 不是換模型。** `stabilityai/stable-diffusion-xl-base-1.0`
        的官方 repo 同時提供兩組權重檔：預設的 fp32（`*.safetensors`）與
        fp16 變體（`*.fp16.safetensors`）。兩者都是官方發布的同一個模型，
        差別只在存檔精度。這與 `sdxl-vae-fp16-fix` 完全不同——後者是**第三方
        重新訓練的 VAE**，換它就是換模型，故被禁止（`ARCH` §7.1）。

        預設取 fp16 變體，理由有二：

        1. **更貼近真實威脅模型。** 攻擊方是「使用 stock SDXL 的一般使用者」，
           而實務上絕大多數 SDXL 應用載入的就是 fp16 變體——它只有一半大小
           且是 diffusers 文件的建議用法。
        2. 本輪一律以 bf16 執行（RTX 5090 支援），fp32 權重檔的多餘精度
           在載入時就會被截掉，多下載 7 GB 換不到任何數值差異。

        `variant=None` 取 fp32 檔。段 0 的精度等價性驗證若要以 fp32 權重
        為基準，由呼叫端明給。**該驗證比較的是「同一組權重在不同計算精度下
        的結果」**，故用哪一組權重存檔並不影響其結論。

        2026-08-05 新增。before：無 `variant` 參數，載入 ModelScope 鏡像的
        fp16 檔案時以 `OSError: no file named model.safetensors` 失敗。
        """
        from diffusers import StableDiffusionXLImg2ImgPipeline

        return StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_name, torch_dtype=dtype, add_watermarker=False,
            variant=variant,
        )

    def _frozen_modules(self) -> list:
        return [self.unet, self.vae, self.text_encoder, self.text_encoder_2]

    def _text_encoders(self) -> list:
        return [self.text_encoder, self.text_encoder_2]

    # ---- 元件 ----

    @property
    def text_encoder_2(self):
        return self.pipe.text_encoder_2

    @property
    def tokenizer_2(self):
        return self.pipe.tokenizer_2

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    # ---- 文字條件 ----

    @property
    def force_zeros_for_empty_prompt(self) -> bool:
        """stock SDXL base 的 `model_index.json` 記為 `true`（已查證）。

        其意義是：**空 prompt 的無條件分支不是 `encode_text("")`，而是
        `torch.zeros_like`**。diffusers 的 `encode_prompt` 依此旗標分派。

        這對本專案有兩個後果：

        1. 攻擊方必須是 stock SDXL。若我們用 `encode_text("")` 當 CFG 的
           無條件分支，模擬的就不是 stock 行為，威脅模型不成立。
        2. **prompt-free 的著力點需要重新檢視**（`DESIGN` §2.1）。
           在 SD v1.x 上，空 prompt 的 CLIP 編碼是 `[BOS][EOS][PAD]×75` 的
           非零嵌入，把注意力質量導向 `[BOS]` 是有意義的；在 SDXL 上，
           若無條件分支恆為零，該 token 位置在無條件分支中不承載任何東西。
           N1 的注意力目標必須取自**條件分支**而非無條件分支。

        由 pipeline 的設定讀取而非寫死：refiner 與其他檢查點的值可能不同，
        寫死會在換模型時靜默給出錯誤的無條件分支。
        """
        cfg = getattr(self.pipe, "config", None)
        if cfg is None:
            raise RuntimeError("pipeline 沒有 config，無法判定無條件分支的構造")
        value = getattr(cfg, "force_zeros_for_empty_prompt", None)
        if value is None:
            raise RuntimeError(
                "pipeline 的 config 缺少 force_zeros_for_empty_prompt。"
                "不預設任何一邊——猜錯會讓 CFG 的無條件分支與 stock 行為不符，"
                "而該差異沒有任何症狀"
            )
        return bool(value)

    def uncond_prompt(self, batch: int = 1) -> SDXLPrompt:
        """CFG 的無條件分支，依 stock SDXL 的規則產生。

        `force_zeros_for_empty_prompt=True` 時回傳零張量，與 diffusers 的
        `encode_prompt` 一致；否則回傳 `encode_text("")`。

        **不要直接用 `encode_text("")` 當無條件分支**——在 stock SDXL base 上
        那是錯的，且錯得沒有症狀：影像仍然生得出來，只是攻擊方不再是 stock 模型。
        """
        if not self.force_zeros_for_empty_prompt:
            return self.encode_text("")
        dim = self.unet.config.cross_attention_dim
        pooled_dim = self.text_encoder_2.config.projection_dim
        length = self.tokenizer.model_max_length
        # 取 backbone 的 dtype：無條件分支要餵進 UNet，與條件分支同一路徑。
        # VAE 在 fp16 下另有自己的 dtype（見 `resolve_precision`），與此無關。
        kw = dict(device=self.device, dtype=self.backbone_dtype)
        return SDXLPrompt(
            torch.zeros(batch, length, dim, **kw),
            torch.zeros(batch, pooled_dim, **kw),
        )

    def encode_text(self, prompt: str) -> SDXLPrompt:
        """回傳 (B,77,2048) 的序列嵌入與 (B,1280) 的 pooled 嵌入。

        兩個 encoder 都取**倒數第二層**的隱狀態（`hidden_states[-2]`），與
        diffusers 的 `StableDiffusionXLPipeline.encode_prompt` 一致；pooled
        只取自第二個 encoder（`CLIPTextModelWithProjection` 的 `text_embeds`）。
        取最後一層或改取第一個 encoder 的 pooled 都會得到能跑但語意不同的
        條件，且沒有任何症狀。

        **這不是 CFG 的無條件分支。** stock SDXL base 的
        `force_zeros_for_empty_prompt=True`，其無條件分支是零張量；
        取無條件分支請用 `uncond_prompt()`。
        """
        parts = []
        pooled = None
        pairs = [
            (self.tokenizer, self.text_encoder),
            (self.tokenizer_2, self.text_encoder_2),
        ]
        for tok, enc in pairs:
            ids = tok(
                prompt,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            out = enc(ids, output_hidden_states=True)
            parts.append(out.hidden_states[-2])
            pooled = out[0]          # 迴圈結束後留下的是第二個 encoder 的
        embeds = torch.cat(parts, dim=-1)

        expect = self.unet.config.cross_attention_dim
        if embeds.shape[-1] != expect:
            raise ValueError(
                f"兩個 text encoder 串接後為 {embeds.shape[-1]} 維，UNet 期待 "
                f"{expect} 維。表示載入的 encoder 組合與此 UNet 不相配"
            )
        return SDXLPrompt(embeds, pooled)

    # ---- micro-conditioning ----

    def _time_ids(self, z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """SDXL base 的 6 維 micro-conditioning：
        (original_h, original_w, crop_top, crop_left, target_h, target_w)。

        由 latent 的空間尺寸乘以 VAE 的下採樣倍率反推，故不需要另外保存
        解析度狀態，也不可能與實際輸入的影像對不上。crop 取 (0,0)：本專案
        不做隨機裁切，宣告成其他值等於對模型謊報影像的來源。

        refiner 用的是 5 維（含 aesthetic score），本封裝只用 base，
        `add_embedding` 的輸入維度會在下方核對，配錯會當場拋出而非跑出
        品質莫名其妙的結果。
        """
        f = self.vae_scale_factor
        h, w = z.shape[-2] * f, z.shape[-1] * f
        ids = torch.tensor([[h, w, 0, 0, h, w]], dtype=dtype, device=z.device)
        return ids.expand(z.shape[0], -1)

    def _check_added_cond_dim(self, pooled: torch.Tensor, n_time_ids: int) -> None:
        """UNet 的 additive embedding 輸入寬度必須等於
        `addition_time_embed_dim × time_ids 個數 + pooled 維度`。

        SDXL base 為 256×6 + 1280 = 2816。time_ids 給成 refiner 的 5 維時
        總寬變 2560，diffusers 會以形狀不符中止——但錯誤發生在 UNet 深處，
        訊息看不出是 micro-conditioning 配錯。此處先核對，讓原因明確。
        """
        cfg = self.unet.config
        total = cfg.addition_time_embed_dim * n_time_ids + pooled.shape[-1]
        if total != cfg.projection_class_embeddings_input_dim:
            raise ValueError(
                f"added_cond 的總維度 {total}（time_ids {n_time_ids}×"
                f"{cfg.addition_time_embed_dim} + pooled {pooled.shape[-1]}）"
                f"與 UNet 的 projection_class_embeddings_input_dim "
                f"{cfg.projection_class_embeddings_input_dim} 不符"
            )

    # ---- UNet 呼叫 ----

    def _cond_tensors(self, emb) -> Tuple[torch.Tensor, ...]:
        if not isinstance(emb, SDXLPrompt):
            raise TypeError(
                "SDXL 的條件必須是 SDXLPrompt（序列嵌入 + pooled 嵌入）。"
                "收到裸張量表示呼叫端只帶了 cross-attention 那一半，"
                f"added_cond_kwargs 會缺 text_embeds。實得 {type(emb).__name__}"
            )
        return (emb.embeds, emb.pooled)

    def _unet_call(self, z, t, *cond, **unet_kwargs) -> torch.Tensor:
        embeds, pooled = cond
        time_ids = self._time_ids(z, pooled.dtype)
        self._check_added_cond_dim(pooled, time_ids.shape[-1])
        if pooled.shape[0] != z.shape[0]:
            # `_time_ids` 依 z 的批次大小產生，pooled 由呼叫端提供。兩者不符
            # 表示自建批次的呼叫端（AdvPaint／PromptFlare）只複製了 latent
            # 沒複製條件。additive embedding 會拿到與 latent 對不上的列數，
            # 而 UNet 不一定會拒絕——廣播之後兩列共用同一組條件，症狀是
            # 「CFG 好像沒作用」，沒有錯誤訊息。
            raise ValueError(
                f"latent 的批次為 {z.shape[0]}，pooled 嵌入為 {pooled.shape[0]}。"
                "自建批次時 latent 與條件必須複製成相同的列數"
            )
        return self.unet(
            z,
            t,
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            **unet_kwargs,
        ).sample
