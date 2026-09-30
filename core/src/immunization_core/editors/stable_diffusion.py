"""Stable Diffusion 的編碼、反演、去噪與編輯 adapter。

影像為 (N,3,H,W)、[0,1]；遮罩為 1 表示重繪。透過 callback 注入殘差，
不依賴防禦或 baseline 實作。建構時依指定 model_name 載入權重；匯入不載入。
取樣、梯度、VAE scaling 與 precision 契約保留於各方法。"""

import contextlib
from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint as ckpt

from immunization_core.runtime.device import bf16_supported, get_device, resolve_precision

EpsHook = Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]


class SDWrapper:
    """單一 SD 模型。防禦與編輯共用同一組權重，差別只在殘差模塊開關。

    `dtype` 是**計算精度**，不是全部子模組的 dtype。VAE 的 dtype 由
    `immunization_core.runtime.device.resolve_precision` 決定：fp16 時 VAE 必須留在 fp32，
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

        # 9 通道權重下 `_eps` 要補的後 5 個通道，由 `inpaint_conditioning()`
        # 設定。**沒有它時 `_eps` 拒絕跑**，理由見該方法。
        self._inpaint_cond: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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

    @contextlib.contextmanager
    def offloaded(self):
        """暫時把權重搬到 CPU，讓另一份權重能獨占顯存。離開時搬回。

        唯一的用途是段 0 的 `calibrate_precision_equiv`：它要在同一台機器上
        比對 bf16 與 fp32 兩條計算路徑，而兩份 SDXL 同時常駐在 24 GB 的卡上
        放不下（2026-08-06 於 RTX 3090 實測 OOM，見下）。

        **只搬裝置不動 dtype。** `nn.Module.to("cpu")` 不帶 dtype 引數時只換
        裝置，`_apply_precision` 施加的混合精度（backbone 半精度、VAE 依規則
        可能留 fp32）原樣保留，故搬回之後的數值路徑與搬走之前逐位元相同。

        走子模組而不走 `self.pipe.to()`：後者是 diffusers 的管線層策略，
        對半精度管線移往 CPU 另有規則；此處要的只是「把張量挪開」這件事。

        **搬走期間本封裝不可用。** `self.device` 仍指向 cuda，此時任何一次
        前向都會以裝置不符當場拋出——那正是要的失敗方式，不是靜默算錯。
        故本方法不做任何保護，由呼叫端保證區塊內不碰這個封裝。

        2026-08-06 新增。before：無此方法，`calibrate_precision_equiv` 在
        bf16 權重仍常駐時直接建構第二個 fp32 封裝，於 RTX 3090（23.56 GB）
        以 `torch.OutOfMemoryError` 失敗於 `decode_latent`（bf16 5.2 GB
        ＋ fp32 14 GB ＋ VAE 解碼活化，合計超過容量）。RTX 5090 的
        31.4 GB 放得下，故修正前未暴露。
        """
        if self.device.type != "cuda":
            yield
            return
        for m in self._frozen_modules():
            m.to("cpu")
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            for m in self._frozen_modules():
                m.to(self.device)

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
        實測後果見原 `docs/RESULTS_E25-E31.md`（不在 repo 內）：CLIP(原圖) 0.2030 →
        CLIP(所謂的編輯結果) 0.2132，只升 0.0101 而標準差 0.0169，即編輯
        根本沒有發生；使用者對 原 `runs/p5_semantic_axis/compare.html`（不在 repo 內） 的判讀是
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
        zc = self._latent_in(z.to(self.unet.dtype))
        cond = tuple(c.to(self.unet.dtype) for c in self._cond_tensors(emb))
        return self._unet_call(zc, t, *cond, **unet_kwargs).to(z.dtype)

    # ---- 9 通道權重的後 5 個通道 ----

    @contextlib.contextmanager
    def inpaint_conditioning(self, x01: torch.Tensor,
                             mask: Optional[torch.Tensor] = None,
                             vae_ckpt: bool = False):
        """在此區塊內，`_eps` 與 `unet_forward` 會自動補上後 5 個通道。

        2026-08-08 新增。inpainting 權重的 UNet 是 9 通道，而本專案有**十幾處**
        呼叫端餵的是 4 通道的 latent：防禦方自己的生成路徑（`DefenseGenerator`
        的反演與去噪）、Mist 與 DIA 的代理前向、精度等價診斷、注意力擷取。
        ip3 段 0 在跑了 2.5 小時之後才死在其中一處（N3 的階段一），錯誤是
        `expected input[1, 4, 64, 64] to have 9 channels`。

        **`mask=None` 表示「不重畫任何區域」**，即後 5 通道為 (全 0 遮罩,
        原圖的 latent)。那是**重建**路徑該用的條件：`DefenseGenerator` 要
        G(x; φ=0) 盡量等於 x，而整個保真預算的下限就建立在這件事上；餵全 1
        遮罩等於叫模型從噪聲重畫整張，G(x;0) 會離 x 很遠。同理，Mist 與 DIA
        原作把 UNet 當**整張影像的去噪器**用，它們沒有遮罩這個概念，換到
        9 通道權重之後最忠實的對應也是「不重畫任何區域」。

        要模擬**攻擊方**那一次呼叫時才傳真正的遮罩（評測期的注意力擷取），
        而攻擊本身走 `inpaint()`，它自己拼 9 通道，不經過這裡。

        巢狀時內層覆蓋外層，離開後還原。
        """
        if not self.is_inpainting:
            # 4 通道權重下設這個沒有意義。靜默接受會讓「這批到底是不是
            # inpainting」在呼叫端變得看不出來。
            raise RuntimeError(
                f"{self.model_name} 不是 inpainting 權重（UNet in_channels = "
                f"{self.unet.config.in_channels}），不需要也不接受這個 context"
            )
        if mask is None:
            mask = torch.zeros_like(x01[:, :1])
        prev = self._inpaint_cond
        self._inpaint_cond = self.mask_latents(x01, mask, vae_ckpt=vae_ckpt)
        try:
            yield
        finally:
            self._inpaint_cond = prev

    def conditioning_for(self, x01: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         vae_ckpt: bool = False):
        """與權重無關的入口：4 通道權重回傳空 context，9 通道走上面那個。

        呼叫端多半是模型無關的（防禦生成、baseline 的代理前向、診斷），
        寫 `if sd.is_inpainting` 分岔會在每一處重複同一段判斷。
        """
        if not self.is_inpainting:
            return contextlib.nullcontext()
        return self.inpaint_conditioning(x01, mask, vae_ckpt=vae_ckpt)

    def _latent_in(self, zc: torch.Tensor) -> torch.Tensor:
        """9 通道權重下把 4 通道的 latent 補成 9 通道。

        **沒有 `inpaint_conditioning` 就拋出，不預設補零。** 預設值會讓每一個
        還沒轉換的呼叫端靜默拿到一個「不重畫任何區域」的條件——對重建路徑
        那恰好是對的，對評測期的注意力擷取卻是錯的（那裡要的是攻擊方真正
        用的遮罩），而兩者的輸出都是一張合理的圖。
        """
        if not self.is_inpainting or zc.shape[1] != self.latent_channels:
            return zc
        m, z_masked = self._require_inpaint_cond()
        # CFG 把 batch 疊成兩份（條件與無條件），而條件本身逐影像只有一份。
        # 兩支走的是同一張影像與同一個遮罩，故沿 batch 複製即為正確。
        if m.shape[0] != zc.shape[0]:
            if zc.shape[0] % m.shape[0] != 0:
                raise ValueError(
                    f"latent 的 batch {zc.shape[0]} 不是 inpainting 條件 batch "
                    f"{m.shape[0]} 的整數倍，無法決定哪一份條件配哪一個樣本"
                )
            rep = zc.shape[0] // m.shape[0]
            m = m.repeat(rep, 1, 1, 1)
            z_masked = z_masked.repeat(rep, 1, 1, 1)
        return torch.cat([zc, m.to(zc.dtype), z_masked.to(zc.dtype)], dim=1)

    def _require_inpaint_cond(self):
        if self._inpaint_cond is None:
            raise RuntimeError(
                f"{self.model_name} 的 UNet 是 {self.unet.config.in_channels} "
                f"通道，收到 {self.latent_channels} 通道的 latent，而沒有作用中的 "
                "`inpaint_conditioning`。後 5 個通道必須由呼叫端明確決定："
                "重建路徑用 `mask=None`（不重畫任何區域），模擬攻擊方時傳"
                "攻擊真正用的遮罩。兩者都會產出一張合理的圖，補錯看不出來"
            )
        return self._inpaint_cond

    def _eps(self, z, t, emb, use_ckpt: bool = False) -> torch.Tensor:
        """ε 預測。半精度時輸入轉成骨幹 dtype，輸出轉回 z 的 dtype。

        輸出轉回去是刻意的：DDIM 與 BDIA 的遞迴在 `alphas_cumprod`（fp32）
        上做加減，狀態張量留在 fp32 才不會逐步累積半精度的捨入誤差，而
        BDIA 的整個存在理由就是數值精確性。fp32 全程時兩次 `.to` 都回傳
        原張量，故 E15–E23 的既有數字逐位元不變。

        9 通道權重下，4 通道的輸入由 `_latent_in` 依作用中的
        `inpaint_conditioning` 補齊；已經是 9 通道的（`inpaint()` 自己拼的、
        N1 的注意力前向）原樣通過。
        """
        zc = self._latent_in(z.to(self.unet.dtype))
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
        `step_idx` 為 0…K−2，latent 逐步注入 那類以 steps 為第一維的模塊會有一格
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
        keep01: Optional[torch.Tensor] = None,
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

        `keep01` (1,1,H,W) 給定時，值為 1 的區域在每一步都被換回**原圖在該
        時刻的帶噪 latent**，於是那塊區域不被編輯。這是 latent 混合，不是
        inpainting——UNet 仍是 4 通道的 stock 權重，威脅模型仍是 img2img。
        存在的理由：strength 0.8 下 SDEdit 會把人臉整個換掉，未防禦的編輯
        連身分都不剩，「防禦保護了什麼」就無從談起（使用者 2026-08-17 裁定
        人臉要遮罩）。遮罩要羽化，硬邊會在混合處留下接縫。

        回傳 (1,3,H,W) [0,1]，計算圖保留。
        """
        abar = self.alphas_cumprod(x01.device)
        t0 = min(int(self.num_train_timesteps * strength), self.num_train_timesteps - 1)

        z_src = self.encode_image(x01, use_ckpt=vae_ckpt)
        z = abar[t0].sqrt() * z_src + (1 - abar[t0]).sqrt() * noise

        keep = None
        if keep01 is not None:
            keep = torch.nn.functional.interpolate(
                keep01.to(device=z.device, dtype=z.dtype),
                size=z.shape[-2:], mode="area")

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
            if keep is not None:
                z_keep = (abar[t_prev].sqrt() * z_src
                          + (1 - abar[t_prev]).sqrt() * noise)
                z = keep * z_keep + (1.0 - keep) * z

        return self.decode_latent(z, use_ckpt=vae_ckpt)

    def sample_edit_noise(self, z_like: torch.Tensor, seed: int) -> torch.Tensor:
        """以固定 seed 產生編輯噪聲，供兩條分支共用。"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            z_like.shape, generator=g, dtype=z_like.dtype
        ).to(z_like.device)

    # ---- inpainting（威脅模型的第二種形態）----

    @property
    def is_inpainting(self) -> bool:
        """本模型的 UNet 吃不吃 9 通道輸入。

        SD 的 inpainting 權重把輸入擴成 `4（帶噪 latent）+ 1（遮罩）+ 4
        （遮罩後影像的 latent）= 9`；一般權重是 4。這個數是**模型自身的
        性質**，不是設定，故由 config 讀而不由呼叫端宣告——宣告錯的症狀是
        UNet 以形狀不符中止，而在通道數恰好對上的情況下會安靜地算錯。
        """
        return int(self.unet.config.in_channels) == self.inpaint_in_channels

    @property
    def inpaint_in_channels(self) -> int:
        """inpainting UNet 的輸入通道數：latent + 遮罩 + 遮罩後影像的 latent。"""
        return self.latent_channels + 1 + self.latent_channels

    @property
    def latent_channels(self) -> int:
        """latent 的通道數。**不可用 `unet.config.in_channels` 代替。**

        一般 UNet 兩者相同（都是 4），inpainting UNet 的 `in_channels` 是 9
        而 latent 仍是 4。`latent_shape` 修正前直接取 `in_channels`，在
        inpainting 權重上會回傳 9 通道的形狀，於是 `sample_edit_noise` 產生
        的噪聲與 latent 對不起來。VAE 的 `latent_channels` 才是真相來源。
        """
        return int(self.vae.config.latent_channels)

    def mask_latents(self, x01: torch.Tensor, mask: torch.Tensor,
                     vae_ckpt: bool = False
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """inpainting UNet 後 5 個通道的內容：(下採樣的遮罩, 遮罩後影像的 latent)。

        `mask` 為 (1,1,H,W)、值域 [0,1]，**1 表示要重畫的區域**（diffusers
        的 inpainting pipeline 同一約定）。

        遮罩以最近鄰下採樣到 latent 邊長。不用雙線性：遮罩是二值的指示函數，
        插值會在邊界產生 0 與 1 之間的值，那些格子既不算保留也不算重畫，
        而該誤差只表現為邊界一圈的品質異常，看不出來源。

        **遮罩後影像取「遮罩區歸零」再編碼**，不是先編碼再遮罩：VAE 是
        非線性的，兩者不等價，而 diffusers 的 pipeline 做的是前者。

        **歸零必須在模型的值域 `[-1, 1]` 上做，不是在 `[0, 1]` 上。**
        diffusers 的 `prepare_mask_latents` 對已經換算成 `[-1, 1]` 的影像乘
        `(mask < 0.5)`，故遮罩區成為 `[-1, 1]` 的 0，即**中灰**。在 `[0, 1]`
        上乘 `(1 − mask)` 會得到 0，經 `encode_image` 的 `x*2−1` 變成 **−1，
        也就是黑**——模型被告知「這個洞是黑的」，就照著把它畫黑。
        2026-08-14 實測：遮罩區亮度 0.0992，官方 pipeline 為 0.7361。
        故此處填 0.5（`[0, 1]` 的中灰），換算後恰為 0。
        """
        f = 2 ** (len(self.vae.config.block_out_channels) - 1)
        h, w = x01.shape[-2] // f, x01.shape[-1] // f
        m = torch.nn.functional.interpolate(
            mask.to(x01.device, x01.dtype), size=(h, w), mode="nearest")
        mk = mask.to(x01)
        z_masked = self.encode_image(x01 * (1.0 - mk) + 0.5 * mk,
                                     use_ckpt=vae_ckpt)
        return m, z_masked

    def inpaint(
        self,
        x01: torch.Tensor,
        mask: torch.Tensor,
        emb: torch.Tensor,
        noise: torch.Tensor,
        num_steps: int,
        use_ckpt: bool = False,
        vae_ckpt: bool = False,
        guidance_scale: float = 1.0,
        emb_uncond: Optional[torch.Tensor] = None,
        step_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """可微分的 inpainting。回傳 (1,3,H,W) [0,1]，計算圖保留。

        與 `sdedit` 的三個差別，每一個都改變防禦方的著力點：

        1. **沒有 strength。** 由純噪聲起跑、跑滿 `num_steps`，這是
           inpainting pipeline 的定義。`sdedit` 的 strength 是本專案為了把
           inpainting 專用的三篇 baseline 移植到 img2img 才引入的參數——
           五篇原始碼裡都沒有這個數（`photoguard.py:124`、`advpaint.py:227`、
           `promptflare.py:350` 各自拒絕預設值並寫明理由）。換到本路徑之後
           那個自由度消失。
        2. **原圖經由後 4 個通道進入**，而不是經由初始 latent。這正是
           AdvPaint 與 PromptFlare 原作梯度所走的那條路
           （`advpaint.py:43`「`masked_image_latents`（9 通道 inpainting
           輸入的後 4 通道）」）。
        3. **未遮罩的區域不在取樣迴圈裡貼回。** 9 通道模型靠的是後 5 個
           條件通道（遮罩 ＋ masked-image latent）告訴它哪裡要保留，官方
           pipeline 的逐步貼回被 `num_channels_unet == 4` 擋住
           （`diffusers 0.39.0` 的 `pipeline_stable_diffusion_inpaint.py`
           第 1294 行守衛、第 1307 行才是那一行混合）。
           **本方法曾經逐步貼回，那是錯的**：它把遮罩外的 latent 每一步壓回
           原圖，而那一塊在本專案的人像上佔 49–67%，模型幾乎沒有空間照
           prompt 生成，症狀是「輸出看起來就是原圖、只有背景微微變化」，
           而且跑的鏈與攻擊方實際跑的不是同一條。
           需要逐位元保留遮罩外的像素時，**在解碼之後合成一次**，由呼叫端做。

        `noise` 由呼叫端提供且必須在比較的兩條分支間共用，理由同 `sdedit`。
        """
        if not self.is_inpainting:
            raise RuntimeError(
                f"{self.model_name} 的 UNet in_channels = "
                f"{self.unet.config.in_channels}，不是 inpainting 權重"
                f"（需 {self.inpaint_in_channels}）。"
                "inpainting 威脅模型必須載入 inpainting 專用權重，"
                "一般權重接不了 9 通道輸入"
            )
        if mask.shape[-2:] != x01.shape[-2:]:
            raise ValueError(
                f"遮罩 {tuple(mask.shape)} 與影像 {tuple(x01.shape)} 的空間"
                "尺寸不符。遮罩必須在影像格點上給定，縮放由本方法負責——"
                "呼叫端自行縮放會出現兩種插值方式並存而無從得知用了哪一種"
            )

        abar = self.alphas_cumprod(x01.device)
        m, z_masked = self.mask_latents(x01, mask, vae_ckpt=vae_ckpt)

        ts = torch.linspace(self.num_train_timesteps - 1, 0,
                            num_steps + 1).round().long()
        z = noise.to(z_masked.dtype)
        for i in range(num_steps):
            t, t_prev = ts[i], ts[i + 1]
            if step_hook is not None:
                step_hook(i, t, None)
            zin = torch.cat([z, m.to(z.dtype), z_masked.to(z.dtype)], dim=1)
            eps = self._eps_cfg(zin, t, emb, guidance_scale, emb_uncond,
                                use_ckpt=use_ckpt)
            pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
            if step_hook is not None:
                step_hook(i, t, pred_x0)
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps
            # **這裡沒有逐步貼回**，理由見本方法 docstring 第 3 點。

        return self.decode_latent(z, use_ckpt=vae_ckpt)

    def edit(
        self,
        x01: torch.Tensor,
        emb: torch.Tensor,
        noise: torch.Tensor,
        num_steps: int,
        *,
        mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = None,
        **kw,
    ) -> torch.Tensor:
        """攻擊方的編輯，依載入的權重分派到 `sdedit` 或 `inpaint`。

        存在理由是**呼叫端有九處**（訓練的代理編輯鏈、評測、對照、baseline、
        段 0 的計時…）。在每一處各加一個 `if` 等於讓「這批跑的是哪一種威脅
        模型」散落在九個地方，其中任何一處漏掉都不會報錯——只會安靜地對同
        一批資料混用兩種攻擊。此處是唯一的分派點。

        兩種形態各自缺參數時一律拋出，不互相沿用預設值：

        - **img2img** 需要 `strength`、且**不接受** `mask`。
        - **inpainting** 需要 `mask`、且**不接受** `strength`——那個參數在
          inpainting 中不存在（pipeline 由純噪聲起跑、跑滿自己的排程）。
          五篇 baseline 的原始碼裡也都沒有這個數，見 `inpaint` 的 docstring。

        故 inpainting 批次的 `RunConfig.strength` 應為 `None`，讓「這個威脅
        模型沒有 strength」這件事出現在 `config_hash` 與每一格的紀錄裡，而不是
        留一個沿用下來卻不起作用的 0.6。
        """
        if self.is_inpainting:
            if mask is None:
                raise ValueError(
                    "inpainting 威脅模型需要遮罩。它決定攻擊方能改哪一塊，"
                    "而防禦方的著力點是遮罩**外**的脈絡——沒有遮罩就沒有"
                    "定義好的攻擊")
            if strength is not None:
                raise ValueError(
                    f"inpainting 沒有 strength（收到 {strength}）。pipeline "
                    "由純噪聲起跑並跑滿自己的排程；沿用一個不起作用的值會讓"
                    "紀錄看起來像是設定過")
            return self.inpaint(x01, mask, emb, noise, num_steps, **kw)

        if mask is not None:
            raise ValueError(
                "img2img 威脅模型不吃遮罩（全圖、無 mask，見 `DESIGN` §2）。"
                "傳了遮罩表示呼叫端以為在跑 inpainting，而載入的是一般權重")
        if strength is None:
            raise ValueError(
                "img2img 需要 strength。五篇 baseline 的原始碼都沒有這個數，"
                "它由本專案的威脅模型指定，故無預設值")
        return self.sdedit(x01, emb, noise, num_steps, strength=strength, **kw)

    def latent_shape(self, height: int, width: int):
        """latent 的形狀。

        2026-08-07 修正。before：通道數取 `self.unet.config.in_channels`。
        一般權重上兩者都是 4 故無症狀，但 inpainting 權重的 `in_channels`
        是 9（4 + 1 + 4），該寫法會回傳 9 通道的形狀，`sample_edit_noise`
        產生的噪聲於是與 latent 對不起來。改取 VAE 的 `latent_channels`，
        那才是 latent 通道數的真相來源。
        """
        f = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return (1, self.latent_channels, height // f, width // f)
