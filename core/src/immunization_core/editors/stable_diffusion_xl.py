"""SDXL adapter；保留雙文字編碼器、pooled 嵌入、time_ids 及權重 variant。

擴散迴圈沿用 SDWrapper，SDXL 的 BDIA 數值性質仍需真實權重另行驗證。"""

from typing import Optional, Tuple

import torch

from .conditioning import SDXLPrompt
from .stable_diffusion import SDWrapper


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
        2. 本專案一律以 bf16 執行（RTX 5090 支援），fp32 權重檔的多餘精度
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

        1. 攻擊方必須是 stock SDXL。若本專案用 `encode_text("")` 當 CFG 的
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
