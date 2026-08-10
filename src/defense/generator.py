"""防禦圖生成器 G(x; φ) — spec §4.3。

依模塊提供的能力分派，不對 site 寫死分支：

- 提供 `pixel_residual` → 直接取得防禦圖（site P）
- 提供 `eps_hook`      → 走 inversion + 去噪路徑（site L、site W）

效率設計：inversion 段殘差模塊關閉，故 z_{k_inv} 不依賴 φ，可在優化
開始前計算一次並快取，省去每個 iteration 一條 k_inv 步的 UNet 前向。
快取由 `prepare()` 產生、`generate()` 接收。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.models.sd import SDWrapper
from src.residual.base import ResidualModule


@dataclass
class DefenseContext:
    """對 φ 為常數的前置計算結果，每張影像算一次。"""

    z_inv: Optional[torch.Tensor] = None  # inversion 終點，site P 為 None
    # BDIA 的遞迴狀態是相鄰兩點，故精確反演時額外帶回 z_{K−1}。DDIM 路徑
    # 為 None，`generate()` 以它是否存在決定走哪一條去噪，不另設旗標：
    # 有沒有第二個狀態本身就是「用的是哪種反演」的充分資訊。
    z_prev: Optional[torch.Tensor] = None
    emb: Optional[torch.Tensor] = None
    ts: Optional[torch.Tensor] = None
    steps: int = 0
    x0_trace: List[torch.Tensor] = field(default_factory=list)


class DefenseGenerator:
    def __init__(
        self,
        sd: SDWrapper,
        module: ResidualModule,
        k_inv: int = 10,
        t_max: Optional[int] = None,
        exact_inversion: bool = False,
        recon=None,
    ):
        self.sd = sd
        self.module = module
        self.k_inv = k_inv
        # A 段（DEC-016）的產物：生成路徑的新起點 z* 與解碼器的逐圖微調值。
        # 依「有沒有提供這個能力」分派，與本檔既有的慣例一致：`None` 時
        # 每一步都與先前逐位元相同，故既有批次不受影響。
        #
        # 它同時改動這條路徑的**兩端**：`prepare()` 的起點與 `generate()` 的
        # 解碼。只換其中一端是沒有意義的——A2 的那組解碼器參數是在 z* 上
        # 擬合出來的，套在 `encode(x)` 上等於用一個為別的輸入調好的解碼器。
        self.recon = recon
        # True 改用 BDIA（arXiv 2307.10829）。latent 空間來回誤差由 DDIM 的
        # 1.4（k=20, t_max=500，tiny-SD 實測最大絕對值）降到 1.4e-04。
        # 但影像空間的下限不會歸零：VAE 的編解碼來回誤差（實測 27.51 dB /
        # LPIPS 0.143）不在 BDIA 的作用範圍內。
        self.exact_inversion = exact_inversion
        # t_max=None 表示走滿 [0, 999]。E0c 實測該設定下 k_inv=10 的重建
        # 已達 LPIPS 0.70，即 phi=0 時 x_def 與 x 已是兩張不同的圖，故
        # 此參數必須由呼叫端依 E0c 的量測結果指定，不可沿用預設值。
        self.t_max = t_max

    def prepare(self, x01: torch.Tensor, prompt_def: str = "") -> DefenseContext:
        """計算不依賴 φ 的部分。像素側的模塊不需要任何前置。"""
        # 依能力分派，不比對 site 名稱。原本此處寫的是
        #     pixel is not None and self.module.site == "P"
        # 該條件在新增全秩對照（site "PF"）時失效：模塊確實提供
        # `pixel_residual`，卻因名稱不是 "P" 而被送進 inversion + 去噪路徑，
        # 而該路徑取的是 `eps_hook`（基底類別回傳 None），於是 φ 完全沒有
        # 進入計算圖，backward 只報 "does not require grad"，看不出真正原因。
        # 提供 `pixel_residual` 即為像素側模塊，這正是模組 docstring 所定的
        # 判準；site 名稱不得參與分派。
        if self.module.pixel_residual(x01) is not None:
            return DefenseContext()

        emb = self.sd.encode_text(prompt_def).detach()
        ts = self.sd.timesteps(self.k_inv, t_max=self.t_max)

        # inversion 期間必須關閉模塊：z_inv 不得依賴 φ，否則快取失效
        was_enabled = self.module.enabled
        self.module.disable()
        try:
            # 9 通道權重下反演也要餵後 5 個通道。取「不重畫任何區域」：這條
            # 是**重建**路徑，G(x; φ=0) 要盡量等於 x，而整個保真預算的下限
            # 就建立在那件事上（`SDWrapper.inpaint_conditioning`）。
            with torch.no_grad(), self.sd.conditioning_for(x01):
                # A 段接上時起點改為解出來的 z*，不再呼叫編碼器。這一行
                # 就是 A1 的全部作用：參數量相同、模組不變，只是不從
                # `encode(x)` 出發。
                z0 = (self.recon.z_star.to(x01.device, x01.dtype)
                      if self.recon is not None
                      else self.sd.encode_image(x01))
                if self.exact_inversion:
                    z_inv, z_prev = self.sd.bdia_inversion(z0, emb, ts, self.k_inv)
                else:
                    z_inv = self.sd.ddim_inversion(z0, emb, ts, self.k_inv)
                    z_prev = None
        finally:
            if was_enabled:
                self.module.enable()

        return DefenseContext(z_inv=z_inv, z_prev=z_prev, emb=emb, ts=ts,
                              steps=self.k_inv)

    def generate(
        self,
        x01: torch.Tensor,
        ctx: DefenseContext,
        use_ckpt: bool = False,
        vae_ckpt: bool = False,
        collect_x0: bool = False,
    ) -> torch.Tensor:
        """回傳 x_def，計算圖保留至 φ。"""
        pixel = self.module.pixel_residual(x01)
        if pixel is not None:
            return pixel

        if ctx.z_inv is None:
            raise ValueError("此模塊需要 prepare() 產生的 inversion 快取")

        # 條件嵌入的擾動只作用於去噪段。inversion 在 prepare() 內已用未擾動
        # 的 ctx.emb 算完，故 z_inv 不依賴 φ，快取仍然有效。
        emb = ctx.emb
        d_emb = self.module.emb_residual(emb)
        if d_emb is not None:
            emb = emb + d_emb

        hook = self.module.eps_hook(ctx.ts, ctx.steps)
        if (hook is None and d_emb is None
                and not self.module.patches_model() and self.module.enabled):
            # 啟用中卻沒有任何一種能力，代表 φ 不會進入計算圖。在此明講，否則
            # 症狀會延後到 backward 才以 "does not require grad" 的形式出現，
            # 看不出真正原因。
            #
            # site W 不在此列：它以 forward hook 直接改動 UNet，φ 經由模型
            # 本身進入計算圖，不走上面三種殘差路徑，故以 patches_model() 表明。
            #
            # 停用時亦不在此列：停用的定義就是「行為與模塊不存在完全一致」，
            # eps_hook 回傳 None 是正確行為，且 x_base = G(x; φ=0) 正是靠這條
            # 路徑取得的。把停用也算成錯誤會讓保真基準無法計算。
            raise ValueError(
                f"模塊 {type(self.module).__name__}（site {self.module.site}）"
                "啟用中卻未提供 pixel_residual、eps_hook、emb_residual、"
                "patches_model 之任一，φ 無法進入計算圖"
            )
        # 去噪與反演必須用同一組後 5 通道，否則兩半走的是不同的條件而
        # G(x; φ=0) 不再逼近 x——那個偏差看起來只是「重建差一點」。
        with self.sd.conditioning_for(x01, vae_ckpt=vae_ckpt):
            if ctx.z_prev is not None:
                z, x0_list = self.sd.bdia_denoise(
                    (ctx.z_inv, ctx.z_prev),
                    emb,
                    ctx.ts,
                    ctx.steps,
                    eps_hook=hook,
                    use_ckpt=use_ckpt,
                    collect_x0=collect_x0,
                )
            else:
                z, x0_list = self.sd.denoise(
                    ctx.z_inv,
                    emb,
                    ctx.ts,
                    ctx.steps,
                    eps_hook=hook,
                    use_ckpt=use_ckpt,
                    collect_x0=collect_x0,
                )
        ctx.x0_trace = x0_list
        if self.recon is None:
            return self.sd.decode_latent(z, use_ckpt=vae_ckpt)
        # 微調值只在這一次解碼期間生效，離開即還原：同一個 SDWrapper 會被
        # 整批影像共用，留在權重上等於讓上一張圖的過擬合污染下一張。
        with self.recon.applied(self.sd.vae.decoder):
            return self.sd.decode_latent(z, use_ckpt=vae_ckpt)
