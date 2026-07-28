"""防禦圖生成器 G(x; φ) — spec §4.3。

依模塊提供的能力分派，不對 site 寫死分支：

- 提供 `pixel_residual` → 直接取得防禦圖（site P）
- 提供 `eps_hook`      → 走 inversion + 去噪路徑（site L、site W）

**效率設計**：inversion 段殘差模塊關閉，故 z_{k_inv} 不依賴 φ，可在優化
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
    ):
        self.sd = sd
        self.module = module
        self.k_inv = k_inv
        # t_max=None 表示走滿 [0, 999]。E0c 實測該設定下 k_inv=10 的重建
        # 已達 LPIPS 0.70，即 phi=0 時 x_def 與 x 已是兩張不同的圖，故
        # 此參數必須由呼叫端依 E0c 的量測結果指定，不可沿用預設值。
        self.t_max = t_max

    def prepare(self, x01: torch.Tensor, prompt_def: str = "") -> DefenseContext:
        """計算不依賴 φ 的部分。site P 不需要任何前置。"""
        if self.module.pixel_residual(x01) is not None and self.module.site == "P":
            return DefenseContext()

        emb = self.sd.encode_text(prompt_def).detach()
        ts = self.sd.timesteps(self.k_inv, t_max=self.t_max)

        # inversion 期間必須關閉模塊：z_inv 不得依賴 φ，否則快取失效
        was_enabled = self.module.enabled
        self.module.disable()
        try:
            with torch.no_grad():
                z0 = self.sd.encode_image(x01)
                z_inv = self.sd.ddim_inversion(z0, emb, ts, self.k_inv)
        finally:
            if was_enabled:
                self.module.enable()

        return DefenseContext(z_inv=z_inv, emb=emb, ts=ts, steps=self.k_inv)

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
        if pixel is not None and self.module.site == "P":
            return pixel

        if ctx.z_inv is None:
            raise ValueError("此模塊需要 prepare() 產生的 inversion 快取")

        hook = self.module.eps_hook(ctx.ts, ctx.steps)
        z, x0_list = self.sd.denoise(
            ctx.z_inv,
            ctx.emb,
            ctx.ts,
            ctx.steps,
            eps_hook=hook,
            use_ckpt=use_ckpt,
            collect_x0=collect_x0,
        )
        ctx.x0_trace = x0_list
        x_gen = self.sd.decode_latent(z, use_ckpt=vae_ckpt)

        # site LA：把生成結果錨定回原圖，使 φ=0 時 x_def = x 逐元素相等。
        # 以能力查詢而非 site 字串判斷，維持 §4.4 的介面判準。
        if hasattr(self.module, "anchor") and getattr(self.module, "has_baseline", False):
            return self.module.anchor(x01, x_gen)
        return x_gen
