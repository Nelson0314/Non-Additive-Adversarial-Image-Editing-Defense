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
    def __init__(self, sd: SDWrapper, module: ResidualModule, k_inv: int = 10):
        self.sd = sd
        self.module = module
        self.k_inv = k_inv

    def prepare(self, x01: torch.Tensor, prompt_def: str = "") -> DefenseContext:
        """計算不依賴 φ 的部分。site P 不需要任何前置。"""
        if self.module.pixel_residual(x01) is not None and self.module.site == "P":
            return DefenseContext()

        emb = self.sd.encode_text(prompt_def).detach()
        ts = self.sd.timesteps(self.k_inv)

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
        return self.sd.decode_latent(z)
