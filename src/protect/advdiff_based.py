"""AdvDiff-based 非加性保護 — SPEC.md §4.3（v4，依官方 ddim_adv.py 修正）。

- DDIM inversion 取得 z_T（[本專案設計]；原文由隨機噪聲出發）
- 每步注入（§4.3.1）：標準 DDIM 一步後「後置加法」z ← z + s·g，
  加在 x_{t-1} 上、無任何係數（官方將 σ²_t、√(1−ᾱ) 等全部註解停用）
- 起始噪聲注入（§4.3.2）：skip-gradient——梯度於最終 latent 計算後
  直接套用 z_T ← z_T + a·g，不反傳穿過採樣鏈（無需保留計算圖）
- guidance 僅於反向過程末段 (0, 0.2]（附錄 E）
- z_T 投影回相對 L2 ball（[本專案設計]；官方無投影；半徑 eps_latent 為
  ‖z_T‖ 之比例，由 stage0 校準）
- s 採論文值 0.7 為起點（官方自身不一致：driver 1.0／簽名 0.75）；
  本專案 reward 定義不同，預期需重調
- reward 採 SPEC §4.2 方案一（注意力抑制，rewards.py）
"""

import torch

from src.protect.base import ProtectionMethod
from src.protect.rewards import attention_reward_latent
from src.utils.device import get_device

_REWARD_T = 1  # 最終 latent 為乾淨估計，reward 於小 timestep 評估


class AdvDiffProtection(ProtectionMethod):
    @property
    def is_additive(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "advdiff_based"

    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        cfg = self.cfg
        sd = self.sd
        self._reset_peak_memory()
        device = get_device()

        x0 = image.to(device) * 2.0 - 1.0
        emb = sd.encode_text("").detach()  # 原文以 unconditional 生成
        abar = sd.scheduler.alphas_cumprod.to(device)
        n_train = len(abar)
        T = cfg["T"]
        ts = sd.ddim_timesteps(T)
        lo, hi = cfg["guidance_range"]

        with torch.no_grad():
            z0 = sd.encode_image(x0)
            z_T = sd.ddim_inversion(z0, emb, T)
        z_T_orig = z_T.detach().clone()
        max_radius = cfg["eps_latent"] * z_T_orig.flatten().norm()

        x_prot = None
        with torch.no_grad():  # v4：全程不保留採樣計算圖（skip-gradient）
            for _ in range(cfg["N"]):
                z = z_T.clone()
                for i in reversed(range(T)):
                    t, t_prev = ts[i + 1], ts[i]
                    # 標準 DDIM 一步（eta=0），不修改 epsilon
                    eps = sd.unet(z, t, encoder_hidden_states=emb).sample
                    pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
                    z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps

                    # v4：後置加法，無係數（官方 ddim_adv.py，加在 x_{t-1} 上）
                    if lo < t_prev.item() / n_train <= hi:
                        with torch.enable_grad():
                            z_g = z.detach().requires_grad_(True)
                            r = attention_reward_latent(z_g, t_prev, sd, concept)
                            g = torch.autograd.grad(r, z_g)[0]
                        z = z + cfg["s"] * g

                x_prot = sd.decode_latents(z)

                # v4：skip-gradient——梯度於最終 latent 計算，直接套用至 z_T
                with torch.enable_grad():
                    z_final = z.detach().requires_grad_(True)
                    r_final = attention_reward_latent(z_final, _REWARD_T, sd, concept)
                    g_T = torch.autograd.grad(r_final, z_final)[0]
                z_T = z_T + cfg["a"] * g_T

                # [本專案設計] 投影回以初始 inversion 為中心之相對 L2 ball
                d = z_T - z_T_orig
                d_norm = d.flatten().norm()
                if d_norm > max_radius:
                    d = d * (max_radius / d_norm)
                z_T = z_T_orig + d

        self._capture_peak_memory()
        return ((x_prot.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
