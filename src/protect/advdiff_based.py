"""AdvDiff-based 非加性保護 — SPEC.md §4.3（DDIM 形式，逐行對應原文 Alg.2）。

- DDIM inversion 取得 z_T（[本專案設計]；原文由隨機噪聲出發）
- 每步注入（Alg.2 line 6）：ε̂ = ε̃ − s·√(1−ᾱ_t)·∇_z R，僅於 guidance_range
  （原文附錄 E：反向過程末段 (0, 0.2]）
- 起始噪聲注入（Alg.2 line 11）：z_T ← z_T + a·∇_{z_T} R_final
- z_T 投影回相對 L2 ball（[本專案設計]；原文無投影，保護任務須貼近原圖；
  半徑 eps_latent 為 ‖z_T‖ 之比例，由 stage0 校準）
- reward 採 SPEC §4.2 方案一（注意力抑制，rewards.py）
"""

import torch

from src.protect.base import ProtectionMethod
from src.protect.rewards import attention_reward_image, attention_reward_latent
from src.utils.device import get_device


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
        ts = sd.ddim_timesteps(cfg["T"])
        lo, hi = cfg["guidance_range"]

        with torch.no_grad():
            z0 = sd.encode_image(x0)
            z_T = sd.ddim_inversion(z0, emb, cfg["T"])
        z_T_orig = z_T.detach().clone()
        max_radius = cfg["eps_latent"] * z_T_orig.flatten().norm()

        x_prot = None
        for _ in range(cfg["N"]):
            z_T_req = z_T.detach().requires_grad_(True)
            z = z_T_req
            for i in reversed(range(cfg["T"])):  # 降冪走訪同一格點
                t, t_prev = ts[i + 1], ts[i]
                eps = sd.unet(z, t, encoder_hidden_states=emb).sample

                if lo < t.item() / n_train <= hi:  # 附錄 E：僅末段施加
                    z_g = z.detach().requires_grad_(True)
                    r = attention_reward_latent(z_g, t, sd, concept)
                    g = torch.autograd.grad(r, z_g)[0]
                    eps = eps - cfg["s"] * (1 - abar[t]).sqrt() * g  # Alg.2 line 6

                pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
                z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps

            x_prot = sd.decode_latents(z)
            r_final = attention_reward_image(x_prot, sd, concept)
            g_T = torch.autograd.grad(r_final, z_T_req)[0]

            with torch.no_grad():
                z_T = z_T_req + cfg["a"] * g_T  # Alg.2 line 11，無係數
                # [本專案設計] 投影回以初始 inversion 為中心之相對 L2 ball
                d = z_T - z_T_orig
                d_norm = d.flatten().norm()
                if d_norm > max_radius:
                    d = d * (max_radius / d_norm)
                z_T = z_T_orig + d

        self._capture_peak_memory()
        return ((x_prot.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
