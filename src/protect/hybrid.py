"""Hybrid 非加性保護 — SPEC.md §4.5（本專案設計，v4 對齊）。

以 APA 為骨架（inversion（依 variant 完整/部分）+ Stage 1 LoRA + 兩階段解耦、
ℓ∞ ball 投影），注入點改用 AdvDiff v4 之官方形式：

- step-level：標準一步後「後置加法」z ← z + s·g（無係數、無動量；
  取代 APA 之 ε 動量 sgn 修改）
- trajectory-level：z_T ← z_T + a·g（無動量、無 sgn、無 scale_constant；
  梯度來源依 variant：SG skip / GC checkpoint），投影維持 APA 骨架之
  ℓ∞ ball（z⁰_T ± ε_a）

假設兩者優勢互補；實驗須驗證是否確實優於單獨任一者，若未優於，如實報告。
"""

import torch

from src.protect.apa_based import APAProtection
from src.protect.rewards import attention_reward_latent


class HybridProtection(APAProtection):
    _needs_pre_grad = False  # 不用 APA 之 step-level ε 修改
    _use_sg_scale = False    # 軌跡採 AdvDiff 形式 a·g，不乘 14.58

    @property
    def name(self) -> str:
        return "hybrid"

    def _post_step_hook(self, z_new, t_prev, concept, state):
        """AdvDiff v4 後置加法：於更新後之 x_{t-1} 上計算 reward 梯度。"""
        with torch.enable_grad():
            z_g = z_new.detach().requires_grad_(True)
            r = attention_reward_latent(z_g, t_prev, self.sd, concept)
            g = torch.autograd.grad(r, z_g)[0]
        return self.cfg["s"] * g

    def _traj_zT_update(self, z_T, g_tr, z_T0, state):
        z_T = z_T + self.cfg["a"] * g_tr
        eps_a = self.cfg["eps_a"]
        return torch.minimum(torch.maximum(z_T, z_T0 - eps_a), z_T0 + eps_a)
