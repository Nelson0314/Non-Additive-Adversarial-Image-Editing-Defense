"""Hybrid 非加性保護 — SPEC.md §4.5（本專案設計）。

以 APA 為骨架（inversion + Stage 1 LoRA + 兩階段解耦），Stage 2 的兩個
注入點改用 AdvDiff 形式：

- step-level：ε ← ε − s·√(1−ᾱ_t)·g（原始梯度，AdvDiff Alg.2 line 6；
  非 APA 之動量取 sgn）
- trajectory-level：z_T ← z_T + a·g_tr（AdvDiff Alg.2 line 11），
  投影維持 APA 骨架之 ℓ∞ ball（z⁰_T ± ε_a）

假設兩者優勢互補；實驗須驗證是否確實優於單獨任一者，若未優於，如實報告。
"""

import torch

from src.protect.apa_based import APAProtection


class HybridProtection(APAProtection):
    @property
    def name(self) -> str:
        return "hybrid"

    def _step_eps_update(self, eps, g_st, sqrt_1mabar, state) -> torch.Tensor:
        return eps - self.cfg["s"] * sqrt_1mabar * g_st

    def _traj_zT_update(self, z_T, g_tr, z_T0, state) -> torch.Tensor:
        z_T = z_T + self.cfg["a"] * g_tr
        eps_a = self.cfg["eps_a"]
        return torch.minimum(torch.maximum(z_T, z_T0 - eps_a), z_T0 + eps_a)
