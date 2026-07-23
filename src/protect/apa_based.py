"""APA-based 非加性保護 — SPEC.md §4.4（式 5–12）。

- Stage 1（式 6）：LoRA ∆θ 以 diffusion loss 更新（模擬 victim 適應；
  手刻 LoRA，不引入 peft；rank 由 config，SPEC §8 第 4 項待確認）
- Stage 2：
  - step-level（式 8、9、10、11）：ε ← ε − √(1−ᾱ_t)·sgn(m_st)，
    m_st 為 ℓ1 正規化梯度之動量；梯度來自中間步淨化估計
    z^t_in = √(1−ᾱ_t)·z_0 + (1−√(1−ᾱ_t))·z^t_0 上之 reward
  - trajectory-level（式 7、12）：g_tr = ∇_{z_T}(1/T_a)Σ_t R((z^t_0+z̄_0)/2)，
    z_T ← Π_{z⁰_T±ε_a}(z_T + µ·sgn(m_tr))（ℓ∞ ball）
- attack guidance 僅於最後 T_a 步施加；variant "sg" T=50 / "gc" T=10
- reward 採 SPEC §4.2 方案一（注意力抑制）

實作簡化（記錄於 NOTES.md）：式 (11)/(12) 之 f(·) 原為分類器（作用於解碼影像）；
本專案 reward 本身即以 UNet 注意力定義，故直接於 latent（z^t_in、(z^t_0+z̄_0)/2）
上評估、不經 VAE 解碼—編碼往返，語意等價且大幅降低內迴圈記憶體。

Hybrid（SPEC §4.5）由子類別覆寫兩個注入 hook 實作。
"""

import torch
import torch.nn as nn

from src.protect.base import ProtectionMethod
from src.protect.rewards import attention_reward_latent
from src.utils.device import get_device

_REWARD_T = 1  # x0 估計為乾淨 latent，reward 於小 timestep 評估


class LoRALinear(nn.Module):
    """W·x + up(down(x))，up 零初始化（注入當下不改變輸出）。"""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)
        self.to(base.weight.device, base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))


def inject_lora(unet, rank: int):
    """對所有 attention 之 to_q/to_k/to_v 注入 LoRA。

    回傳 (可訓練參數清單, 還原函式)。
    """
    params, restores = [], []
    for module in unet.modules():
        if hasattr(module, "to_q") and isinstance(module.to_q, nn.Linear):
            for attr in ("to_q", "to_k", "to_v"):
                base = getattr(module, attr)
                lora = LoRALinear(base, rank)
                setattr(module, attr, lora)
                params += [lora.down.weight, lora.up.weight]
                restores.append((module, attr, base))

    def restore():
        for m, attr, base in restores:
            setattr(m, attr, base)

    return params, restore


class APAProtection(ProtectionMethod):
    @property
    def is_additive(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "apa_based"

    # ---- 注入 hook（Hybrid 覆寫這兩個） ----

    def _step_eps_update(self, eps, g_st, sqrt_1mabar, state) -> torch.Tensor:
        """式 (8) + (11)：ℓ1 動量後取 sgn。"""
        state["m_st"] = state.get("m_st", 0.0) + g_st / (g_st.abs().sum() + 1e-12)
        return eps - sqrt_1mabar * state["m_st"].sign()

    def _traj_zT_update(self, z_T, g_tr, z_T0, state) -> torch.Tensor:
        """式 (7)：ℓ1 動量、µ·sgn、投影回 z⁰_T ± ε_a（ℓ∞）。"""
        state["m_tr"] = state.get("m_tr", 0.0) + g_tr / (g_tr.abs().sum() + 1e-12)
        z_T = z_T + self.cfg["mu"] * state["m_tr"].sign()
        eps_a = self.cfg["eps_a"]
        return torch.minimum(torch.maximum(z_T, z_T0 - eps_a), z_T0 + eps_a)

    # ---- 主流程 ----

    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        cfg = self.cfg
        sd = self.sd
        self._reset_peak_memory()
        device = get_device()

        x0 = image.to(device) * 2.0 - 1.0
        emb = sd.encode_text(concept).detach()
        with torch.no_grad():
            z0 = sd.encode_image(x0)

        params, restore = inject_lora(sd.unet, cfg["lora_rank"])
        try:
            self._stage1_lora(z0, emb, params)
            out = self._stage2(z0, emb, concept)
        finally:
            restore()
            sd.unet.requires_grad_(True)  # 恢復預設（stage1 曾凍結基底參數）
        self._capture_peak_memory()
        return out

    def _stage1_lora(self, z0, emb, params) -> None:
        """式 (6)：maximize R_s = −E‖ε − ε_{θ+∆θ}‖²，即最小化 diffusion loss。"""
        sd = self.sd
        cfg = self.cfg
        abar = sd.scheduler.alphas_cumprod.to(z0.device)
        for p in sd.unet.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=cfg["lora_lr"])
        for _ in range(cfg["lora_steps"]):
            t = torch.randint(0, len(abar), (1,), device=z0.device)
            noise = torch.randn_like(z0)
            z_t = abar[t].sqrt() * z0 + (1 - abar[t]).sqrt() * noise
            eps_pred = sd.unet(z_t, t, encoder_hidden_states=emb).sample
            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def _stage2(self, z0, emb, concept: str) -> torch.Tensor:
        cfg = self.cfg
        sd = self.sd
        device = z0.device
        abar = sd.scheduler.alphas_cumprod.to(device)
        T, T_a = cfg["T"], cfg["T_a"]
        ts = sd.ddim_timesteps(T)

        with torch.no_grad():
            z_T = sd.ddim_inversion(z0, emb, T)
        z_T0 = z_T.detach().clone()
        state: dict = {}

        for _ in range(cfg["N"]):
            z_T_req = z_T.detach().requires_grad_(True)
            _, r_traj = self._sample_guided(z_T_req, z0, emb, concept, ts, abar, T, T_a, state)
            g_tr = torch.autograd.grad(r_traj, z_T_req)[0]  # 式 (12)
            with torch.no_grad():
                z_T = self._traj_zT_update(z_T_req.detach(), g_tr, z_T0, state)

        with torch.no_grad():  # 以最終 z_T 產出（僅 step-level 注入，無軌跡梯度）
            z_final, _ = self._sample_guided(
                z_T, z0, emb, concept, ts, abar, T, T_a, state, need_traj=False
            )
            x_prot = sd.decode_latents(z_final)
        return ((x_prot + 1.0) / 2.0).clamp(0.0, 1.0)

    def _sample_guided(self, z_T, z0, emb, concept, ts, abar, T, T_a, state, need_traj=True):
        """由 z_T 依降冪格點採樣；最後 T_a 步施加 step-level 注入並累積軌跡 reward。"""
        sd = self.sd
        z = z_T
        r_traj = torch.zeros((), device=z0.device)
        n_guided = 0
        state["m_st"] = 0.0  # step-level 動量每條軌跡重置（式 11 依 t 遞迴）

        for i in reversed(range(T)):
            t, t_prev = ts[i + 1], ts[i]
            sqrt_1mabar = (1 - abar[t]).sqrt()
            eps = sd.unet(z, t, encoder_hidden_states=emb).sample

            if i < T_a:  # 僅去噪最後 T_a 步（低噪聲端；高 t 時 guidance 無效，AdvDiff 附錄 E 同理）
                with torch.enable_grad():
                    z_g = z.detach().requires_grad_(True)
                    eps_g = sd.unet(z_g, t, encoder_hidden_states=emb).sample
                    z_t0_g = (z_g - sqrt_1mabar * eps_g) / abar[t].sqrt()      # 式 (9)
                    w = sqrt_1mabar
                    z_in = w * z0 + (1 - w) * z_t0_g                            # 式 (10)
                    r_st = attention_reward_latent(z_in, _REWARD_T, sd, concept)
                    g_st = torch.autograd.grad(r_st, z_g)[0]                    # 式 (11)
                eps = self._step_eps_update(eps, g_st, sqrt_1mabar, state)

                if need_traj:  # 式 (12)：diffusion augmentation（latent 簡化）
                    z_t0 = (z - sqrt_1mabar * eps) / abar[t].sqrt()
                    r_traj = r_traj + attention_reward_latent(
                        (z_t0 + z0) / 2, _REWARD_T, sd, concept
                    )
                    n_guided += 1

            pred_x0 = (z - sqrt_1mabar * eps) / abar[t].sqrt()
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps

        if need_traj:
            r_traj = r_traj / max(n_guided, 1)
        return z, r_traj
