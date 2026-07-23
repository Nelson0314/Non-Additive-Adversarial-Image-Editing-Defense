"""APA-based 非加性保護 — SPEC.md §4.4（v4，依官方 repo 修正）。

- Stage 1（§4.4.2，官方 visual_alignment.py）：peft LoRA
  （rank=8、alpha=8、to_k/to_q/to_v/to_out.0、gaussian 初始化、AdamW +
  weight decay + grad clip、noise_offset、latent 取 .sample()），
  標準 diffusion loss 最小化（式 6 等價形式）
- Stage 2（§4.4.1、§4.4.3）：
  - step-level（式 8–11）：ε ← ε − √(1−ᾱ_t)·sgn(m_st)，ℓ1 動量每輪重置；
    梯度來自中間步淨化估計 z^t_in 上之 reward，僅最後 T_a 步施加
  - trajectory-level（式 7）：ℓ1 動量跨輪累積、µ·sgn、ℓ∞ clamp（z⁰_T ± ε_a）
  - **SG**：完整 inversion（50 步格點）；軌跡梯度 skip——於最終 latent 計算、
    乘 scale_constant（官方 14.58，經驗值）後直接套用，不穿鏈
  - **GC**：部分 inversion（50 步格點只走前 inversion_steps 步，t ≈ 0.2T）、
    只採樣最後數步、torch.utils.checkpoint 反傳至 z_T；
    reward 另含 −mse_reg_weight·MSE(z_ori, z_final)
- 式 (12) diffusion augmentation（§4.4.4）：增強樣本 =
  (x̂_0^t[detached] + 最終生成 latent)/2，ϱ = random resize(90–100%) +
  random 零 padding + interpolate（brightness 官方停用，不啟用）
- reward 採 §4.2 方案一（注意力抑制）；官方 f(·) 為解碼影像上之分類器，
  本專案 reward 以 UNet 注意力定義，故 ϱ 與混合均於 latent 上進行
  （既有設計決定，NOTES.md 2026-07-23）
- guidance_scale=1：inversion 與攻擊均無 CFG（官方）

Hybrid（SPEC §4.5）由子類別覆寫注入 hook 實作。
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.protect.base import ProtectionMethod
from src.protect.rewards import attention_reward_latent
from src.utils.device import get_device

_REWARD_T = 1  # x0 估計為乾淨 latent，reward 於小 timestep 評估
_ADAPTER_NAME = "apa_stage1"


class APAProtection(ProtectionMethod):
    @property
    def is_additive(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"apa_{self.cfg.get('variant', 'gc')}"

    # ---- 注入 hook（Hybrid 覆寫） ----

    def _pre_step_hook(self, eps, g_st, sqrt_1mabar, state):
        """式 (8)+(11)：ℓ1 動量後取 sgn，修改 epsilon。回傳修改後 eps。"""
        state["m_st"] = state.get("m_st", 0.0) + g_st / (g_st.abs().sum() + 1e-12)
        return eps - sqrt_1mabar * torch.sign(state["m_st"])

    def _post_step_hook(self, z_new, t_prev, concept, state):
        """步後加法項（APA 無；Hybrid 用 AdvDiff 後置加法）。回傳 None 或加項。"""
        return None

    def _traj_zT_update(self, z_T, g_tr, z_T0, state):
        """式 (7)：ℓ1 動量跨輪累積、µ·sgn、ℓ∞ clamp。"""
        state["m_tr"] = state.get("m_tr", 0.0) + g_tr / (g_tr.abs().sum() + 1e-12)
        z_T = z_T + self.cfg["mu"] * torch.sign(state["m_tr"])
        eps_a = self.cfg["eps_a"]
        return torch.minimum(torch.maximum(z_T, z_T0 - eps_a), z_T0 + eps_a)

    # ---- 主流程 ----

    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        cfg = self.cfg
        sd = self.sd
        self._reset_peak_memory()
        device = get_device()

        x0 = image.to(device) * 2.0 - 1.0
        emb = sd.encode_text(concept).detach()  # caption = class 名稱
        with torch.no_grad():
            z_ori = sd.encode_image(x0)  # 官方 ori_latents：原圖 latent（mean）

        self._inject_lora()
        try:
            self._stage1_lora(x0, emb)
            out = self._stage2(z_ori, emb, concept)
        finally:
            sd.unet.delete_adapters(_ADAPTER_NAME)
            sd.unet.requires_grad_(True)  # 恢復預設（stage1 曾凍結基底參數）
        self._capture_peak_memory()
        return out

    # ---- Stage 1：peft LoRA（官方 visual_alignment.py 設定）----

    def _inject_lora(self) -> None:
        from peft import LoraConfig

        cfg = self.cfg
        self.sd.unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            init_lora_weights=cfg.get("lora_init", "gaussian"),
            target_modules=list(cfg["lora_target_modules"]),
        )
        self.sd.unet.add_adapter(lora_config, adapter_name=_ADAPTER_NAME)

    def _stage1_lora(self, x0: torch.Tensor, emb: torch.Tensor) -> None:
        """式 (6)：maximize R_s ⇔ 最小化 diffusion loss（單圖）。"""
        sd = self.sd
        cfg = self.cfg
        abar = sd.scheduler.alphas_cumprod.to(x0.device)
        params = [p for p in sd.unet.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=cfg["lora_lr"], weight_decay=cfg["lora_weight_decay"]
        )
        for _ in range(cfg["lora_steps"]):
            with torch.no_grad():  # 官方每步重新 .sample()
                z0 = (
                    sd.vae.encode(x0).latent_dist.sample() * sd.scaling_factor
                )
            noise = torch.randn_like(z0)
            if cfg.get("noise_offset"):
                noise = noise + cfg["noise_offset"] * torch.randn(
                    z0.shape[0], z0.shape[1], 1, 1, device=z0.device
                )
            t = torch.randint(0, len(abar), (1,), device=z0.device)
            z_t = abar[t].sqrt() * z0 + (1 - abar[t]).sqrt() * noise
            eps_pred = sd.unet(z_t, t, encoder_hidden_states=emb).sample
            loss = F.mse_loss(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["lora_grad_clip"])
            opt.step()

    # ---- Stage 2 ----

    def _stage2(self, z_ori, emb, concept: str) -> torch.Tensor:
        cfg = self.cfg
        sd = self.sd
        variant = cfg.get("variant", "gc")
        self._z_ori = z_ori  # 供 _rollout 之式 (10) 使用
        grid = cfg.get("grid_steps", 50)  # 官方 50 步排程
        ts = sd.ddim_timesteps(grid)
        top = grid if variant == "sg" else cfg["gc"]["inversion_steps"]

        with torch.no_grad():
            # SG：完整 inversion；GC：部分 inversion（只走前 top 步，t ≈ ts[top]）
            z_T = self._partial_ddim_inversion(z_ori, emb, ts, top)
        z_T0 = z_T.detach().clone()
        state: dict = {}

        z_last = z_T
        for _ in range(cfg["N"]):
            if variant == "sg":
                with torch.no_grad():
                    z_fin, la_list = self._rollout(
                        z_T, ts, top, emb, concept, state, keep_graph=False
                    )
                with torch.enable_grad():
                    z_g = z_fin.detach().requires_grad_(True)
                    r = self._traj_reward(z_g, la_list, z_ori, concept)
                    g_tr = torch.autograd.grad(r, z_g)[0]
                if self._use_sg_scale:
                    g_tr = cfg["sg"]["scale_constant"] * g_tr  # 官方 14.58，經驗值
            else:  # gc：checkpoint 反傳穿過（短）採樣鏈
                z_T_req = z_T.detach().requires_grad_(True)
                z_fin, la_list = self._rollout(
                    z_T_req, ts, top, emb, concept, state, keep_graph=True
                )
                r = self._traj_reward(z_fin, la_list, z_ori, concept)
                r = r - cfg["gc"]["mse_reg_weight"] * F.mse_loss(z_ori, z_fin)
                g_tr = torch.autograd.grad(r, z_T_req)[0]

            z_last = z_fin.detach()
            with torch.no_grad():
                z_T = self._traj_zT_update(z_T.detach(), g_tr.detach(), z_T0, state)

        with torch.no_grad():  # 官方回傳最後一輪之生成結果（最終 z_T 更新不再採樣）
            x_prot = sd.decode_latents(z_last)
        return ((x_prot + 1.0) / 2.0).clamp(0.0, 1.0)

    def _partial_ddim_inversion(self, z0, emb, ts, top):
        """於格點 ts 上走前 top 步之 DDIM inversion（top=len(ts)-1 即完整）。"""
        sd = self.sd
        abar = sd.scheduler.alphas_cumprod.to(z0.device)
        z = z0
        for i in range(top):
            t_prev, t = ts[i], ts[i + 1]
            eps = sd.unet(z, t_prev, encoder_hidden_states=emb).sample
            pred_x0 = (z - (1 - abar[t_prev]).sqrt() * eps) / abar[t_prev].sqrt()
            z = abar[t].sqrt() * pred_x0 + (1 - abar[t]).sqrt() * eps
        return z

    def _rollout(self, z_start, ts, top, emb, concept, state, keep_graph):
        """由 ts[top] 依格點降冪採樣至 0；最後 T_a 步施加 step-level 注入。

        回傳 (最終 latent, la_list)。la_list 為 guided 步之 x̂_0 估計（detached，
        式 12 之增強素材）。keep_graph=True 時以 checkpoint 保留至 z_start 之梯度。
        """
        sd = self.sd
        cfg = self.cfg
        abar = sd.scheduler.alphas_cumprod.to(z_start.device)
        T_a = cfg["T_a"]
        state["m_st"] = 0.0  # 式 (11)：動量每輪軌跡重置
        la_list = []
        z = z_start

        for i in reversed(range(top)):
            t, t_prev = ts[i + 1], ts[i]
            sqrt_1mabar = (1 - abar[t]).sqrt()
            if keep_graph:
                eps = checkpoint.checkpoint(
                    lambda a, b: sd.unet(a, b, encoder_hidden_states=emb).sample,
                    z, t, use_reentrant=False,
                )
            else:
                eps = sd.unet(z, t, encoder_hidden_states=emb).sample

            guided = i < T_a  # 官方 index_cond=40/50：僅去噪最後 T_a 步
            if guided:
                if self._needs_pre_grad:
                    with torch.enable_grad():
                        z_g = z.detach().requires_grad_(True)
                        eps_g = sd.unet(z_g, t, encoder_hidden_states=emb).sample
                        z_t0_g = (z_g - sqrt_1mabar * eps_g) / abar[t].sqrt()  # 式 (9)
                        w = sqrt_1mabar
                        z_in = w * self._z_ori + (1 - w) * z_t0_g              # 式 (10)
                        r_st = attention_reward_latent(z_in, _REWARD_T, sd, concept)
                        g_st = torch.autograd.grad(r_st, z_g)[0]               # 式 (11)
                    eps = self._pre_step_hook(eps, g_st, sqrt_1mabar, state)
                # 式 (12) 素材：修改後 eps 之 x̂_0 估計（官方為 detached）
                la_list.append(((z - sqrt_1mabar * eps) / abar[t].sqrt()).detach())

            pred_x0 = (z - sqrt_1mabar * eps) / abar[t].sqrt()
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps

            if guided:
                post = self._post_step_hook(z, t_prev, concept, state)
                if post is not None:
                    z = z + post
        return z, la_list

    def _traj_reward(self, z_fin, la_list, z_ori, concept):
        """式 (12)：增強樣本 = (x̂_0^t + 最終生成)/2 經 ϱ 後之平均 reward。

        官方於解碼影像上做；本專案 reward 以注意力定義，故於 latent 上進行。
        梯度僅經 z_fin 半邊流動（la 為 detached，官方同）。
        """
        sd = self.sd
        if not la_list:
            return attention_reward_latent(z_fin, _REWARD_T, sd, concept)
        rewards = []
        for la in la_list:
            z_mix = (la + z_fin) / 2.0
            rewards.append(
                attention_reward_latent(self._rho(z_mix), _REWARD_T, sd, concept)
            )
        return torch.stack(rewards).mean()

    def _rho(self, z):
        """ϱ：random resize（resize_range）+ random 零 padding 回原尺寸。
        brightness 官方註解停用，不啟用。"""
        lo, hi = self.cfg["aug"]["resize_range"]
        size = z.shape[-1]
        new = max(1, int(torch.empty(1).uniform_(lo, hi).item() * size))
        if new >= size:
            return z
        small = F.interpolate(z, size=(new, new), mode="bilinear", align_corners=False)
        pad_total = size - new
        top = int(torch.randint(0, pad_total + 1, (1,)).item())
        left = int(torch.randint(0, pad_total + 1, (1,)).item())
        return F.pad(small, (left, pad_total - left, top, pad_total - top))

    # 供 _rollout 之式 (10)；_stage2 開頭設定
    _z_ori: torch.Tensor = None
    # APA 需要 step-level 梯度（式 8–11）；Hybrid 覆寫為 False（改用後置加法）
    _needs_pre_grad: bool = True
    # SG 軌跡梯度乘 scale_constant（官方 14.58）；Hybrid 覆寫為 False（AdvDiff 形式用 a·g）
    _use_sg_scale: bool = True
