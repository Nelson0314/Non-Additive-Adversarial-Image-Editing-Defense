"""APA 原生階段二重現：dual-path attack guidance + L∞ 球投影。

逐字對照 `docs/reference/dia_apa.md` §3（`pipe_ours.py::attack_optimization_
checkpoint`／`cond_guidance`，2026-08-12 直接取官方原始碼核對）。這裡走
**APA-GC**（gradient checkpointing，真實梯度），不是 APA-SG（skip-gradient +
未查證用途的縮放常數 14.58）——原文自己的數字顯示 APA-GC 較好，且它不需要
一個換了 reward 之後意義不明的常數，兩者都是「原生做法」的合法變體。

## 與官方程式碼的對應

| 官方 | 這裡 |
|---|---|
| `attack_optimization_checkpoint` 的外層迴圈 | `attack_native` 的 `for ii in range(niters)` |
| trajectory-level：`la_t_g = grad(reward, adv_latents)` | `trajectory_grad` |
| `momentum = momentum + l1_grad; adv_latents += sign(momentum)*alpha` | 同一段，逐字保留 |
| `noise = clamp(adv_latents-la_0, -eps, eps)` | 同一段 |
| `cond_guidance`（step-level，Eq.8/9/10/11） | `_step_guidance` |
| `reward = CE(...) - 10*MSE(ori_latents, la_t)` | `reward = R_attn - fidelity_lambda*MSE(...)`，**保留 -10·MSE 這項**——
  它字面上是官方 reward 公式的一部分，不是 cross-entropy 的一部分，換 reward
  不動它 |

## Reward 替換：cross-entropy → attention 抑制損失

原文 `R_a` 是替代分類器對解碼影像的 cross-entropy。這裡換成 Lo et al. 式(5)
的 `‖Att(x,c_a)⊙M‖₁`（`suppress_attn_ca` 已有的損失），但**這是要最小化的
量**，跟官方「maximize R_a」的號約定相反，故程式內部一律取
`R_attn = -masked_attention_l1(...)`，讓「對 R_attn 取梯度、sign、上升」與
官方逐字同構——這是唯一的號處理，其餘更新規則不變。

**兩處都需要一次額外的、帶 c_a 條件的 UNet forward**（原文兩處都是額外呼叫
分類器 `classfier(img)`）：

- trajectory-level：對最終 `z̄_0`（在一個代表性的低噪聲 timestep，預設
  `T_ref = round(0.1 * T)`）取一次注意力圖。
- step-level：對 Eq.10 的 `z_t^in` 混合（`ori_latents` 與該步 x̂₀ 的加權和）
  在**該步自己的 t** 取一次注意力圖——這一步的 t 已經是「噪聲程度」，
  不需要另外決定參照 timestep。

遮罩 M 在攻擊開始前算一次，來自**原圖自己**（`ori_latents`）在 `T_ref`
的注意力（`attention_region_mask`，跟 trajectory-level 用同一個 T_ref），
對防禦訊號為常數，符合 Lo et al. 式(4) 與本專案既有 `suppress_attn_ca`
的慣例。

## 架構變因（DDIM vs BDIA）怎麼接

`use_bdia=False`：標準 DDIM 步進（`sd._ddim_step` 的單方向版本），對應
APA-SG 的設定（T=50，`t_max=None`，全範圍）——不追加「只跑最後 inversion_
step 步」那個分段，理由是那個分段的目的（partial inversion 對齊 partial
denoise）在我們兩種架構的比較裡不是要控制的變因，追加只會多一層跟本實驗
無關的耦合。

`use_bdia=True`：BDIA 雙向遞迴（`sd._ddim_step` 呼叫兩次，係數與
`sd.bdia_inversion`/`bdia_denoise` 相同），`k_inv=20, t_max=500`——本專案
其餘實驗一路採用的設定（FND-016）。**兩種架構故意用各自「本來就在跑」的
步數/範圍組合，不是把數值刻意對齊**：這裡要問的是「每種架構在它自己
運作良好的範圍內，dual-path 攻擊訓不訓得動、保真度如何」，不是「步數
相同時哪個更好」——後者是另一個問題，不是這輪要回答的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.models.attention import (
    CrossAttentionRecorder,
    aggregate_token_attention,
    attention_region_mask,
    masked_attention_l1,
    token_span,
)


@dataclass
class NativeStage2Config:
    """對應 `attack_config`（`eps`/`alpha`/`niters`）與 `index_cond` 等。"""

    eps_a: float = 0.4          # ε_a，Eq.7 的 L∞ 球半徑
    mu: float = 0.04            # µ，trajectory-level 步長
    niters: int = 10            # N，外層迭代數
    steps: int = 50             # 去噪步數（DDIM 架構用；BDIA 架構見 t_max/k_inv）
    t_max: Optional[int] = None
    index_cond_frac: float = 0.8   # 對應官方 index_cond=40/50=0.8：最後 20% 的步驟做 step-level guidance
    guidance_scale: float = 7.5    # 攻擊方 CFG（跟本專案評測期 SDEdit 同一個值）
    fidelity_lambda: float = 10.0  # 官方 APA-GC reward 裡 -10·MSE(ori_latents, z̄_0) 那一項
    attn_mask_tau: float = 0.5
    ref_timestep_frac: float = 0.1  # 算 trajectory-level 注意力圖用的參照 timestep，相對 t_max（或 999）
    use_ckpt: bool = True
    use_bdia: bool = False


def _ref_timestep(cfg: NativeStage2Config, top: int) -> int:
    return max(1, round(cfg.ref_timestep_frac * top))


def _attention_reward(
    sd, z: torch.Tensor, t: torch.Tensor, emb_ca: torch.Tensor,
    span: tuple, mask: torch.Tensor, side: int,
) -> torch.Tensor:
    """`-‖Att(z,c_a)⊙M‖₁`。額外一次帶 c_a 條件的 UNet forward，對應官方的
    `classfier(img)`。"""
    rec = CrossAttentionRecorder(sd.unet)
    with rec:
        sd._eps(z, t, emb_ca)
    att = aggregate_token_attention(rec.maps, span, side=side, reduce="sum")
    return -masked_attention_l1(att, mask)


def _step_guidance(
    sd, z_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor,
    ori_latents: torch.Tensor, m_state: List[torch.Tensor],
    emb_ca: torch.Tensor, span: tuple, mask: torch.Tensor, side: int,
    abar: torch.Tensor,
) -> torch.Tensor:
    """`cond_guidance` 的逐字對應（Eq.8/9/10/11），reward 換成注意力抑制。

    `z_t` 進來前**必須先 detach**——官方在這裡開一個獨立於外層計算圖的
    局部梯度，只用來算 sign-momentum 修正量，不讓這段反傳污染 trajectory-
    level 那條線（兩者是並行的兩條路徑，這正是「dual-path」的字面意思）。
    """
    with torch.enable_grad():
        z_local = z_t.detach().requires_grad_()
        alpha_prod_t = abar[t]
        beta_prod_t = 1.0 - alpha_prod_t
        pred_x0 = (z_local - beta_prod_t.sqrt() * noise_pred.detach()) / alpha_prod_t.sqrt()
        fac = beta_prod_t.sqrt()
        z_in = ori_latents * fac + pred_x0 * (1.0 - fac)          # Eq.10
        reward = _attention_reward(sd, z_in, t, emb_ca, span, mask, side)
        grad = torch.autograd.grad(reward, z_local)[0]
    l1_grad = grad / grad.norm(p=1).clamp_min(1e-12)
    m_state[0] = m_state[0] + l1_grad.detach()
    return beta_prod_t.sqrt() * torch.sign(m_state[0])


def _trajectory_pass(
    sd, adv_latents: torch.Tensor, emb_cond: torch.Tensor,
    emb_uncond: torch.Tensor, emb_ca: torch.Tensor, ori_latents: torch.Tensor,
    cfg: NativeStage2Config, span: tuple, mask: torch.Tensor, side: int,
    ts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """跑一次完整去噪（可反傳），回傳 `(z_0, reward, log)`。

    `ts` 升冪、長度 `steps+1`（同 `sd.timesteps` 的慣例）。DDIM 與 BDIA
    共用這一個函式，差別只在 `cfg.use_bdia` 切換單步公式。
    """
    device = adv_latents[0].device if cfg.use_bdia else adv_latents.device
    abar = sd.alphas_cumprod(device)
    steps = cfg.steps
    index_cond = round(cfg.index_cond_frac * steps)
    m_state = [torch.zeros(1, device=device)]

    if cfg.use_bdia:
        # 對齊 `sd.bdia_inversion` 的慣例：z_T 只是遞迴的一半，另一半
        # （z_{K-1}）從標準 DDIM 反演補一步取得，跟 `apa_native_lora_ddim_
        # control.py`／既有的 `generator.py` 同一個作法。呼叫端已經把兩者
        # 一起準備好（見 `attack_native`），這裡直接吃 `adv_latents` 為
        # `(z_K, z_{K-1})` 這一對，不在函式內重建。
        z_next, z_cur = adv_latents
        x0_for_reward = None
        for step_idx, i in enumerate(range(steps - 1, 0, -1)):
            t = ts[i]
            eps = sd._eps_cfg(z_cur, t, emb_cond, cfg.guidance_scale,
                              emb_uncond, use_ckpt=cfg.use_ckpt)
            if step_idx >= index_cond:
                delta = _step_guidance(sd, z_cur, t, eps, ori_latents,
                                       m_state, emb_ca, span, mask, side, abar)
                eps = eps - delta
            a_plus = sd._ddim_step(z_cur, eps, ts[i], ts[i + 1], abar)
            a_minus = sd._ddim_step(z_cur, eps, ts[i], ts[i - 1], abar)
            z_prev = (z_next - a_plus) + a_minus     # gamma=1
            z_next, z_cur = z_cur, z_prev
        z_0 = z_cur
    else:
        z = adv_latents
        for step_idx, i in enumerate(range(steps - 1, -1, -1)):
            t, t_prev = ts[i + 1], ts[i]
            eps = sd._eps_cfg(z, t, emb_cond, cfg.guidance_scale,
                              emb_uncond, use_ckpt=cfg.use_ckpt)
            if step_idx >= index_cond:
                delta = _step_guidance(sd, z, t, eps, ori_latents,
                                       m_state, emb_ca, span, mask, side, abar)
                eps = eps - delta
            pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps
        z_0 = z

    ref_t = torch.as_tensor(_ref_timestep(cfg, cfg.t_max or (sd.num_train_timesteps - 1)),
                            device=z_0.device)
    r_attn = _attention_reward(sd, z_0, ref_t, emb_ca, span, mask, side)
    fidelity_pen = cfg.fidelity_lambda * torch.nn.functional.mse_loss(
        ori_latents.float(), z_0.float())
    reward = r_attn - fidelity_pen
    log = {"r_attn": float(r_attn.detach()), "fidelity_pen": float(fidelity_pen.detach())}
    return z_0, reward, log


def attack_native(
    sd,
    z_T: torch.Tensor,
    z_prev: Optional[torch.Tensor],
    ori_latents: torch.Tensor,
    class_name: str,
    c_a: str,
    cfg: NativeStage2Config,
    seed: int = 0,
    log_every: int = 1,
) -> Tuple[torch.Tensor, List[Dict]]:
    """對應 `attack_optimization_checkpoint` 的外層迴圈（Eq.7）。

    `z_T`：反演終點。DDIM 架構下就是唯一的起點；BDIA 架構下要另外傳
    `z_prev`（`z_{K-1}`），兩者當一個 pair 處理。

    `class_name`：CFG 的條件 prompt，跟階段一 LoRA 用同一個官方 class 名稱
    （原文 Figure 2：z_T 的去噪與 attack guidance 都在這個條件下進行）。
    `c_a`：注意力抑制損失要保護的詞——這裡兩者刻意給同一個字串（這三張圖
    只有一個顯著主體，class 名稱本身就是最自然的 c_a），但介面上兩者分開，
    未來要測「防禦方保護的詞跟編輯 prompt 的主體不同」時不必改函式簽名。

    回傳 `(x_def, history)`，`x_def` 為 (1,3,H,W) [0,1]，`history` 逐 iteration
    的 reward 與投影後的 L∞。
    """
    device = z_T.device
    emb_cond = sd.encode_text(class_name)
    emb_uncond = sd.uncond_prompt()
    emb_ca = sd.encode_text(c_a)
    span = token_span(sd.tokenizer, c_a)

    top = cfg.t_max or (sd.num_train_timesteps - 1)
    ts = sd.timesteps(cfg.steps, t_max=cfg.t_max)
    ref_t = torch.as_tensor(_ref_timestep(cfg, top), device=device)

    side = None  # aggregate_token_attention 用掃到的最大解析度
    with torch.no_grad():
        rec = CrossAttentionRecorder(sd.unet)
        with rec:
            sd._eps(ori_latents, ref_t, emb_ca)
        ref_att = aggregate_token_attention(rec.maps, span, side=side, reduce="sum")
        side = ref_att.shape[-1]
        mask = attention_region_mask(ref_att, tau=cfg.attn_mask_tau)

    la_0 = (z_T.detach().clone() if not cfg.use_bdia
           else (z_T.detach().clone(), z_prev.detach().clone()))
    # `adv` 必須跟 `la_0` 是不同的張量物件——下面 `adv.requires_grad_()`
    # 是 in-place 呼叫，兩者若共用同一個物件，`la_0`（後面當 `target0`，
    # L∞ 投影的固定中心）也會被標成需要梯度，讓「投影中心是常數」這個
    # 前提悄悄失效，且會在把 `noise` 轉成 float 記錄時跳出隱晦的警告。
    adv = (la_0.clone() if not cfg.use_bdia
          else (la_0[0].clone(), la_0[1].clone()))
    momentum = torch.zeros_like(z_T)

    history: List[Dict] = []
    for ii in range(cfg.niters):
        if cfg.use_bdia:
            a0, a1 = adv
            a0 = a0.requires_grad_()
            adv_in = (a0, a1)
            opt_var = a0
        else:
            adv = adv.requires_grad_()
            adv_in = adv
            opt_var = adv

        z_0, reward, log = _trajectory_pass(
            sd, adv_in, emb_cond, emb_uncond, emb_ca, ori_latents,
            cfg, span, mask, side, ts)

        grad = torch.autograd.grad(reward, opt_var, retain_graph=False,
                                   create_graph=False)[0].detach()
        l1_grad = grad / grad.norm(p=1).clamp_min(1e-12)
        momentum = momentum + l1_grad
        opt_var = opt_var.detach() + torch.sign(momentum) * cfg.mu
        target0 = la_0[0] if cfg.use_bdia else la_0
        noise = (opt_var - target0).clamp(-cfg.eps_a, cfg.eps_a)
        opt_var = target0 + noise

        if cfg.use_bdia:
            adv = (opt_var, la_0[1])
        else:
            adv = opt_var

        log.update({"iter": ii, "reward": float(reward.detach()),
                   "linf": float(noise.abs().max())})
        history.append(log)
        if ii % log_every == 0 or ii == cfg.niters - 1:
            print(f"  [attack_native] iter {ii:>3d}  reward={log['reward']:+.4f}  "
                 f"r_attn={log['r_attn']:+.4f}  fid_pen={log['fidelity_pen']:.4f}  "
                 f"linf={log['linf']:.3f}", flush=True)

    with torch.no_grad():
        z_final, _, _ = _trajectory_pass(
            sd, adv, emb_cond, emb_uncond, emb_ca, ori_latents,
            cfg, span, mask, side, ts)
        x_def = sd.decode_latent(z_final)
    return x_def, history
