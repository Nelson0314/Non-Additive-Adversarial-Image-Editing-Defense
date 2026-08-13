"""弱 baseline 的階段二：APA 原生 dual-path attack guidance + latent L∞ 球。

逐字對照 `docs/reference/dia_apa.md` §3（官方 `pipe_ours.py` 的
`attack_optimization_checkpoint` 與 `cond_guidance`，2026-08-12 取原始碼核對）。
走 **APA-GC**（gradient checkpointing，真實梯度），不是 APA-SG——後者需要一個
換掉 reward 之後意義不明的縮放常數 14.58，而原文自己的數字顯示 GC 較好。

本檔在 2026-08-12 縮到**只剩 baseline 這一條路徑**。曾經在此並存、且都已測過
並否決的變體（注意力抑制／分類器 CE／latent／CLIP 四種 reward、DISTS 進 loss
的軟約束、Adam 更新規則）已移除，結論記在 FND-027…030。要取回：

    git checkout a4f93451f -- src/defense/apa_native_stage2.py

## 與官方程式碼的對應

| 官方 | 這裡 |
|---|---|
| `attack_optimization_checkpoint` 的外層迴圈 | `attack_native` 的 `for ii in range(niters)` |
| `la_t_g = grad(reward, adv_latents)` | trajectory-level 的 `torch.autograd.grad` |
| `momentum += g/‖g‖₁；adv += sign(momentum)*alpha` | 同一段，逐字保留 |
| `noise = clamp(adv-la_0, -eps, eps)` | 同一段 |
| `cond_guidance`（Eq.8/9/10/11） | `_step_guidance` |
| `reward = CE(...) − 10*MSE(ori_latents, la_t)` | `reward = R_target − fidelity_lambda*MSE(...)` |

## 相對 APA 原文的三個有記錄偏離

1. **reward 換成 targeted output**：`−‖D(z̄_0) − y_target‖²`。原文的 reward 是
   替代分類器的 cross-entropy，而抗編輯場景沒有分類器，該式字面上不可執行。
   換用的形式取自 PhotoGuard-c 的 diffusion attack 與 Mist 的 textural loss，
   故與那兩個加性 baseline 在損失形式上可直接對照。
   **原文自己的分類器 reward 已測過**：失真最高而抗編輯無效（FND-030）。
2. **reward 正規化**：除以它自己在第 0 次迭代的絕對值，使該項由 −1 起步，
   `fidelity_lambda` 維持官方的 10.0（DEC-021）。不做的話換 reward 會讓兩項
   量級差三個數量級，等於關掉官方內建的那道保真煞車。
3. **不做 diffusion augmentation**：它服務的是黑盒可遷移性，不在本專案的
   威脅模型內。

## 兩個必須照官方、否則數值會錯的地方

- **CFG = 1.0**：官方反演與去噪都關掉 CFG（`attack_alignment.py:146,148,152,156`）。
  誤用評測期的 7.5 會讓文字條件把 latent 擾動放大成大幅語意改變，實測
  `fid_lpips` 由 0.23–0.34 惡化到 0.51–0.82。
- **淺噪聲帶**：官方 `set_timesteps(50)` 但反演在 `i == inversion_step`（GC 為
  10）就 break，只走前 11 格、停在中等噪聲；去噪對稱地只跑最後 11 格。
  做成「完整反演到 z_T 再完整去噪」會讓同一個 ε_a 的失真量級完全不同。

## 已知性質：latent 球其實從未生效

`ε_a = 0.4` 而 `µ × N = 0.04 × 10 = 0.4`，兩者**恰好相等**；實測 `linf` 逐迭代
精確等於 `µ × iter`，投影 `Π` 全程是空操作。失真幅度完全由「步長 × 迭代數」
決定，與 reward 無關——這是 sign 更新丟掉梯度大小的必然結果（FND-028）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class NativeStage2Config:
    """對應官方的 `attack_config`（`eps`/`alpha`/`niters`）與 `index_cond`。"""

    eps_a: float = 0.4          # ε_a，Eq.7 的 L∞ 球半徑
    mu: float = 0.04            # µ，trajectory-level 步長
    niters: int = 10            # N，外層迭代數
    # **排程粒度與實際執行的步數是兩件事**（見檔頭「淺噪聲帶」）：官方
    # `set_timesteps(50)` 給 50 格，但只執行前 11 格。用 `sd.timesteps(steps)`
    # 產生排程是錯的——那會把 11 格均分到整個 [0, t_max]，每格的噪聲跨度
    # 變成官方的 4.5 倍，等於在完全不同的噪聲帶上運作。
    schedule_steps: int = 50
    steps: int = 11             # 實際執行的反演／去噪步數（inversion_step=10 → 11 格）
    guidance_steps: int = 10    # T_a，最後幾步做 step-level guidance
    t_max: Optional[int] = None
    guidance_scale: float = 1.0    # 官方設定，見檔頭
    fidelity_lambda: float = 10.0  # 官方 reward 裡 −10·MSE(z_0, z̄_0) 那一項
    normalize_reward: bool = True  # DEC-021
    use_ckpt: bool = True
    use_bdia: bool = False


def _targeted_reward(sd, z: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """`−‖D(z) − y_target‖²`：把輸出推向一張固定的目標影像。

    負號使它與官方「maximize R_a」的號約定一致。
    """
    return -torch.nn.functional.mse_loss(sd.decode_latent(z).clamp(0, 1), y_target)


def _step_guidance(
    sd, z_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor,
    ori_latents: torch.Tensor, m_state: List[torch.Tensor],
    y_target: torch.Tensor, abar: torch.Tensor,
) -> torch.Tensor:
    """`cond_guidance` 的逐字對應（Eq.8/9/10/11）。

    `z_t` 進來後**先 detach**：官方在這裡開一個獨立於外層計算圖的局部梯度，
    只用來算 sign-momentum 修正量，不讓這段反傳污染 trajectory-level 那條線
    ——兩者是並行的兩條路徑，這正是「dual-path」的字面意思。
    """
    with torch.enable_grad():
        z_local = z_t.detach().requires_grad_()
        alpha_prod_t = abar[t]
        beta_prod_t = 1.0 - alpha_prod_t
        pred_x0 = (z_local - beta_prod_t.sqrt() * noise_pred.detach()) / alpha_prod_t.sqrt()
        fac = beta_prod_t.sqrt()
        z_in = ori_latents * fac + pred_x0 * (1.0 - fac)          # Eq.10
        reward = _targeted_reward(sd, z_in, y_target)
        grad = torch.autograd.grad(reward, z_local)[0]
    l1_grad = grad / grad.norm(p=1).clamp_min(1e-12)
    m_state[0] = m_state[0] + l1_grad.detach()
    return beta_prod_t.sqrt() * torch.sign(m_state[0])


def _trajectory_pass(
    sd, adv_latents, emb_cond: torch.Tensor, emb_uncond: torch.Tensor,
    ori_latents: torch.Tensor, y_target: torch.Tensor,
    cfg: NativeStage2Config, ts: torch.Tensor,
    norm: Optional[List[Optional[float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """跑一次完整去噪（可反傳），回傳 `(z_0, reward, log)`。

    `ts` 升冪、長度 `schedule_steps+1`，本函式只走前 `cfg.steps` 格。
    DDIM 與 BDIA 共用，差別只在單步公式。

    `norm` 是單元素 list 的可變狀態：第一次呼叫時填入 `|R|`，其後沿用同一個值
    ——每次迭代各自正規化會讓 R 恆為 −1、梯度尺度被抹平，與 DEC-021 要做的
    事相反。
    """
    device = adv_latents[0].device if cfg.use_bdia else adv_latents.device
    abar = sd.alphas_cumprod(device)
    steps = cfg.steps
    # BDIA 的遞迴比 DDIM 少跑一格（不需反解反演的第 0 步，見 `sd.bdia_denoise`），
    # 故迴圈長度逐架構不同。guidance 一律施加在**最後 `guidance_steps` 步**，
    # 使兩種架構的 T_a 是同一個數而非同一個比例。
    loop_len = (steps - 1) if cfg.use_bdia else steps
    index_cond = max(0, loop_len - cfg.guidance_steps)
    m_state = [torch.zeros(1, device=device)]

    if cfg.use_bdia:
        z_next, z_cur = adv_latents
        for step_idx, i in enumerate(range(steps - 1, 0, -1)):
            t = ts[i]
            eps = sd._eps_cfg(z_cur, t, emb_cond, cfg.guidance_scale,
                              emb_uncond, use_ckpt=cfg.use_ckpt)
            if step_idx >= index_cond:
                eps = eps - _step_guidance(sd, z_cur, t, eps, ori_latents,
                                           m_state, y_target, abar)
            a_plus = sd._ddim_step(z_cur, eps, ts[i], ts[i + 1], abar)
            a_minus = sd._ddim_step(z_cur, eps, ts[i], ts[i - 1], abar)
            z_next, z_cur = z_cur, (z_next - a_plus) + a_minus   # gamma=1
        z_0 = z_cur
    else:
        z = adv_latents
        for step_idx, i in enumerate(range(steps - 1, -1, -1)):
            t, t_prev = ts[i + 1], ts[i]
            eps = sd._eps_cfg(z, t, emb_cond, cfg.guidance_scale,
                              emb_uncond, use_ckpt=cfg.use_ckpt)
            if step_idx >= index_cond:
                eps = eps - _step_guidance(sd, z, t, eps, ori_latents,
                                           m_state, y_target, abar)
            pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps
        z_0 = z

    r_raw = _targeted_reward(sd, z_0, y_target)
    r_main = r_raw
    if cfg.normalize_reward:
        if norm is None:
            raise ValueError("normalize_reward=True 但呼叫端沒有提供 norm 狀態，"
                             "正規化常數必須跨迭代共用（DEC-021）")
        if norm[0] is None:
            norm[0] = float(r_raw.detach().abs().clamp_min(1e-12))
        r_main = r_raw / norm[0]
    fidelity_pen = cfg.fidelity_lambda * torch.nn.functional.mse_loss(
        ori_latents.float(), z_0.float())
    return z_0, r_main - fidelity_pen, {
        "reward_main": float(r_main.detach()),
        "reward_raw": float(r_raw.detach()),
        "fidelity_pen": float(fidelity_pen.detach()),
    }


def attack_native(
    sd,
    z_T: torch.Tensor,
    z_prev: Optional[torch.Tensor],
    ori_latents: torch.Tensor,
    class_name: str,
    y_target: torch.Tensor,
    cfg: NativeStage2Config,
    log_every: int = 2,
) -> Tuple[torch.Tensor, List[Dict]]:
    """官方 `attack_optimization_checkpoint` 外層迴圈（Eq.7）的對應。

    `z_T` 為反演終點；BDIA 架構下另需 `z_prev`（`z_{K−1}`），兩者當一對處理
    ——BDIA 的遞迴狀態是相鄰兩點，只給 z_T 解不出下一步。

    `class_name` 是去噪與 attack guidance 的文字條件（原文 Figure 2 兩者同條件）。
    `y_target` 是 targeted reward 的目標影像，(1,3,H,W)、[0,1]。

    回傳 `(x_def, history)`，`history` 逐 iteration 的 reward 與投影後的 L∞。
    """
    emb_cond = sd.encode_text(class_name)
    emb_uncond = sd.uncond_prompt()
    ts = sd.timesteps(cfg.schedule_steps, t_max=cfg.t_max)

    la_0 = (z_T.detach().clone() if not cfg.use_bdia
            else (z_T.detach().clone(), z_prev.detach().clone()))
    # `adv` 必須與 `la_0` 是不同物件：下面的 `requires_grad_()` 是 in-place，
    # 共用會讓 L∞ 投影的固定中心也被標成需要梯度。
    adv = (la_0.clone() if not cfg.use_bdia
           else (la_0[0].clone(), la_0[1].clone()))
    momentum = torch.zeros_like(z_T)
    norm: List[Optional[float]] = [None]

    history: List[Dict] = []
    for ii in range(cfg.niters):
        if cfg.use_bdia:
            a0 = adv[0].requires_grad_()
            adv_in, opt_var = (a0, adv[1]), a0
        else:
            adv = adv.requires_grad_()
            adv_in = opt_var = adv

        z_0, reward, log = _trajectory_pass(
            sd, adv_in, emb_cond, emb_uncond, ori_latents, y_target, cfg, ts, norm)
        grad = torch.autograd.grad(reward, opt_var)[0].detach()

        # 官方更新規則：L1 正規化動量 + sign + 固定步長，再投影回 L∞ 球。
        l1_grad = grad / grad.norm(p=1).clamp_min(1e-12)
        momentum = momentum + l1_grad
        opt_var = opt_var.detach() + torch.sign(momentum) * cfg.mu
        target0 = la_0[0] if cfg.use_bdia else la_0
        noise = (opt_var - target0).clamp(-cfg.eps_a, cfg.eps_a)
        opt_var = target0 + noise
        adv = (opt_var, la_0[1]) if cfg.use_bdia else opt_var

        log.update({"iter": ii, "reward": float(reward.detach()),
                    "linf": float(noise.abs().max())})
        history.append(log)
        if ii % log_every == 0 or ii == cfg.niters - 1:
            print(f"  [attack_native] iter {ii:>3d}  reward={log['reward']:+.4f}  "
                  f"R={log['reward_main']:+.4f}  fid_pen={log['fidelity_pen']:.4f}  "
                  f"linf={log['linf']:.3f}", flush=True)

    with torch.no_grad():
        z_final, _, _ = _trajectory_pass(
            sd, adv, emb_cond, emb_uncond, ori_latents, y_target, cfg, ts, norm)
        x_def = sd.decode_latent(z_final)
    return x_def, history
