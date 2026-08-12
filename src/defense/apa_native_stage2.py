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

### 換 reward 之後必須重新標定的一件事（DEC-021）

官方 APA-GC 的 reward 是 `CE − 10·MSE(z_0, z̄_0)`，**兩項同一個量級**，
後面那個 MSE 是有作用的保真煞車。注意力抑制損失是「16 層注意力圖在遮罩區
內的 L1 總和」，量級差三個數量級——實測 `r_attn` 455–1588 對
`fid_pen` 0.15–0.71，比值 1000–2000 倍，於是**那道煞車在換 reward 之後
等於被關掉**。這不是實作缺陷，是「只換 reward、其餘照抄」的必然後果，
而且不會有症狀：訓練跑得完、L∞ 球照樣綁住幅度，只是失真比原文大得多。

處置（使用者 2026-08-12 裁決）：把 `R_attn` 除以它自己在**第 0 次迭代**的
絕對值，使該項由 −1 起步，`fidelity_lambda` 維持官方的 10.0 不動
（`normalize_attn_reward`，預設開啟）。正的常數縮放不改變該項的梯度方向，
改變的只有它與保真項的相對權重。

    before（v1/v2 兩批）：reward = R_attn − 10·MSE
                          實測 r_attn ≈ 455–1588、fid_pen ≈ 0.15–0.71
    after （v3）        ：reward = R_attn/|R_attn⁰| − 10·MSE
                          第 0 次迭代該項恆為 −1，兩項回到同量級

**這是相對 APA 原文的第三個有記錄偏離**（另外兩個是 reward 本身的替換、
以及不做 diffusion augmentation），論文須載明。

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
    # **timestep 排程的粒度與實際執行的步數是兩件事，APA-GC 兩者不同。**
    #
    # 官方（`pipe_ours.py:186-210` 的 `ddim_inversion` 與
    # `attack_optimization_checkpoint` 的去噪迴圈，2026-08-12 逐行核對）：
    # `set_timesteps(50)` 給出 50 格排程，但反演走 `reversed(timesteps)`
    # 並在 `i == inversion_step`（GC 為 10）**break**，即只反演前 11 格、
    # 停在中等噪聲而非完整 z_T；去噪迴圈以
    # `if i < len(timesteps) - inversion_step - 1: continue` 跳過前 39 格，
    # 只跑最後 11 格——與反演的深度剛好對上。
    #
    # 這個「淺噪聲帶」正是 APA 保住保真度的機制：同一個 ε_a=0.4 的球，
    # 加在中等噪聲的 latent 上、只經 11 步去噪，與加在最大噪聲上、
    # 經 50 步放大，是完全不同的失真量級。
    #
    # 2026-08-12 修正。before：`steps=50` 且完整反演到最大噪聲再完整去噪
    # （本模組 docstring 原本記為「那個分段不是要控制的變因，追加只會多一層
    # 耦合」——**那個判斷是錯的**，它不是自由參數而是方法的一部分）。
    # 實測後果：CFG 修正後 12 個原生格的 `fid_lpips` 仍落在 0.50–0.71，
    # 而 APA 原文 Table 3 是 0.23–0.25。
    schedule_steps: int = 50    # 排程粒度（`set_timesteps(50)`）
    steps: int = 11             # 實際執行的反演／去噪步數（inversion_step=10 → 11 格）
    guidance_steps: int = 10    # T_a，最後幾步做 step-level guidance
    t_max: Optional[int] = None
    # **CFG 必須是 1.0，這是 APA 原生設定**（`dia_apa.md` §3.2：
    # `attack_alignment.py:146,148,152,156` 反演與去噪都是 `guidance_scale=1`）。
    #
    # 2026-08-12 修正。before：7.5（誤用本專案**評測期** SDEdit 的攻擊方 CFG）。
    # 那個值屬於另一條路徑——評測期模擬的是攻擊方用 stock SD 做文字引導編輯，
    # 需要高 CFG 才會服從 prompt（E26）；階段二是防禦方自己在跑生成鏈，
    # APA 在這裡刻意關掉 CFG。
    #
    # 症狀不是報錯而是數值：CFG=7.5 下文字條件把 latent 擾動放大成大幅語意
    # 改變，實測 12 個原生格的 `fid_lpips` 落在 0.51–0.82，而 APA 原文
    # Table 3 報的是 0.23–0.25。差距全部來自這一個旗標。
    guidance_scale: float = 1.0
    fidelity_lambda: float = 10.0  # 官方 APA-GC reward 裡 -10·MSE(ori_latents, z̄_0) 那一項
    # **把注意力 reward 正規化到 cross-entropy 原本佔的量級**（使用者
    # 2026-08-12 裁決，見 DEC-021）。`True` 時 `R_attn` 除以它自己在第 0 次
    # 迭代的絕對值，使該項由 −1 起步；`fidelity_lambda` 維持官方的 10.0 不動。
    #
    # 為什麼需要：官方 reward 是 `CE − 10·MSE(z_0, z̄_0)`，兩項同量級、
    # 那個 MSE 真的在制衡。換成注意力抑制損失（16 層注意力圖在遮罩區內的
    # L1 **總和**）之後量級差三個數量級——v2 實測 `r_attn` 455–1588 而
    # `fid_pen` 0.15–0.71，比值 1000–2000 倍，**APA 內建的那道保真煞車
    # 等於失效**。這不是實作錯誤，是「只換 reward、其他照抄」的必然結果。
    #
    # 為什麼是正規化 reward 而不是放大 λ：這樣官方的 λ=10 逐字保留，
    # 偏離只有一處且可逆；改 λ 則要挑一個 6.6×10⁴ 量級的數字，那個數字
    # 沒有出處、也無法對照回原文。正的常數縮放不改變該項的梯度方向，
    # 改變的只有它與保真項的相對權重——那正是要修的東西。
    #
    # 正規化常數取第 0 次迭代的值而非固定常數：`r_attn` 的絕對值隨影像的
    # 注意力質量而變（遮罩大小、主體佔比都會影響），固定常數會讓不同影像
    # 拿到不同的有效權重，而那個差異不會有症狀。
    normalize_attn_reward: bool = True
    attn_mask_tau: float = 0.5

    # ---- 三個新的變因軸（2026-08-12，使用者裁決一併跑）----
    #
    # `reward_mode` — 階段二在最大化什麼：
    #   "attn"       Lo et al. 式(5) 的注意力抑制（本輪主線，FND-024 已證明
    #                它與「編輯失敗」沒有因果關係）
    #   "targeted"   `−‖D(z) − y_target‖²`，形式取自 PhotoGuard-c 的 diffusion
    #                attack 與 Mist 的 textural loss。動機：本輪八條件中唯一
    #                在兩張圖、兩個語意指標上都為正的是 Mist，而它用的正是
    #                這個形式；FND-024 已排除「大失真本身就夠」，故剩下的
    #                可解釋變因是目標函數
    #   "classifier" **APA 原文的原生 reward**：替代分類器的 cross-entropy，
    #                untargeted（最大化 CE 使其偏離原類別）。這一臂是拿
    #                「誤導分類器」這件事本身去看它對文字引導編輯有沒有
    #                意外的抵抗效果——原文的攻擊目標與本專案的威脅模型
    #                不同，兩者有沒有交集是一個沒人量過的問題
    reward_mode: str = "attn"
    #
    # `fidelity_mode` — 保真度怎麼被控制：
    #   "ball"  latent L∞ 球（APA 原生）。**實測它從來沒有綁住過任何東西**：
    #           ε_a=0.4 而 µ×N=0.04×10=0.4，兩者恰好相等，log 中 linf 逐迭代
    #           精確等於 µ×iter，投影 Π 全程是空操作
    #   "soft"  把 DISTS 直接加進 reward（使用者提案的實驗二），取消球約束。
    #           `dists_lambda` 決定強度
    fidelity_mode: str = "ball"
    dists_lambda: float = 1.0
    #
    # `update_rule` — 怎麼更新 z_T：
    #   "sign"  APA 原生的 L1 動量 + sign + 固定步長 µ。**sign 丟掉梯度大小**，
    #           故此規則下失真幅度恆為 µ×N，與 reward 裡放什麼無關——這正是
    #           「soft + sign」被設計成對照組的理由：它應該與 ball 幾乎同失真，
    #           只有方向不同
    #   "adam"  真實梯度 + Adam。只有換掉 sign，loss 裡的保真項才真的能控制幅度
    update_rule: str = "sign"
    adam_lr: float = 0.02
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


# 分類器輸入的正規化。APA 的 `WrapperModel`（`utils.py:20-28`）依模型而異，
# ResNet 系用 ImageNet 的均值／標準差——本輪只用 ResNet-50（官方
# `--source_model` 的預設），故此處只放這一組，不做「依名稱查表」的分派：
# 換模型時要明寫，不得靜默沿用別的模型的常數。
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSIFIER_SIZE = 224


def load_source_classifier(device):
    """APA 原生 reward 用的替代分類器（ResNet-50，ImageNet 預訓練）。

    對應官方 `attack_alignment.py` 的 `--source_model ResNet50` 預設。
    回傳的模組已 `eval()` 且凍結——它是評分器不是被優化的對象。
    """
    import torchvision.models as tvm

    clf = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    clf.eval().to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


def _classifier_reward(sd, z: torch.Tensor, clf, label: torch.Tensor
                       ) -> torch.Tensor:
    """**APA 原文的原生 reward**：`R_a = CE(f_φ(D(z)), y)`，untargeted。

    最大化 cross-entropy 即把影像推離其真實類別，與官方
    `attack_optimization_checkpoint` 的 `reward = CrossEntropyLoss()(out, label)`
    逐字相同（官方另有 `−10·MSE` 那一項，由 `_trajectory_pass` 統一施加）。

    `decode_latent` 已回傳 [0,1]，故此處只做尺寸與 ImageNet 正規化，
    與官方 `decode_la` + `F.interpolate` + `WrapperModel` 的順序一致。
    """
    img = sd.decode_latent(z).clamp(0, 1)
    img = torch.nn.functional.interpolate(
        img, size=(CLASSIFIER_SIZE, CLASSIFIER_SIZE), mode="bilinear",
        align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    out = clf((img - mean) / std)
    return torch.nn.functional.cross_entropy(out, label)


def _targeted_reward(sd, z: torch.Tensor, y_target: torch.Tensor
                     ) -> torch.Tensor:
    """`−‖D(z) − y_target‖²`：把輸出推向一張固定的目標影像。

    形式取自 PhotoGuard-c 的 diffusion attack 與 Mist 的 textural loss，
    與本專案既有的 `targeted_output` 模式同一個量（`DESIGN` §4），故與
    B1／B2 可直接對照。負號使它與其餘 reward 同為「越大越好」。
    """
    img = sd.decode_latent(z).clamp(0, 1)
    return -torch.nn.functional.mse_loss(img, y_target)


def _reward_at(sd, z, t, cfg, ctx) -> torch.Tensor:
    """依 `cfg.reward_mode` 分派。`ctx` 帶各模式各自需要的東西。

    不對未知模式回退到預設：回退會讓「模式名稱打錯」變成「跑了另一個實驗」，
    而輸出上完全看不出來。
    """
    if cfg.reward_mode == "attn":
        return _attention_reward(sd, z, t, ctx["emb_ca"], ctx["span"],
                                 ctx["mask"], ctx["side"])
    if cfg.reward_mode == "classifier":
        return _classifier_reward(sd, z, ctx["clf"], ctx["label"])
    if cfg.reward_mode == "targeted":
        return _targeted_reward(sd, z, ctx["y_target"])
    raise ValueError(
        f"未知的 reward_mode {cfg.reward_mode!r}；只接受 "
        "'attn'／'classifier'／'targeted'，不回退到預設")


def _step_guidance(
    sd, z_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor,
    ori_latents: torch.Tensor, m_state: List[torch.Tensor],
    cfg: NativeStage2Config, ctx: Dict, abar: torch.Tensor,
) -> torch.Tensor:
    """`cond_guidance` 的逐字對應（Eq.8/9/10/11），reward 由 `cfg.reward_mode` 決定。

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
        reward = _reward_at(sd, z_in, t, cfg, ctx)
        grad = torch.autograd.grad(reward, z_local)[0]
    l1_grad = grad / grad.norm(p=1).clamp_min(1e-12)
    m_state[0] = m_state[0] + l1_grad.detach()
    return beta_prod_t.sqrt() * torch.sign(m_state[0])


def _trajectory_pass(
    sd, adv_latents: torch.Tensor, emb_cond: torch.Tensor,
    emb_uncond: torch.Tensor, ori_latents: torch.Tensor,
    cfg: NativeStage2Config, ctx: Dict,
    ts: torch.Tensor, attn_norm: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """跑一次完整去噪（可反傳），回傳 `(z_0, reward, log)`。

    `ts` 升冪、長度 `schedule_steps+1`（同 `sd.timesteps` 的慣例），本函式
    只走前 `cfg.steps` 格。DDIM 與 BDIA 共用這一個函式，差別只在
    `cfg.use_bdia` 切換單步公式。

    `attn_norm` 是單元素 list 的可變狀態，供 `normalize_attn_reward` 用：
    第一次呼叫時由本函式填入 `|R_attn|`，其後各次沿用同一個值。用可變容器
    而不是回傳值，是為了讓「整輪攻擊共用同一個正規化常數」成為呼叫端不必
    自己維護的性質——每次迭代各自正規化會讓 `R_attn` 恆為 −1，梯度尺度
    被抹平，那與這個旗標要做的事相反。
    """
    device = adv_latents[0].device if cfg.use_bdia else adv_latents.device
    abar = sd.alphas_cumprod(device)
    steps = cfg.steps
    # BDIA 的遞迴比 DDIM 少跑一格（它不需要反解反演的第 0 步，見
    # `sd.bdia_denoise`），故迴圈長度逐架構不同。guidance 一律施加在**最後
    # `guidance_steps` 步**，這樣兩種架構的 T_a 是同一個數，而不是同一個比例
    # ——後者會讓迴圈短的那一邊拿到比較少的 guidance 步數，那是架構之外的
    # 第二個變因。
    loop_len = (steps - 1) if cfg.use_bdia else steps
    index_cond = max(0, loop_len - cfg.guidance_steps)
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
                                       m_state, cfg, ctx, abar)
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
                                       m_state, cfg, ctx, abar)
                eps = eps - delta
            pred_x0 = (z - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
            z = abar[t_prev].sqrt() * pred_x0 + (1 - abar[t_prev]).sqrt() * eps
        z_0 = z

    ref_t = torch.as_tensor(_ref_timestep(cfg, cfg.t_max or (sd.num_train_timesteps - 1)),
                            device=z_0.device)
    r_raw = _reward_at(sd, z_0, ref_t, cfg, ctx)
    r_main = r_raw
    # 正規化對三種 reward 一律適用（DEC-021 的理由與 reward 是哪一種無關：
    # 它要的是「主項與官方 λ=10 的保真項同量級」）。cross-entropy 本來就在
    # O(1–10)，正規化後仍是 O(1)，故 classifier 臂與官方的相對權重最接近。
    if cfg.normalize_attn_reward:
        if attn_norm is None:
            raise ValueError(
                "normalize_attn_reward=True 但呼叫端沒有提供 attn_norm 狀態。"
                "正規化常數必須跨迭代共用，見 DEC-021"
            )
        if attn_norm[0] is None:
            attn_norm[0] = float(r_raw.detach().abs().clamp_min(1e-12))
        r_main = r_raw / attn_norm[0]
    fidelity_pen = cfg.fidelity_lambda * torch.nn.functional.mse_loss(
        ori_latents.float(), z_0.float())
    reward = r_main - fidelity_pen
    log = {"r_attn": float(r_main.detach()),
           "r_attn_raw": float(r_raw.detach()),
           "attn_norm": attn_norm[0] if attn_norm else None,
           "fidelity_pen": float(fidelity_pen.detach())}

    # soft 模式：把 DISTS 直接加進 reward（實驗二）。**只有在這個模式下
    # 才計算**——它需要一次 VAE 解碼加一次 DISTS 前向，ball 模式不該付這個成本。
    if cfg.fidelity_mode == "soft":
        dists_fn = ctx.get("dists_module")
        if dists_fn is None:
            raise ValueError(
                "fidelity_mode='soft' 需要可微的 DISTS（ctx['dists_module']）。"
                "不回退到 LPIPS：預算軸是哪一個度量必須與段 2 逐字相同")
        x_gen = sd.decode_latent(z_0).clamp(0, 1).float()
        d = dists_fn(x_gen, ctx["x01"].float())
        reward = reward - cfg.dists_lambda * d
        log["dists"] = float(d.detach())
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
    x01: Optional[torch.Tensor] = None,
    y_target: Optional[torch.Tensor] = None,
    clf=None,
    label: Optional[torch.Tensor] = None,
    dists_module=None,
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
    # **排程用 `schedule_steps` 產生、只執行前 `steps` 格**（官方
    # `set_timesteps(50)` + `break at i==inversion_step` 的對應）。
    # 用 `sd.timesteps(cfg.steps)` 產生排程是錯的：那會把 11 格均分到
    # 整個 [0, t_max]，每格的噪聲跨度變成官方的 4.5 倍，等於在完全不同的
    # 噪聲帶上運作。呼叫端（`apa_native_full_pipeline.run_native`）的反演
    # 也吃這一條 `ts`，兩邊必須是同一個排程。
    ts = sd.timesteps(cfg.schedule_steps, t_max=cfg.t_max)
    ref_t = torch.as_tensor(_ref_timestep(cfg, top), device=device)

    # 遮罩只有 attn 模式用得到，另兩種 reward 不需要它——但仍然一律計算，
    # 因為它同時是 `[suppress]` 那行診斷的來源（遮罩覆蓋率是判讀 c_a 選得
    # 對不對的唯一線索），成本是一次 UNet 前向。
    side = None  # aggregate_token_attention 用掃到的最大解析度
    with torch.no_grad():
        rec = CrossAttentionRecorder(sd.unet)
        with rec:
            sd._eps(ori_latents, ref_t, emb_ca)
        ref_att = aggregate_token_attention(rec.maps, span, side=side, reduce="sum")
        side = ref_att.shape[-1]
        mask = attention_region_mask(ref_att, tau=cfg.attn_mask_tau)

    ctx = {"emb_ca": emb_ca, "span": span, "mask": mask, "side": side,
           "x01": x01, "y_target": y_target, "clf": clf, "label": label,
           "dists_module": dists_module}
    # 缺件一律當場拋出，不讓它延後到迴圈裡才以 KeyError／None 的形式出現。
    need = {"targeted": ("y_target",), "classifier": ("clf", "label")}.get(
        cfg.reward_mode, ())
    missing = [k for k in need if ctx.get(k) is None]
    if missing:
        raise ValueError(
            f"reward_mode={cfg.reward_mode!r} 需要 {missing}，呼叫端沒有提供")
    if cfg.fidelity_mode == "soft" and (x01 is None or dists_module is None):
        raise ValueError("fidelity_mode='soft' 需要 x01 與 dists_module")

    la_0 = (z_T.detach().clone() if not cfg.use_bdia
           else (z_T.detach().clone(), z_prev.detach().clone()))
    # `adv` 必須跟 `la_0` 是不同的張量物件——下面 `adv.requires_grad_()`
    # 是 in-place 呼叫，兩者若共用同一個物件，`la_0`（後面當 `target0`，
    # L∞ 投影的固定中心）也會被標成需要梯度，讓「投影中心是常數」這個
    # 前提悄悄失效，且會在把 `noise` 轉成 float 記錄時跳出隱晦的警告。
    adv = (la_0.clone() if not cfg.use_bdia
          else (la_0[0].clone(), la_0[1].clone()))
    momentum = torch.zeros_like(z_T)
    adam_state = {"m": torch.zeros_like(z_T), "v": torch.zeros_like(z_T), "t": 0}
    # 整輪共用一個正規化常數，由第 0 次迭代填入（DEC-021）。
    attn_norm: List[Optional[float]] = [None]

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
            sd, adv_in, emb_cond, emb_uncond, ori_latents, cfg, ctx,
            ts, attn_norm)

        grad = torch.autograd.grad(reward, opt_var, retain_graph=False,
                                   create_graph=False)[0].detach()
        target0 = la_0[0] if cfg.use_bdia else la_0

        if cfg.update_rule == "sign":
            # APA 原生：L1 正規化動量 + sign + 固定步長。**sign 丟掉梯度大小**，
            # 故位移量恆為 µ×iter，與 reward 的內容無關。
            l1_grad = grad / grad.norm(p=1).clamp_min(1e-12)
            momentum = momentum + l1_grad
            opt_var = opt_var.detach() + torch.sign(momentum) * cfg.mu
        elif cfg.update_rule == "adam":
            # 真實梯度上升。用 Adam 的狀態手動實作而非 `torch.optim.Adam`：
            # 被優化的 `opt_var` 每次迭代都是新張量（重建計算圖），優化器
            # 綁在舊張量上會靜默地什麼都不更新。
            adam_state["t"] += 1
            b1, b2, eps_ = 0.9, 0.999, 1e-8
            g = -grad          # 最大化 reward = 最小化 −reward
            adam_state["m"] = b1 * adam_state["m"] + (1 - b1) * g
            adam_state["v"] = b2 * adam_state["v"] + (1 - b2) * g * g
            mh = adam_state["m"] / (1 - b1 ** adam_state["t"])
            vh = adam_state["v"] / (1 - b2 ** adam_state["t"])
            opt_var = opt_var.detach() - cfg.adam_lr * mh / (vh.sqrt() + eps_)
        else:
            raise ValueError(
                f"未知的 update_rule {cfg.update_rule!r}；只接受 'sign' 或 'adam'")

        if cfg.fidelity_mode == "ball":
            noise = (opt_var - target0).clamp(-cfg.eps_a, cfg.eps_a)
            opt_var = target0 + noise
        elif cfg.fidelity_mode == "soft":
            # 不投影：幅度由 reward 裡的 DISTS 項承擔（這正是本模式要測的）。
            noise = opt_var - target0
        else:
            raise ValueError(
                f"未知的 fidelity_mode {cfg.fidelity_mode!r}；只接受 'ball' 或 'soft'")

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
            sd, adv, emb_cond, emb_uncond, ori_latents, cfg, ctx,
            ts, attn_norm)
        x_def = sd.decode_latent(z_final)
    return x_def, history
