"""L∞ 球上的 PGD 免疫 —— Lo et al. (CVPR 2024) 的協定，本專案的文獻基準。

出處：Ling Lo, Cheng Yu Yeo, Hong-Han Shuai, Wen-Huang Cheng,
"Distraction is All You Need: Memory-Efficient Image Immunization against
Diffusion-Based Image Editing", CVPR 2024, pp. 24462–24471。

為什麼要另開一個模組，而不是在 `optimize.py` 加一個分支
─────────────────────────────────────────────────────────────────────

這條路徑與本專案既有的三條最佳化在**結構**上不同，不只是換一個損失：

| | `optimize()` 等既有路徑 | 本模組 |
|---|---|---|
| 被優化的對象 | residual module 的參數 φ | 像素擾動 δ 本身 |
| 更新規則 | Adam，步長由梯度大小決定 | `sign(grad)`，步長固定 |
| 保真約束 | LPIPS／鈍化／色度三道 hinge，軟性 | L∞ 球，硬性投影 |
| 防禦圖的產生 | `G(x; φ)`，含 VAE 編解碼來回 | `x + δ`，無任何來回 |
| 重建誤差下限 | 有（純 VAE 來回即 LPIPS 0.1434） | **無** |

最後一列是關鍵。本專案 A 類（latent／embedding／weight）的防禦圖都要經過
`decode(... encode(x) ...)`，因此背負一個與防禦無關的重建誤差下限；Lo 的方法是
純像素加性，`x_adv = x + δ`，沒有這個下限。把兩者塞進同一個函式會讓
「有沒有下限」這個結構差異藏在旗標裡。

與論文的四處偏離（全部有記錄，無一是靜默的）
─────────────────────────────────────────────────────────────────────

1. **Algorithm 1 第 13 行的 `x_adv ← x_adv − δ`**。字面執行會讓 x_adv 每一
   輪都再減一次**累積後**的 δ，n 輪之後偏離量是 Σδ_i 而非 δ，且與 L∞ 投影
   `clip(δ, −κ, κ)` 的用意矛盾——投影限制的是 δ，若 x_adv 累積偏移，實際
   偏離量根本不受 κ 限制。本實作採 PGD 的標準形式
   `x_adv ← clamp(x − δ, 0, 1)`，即每輪由**原圖**重新投影。判定為論文的
   誤植而非設計，因為只有這樣 κ 才真的是擾動預算。

2. **步長 s 論文未給值**。`step_size` 預設 κ/10，即 100 步中任一像素可在
   球內來回約十次，足以走遍球面。此值可由呼叫端指定。

3. **遮罩門檻 τ 論文未給值**，且式 (3) 的 Att 是 L 層相加、值域上界隨層數
   而變。本實作先把 Att 除以自身最大值再比較，τ 因此作用在 [0,1] 上，
   預設 0.5。理由與實作見 `src.models.attention.attention_region_mask`。

4. **前向雜訊 ε 論文未指定是否固定**。本實作對每個取樣到的 timestep 固定
   一個由 seed 決定的 ε，使損失是 δ 的確定性函數。若每步重抽，量到的
   逐步變化會混入雜訊差異，`plateau` 一類的判斷會失去意義。

`clamp(·, 0, 1)` 論文未提及，但影像必須落在有效值域，否則 `x_adv` 存成
PNG 時會被截斷，實際受攻擊的影像與最佳化的對象不是同一張。

記憶體：逐 timestep 反傳再累加梯度
─────────────────────────────────────────────────────────────────────

Algorithm 1 第 5–10 行對每個 t 各自算一次 `∇_{x_adv} L` 再取平均。數學上
等同於「把 T 個損失加總後反傳一次」（梯度是線性的），但**記憶體不同**：
後者要同時持有 T 份計算圖。論文的主要賣點正是記憶體效率，故此處照其
逐項反傳的寫法，以 `torch.autograd.grad` 取梯度並即時釋放該項的圖。

損失函式因此不回傳單一純量，而是回傳一個逐項產生純量的 generator。
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional

import torch

from src.models.sd import SDWrapper


@dataclass
class LinfAttackConfig:
    """Lo et al. §4.1 的設定。預設值即論文寫死的數值。"""

    # 論文：「For a fair comparison, we set the perturbation budget κ = 0.06,
    # number of iterations N = 100 for all attacks.」
    kappa: float = 0.06
    steps: int = 100
    # 論文未給步長，見模組 docstring 偏離之二。None 時取 κ/10 = 0.006，
    # 即 10 步可由 0 走到球面、其餘 90 步用於調整方向。比例 10:90 使「走得到
    # 邊界」與「有足夠步數修正方向」兩者都成立；1/255（≈0.0039）亦屬合理範圍。
    step_size: Optional[float] = None

    # 論文：「For the diffusion attack and our semantic attack, the number of
    # diffusion timesteps T is set to 10.」
    timesteps: int = 10

    # ---- semantic attack 專用 ----
    # 要保護的內容（論文的 c_a，例如 "woman"）。空字串在 semantic attack 下
    # 沒有意義，`make_semantic_attack_loss` 會拒絕。
    content: str = ""
    mask_tau: float = 0.5

    # ---- 攻擊方的設定 ----
    # 用於產生 t 的區間與 diffusion attack 的編輯鏈。
    #
    # strength 論文未公布。取 0.3（2026-08-03 決定，先前為 0.5）：論文 §4.3
    # 明說資料集與評測「adopt experimental settings similar to [23]」，而 [23]
    # 是 PhotoGuard，其 img2img 評測用 0.2／0.3。L1 的目的是重現他的 Table 1，
    # 故取最可能與他一致的值，而非本專案自選的 0.5。
    #
    # **這使 L1 與本專案既有的 run 不在同一個 strength 上**，兩者不可直接
    # 並列；L1 對照的是論文公布的數字，不是我們自己的舊資料。
    strength: float = 0.3
    # 論文亦未公布。依 E26 的結論固定 7.5——w=1 時 SD v1.4 幾乎不服從 prompt，
    # 該設定下的「編輯」與「什麼都沒做」在指標上分不出來。7.5 亦為 diffusers
    # 對 SD 的預設值。
    guidance_scale: float = 7.5
    n_edit: int = 10
    prompt_edit: str = "a photo"

    seed: int = 20260803
    log_every: int = 10
    vae_ckpt: bool = True


@dataclass
class LinfAttackResult:
    x_adv: torch.Tensor
    delta: torch.Tensor
    history: List[Dict]
    seconds: float = 0.0
    peak_mb: float = 0.0


# 損失介面：吃 x_adv，逐項產生要「最小化」的純量。
# 逐項而非單一純量的理由見模組 docstring 最後一節。
#
# **每一項必須自己擁有完整的計算圖。** `pgd_linf` 對每一項各呼叫一次
# `autograd.grad(..., retain_graph=False)`，共用的上游子圖會在第一項反傳後
# 被釋放，第二項即拋出「Trying to backward through the graph a second time」。
# 若有共用的前置計算（例如 VAE 編碼），要放進逐項的迴圈內重算，不是提到外面。
LossTerms = Callable[[torch.Tensor], Iterator[torch.Tensor]]


def pgd_linf(
    x01: torch.Tensor,
    loss_terms: LossTerms,
    cfg: LinfAttackConfig,
    tag: str = "pgd",
) -> LinfAttackResult:
    """Lo et al. Algorithm 1：timestep universal gradient updating。

        δ ← 0
        for n = 1..N:
            all_grad ← mean_t( ∇_{x_adv} L_t )
            δ ← clip( δ + s·sign(all_grad), −κ, +κ )
            x_adv ← clamp( x − δ, 0, 1 )

    `loss_terms` 逐項產生的純量會各自反傳一次，梯度累加後取平均。三種攻擊
    （semantic、PhotoGuard encoder、PhotoGuard diffusion）共用這個迴圈，
    差別只在損失——這與論文「所有攻擊共用 κ 與 N」的公平比較設定一致。

    δ 的符號：損失是要**最小化**的，而 `sign(all_grad)` 指向損失上升的方向，
    故 `x_adv = x − δ` 使 x_adv 沿著損失下降的方向移動。這與論文第 12–13 行
    的符號一致。
    """
    if cfg.kappa <= 0:
        raise ValueError(f"擾動預算 κ 必須為正，收到 {cfg.kappa}")
    s = cfg.step_size if cfg.step_size is not None else cfg.kappa / 10.0

    device = x01.device
    x_ref = x01.detach()
    delta = torch.zeros_like(x_ref)
    history: List[Dict] = []
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in range(cfg.steps):
        x_adv = (x_ref - delta).clamp(0.0, 1.0).detach().requires_grad_(True)

        all_grad = torch.zeros_like(x_ref)
        n_terms = 0
        loss_sum = 0.0
        for term in loss_terms(x_adv):
            g = torch.autograd.grad(term, x_adv, retain_graph=False)[0]
            all_grad += g
            loss_sum += float(term.detach())
            n_terms += 1
            del term, g
        if n_terms == 0:
            raise RuntimeError(
                "損失函式沒有產生任何項；梯度恆為零，最佳化不會做任何事"
            )
        all_grad /= n_terms

        delta = (delta + s * all_grad.sign()).clamp(-cfg.kappa, cfg.kappa)

        rec = {
            "step": step,
            "loss": loss_sum / n_terms,
            "grad_norm": float(all_grad.norm()),
            # 實際的 L∞ 偏離量。clamp 到 [0,1] 之後可能小於 ‖δ‖∞，故兩者
            # 都記：前者是投影後的參數，後者是影像上真的看得到的偏離。
            "delta_linf": float(delta.abs().max()),
            "eff_linf": float(
                ((x_ref - delta).clamp(0, 1) - x_ref).abs().max()
            ),
        }
        history.append(rec)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(
                f"  [{tag}] step {step:>4d}  L={rec['loss']:.6e}  "
                f"|g|={rec['grad_norm']:.3e}  "
                f"Linf={rec['eff_linf']:.4f}",
                flush=True,
            )

        del all_grad, x_adv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    x_adv = (x_ref - delta).clamp(0.0, 1.0)
    peak = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available()
        else 0.0
    )
    return LinfAttackResult(
        x_adv=x_adv.detach(),
        delta=(x_adv - x_ref).detach(),
        history=history,
        seconds=time.perf_counter() - t0,
        peak_mb=peak,
    )


# ---------------------------------------------------------------------------
# 三種損失。全部共用上面的 PGD 迴圈，與論文 Table 1 的三個欄位一一對應。
# ---------------------------------------------------------------------------


def make_semantic_attack_loss(
    sd: SDWrapper, x01: torch.Tensor, cfg: LinfAttackConfig
) -> LossTerms:
    """Lo et al. 的 semantic attack：`L = ‖Att(M ⊙ x_adv, c_a)‖₁`（式 5）。

    對應 Table 1 的 "Ours" 欄。

    遮罩 M 取自**原圖**（式 4），故對 δ 為常數，此處算一次後快取。論文的
    式 (4) 沒有指定在哪個 timestep 上取——注意力圖是 t 的函數，而 Att(x, c_a)
    的記號裡沒有 t。本實作對取樣到的全部 t 各算一次原圖的聚合注意力後取
    平均，再據以二值化：這樣遮罩代表「跨越整個去雜訊區間，模型認為這個詞
    在哪裡」，而不是某一個 t 的偶然結果。

    timestep 取樣點均分於 (0, t_edit]，t_edit 是 SDEdit 在該 strength 下的
    起始步。超出該區間的 t 攻擊方根本不會走到，在那裡施力是浪費預算。
    """
    from src.models.attention import (
        CrossAttentionRecorder,
        aggregate_token_attention,
        attention_region_mask,
        masked_attention_l1,
        token_span,
    )

    if not cfg.content.strip():
        raise ValueError(
            "semantic attack 需要指定要保護的內容 c_a（LinfAttackConfig.content）。"
            "論文的方法是壓低『那個詞』在其對應區域的注意力，沒有詞就沒有目標"
        )

    device = x01.device
    rec = CrossAttentionRecorder(sd.unet)
    emb = sd.encode_text(cfg.content).detach()
    span = token_span(sd.tokenizer, cfg.content)
    if span[1] <= span[0]:
        raise ValueError(
            f"內容 {cfg.content!r} 沒有產生任何內容 token（span={span}）"
        )

    t_edit = min(
        int(sd.num_train_timesteps * cfg.strength), sd.num_train_timesteps - 1
    )
    t_list = torch.linspace(0, t_edit, cfg.timesteps + 1)[1:].round().long()
    abar = sd.alphas_cumprod(device)
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    noises = [
        sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed + i)
        for i in range(len(t_list))
    ]

    # 參照注意力 → 遮罩。原圖，不含 δ，故只算一次。
    with torch.no_grad():
        z0 = sd.encode_image(x01)
        atts = []
        for t, n in zip(t_list, noises):
            zt = abar[t].sqrt() * z0 + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb)
            atts.append(aggregate_token_attention(rec.maps, span))
            rec.clear()
        att_ref = torch.stack(atts).mean(dim=0)
    mask = attention_region_mask(att_ref, cfg.mask_tau)
    coverage = float(mask.mean())
    print(
        f"  [semantic] c_a={cfg.content!r}  遮罩覆蓋 "
        f"{coverage * 100:.1f}% 的格點  "
        f"（{int(mask.sum())} / {mask.numel()}）",
        flush=True,
    )

    def terms(x_adv: torch.Tensor) -> Iterator[torch.Tensor]:
        for t, n in zip(t_list, noises):
            # VAE 編碼在迴圈**內**，每個 timestep 各自建一次圖。
            #
            # 放在迴圈外只算一次會壞：`pgd_linf` 對每一項各呼叫一次
            # `autograd.grad(..., retain_graph=False)`，第一次就把共用的編碼器
            # 子圖釋放掉，第二項拋出「Trying to backward through the graph a
            # second time」。本機實測確認（2026-08-03）。
            #
            # 代價是每個 PGD 步多 T−1 次 VAE 編碼。不改用 retain_graph 的理由
            # 是那會讓 T 份圖同時存活，而逐項反傳的整個用意就是不要如此——
            # 論文的賣點是記憶體效率（其 Figure 6）。這也與論文把去雜訊步
            # 視為「independent denoising steps」的框架一致。
            z = sd.encode_image(x_adv, use_ckpt=cfg.vae_ckpt)
            zt = abar[t].sqrt() * z + (1 - abar[t]).sqrt() * n
            with rec:
                # 此處不開 UNet checkpoint：hook 在原前向時掛著、重算時已卸除，
                # 兩次存檔的張量數對不上會讓 backward 直接中止。詳見
                # optimize_crossattn 的 docstring。
                sd._eps(zt, t, emb)
            att = aggregate_token_attention(rec.maps, span)
            rec.clear()
            yield masked_attention_l1(att, mask)

    # 遮罩覆蓋率要進 CSV，不能只印出來。`mask_tau` 是論文未公布、由本專案
    # 選定的超參數（見模組 docstring 偏離之三），其效果必須在事後可查：
    # 覆蓋率若接近 0 或接近 1，代表門檻選錯——前者等於幾乎沒有梯度，後者
    # 等於攻擊了整張圖而不是「那個詞所在的區域」，兩者都不是論文的方法。
    terms.mask_coverage = coverage
    return terms


def make_photoguard_encoder_loss(
    sd: SDWrapper, x_target: torch.Tensor, cfg: LinfAttackConfig
) -> LossTerms:
    """PhotoGuard 的 encoder attack：`L = ‖E(x_adv) − E(x_target)‖₂²`。

    對應 Table 1 的 "Encoder Attack" 欄。出處 Salman et al., ICML 2023。
    只攻擊 VAE 的 encoder，不碰 UNet，故只有一項、也不需要 timestep 迴圈。
    """

    with torch.no_grad():
        z_t = sd.encode_image(x_target).detach()

    def terms(x_adv: torch.Tensor) -> Iterator[torch.Tensor]:
        z = sd.encode_image(x_adv, use_ckpt=cfg.vae_ckpt)
        yield ((z - z_t) ** 2).mean()

    return terms


def make_photoguard_diffusion_loss(
    sd: SDWrapper, x_target: torch.Tensor, cfg: LinfAttackConfig
) -> LossTerms:
    """PhotoGuard 的 diffusion attack：`L = ‖SDEdit(x_adv) − x_target‖₂²`。

    對應 Table 1 的 "Diffusion Attack" 欄。要反傳穿過整條 n_edit 步的去雜訊
    鏈，這正是論文指出的記憶體瓶頸——與 semantic attack 的記憶體對比就在
    這一項上（論文 Figure 6）。

    只有一項（整條鏈是一個目標，不像 semantic attack 可以拆成獨立的 t），
    故 `pgd_linf` 的逐項反傳在此退化成單次反傳。這個退化本身就是論文要
    呈現的差異，不是實作的缺陷。
    """

    device = x_target.device
    emb = sd.encode_text(cfg.prompt_edit).detach()
    emb_un = sd.encode_text("").detach()
    lat = sd.latent_shape(x_target.shape[-2], x_target.shape[-1])
    noise = sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed)
    tgt = x_target.detach()

    def terms(x_adv: torch.Tensor) -> Iterator[torch.Tensor]:
        y = sd.sdedit(
            x_adv, emb, noise, cfg.n_edit, strength=cfg.strength,
            use_ckpt=True, vae_ckpt=cfg.vae_ckpt,
            guidance_scale=cfg.guidance_scale, emb_uncond=emb_un,
        )
        yield ((y - tgt) ** 2).mean()

    return terms


# Table 1 的三個欄位。三個建構函式的第二個位置引數都是一張影像，但**語意
# 不同**：semantic attack 要的是原圖（用來取遮罩），兩個 PhotoGuard 變體要的
# 是目標影像（要把輸出推向它，PhotoGuard 用灰圖）。呼叫端必須依此分派，
# 傳錯不會報錯只會量到別的東西，故 `ATTACKS` 一併記下該引數的角色。
ATTACKS = {
    "semantic": (make_semantic_attack_loss, "x01"),
    "pg_encoder": (make_photoguard_encoder_loss, "x_target"),
    "pg_diffusion": (make_photoguard_diffusion_loss, "x_target"),
}


def build_attack(
    name: str,
    sd: SDWrapper,
    cfg: LinfAttackConfig,
    x01: torch.Tensor,
    x_target: Optional[torch.Tensor] = None,
) -> LossTerms:
    """依名稱建構損失，並把「第二引數該傳哪張圖」的分派收在這裡。"""
    if name not in ATTACKS:
        raise ValueError(
            f"未知的攻擊 {name!r}；可用的是 {sorted(ATTACKS)}"
        )
    fn, which = ATTACKS[name]
    if which == "x01":
        return fn(sd, x01, cfg)
    if x_target is None:
        raise ValueError(
            f"{name} 是有目標的攻擊（PhotoGuard），必須提供 x_target。"
            "論文用的是灰圖，見 data/targets/gray.png"
        )
    return fn(sd, x_target, cfg)
