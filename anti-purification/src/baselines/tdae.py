"""TDAE —— Zhang, Dong, Shan, Chen，*Towards Transferable Defense Against
Malicious Image Edits*，arXiv:2512.14341v2（已被 IEEE TPAMI 接受）。

**這是依論文重建的，不是重現官方實作。** TDAE 沒有釋出程式碼、沒有
checkpoint：論文全文（HTML v2）不含 Appendix、不含補充資料，全文檢索
`github.com`／`code is available`／`supplementary` 皆為零筆。逐條查證見
`docs/reference/AUDIT_TDAE.md`。因此本模組的每一個數只有兩種來源：
**論文正文寫出來的**，或**本專案指定的**——後者一律標成必填參數或寫進
`modification_note`，沒有第三種。

論文的方法
──────────────────────────────────────────────────────────────────────

TDAE 是 plug-and-play 模組（§I 貢獻列表），套在既有免疫方法（ACE、Mist、
PGD、PGE、SA）之上，由兩個機制組成：

`FDM`（FlatGrad Defense Mechanism，§III-C）
    把梯度範數正則項放進對抗目標，把 δ_v 推向平坦極小值。**不是** SAM 式的
    ascent-then-descent：SAM 用鄰域最大損失點的梯度取代原梯度，FDM 是在
    原目標上**加一個梯度範數懲罰項**，再用有限差分把它化成兩次一階梯度。
    論文 Eq. (4) 的內層 max 被明文放棄（§III-C「computationally
    intractable」），實跑的是 Eq. (5)：

        min_{‖δ_v‖_p ≤ ε_v}  −L(f_θ(x₀+δ_v, e), y₀)
                             + λ·‖∇_{δ_v} L(f_θ(x₀+δ_v, e), y₀)‖₂

    其中 s = ∇L/‖∇L‖₂（Eq. 6；∇L = 0 時 s = 0），以有限差分近似方向導數
    （Eq. 7），取絕對值以保證非負（Eq. 8）：

        J(δ_v) = −L_{δ_v} + (λ/h)·| L_{δ_v+h·s} − L_{δ_v} |            (8)

        g_FDM  = −∇L_{δ_v} + (λ/h)·sign(z)·( ∇L_{δ_v+h·s} − ∇L_{δ_v} )  (9)
                 其中 z = L_{δ_v+h·s} − L_{δ_v}

    Eq. (10)：∇L_{δ_v+h·s} 直接取「在 δ' = δ_v + h·s 這一點對 δ 的一階
    梯度」，不算二階導數（Hessian-free）。Algorithm 1 第 18–25 行逐字如此：
    g₁ → s → δ' → g₂ → z → g_FDM → `δ_v ← δ_v − α·sign(g_FDM)` → 投影。

`DPD`（Dynamic Prompt Defense，§III-D）
    每 S 步把 δ_v 固定住，對文字嵌入的擾動 δ_p 做 M 步 PGD，目標是
    **最小化** L（Eq. 11），即找一個能繞過當前防禦的 prompt；接著用
    e = c + δ_p 回去更新 δ_v（Eq. 12）。δ_p 受 ‖δ_p‖_∞ ≤ ε_p 約束。

損失 L 本身
──────────────────────────────────────────────────────────────────────

論文 §III-A 只說 L 是「可微分的差異度量（e.g. ℓ₂-distance）」，量的是
編輯輸出 f_θ(x_adv, c) 與 **benign target output y₀** 的差距；§III-C 之後
一律寫成 `L(f_θ(·), y₀)`，y₀ 在 Algorithm 1 的 Input 列裡是給定的常量。
TDAE 當 plug-and-play 用時，L 就是被套的那一篇的損失（§III-D「where L is
the loss function used throughout the framework」）。

本模組取 §III-A 的字面定義作為 L：

    L(x) = ‖ f_θ(x, e) − y₀ ‖₂ ，  y₀ = f_θ(x₀, e)（乾淨圖的編輯結果）

即「未防禦的編輯」當作 y₀，這正是本 repo 評測欄 `edit_*` 的分母定義
（`scripts/baseline_run.py::evaluate`）。範數不平方，與 §III-A 的
「ℓ₂-distance」字面一致。

與論文對不上的地方（完整清單見 `AUDIT_TDAE.md`）
──────────────────────────────────────────────────────────────────────

1. **文字側（DPD）與本專案的威脅模型衝突。** 本專案的防禦方**看不到攻擊
   指令**，而 DPD 的 c 是「該張圖在 InstructPix2Pix-clip-filtered 裡附帶的
   原始編輯指令」（§IV-A Target Models and Dataset）。拿到那個指令才談得上
   在它的鄰域裡搜 δ_p。故本模組**只實作影像側（FDM）**，文字側以固定的
   中性 prompt（空字串，`TDAE_PROMPT`）代入、δ_p ≡ 0。這個組態不是我們發明
   的：論文 Table VI 的 `SA + FDM`、Table I–III 的 `PGE + FDM` 就是只開 FDM
   的那一列。`prepare(dpd=...)` 只要不是 `None` 就拋
   `NotImplementedError`，把「要做 DPD 必須先取得攻擊指令、且論文沒給
   S／M／ε_p／η 四個數」寫在訊息裡，不預設一組看起來合理的值。
2. **空 prompt 在 SD v1.x 上讓 CFG 退化。** `encode_text("")` 與
   `uncond_prompt()` 在 SD v1.x 上相同，兩支前向數值相等，CFG 恆等於無條件
   預測（與 `photoguard.py` 記過的同一件事）。於是 y₀ 是一次**無指令的
   SDEdit 重建**，L 量的是「免疫圖的無指令重建」對「乾淨圖的無指令重建」的
   ℓ₂ 距離。這是威脅模型的直接後果，不是論文的設定。
3. **論文沒說 SD14／SD3 是怎麼當編輯模型用的。** §IV-A 只寫「multiple
   versions of StableDiffusion (SD) configured for editing tasks」。全文
   `SDEdit` 只出現在 §II-A 的引用、`img2img` 零筆、`strength`／`512`／`1024`
   只出現在參考文獻字串裡。本 repo 先前查證記的「未說明」核對為正確。故
   `strength` 與 `num_inference_steps` 在本模組**無預設、必填**。
4. **SD3 這一側本輪不做。** `src/models/sd.py` 只有 SD v1.x／SDXL／inpainting
   三種封裝，沒有 SD3（MMDiT、三個 text encoder、flow-matching 排程），
   全 repo 檢索 `SD3`／`SD3Transformer` 為零筆。論文的 Table II（SD3 為代理）
   與所有 →SD3 的轉移列因此不可重建，缺口記在 `AUDIT_TDAE.md` §6。

每一個超參數的出處
──────────────────────────────────────────────────────────────────────

| 參數 | 值 | 出處 |
|---|---|---|
| `λ/h` | 0.3 | §IV-E「we adopt λ/h = 0.3 for other experiments involving TDAE with PGD and SA」 |
| δ_v 起點 | 0 | Algorithm 1 第 1 行 `δ_v ← 0`，故 `init_rule="none"` |
| 更新規則 | sign | Algorithm 1 第 25 行 `δ_v ← δ_v − α·sign(g_FDM)` |
| 投影 | ℓ∞ | Algorithm 1 第 26 行 `Π_{‖·‖_∞ ≤ ε_v}` |
| 每步梯度數 | 1 | Algorithm 1 一步只算 g₁、g₂ 各一次，無多樣本平均 |
| `h` | **未找到** | 全文只出現在 Algorithm 1 的 Input 列與 λ/h 之中，沒有數值。必填 |
| `λ` | **未找到** | 同上，只有 λ/h 這個比值 |
| `ε_v`、`α`、`N` | **未找到** | Algorithm 1 的 Input 列有這三個符號，正文與圖表都沒有數值 |
| `ε_p`、`η`、`S`、`M` | **未找到** | 同上；DPD 未實作，見上面第 1 點 |
| `strength`、步數 | **未找到** | 見上面第 3 點 |

`ε_v`／`α`／`N` 的處置（本專案的決定，已列入 `modification_note`）：TDAE 是
plug-and-play，預算屬於被套的那一篇。論文的 `PGD [32]` 是 Salman et al.
(arXiv:2302.06588)，本 repo 依其論文 Table 9 重建的臂是 `photoguard_linf`
（ℓ∞ 16/255、step 2/255、200 步，見 `docs/reference/BASELINE_PROVENANCE.md`）。
本模組取同一組，好處是與既有 baseline 在同一個 `eps01` 上頭對頭；代價是
**它不是 TDAE 論文的預算**。

另有一個**反推**的數（依 `BASELINE_PROVENANCE` 規則 4 標星號，反推的數不是
量到的數）：論文 Table V 報 `PGD` 的免疫圖對原圖 PSNR = 36.04 dB、
`PGD + TDAE` = 35.09 dB。若把 sign-PGD 視為飽和（絕大多數像素落在 ±ε_v），
rms ≈ ε_v，則 ε_v* ≈ 255·10^(−36.04/20) ≈ 4.03/255。這與本模組採用的
16/255 差約四倍。此數只供對照，不作為設定值。

實作上的兩個決定
──────────────────────────────────────────────────────────────────────

1. **整次求解固定同一個編輯噪聲。** Eq. (7) 的有限差分要求 L_{δ} 與
   L_{δ+h·s} 是**同一個** f_θ 在兩點上的值；若兩次前向換噪聲，z 就是兩個
   不同函式的差，除以 h 之後不再近似任何方向導數。論文沒有提到噪聲怎麼取
   （全文無 `seed`），此處固定，並把 seed 記進 `extras`。
2. **`flatgrad_objective` 回傳的張量：值是 Eq. (8)、梯度是 Eq. (9)。**
   兩者由同一次呼叫給出，且**只做兩次 f_θ 前向**（與 Algorithm 1 同）。
   作法見該函式的 docstring；那裡有逐項的代數對照。不這樣寫的話，因為 s
   需要 ∇L_{δ_v}，就必須對 L_{δ_v} 的計算圖反向兩次，而該圖上有
   `torch.utils.checkpoint`（`use_reentrant=False`）區塊。

要加到 `scripts/baseline_run.py` 的幾行（本模組不改該檔）
──────────────────────────────────────────────────────────────────────

    1. import 那一行加 `tdae`：
           from src.baselines import dia, mist, photoguard, tdae
    2. `run_additive` 的 spec 表加一列：
           "tdae": tdae.SPEC_PAPER,
    3. `run_additive` 的分派加一支（三個數都無出處，必須顯式寫出來）：
           elif name == "tdae":
               kw = {"strength": strength, "num_inference_steps": 4, "h": 1.0}
    4. `CONDITIONS` 加 `"tdae"`。

`h = 1.0` 是本專案指定的，論文沒有這個數。它是 `[-1,1]` 值域上的 ℓ₂ 半徑
（s 已單位化），可與可行域比對：512² RGB 在 ℓ∞ 32/255 下，球內最遠點的
ℓ₂ 距離是 (32/255)·√(3·512·512) ≈ 111，故 h = 1.0 約是可行域尺度的 1%。
換 h 必須同時記在報表上，因為 λ = 0.3·h 會跟著變。
"""

from typing import Callable, Optional

import torch

from src.baselines.pgd import BaselineSpec, ValueRange

# 論文沒有釋出程式碼，故沒有「原始碼在哪一行把影像映到這個值域」可引。
# 取 `[-1,1]` 的理由是：擴散模型的 VAE 輸入本來就在 `[-1,1]`，本 repo 其餘
# 五篇 baseline 也全部在這個值域上最佳化（`pgd.py` 模組 docstring 的對照表）。
# 這是本專案的決定，已列入 `modification_note`。
TDAE_RANGE = ValueRange(
    -1.0,
    1.0,
    "論文無官方程式碼，值域未載；本專案沿用 diffusion VAE 輸入慣例 [-1,1]，"
    "與 photoguard／mist／dia 同（見 src/baselines/pgd.py 模組 docstring）",
)

# 文字側的中性 prompt。論文的 c 是資料集附帶的原始編輯指令（§IV-A），
# 本專案的防禦方看不到指令，故以空字串代入。見模組 docstring 第 1、2 點。
TDAE_PROMPT = ""

# §IV-E：「we adopt λ/h = 0.3 for other experiments involving TDAE with PGD and SA」。
# 論文只固定這個**比值**，λ 與 h 各自的值全文沒有。
LAMBDA_OVER_H = 0.3


# ---------------------------------------------------------------------------
# FDM：Eq. (5)–(10)
# ---------------------------------------------------------------------------


def flatgrad_objective(
    base_loss: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    lam_over_h: float,
    h: float,
) -> torch.Tensor:
    """回傳一個純量，其**值**為 Eq. (8)、其**梯度**為 Eq. (9)。

    `base_loss(x) -> 純量` 是論文的 L（本模組為 `_edit_distance`）。`x` 必須
    `requires_grad`，且是骨幹 `run_pgd` 傳進來的那一個葉節點。

    為什麼值與梯度要分開組
    ────────────────────────────────────────────────────────────────
    Eq. (6) 的 s = ∇L_{δ}/‖∇L_{δ}‖₂ 本身就要先對 L_{δ} 反向一次，之後
    Eq. (9) 又用到 ∇L_{δ}。若直接把 `−L_cur + (λ/h)·|L_pert − L_cur|` 交給
    autograd，L_cur 的計算圖會被反向**兩次**；而該圖上有
    `torch.utils.checkpoint(use_reentrant=False)` 區塊（`sd.edit` 的
    `use_ckpt`／`vae_ckpt`），第二次反向要重算、行為未經查證。第三次前向
    也不行：那會把每步成本從論文的兩次 f_θ 變成三次。

    故此處把 g₁ 留成常數用：令 k = (λ/h)·sign(z)，

        g_FDM = −g₁ + k·(g₂ − g₁) = −(1+k)·g₁ + k·g₂                    (9)

    右式正是下面 `surrogate` 的梯度——`(g₁·x).sum()` 對 x 的梯度是常向量
    g₁，`k·L_pert` 對 x 的梯度是 k·g₂（Eq. 10 的 Hessian-free 取法：在
    δ' 這一點對 δ 的一階梯度）。`value + (surrogate − surrogate.detach())`
    的值等於 `value`、梯度等於 ∇surrogate，兩者同時成立且只前向兩次。

    z = 0 時 sign(z) = 0、k = 0，梯度退化為 −g₁（純上升），與 Algorithm 1
    第 24 行的 `sign(z)` 一致。∇L = 0 時 s = 0（Eq. 6 的但書），於是
    δ' = δ、z = 0，整項為 0。
    """
    if not x.requires_grad:
        raise ValueError(
            "flatgrad_objective 需要一個 requires_grad 的張量：Eq. (6) 的 s "
            "必須先對它反向一次才算得出來"
        )
    if h <= 0:
        raise ValueError(f"h 必須為正（有限差分的步長），收到 {h}")

    l_cur = base_loss(x)
    if l_cur.dim() != 0:
        raise ValueError(f"base_loss 必須回傳純量，收到 {tuple(l_cur.shape)}")

    # 這一次反向之後 l_cur 的圖即可釋放：g₁ 之後只當常數用。
    g1 = torch.autograd.grad(l_cur, x)[0].detach()
    l_cur_v = l_cur.detach()

    # Eq. (6)。逐樣本取範數（batch=1 時即整張影像的單一 L2 範數，與論文的
    # 純量式子相同）；`prepare` 已擋掉 batch>1。
    flat_dims = [1] * (g1.dim() - 1)
    n = g1.flatten(1).norm(dim=1).view(-1, *flat_dims)
    nonzero = (n > 0).to(g1.dtype)
    s = (g1 / n.clamp_min(torch.finfo(g1.dtype).tiny)) * nonzero

    l_pert = base_loss(x + h * s)
    if l_pert.dim() != 0:
        raise ValueError(f"base_loss 必須回傳純量，收到 {tuple(l_pert.shape)}")

    z = (l_pert - l_cur_v).detach()
    k = lam_over_h * torch.sign(z)

    value = -l_cur_v + lam_over_h * z.abs()          # Eq. (8)
    surrogate = -(1.0 + k) * (g1 * x).sum() + k * l_pert
    return value + (surrogate - surrogate.detach())


# ---------------------------------------------------------------------------
# 與 δ 無關的常量
# ---------------------------------------------------------------------------


class TDAEContext:
    """`y₀`、文字嵌入、編輯噪聲，全部在一次求解中固定。

    `y0` 在**該篇的值域**上（`[-1,1]`），與 `loss_fn` 比較的對象一致。
    """

    def __init__(self, spec, y0, emb, emb_uncond, noise, strength,
                 num_inference_steps, guidance_scale, h, lam_over_h,
                 seed, prompt, use_ckpt, vae_ckpt):
        self.spec = spec
        self.y0 = y0
        self.emb = emb
        self.emb_uncond = emb_uncond
        self.noise = noise
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.h = h
        self.lam_over_h = lam_over_h
        self.seed = seed
        self.prompt = prompt
        self.use_ckpt = use_ckpt
        self.vae_ckpt = vae_ckpt


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    strength: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    h: Optional[float] = None,
    lam_over_h: float = LAMBDA_OVER_H,
    guidance_scale: float = 7.5,
    prompt: str = TDAE_PROMPT,
    seed: int = 0,
    dpd=None,
    use_ckpt: bool = True,
    vae_ckpt: bool = True,
    **_,
) -> TDAEContext:
    """算出 y₀ = f_θ(x₀, e)，並固定編輯噪聲。

    `strength`、`num_inference_steps`、`h` 三者論文皆無，故**無預設、必填**
    （出處與檢索範圍見模組 docstring 與 `AUDIT_TDAE.md`）。

    `guidance_scale` 預設 7.5：§IV-A Threat Model 寫攻擊方用「standard
    inference settings (e.g. default schedulers, CFG scale, step counts from
    the original editing models)」，SD v1.x 的 pipeline 預設 CFG 就是 7.5。
    這是論文唯一對推論設定的敘述，但它講的是**攻擊方**，不是代理模型的
    設定，故仍記在 `extras` 裡當本專案的設定看。

    `use_ckpt`／`vae_ckpt` 預設為真：一步之內有兩次完整的編輯鏈前向，且第二
    次要留圖到反向。兩者只影響記憶體與重算，**數值中性**，故不進任何設定
    雜湊（與 `mist.py` 的同一條處理）。
    """
    if x01.dim() != 4 or x01.shape[0] != 1:
        raise NotImplementedError(
            f"TDAE 的式子逐張定義（Eq. 6 的 s 是單一影像的單位梯度方向、"
            f"Eq. 8 的 z 是單一純量），batch>1 時 z 變成整批的和，"
            f"不是論文那個量。收到 {tuple(x01.shape)}"
        )
    if h is None:
        raise NotImplementedError(
            "TDAE 缺 h（Eq. 7 的有限差分步長）。論文只在 §IV-E 固定比值 "
            "λ/h = 0.3，h 與 λ 各自的數值在全文（Algorithm 1 的 Input 列、"
            "§III-C、§IV-A、§IV-E、Fig. 5）都查不到，且無 Appendix、"
            "無官方程式碼。請由呼叫端明確指定並在報表標為本專案設定"
        )
    if strength is None:
        raise NotImplementedError(
            "TDAE 缺 strength。論文 §IV-A 只寫 SD14／SD3「configured for "
            "editing tasks」，全文沒有 img2img／SDEdit 強度、也沒有任何"
            "推論設定的數字（`SDEdit` 只出現在 §II-A 的引用）。"
            "請由呼叫端依本專案的威脅模型指定"
        )
    if num_inference_steps is None:
        raise NotImplementedError(
            "TDAE 缺代理編輯鏈的步數。論文未載（同 strength 的檢索範圍）。"
            "注意一步 PGD 會跑兩次完整編輯鏈（Algorithm 1 第 18、22 行），"
            "步數直接乘上去"
        )
    if dpd is not None:
        raise NotImplementedError(
            "DPD（文字側，§III-D）未實作，兩個理由都要看："
            "(1) 本專案的威脅模型下防禦方看不到攻擊指令，而 DPD 的 c 是資料集"
            "附帶的原始編輯指令（§IV-A Target Models and Dataset），沒有它就"
            "沒有「在它的鄰域裡搜 δ_p」這件事；"
            "(2) 論文沒給 DPD 的任何一個數——週期 S、內層步數 M、"
            "文字預算 ε_p、文字步長 η 四者只出現在 Algorithm 1 的 Input 列。"
            "只開 FDM 是論文自己的組態（Table VI 的 SA + FDM、Table I–III 的 "
            "PGE + FDM），本模組跑的就是那一列"
        )
    if lam_over_h <= 0:
        raise ValueError(f"λ/h 必須為正，收到 {lam_over_h}")

    vr = spec.value_range
    emb = sd.encode_text(prompt).detach()
    emb_uncond = sd.uncond_prompt().detach()

    z_like = torch.empty(
        sd.latent_shape(x01.shape[-2], x01.shape[-1]),
        device=x01.device,
        dtype=x01.dtype,
    )
    noise = sd.sample_edit_noise(z_like, seed=seed).detach()

    # y₀ = f_θ(x₀, e)：乾淨圖在同一條編輯鏈、同一個噪聲下的輸出。
    # Algorithm 1 的 Input 列把 y₀ 當給定常量，故不進計算圖。
    with torch.no_grad():
        y01 = sd.edit(
            x01.detach(), emb, noise, num_inference_steps,
            mask=None, strength=strength,
            guidance_scale=guidance_scale, emb_uncond=emb_uncond,
        )
    y0 = vr.from01(y01).detach()

    return TDAEContext(
        spec, y0, emb, emb_uncond, noise, strength, num_inference_steps,
        guidance_scale, float(h), float(lam_over_h), seed, prompt,
        bool(use_ckpt), bool(vae_ckpt),
    )


# ---------------------------------------------------------------------------
# 損失
# ---------------------------------------------------------------------------


def _edit_distance(sd, x_paper: torch.Tensor, ctx: TDAEContext) -> torch.Tensor:
    """L(x) = ‖f_θ(x, e) − y₀‖₂，兩邊都在該篇的值域上。

    §III-A 的字面定義（「a differentiable loss metric (e.g. ℓ₂-distance)」），
    不平方。`sd.edit` 回傳 `[0,1]`，故先換回 `[-1,1]` 再比——差一個常數倍會
    直接改變 loss 的量級，也會改變 (λ/h)·|z| 與 L 的相對權重。
    """
    vr = ctx.spec.value_range
    y01 = sd.edit(
        vr.to01(x_paper), ctx.emb, ctx.noise, ctx.num_inference_steps,
        mask=None, strength=ctx.strength,
        use_ckpt=ctx.use_ckpt, vae_ckpt=ctx.vae_ckpt,
        guidance_scale=ctx.guidance_scale, emb_uncond=ctx.emb_uncond,
    )
    return (vr.from01(y01) - ctx.y0).norm(p=2)


def loss_fn(sd, x_adv: torch.Tensor, ctx: TDAEContext) -> torch.Tensor:
    """Eq. (8) 的目標，梯度為 Eq. (9)。

    骨幹以 `objective="minimize"` 走 `x ← x − α·sign(∇J)`，與 Algorithm 1
    第 25 行的 `δ_v ← δ_v − α·sign(g_FDM)` 逐字相同，故 `run_pgd` 不必改。
    """
    return flatgrad_objective(
        lambda t: _edit_distance(sd, t, ctx),
        x_adv,
        lam_over_h=ctx.lam_over_h,
        h=ctx.h,
    )


SPEC_PAPER = BaselineSpec(
    name="tdae",
    source=(
        "arXiv:2512.14341v2（IEEE TPAMI accepted）；無官方程式碼、無 checkpoint，"
        "依論文 Algorithm 1 與 Eq. (5)–(10) 重建，逐條查證見 "
        "docs/reference/AUDIT_TDAE.md"
    ),
    value_range=TDAE_RANGE,
    # 預算取自本 repo 依 Salman et al. 論文 Table 9 重建的 `photoguard_linf`
    # 臂（TDAE 論文的 `PGD [32]` 即該篇）。TDAE 論文本身沒有 ε_v／α／N。
    eps=32.0 / 255.0,          # [-1,1] 值域；= [0,1] 的 16/255
    eps_pixel01=16.0 / 255.0,
    norm="linf",               # Algorithm 1 第 26 行 Π_{‖·‖_∞ ≤ ε_v}
    steps=200,
    step_size=4.0 / 255.0,     # [-1,1] 值域；= [0,1] 的 2/255
    step_schedule="constant",  # Algorithm 1 的 α 不隨 n 變
    update_rule="sign",        # Algorithm 1 第 25 行
    objective="minimize",      # 同上，g_FDM 已含 Eq. (8) 的負號
    init_rule="none",          # Algorithm 1 第 1 行 δ_v ← 0
    grad_reps=1,               # 一步只算 g₁、g₂ 各一次，無多樣本平均
    needs_target_image=False,  # y₀ 由乾淨圖的編輯結果算出，不需外部檔案
    needs_mask=False,
    modified_from_paper=True,
    modification_note=(
        "依論文重建（無官方程式碼、無 checkpoint、論文無 Appendix）。"
        "重建範圍：只做影像側的 FDM（§III-C、Eq. 5–10、Algorithm 1 第 17–26 行），"
        "受害模型只做 SD v1.x／IP2P 這一側。"
        "文字側（DPD，§III-D）不實作，兩個理由：(a) 本專案的威脅模型下防禦方看不到"
        "攻擊指令，而 DPD 的 c 是資料集附帶的原始編輯指令；(b) 論文未給 S／M／ε_p／η "
        "任一個數。故 c 固定為中性的空 prompt、δ_p ≡ 0——這是論文自己的 "
        "`+ FDM` 組態（Table VI 的 SA + FDM、Table I–III 的 PGE + FDM）。"
        "空 prompt 在 SD v1.x 上使 CFG 退化為無條件預測，y₀ 因而是一次無指令的 "
        "SDEdit 重建。"
        "L 取 §III-A 的字面定義 ‖f_θ(x,e) − y₀‖₂（不平方），y₀ = f_θ(x₀,e)。"
        "ε_v 16/255、α 2/255、N 200 步取自本 repo 的 `photoguard_linf`（TDAE 論文的 "
        "`PGD [32]` 即 Salman et al. 該篇，其 Table 9 的預算），"
        "因為 TDAE 論文對 ε_v／α／N 三者隻字未提，而 TDAE 是 plug-and-play、"
        "預算屬於被套的那一篇；如此也與既有 baseline 在同一個 eps01 上頭對頭。"
        "h（Eq. 7 的差分步長）、strength、代理編輯鏈步數三者論文皆無，改為呼叫端必填。"
        "整次求解固定同一個編輯噪聲，否則 Eq. (7) 的 z 會是兩個不同函式的差。"
        "SD3 未做：本 repo 無 SD3 封裝（缺口見 AUDIT_TDAE.md §6）。"
    ),
    discrepancy_note=(
        "論文 Eq. (4) 的內層 max 被論文自己放棄（§III-C），實跑的是 Eq. (5) 的"
        "當前點梯度範數懲罰，本模組跟著實跑的那一式。"
        "論文 Table V 報 PGD 免疫圖對原圖 PSNR 36.04 dB、PGD+TDAE 35.09 dB；"
        "若視 sign-PGD 為飽和則反推 ε_v* ≈ 4/255（反推的數，非量到的數，"
        "見 BASELINE_PROVENANCE 規則 4），與本 spec 採用的 16/255 差約四倍。"
        "論文未說明 SD14／SD3 以何種方式當編輯模型（全文無 img2img／strength／"
        "推論步數），故本 spec 的編輯協定不可寫成「論文的設定」。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "lambda_over_h": LAMBDA_OVER_H,     # §IV-E
        "lambda": "未找到（論文只固定比值 λ/h = 0.3）",
        "h": "未找到（呼叫端必填，[-1,1] 值域上的 ℓ₂ 半徑）",
        "text_side": "DPD 未實作；c = 空 prompt、δ_p ≡ 0（論文的 `+ FDM` 組態）",
        "dpd_S": "未找到",
        "dpd_M": "未找到",
        "dpd_eps_p": "未找到",
        "dpd_eta": "未找到",
        "prompt": TDAE_PROMPT,
        "base_loss": "‖f_θ(x,e) − y₀‖₂（§III-A 字面定義，不平方）",
        "y0": "f_θ(x₀, e)，乾淨圖在同一條編輯鏈與同一個噪聲下的輸出",
        "guidance_scale_default": 7.5,
        "edit_noise": "整次求解固定一個，seed 由呼叫端給（論文全文無 seed）",
        "budget_source": "photoguard_linf（Salman et al. Table 9），非 TDAE 論文",
        "paper_victims": "InstructPix2Pix、SD1.4、SD3；本模組只做 SD v1.x／IP2P 側",
        "paper_dataset": "InstructPix2Pix-clip-filtered 選 100 張（35 人像／35 風景／30 畫作）",
        "paper_metrics": "PSNR↓、SSIM↓、LPIPS↑、VIFP↓、FSIM↓（對 benign 編輯結果）",
    },
)
