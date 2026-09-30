r"""DANP —— Zhang, Dong, Shan, Chen，*Dual Attention Guided Defense Against
Malicious Edits*（arXiv:2512.14333v1，2025-12-16）。

**本檔為依論文重建，無官方程式。** 作者未釋出任何程式碼、設定檔或權重，
論文正文亦未附虛擬碼以外的實作細節。因此本檔的每一行都只能追溯到論文的
某一節或某一式；凡論文未寫、無法由論文推得的項目，一律在下方「論文未給的
項目」表中列出，並在 `BaselineSpec.discrepancy_note` 與
`docs/reference/AUDIT_DANP.md` 重複一次。**不得把這些項目讀成論文設定。**

論文的方法（§IV）
──────────────────────────────────────────────────────────────────────

DANP = DAA（Dual Attention-guided Attack）＋ NBA（Noise-Based Attack），
兩者合成一個要**最小化**的目標，以 sign-PGD 在 `L∞` 球內求解。

1. 聚合注意力（Eq. 3、Eq. 4）

       A_l = softmax(Q_l K_lᵀ / √d_k)
       Att(x, φ(c)) = (1/L) Σ_l Upsample(A_l)

   `A_l` 是 UNet 第 `l` 個中間 block 的 **cross-attention 機率**（softmax
   之後、乘上 V 之前）。

2. 動態門檻（Eq. 7–10）

   先把聚合注意力 min-max 正規化到 `[0,1]`（記為 `N(·)`），再對其直方圖
   （`L` 個強度級距）套 Kapur 最大熵法：

       H₀(τ) = −Σ_{i≤τ}   (p_i/P₀(τ))·log(p_i/P₀(τ))
       H₁(τ) = −Σ_{i>τ}   (p_i/P₁(τ))·log(p_i/P₁(τ))
       τ*    = argmax_{0 ≤ τ < L} [H₀(τ) + H₁(τ)]
       M_t   = 𝟙( N(Att(x_t^imu, φ(c))) > τ_t )

   `M_t = 1` 是「與文字相關」的區域，`M_t = 0` 是「不相關」的區域。
   門檻**每個 timestep 重算一次**（§IV-B 末段）。

3. DAA 損失（Eq. 11）——雙向操作，這是本篇與 SA 的差別

       L_DAA = ‖Att ⊙ M_t‖_F²  −  λ_daa · ‖Att ⊙ (1 − M_t)‖_F²

   最小化它 = **壓低**相關區的注意力（第一項為正）、**抬高**不相關區的
   注意力（第二項為負）。

4. NBA 損失（Eq. 12）

       L_NBA = − ‖ε_θ(x_t, t, c) − ε_θ(x_t^imu, t, c)‖₂²

   最小化它 = 拉大乾淨圖與免疫圖兩條噪聲預測的距離。

5. 總目標（Eq. 13）與 Algorithm 1

       L_total = L_DAA + λ_nba · L_NBA

       δ ← 0
       for n = 1..N:
           g ← 0
           x_imu ← x₀ + δ
           for t in 𝒯:                       # |𝒯| = 10
               ε ~ N(0, I)                    # 兩條分支共用同一個 ε
               x_t     ← √ᾱ_t·x₀    + √(1−ᾱ_t)·ε
               x_t^imu ← √ᾱ_t·x_imu + √(1−ᾱ_t)·ε
               g ← g + ∇_{x_imu} [ L_DAA + λ_nba·L_NBA ]
           g ← g / |𝒯|
           δ ← δ − α · sign(g)
           δ ← clip(δ, −γ, γ)

   注意 Algorithm 1 第 13 行是**減號**（梯度下降），與 §III-C Eq. 6 的加號
   （對上升式目標而言）方向一致，因為 Eq. 11／Eq. 12 已把符號寫進損失。
   本檔的 `objective="minimize"` 即此。

論文給的超參數與其出處
──────────────────────────────────────────────────────────────────────

| 項目 | 值 | 出處 |
|---|---|---|
| 約束 | `L∞`，γ = 0.03 | §V-A「General Settings and Baselines」 |
| 步數 N | 100 | 同上 |
| timestep 數 \|𝒯\| | 10（uniformly sample） | 同上 |
| λ_daa | 1.0 | §V-A「Settings of DANP」 |
| λ_nba | 1.0 | 同上 |
| Kapur 級距數 L | 128 | 同上；§V-F Table VI 掃過 32/64/128/256/512 |
| 受害模型 | SD v1-4、HQ-Edit、InstructPix2pix | §V-A「Target Models and Dataset」 |
| 資料 | InstructPix2Pix-clip-filtered 抽 200 張 | 同上 |
| prompt | 該影像在資料集中**原本的編輯指令** | 同上 |

論文未給的項目（本檔的決定，全部標 `modified_from_paper`）
──────────────────────────────────────────────────────────────────────

以下每一項都查過：論文全文（HTML v1）、Algorithm 1、圖 1–3 的說明文字、
表 I–VI 的欄位與註腳。作者無 GitHub、無附錄、arXiv 無 ancillary files。

| 項目 | 論文 | 本檔 | 理由 |
|---|---|---|---|
| **步長 α** | **未找到**（只在 Algorithm 1 的 Input 列出符號） | `1/255` | 見 `RECONSTRUCTED_STEP_SIZE` |
| **值域** | **未找到**（只寫 γ=0.03） | `[0,1]` | 由 Table IV 的 PSNR 反推，見 `DANP_RANGE` |
| **注意力取自哪些層** | Eq. 4 寫「L 個 U-Net block」，未指定子集 | 全部 `attn2`（cross-attention），不含 `attn1` | Eq. 3 的 K 來自文字嵌入，故只可能是 cross-attention；論文未再縮小範圍 |
| **head 怎麼處理** | **未找到**（Eq. 3 的 `A_l` 沒有 head 軸） | 對 head 取平均 | Eq. 3 把 `A_l` 寫成單一矩陣，平均是唯一不引入新權重的化約 |
| **Upsample 的目標解析度與插值** | **未找到** | 雙線性上採樣到**觀察到的最大** cross-attention 網格 | 見 `DANPAttnController.aggregate` |
| **token 軸怎麼處理** | **未找到**（Eq. 10 的 𝟙 與 Eq. 11 的 ⊙ 都是逐元素，故 `M` 與 `Att` 同形） | 逐元素，涵蓋全部 77 個 token（含 BOS/EOT/PAD） | 照 Eq. 10／Eq. 11 的維度字面讀 |
| **min-max 的範圍** | 「normalize the aggregated attention map to a range of [0,1]」 | 對整張聚合圖取全域 min/max | 字面讀；未寫逐 token 或逐層 |
| **latent 還是像素** | 通篇寫 `x_t = √ᾱ x₀ + …`，未提 VAE | 在 **latent** 上加噪，`z₀ = E(x)` 取後驗均值 | 三個受害模型都是 latent diffusion，Eq. 3 的 `Q_l` 來自 UNet 的影像特徵 |
| **𝒯 是固定格點還是每步重抽** | 「uniformly sample a set of \|𝒯\|=10 timesteps」＋ Algorithm 1 把 𝒯 當 Input | 固定等距格點 `linspace(0, T−1, 10)` | Algorithm 1 的 𝒯 是輸入，不在迴圈內重抽 |
| **CFG** | **未找到** | 不用 CFG，單次條件前向 | Eq. 12 只寫 `ε_θ(·, t, c)` |
| **NBA 的尺度** | §V-F：「we scale the NBA loss to a comparable level with the DAA loss」，**未給係數** | `nba_scale = 1.0`（不縮放），照 Eq. 13 字面 | 該句只描述 §V-F 的消融掃描，不是主設定；填一個係數等於捏造 |

值域的反推（`DANP_RANGE` 的依據）
──────────────────────────────────────────────────────────────────────

論文只寫 γ = 0.03，未說是在 `[0,1]` 還是 `[-1,1]` 上量的，而兩者差一倍。
Table IV（§V-D）給了 DANP 在 InstructPix2pix 上免疫圖對原圖的 **PSNR =
34.76 dB**，反推的 rms 是 `10^(−34.76/20) = 0.018281`（`[0,1]` 尺度）。

- 若 γ 在 `[0,1]` 上量：上界 0.03，`0.018281 < 0.03`，相容。
- 若 γ 在 `[-1,1]` 上量：`[0,1]` 的等價上界是 0.015，而 sign-PGD 的 rms
  **不可能超過 `L∞` 上界**，`0.018281 > 0.015` 矛盾。

故取 `[0,1]`。**這個 0.018281 是反推的數，不是量到的數**
（`docs/reference/BASELINE_PROVENANCE.md` 規則 4）：它由一批影像的**算術
平均 PSNR** 反推，得到的是 rms 的**幾何平均**，引用時必須標星號。

實作上與論文對不上的地方
──────────────────────────────────────────────────────────────────────

1. **乾淨分支不回傳梯度。** Eq. 12 對 `ε_θ(x_t,·)` 與 `ε_θ(x_t^imu,·)`
   是對稱的，但 Algorithm 1 第 7 行的 `x_t` 只依賴 `x₀`，與 δ 無關，
   故 `∇_{x_imu}` 對該分支恆為零。此處以 `no_grad` 計算，**不是簡化**，
   是同一個梯度。
2. **遮罩不回傳梯度。** `M_t` 是指示函數，幾乎處處梯度為零。論文未討論，
   此處明確 `detach()`。
3. **梯度累積以 `grad_reps` 實作。** Algorithm 1 在一次 PGD 迭代內對 |𝒯|
   個 timestep 各求一次梯度再平均；本專案骨幹的 `grad_reps` 做的正是
   「求 `grad_reps` 次梯度取平均」，故 `grad_reps = |𝒯| = 10`，
   `loss_fn` 每次呼叫取 𝒯 的下一個元素並重抽 ε。與 Algorithm 1 逐項對應。
4. **9 通道（inpainting）權重下後 5 個通道取全 1 遮罩**，與 `mist.loss_fn`、
   `dia.loss_fn` 同一處置：論文的 `ε_θ` 沒有影像條件。

要接進 `scripts/baseline_run.py` 的 `CONDITIONS` 需要加的幾行
──────────────────────────────────────────────────────────────────────

**本檔不修改 `scripts/baseline_run.py`，也不註冊進 `immunization_baseline/attacks/__init__.py`
的 `REGISTRY`。** 後者是刻意的：`tests/test_baselines.py::
test_五篇的值域全部是負一到一` 會逐一檢查 `REGISTRY` 的 `value_range`，
而 DANP 的值域是 `[0,1]`（見上），註冊進去會讓那個測試失敗——但那個測試
釘的是「那五篇的值域查證過是 `[-1,1]`」，不該為了本篇放寬。要納入時應
先擴充該測試的涵蓋方式。

接線要加的是：

    # immunization_baseline/attacks/__init__.py
    from immunization_baseline.attacks import advpaint, danp, dia, mist, photoguard, promptflare
    _SPECS = (..., danp.SPEC_PAPER)

    # scripts/baseline_run.py 第 36 行
    from immunization_baseline.attacks import danp, dia, mist, photoguard  # noqa: E402

    # scripts/baseline_run.py 第 60 行
    CONDITIONS = ["photoguard_c", "photoguard_linf", "mist", "dia_r", "danp"]

    # scripts/baseline_run.py `run_additive` 的 spec 表（第 90-93 行）
    spec = {"photoguard_c": photoguard.SPEC,
            "photoguard_linf": photoguard.SPEC_PAPER_LINF,
            "mist": mist.SPEC,
            "dia_r": dia.SPEC_R,
            "danp": danp.SPEC_PAPER}[name]

    # scripts/baseline_run.py `run_additive` 的 kw 分支（第 94-105 行之後）
    elif name == "danp":
        # DANP 的 DAA 以編輯指令為條件，沒有預設 prompt；prepare 會拒絕 None。
        # use_ckpt 讓每次 UNet 前向各成一個 checkpoint 區塊（一次 loss_fn 有
        # 兩次前向，其中一次在 no_grad 內）。
        kw = {"prompt": item["prompt"], "use_ckpt": True}
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from immunization_baseline.attacks.pgd import BaselineSpec, ValueRange

# ---------------------------------------------------------------------------
# 論文給的常數
# ---------------------------------------------------------------------------

DANP_RANGE = ValueRange(
    0.0,
    1.0,
    "論文 §V-A 只寫 γ=0.03、未述值域；由 §V-D Table IV 的 PSNR 34.76 dB "
    "反推 rms 0.018281（[0,1] 尺度）——大於 [-1,1] 讀法的上界 0.015，"
    "故值域只能是 [0,1]。反推值標星號，見模組 docstring「值域的反推」",
)

PAPER_GAMMA = 0.03              # §V-A，L∞ 預算
PAPER_ITERS = 100               # §V-A，N
PAPER_NUM_TIMESTEPS = 10        # §V-A，|𝒯|
PAPER_LAMBDA_DAA = 1.0          # §V-A「Settings of DANP」，Eq. 11
PAPER_LAMBDA_NBA = 1.0          # §V-A「Settings of DANP」，Eq. 13
PAPER_KAPUR_BINS = 128          # §V-A「Settings of DANP」，§V-F Table VI 選定

# §V-F：「we scale the NBA loss to a comparable level with the DAA loss」，
# 但**沒有給係數**，且該句描述的是消融掃描而非主設定。照 Eq. 13 字面取 1.0。
RECONSTRUCTED_NBA_SCALE = 1.0

# 論文全文未給步長 α（只在 Algorithm 1 的 Input 列出符號）。查過：§III-C
# Eq. 6 的說明、§IV-D、§V-A 三段設定、Algorithm 1 全文、表 I–VI 的註腳。
#
# 取 1/255：本 repo 已查證的五篇 L∞ baseline（Mist、DIA、PromptFlare 的
# [-1,1] 尺度、DiffusionGuard）步長全部是 1/255，是這一類方法的慣用值。
# 在 γ=0.03、N=100 下，`100 × 1/255 = 0.39 ≫ 0.03`，預算可達且有餘裕。
#
# **這是本檔挑的，不是論文的。** 換成 γ/N = 3e-4（剛好走滿一次）或
# 2γ/N = 6e-4（常見的 PGD 慣例）都同樣說得通，而三者會給出不同的解。
RECONSTRUCTED_STEP_SIZE = 1.0 / 255.0


# ---------------------------------------------------------------------------
# 動態門檻：min-max 正規化 ＋ Kapur 最大熵（Eq. 7–10）
# ---------------------------------------------------------------------------


def normalize_attention(att: torch.Tensor) -> torch.Tensor:
    """`N(·)`（§IV-B）：對整張聚合注意力圖取全域 min-max，映到 `[0,1]`。

    退化輸入（`max == min`，即整張圖是常數）回傳**全 0**。這不是繞過錯誤，
    是把 min-max 在常數輸入上的極限寫死成一個定值：常數圖不含任何相對
    重要性資訊，而 `(a−a)/(a−a)` 沒有值。全 0 之後 Kapur 的直方圖全部落在
    第 0 格，`kapur_threshold_index` 依其自身規則回傳 `bins−1`，遮罩全 0
    ——即「沒有任何區域被判為與文字相關」，DAA 退化成 `−λ_daa‖Att‖_F²`
    （整張抬高）。此行為由 `tests/test_danp.py` 釘住。
    """
    lo = att.amin()
    hi = att.amax()
    span = hi - lo
    if float(span.detach()) == 0.0:
        return torch.zeros_like(att)
    return (att - lo) / span


def kapur_threshold_index(values01: torch.Tensor, bins: int) -> int:
    """Kapur 最大熵門檻（Eq. 7–9），回傳**級距索引** `τ*`。

    `values01` 必須已在 `[0,1]`。直方圖分 `bins` 格，
    class 0 = 級距 `[0..τ]`、class 1 = 級距 `[τ+1..bins−1]`，
    `τ*` 取 `H₀+H₁` 最大者。

    只有 `P₀(τ) > 0` 且 `P₁(τ) > 0` 的 τ 是合法的——另一側為空時該側的
    條件熵沒有定義（Eq. 7／Eq. 8 的分母為 0）。論文寫 `0 ≤ τ < L`，未處理
    這件事。**全部 τ 都不合法**（所有質量集中在同一格）時回傳 `bins−1`，
    即遮罩全 0；這是唯一與「class 1 為空」相容的答案。
    """
    if bins < 2:
        raise ValueError(f"Kapur 的級距數必須 ≥ 2，收到 {bins}")
    v = values01.detach().reshape(-1).float()
    hist = torch.histc(v, bins=bins, min=0.0, max=1.0)
    total = hist.sum()
    if float(total) == 0.0:
        raise ValueError("直方圖總和為 0：輸入為空張量，Kapur 無從計算")
    p = hist / total

    # p·log p，0·log0 取 0
    plogp = torch.where(p > 0, p * p.clamp_min(1e-300).log(), torch.zeros_like(p))
    c_p = p.cumsum(0)                      # P₀(τ)
    c_plogp = plogp.cumsum(0)              # Σ_{i≤τ} p_i log p_i
    tot_plogp = c_plogp[-1]

    p0 = c_p
    p1 = 1.0 - c_p
    s0 = c_plogp
    s1 = tot_plogp - c_plogp

    # H₀(τ) = log P₀ − (1/P₀)·Σ_{i≤τ} p_i log p_i，H₁ 同理。
    valid = (p0 > 0) & (p1 > 0)
    valid[-1] = False                      # τ = bins−1 時 class 1 為空
    if not bool(valid.any()):
        return bins - 1
    safe0 = p0.clamp_min(1e-300)
    safe1 = p1.clamp_min(1e-300)
    h0 = safe0.log() - s0 / safe0
    h1 = safe1.log() - s1 / safe1
    obj = torch.where(valid, h0 + h1, torch.full_like(h0, -float("inf")))
    return int(obj.argmax())


def kapur_mask(att: torch.Tensor, bins: int) -> Tuple[torch.Tensor, int, float]:
    """Eq. 10 的 `M_t`。回傳 `(mask, τ*, τ_t)`。

    `mask` 與 `att` 同形、值為 0/1、**已 detach**（指示函數幾乎處處梯度為
    零，見模組 docstring「對不上的地方」第 2 點）。

    比較以**級距索引**進行（`bin(v) > τ*`），而不是 Eq. 10 字面的
    `N(Att) > τ_t`。兩者只在恰好落在級距邊界 `τ_t = (τ*+1)/bins` 的元素上
    不同（字面讀法把邊界上的元素歸為 class 0，直方圖把它歸為 class 1），
    此處取與直方圖分割一致的那一種。回傳的 `τ_t` 即該邊界值，供報表引用。
    """
    norm = normalize_attention(att)
    tau_idx = kapur_threshold_index(norm, bins)
    idx = (norm.detach().float() * bins).floor().long().clamp_(max=bins - 1)
    mask = (idx > tau_idx).to(att.dtype)
    return mask, tau_idx, (tau_idx + 1) / bins


# ---------------------------------------------------------------------------
# 注意力擷取：必須拿到 softmax 之後的機率，不能用 SDPA
# ---------------------------------------------------------------------------


class DANPAttnController:
    """收集各 cross-attention 層的**後 softmax 機率** `A_l`，再依 Eq. 4 聚合。

    與 `immunization_baseline/attacks/promptflare.py::AttnController` 的差別：PromptFlare 記的是
    `attn2` 模組**經 `to_out` 之後的輸出**（`A·V·W_out`），走 SDPA 融合核，
    `A` 從未被實體化。DANP 的 Eq. 11 作用在 `A` 本身，故不能沿用那條路，
    必須換成明確算 softmax 的 processor（`DANPAttnProcessor`）。
    兩者不共用擷取層。
    """

    def __init__(self):
        self.maps: List[torch.Tensor] = []

    def __call__(self, probs: torch.Tensor, heads: int, module_name: str) -> None:
        # probs: (B·heads, HW, tokens)
        bh, hw, tok = probs.shape
        if bh % heads != 0:
            raise RuntimeError(
                f"{module_name}：注意力機率的第 0 維 {bh} 不是 head 數 {heads} 的倍數"
            )
        b = bh // heads
        # head 平均：論文 Eq. 3 的 A_l 沒有 head 軸，見模組 docstring。
        self.maps.append(probs.view(b, heads, hw, tok).mean(1))

    def reset(self) -> None:
        self.maps = []

    def aggregate(self) -> torch.Tensor:
        """Eq. 4：`Att = (1/L) Σ_l Upsample(A_l)`，回傳 `(B, G·G, tokens)`。

        `Upsample` 的目標解析度論文未給。此處取**本次前向觀察到的最大**
        cross-attention 網格（SD v1.x／512² 下是 64×64，即 latent 解析度），
        理由是那是不丟失任何一層空間資訊的最小共同網格；取更大的網格只是
        在所有層上插值，取更小的會把最細的那一層降採樣掉。插值用雙線性
        （`align_corners=False`）——論文只寫「Upsample」，未指定方式。

        只支援正方形網格：`HW` 不是完全平方數時直接中止。長寬比不為 1 時
        無法由 token 數還原網格形狀，而猜錯會靜默算出另一張圖。
        """
        if not self.maps:
            raise RuntimeError(
                "沒有收集到任何 cross-attention 機率。processor 未安裝，"
                "或這次前向沒有走到任何 attn2 層"
            )
        sides = []
        for m in self.maps:
            hw = m.shape[1]
            s = int(round(math.sqrt(hw)))
            if s * s != hw:
                raise NotImplementedError(
                    f"cross-attention 的 token 數 {hw} 不是完全平方數；"
                    "非正方形輸入無法由 token 數還原空間網格，"
                    "Eq. 4 的 Upsample 目標形狀必須另行決定"
                )
            sides.append(s)
        grid = max(sides)
        out = None
        for m, s in zip(self.maps, sides):
            b, hw, tok = m.shape
            if s == grid:
                up = m
            else:
                x = m.transpose(1, 2).reshape(b, tok, s, s)
                x = F.interpolate(
                    x, size=(grid, grid), mode="bilinear", align_corners=False
                )
                up = x.reshape(b, tok, grid * grid).transpose(1, 2)
            out = up if out is None else out + up
        return out / len(self.maps)


class DANPAttnProcessor:
    """明確算 softmax 的 cross-attention processor，供 `A_l` 被實體化。

    這段是 diffusers 內建 `AttnProcessor`（非 2.0 版）的逐項展開，唯一的
    增加是把 `attn.get_attention_scores` 的回傳值交給 controller。
    **不能改用 `AttnProcessor2_0`／SDPA**：融合核不回傳注意力機率，而
    Eq. 11 的被優化量正是那個機率。

    `attn1`（self-attention）也會被裝上，但 controller 只在 `attn2` 上被呼叫
    ——Eq. 3 的 `K_l` 來自文字嵌入，self-attention 不在定義內。
    """

    def __init__(self, controller: DANPAttnController, module_name: str):
        self.controller = controller
        self.module_name = module_name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if args or kwargs:
            # 新版 diffusers 若多傳引數，靜默忽略會讓注意力算出別的東西而
            # 完全沒有症狀。與 promptflare 的同一處置。
            raise RuntimeError(
                f"{self.module_name}：processor 收到未預期的引數 "
                f"args={args} kwargs={list(kwargs)}；必須先查證其語意"
            )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 這一行就是 Eq. 3 的 softmax(QKᵀ/√d_k)。
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross and self.module_name.endswith("attn2"):
            self.controller(attention_probs, attn.heads, self.module_name)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


# ---------------------------------------------------------------------------
# 損失（Eq. 11、Eq. 12、Eq. 13）
# ---------------------------------------------------------------------------


def daa_loss(
    att: torch.Tensor, mask: torch.Tensor, lambda_daa: float
) -> torch.Tensor:
    """Eq. 11。**最小化**它會壓低 `mask=1` 處、抬高 `mask=0` 處的注意力。

        ∂L/∂Att = 2·Att⊙M − 2·λ_daa·Att⊙(1−M)

    第一項為正（沿負梯度走即減小相關區的值），第二項為負（沿負梯度走即
    增大不相關區的值）。兩個方向的符號由 `tests/test_danp.py` 釘住。
    """
    relevant = (att * mask).pow(2).sum()
    irrelevant = (att * (1.0 - mask)).pow(2).sum()
    return relevant - lambda_daa * irrelevant


def nba_loss(eps_clean: torch.Tensor, eps_imu: torch.Tensor) -> torch.Tensor:
    """Eq. 12：`−‖ε_θ(x_t) − ε_θ(x_t^imu)‖₂²`。最小化即拉大兩者距離。"""
    return -(eps_clean - eps_imu).pow(2).sum()


# ---------------------------------------------------------------------------
# prepare / loss_fn
# ---------------------------------------------------------------------------


def paper_timesteps(sd, count: int = PAPER_NUM_TIMESTEPS) -> torch.Tensor:
    """§V-A「we uniformly sample a set of |𝒯| = 10 timesteps」。

    取 `[0, T−1]` 上的等距格點。論文把 𝒯 當成 Algorithm 1 的 Input，
    迴圈內不重抽，故此處是固定格點而非每步隨機抽樣（見模組 docstring）。
    """
    if count < 1:
        raise ValueError(f"|𝒯| 必須 ≥ 1，收到 {count}")
    return torch.linspace(
        0, sd.num_train_timesteps - 1, count
    ).round().long()


class DANPContext:
    """安裝／還原 attention processor 的責任在這裡。`run_pgd` 結束時呼叫 `close()`。"""

    def __init__(
        self,
        sd,
        spec: BaselineSpec,
        x01: torch.Tensor,
        emb,
        timesteps: torch.Tensor,
        generator: torch.Generator,
        *,
        lambda_daa: float,
        lambda_nba: float,
        nba_scale: float,
        kapur_bins: int,
        use_ckpt: bool = False,
    ):
        self.spec = spec
        self.x01 = x01.detach()
        self.emb = emb
        self.timesteps = timesteps
        self.generator = generator
        self.lambda_daa = lambda_daa
        self.lambda_nba = lambda_nba
        self.nba_scale = nba_scale
        self.kapur_bins = kapur_bins
        self.use_ckpt = bool(use_ckpt)
        self.controller = DANPAttnController()
        self.abar = sd.alphas_cumprod(x01.device)
        # 乾淨分支的 latent 與 δ 無關（Algorithm 1 第 7 行），先算一次。
        with torch.no_grad():
            self.z0_clean = sd.encode_image(self.x01).detach()
        # Algorithm 1 在一次迭代內走訪 𝒯 的每個元素；骨幹以 grad_reps 次
        # 呼叫 loss_fn 實現，此游標決定這一次取哪個 t。
        self._cursor = 0
        # 每次記下的診斷值，供報表引用（不參與最佳化）。
        self.last_tau_index: Optional[int] = None
        self.last_tau_value: Optional[float] = None
        self.last_mask_fraction: Optional[float] = None

        self._saved = []
        for name, module in sd.unet.named_modules():
            if name.endswith("attn2"):
                self._saved.append((module, module.get_processor()))
                module.set_processor(DANPAttnProcessor(self.controller, name))
        if not self._saved:
            raise RuntimeError("UNet 中找不到任何 attn2 層")

    def next_timestep(self) -> torch.Tensor:
        t = self.timesteps[self._cursor % len(self.timesteps)]
        self._cursor += 1
        return t

    def close(self) -> None:
        """還原原本的 processor。不還原會污染共用同一個 `SDWrapper` 的後續實驗。"""
        for module, proc in self._saved:
            module.set_processor(proc)
        self._saved = []


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    prompt: Optional[str] = None,
    seed: int = 0,
    num_timesteps: int = PAPER_NUM_TIMESTEPS,
    use_ckpt: bool = False,
    **_,
) -> DANPContext:
    """`prompt` 是必填：DAA 的整個機制以編輯指令為條件。

    §V-A 用的是「該影像在 InstructPix2Pix-clip-filtered 中原本的編輯指令」，
    那是逐張不同的東西，本檔沒有、也不該有預設值。
    """
    if prompt is None:
        raise NotImplementedError(
            "DANP 需要編輯指令：Eq. 3 的 K_l 來自 φ(c)，Eq. 10/11 的遮罩與"
            "損失全部以該 prompt 為條件。論文 §V-A 用資料集附帶的逐張指令，"
            "沒有可沿用的預設值，請由呼叫端指定"
        )
    if spec.grad_reps != num_timesteps:
        raise ValueError(
            f"grad_reps={spec.grad_reps} 與 |𝒯|={num_timesteps} 不一致。"
            "Algorithm 1 在一次迭代內對 𝒯 的每個元素各求一次梯度再平均，"
            "本專案以 grad_reps 實現，兩者必須相等，否則有的 t 會被走訪兩次、"
            "有的一次也沒有，而這件事在結果上沒有症狀"
        )
    emb = sd.encode_text(prompt).detach()
    ts = paper_timesteps(sd, num_timesteps).to(x01.device)
    generator = torch.Generator(device=x01.device).manual_seed(seed)
    return DANPContext(
        sd,
        spec,
        x01,
        emb,
        ts,
        generator,
        lambda_daa=float(spec.extras["lambda_daa"]),
        lambda_nba=float(spec.extras["lambda_nba"]),
        nba_scale=float(spec.extras["nba_scale"]),
        kapur_bins=int(spec.extras["kapur_bins"]),
        use_ckpt=use_ckpt,
    )


def _unet(sd, ctx: DANPContext, z, t):
    if ctx.use_ckpt:
        return ckpt.checkpoint(
            lambda a: sd.unet_forward(a, t, ctx.emb), z, use_reentrant=False
        )
    return sd.unet_forward(z, t, ctx.emb)


def loss_fn(sd, x_adv: torch.Tensor, ctx: DANPContext) -> torch.Tensor:
    """Algorithm 1 第 6–9 行的一個 `t`，回傳 `L_DAA + λ_nba · L_NBA`。

    `x_adv` 在 `[0,1]`（`DANP_RANGE`），直接餵 `sd.encode_image`。
    乾淨分支與免疫分支**共用同一個 ε**（Algorithm 1 第 6 行只抽一次）。
    """
    t = ctx.next_timestep()
    z0_imu = sd.encode_image(x_adv)
    abar = ctx.abar[int(t)].to(z0_imu.dtype)
    noise = torch.randn(
        z0_imu.shape,
        generator=ctx.generator,
        device=z0_imu.device,
        dtype=z0_imu.dtype,
    )
    root, coroot = abar.sqrt(), (1.0 - abar).sqrt()
    z_imu = root * z0_imu + coroot * noise
    z_clean = root * ctx.z0_clean.to(z0_imu.dtype) + coroot * noise

    # 免疫分支：同一次前向拿到注意力機率與 ε 預測。
    ctx.controller.reset()
    ones = torch.ones_like(x_adv[:, :1])
    with sd.conditioning_for(x_adv, mask=ones):
        eps_imu = _unet(sd, ctx, z_imu, t)
        att = ctx.controller.aggregate()
    ctx.controller.reset()

    # 乾淨分支：x_t 只依賴 x₀，∇_{x_imu} 對它恆為零，故不建圖。
    with torch.no_grad(), sd.conditioning_for(ctx.x01, mask=ones):
        eps_clean = _unet(sd, ctx, z_clean, t).detach()
    ctx.controller.reset()

    mask, tau_idx, tau_val = kapur_mask(att, ctx.kapur_bins)
    ctx.last_tau_index = tau_idx
    ctx.last_tau_value = tau_val
    ctx.last_mask_fraction = float(mask.mean())

    l_daa = daa_loss(att, mask, ctx.lambda_daa)
    l_nba = nba_loss(eps_clean, eps_imu)
    return l_daa + ctx.lambda_nba * ctx.nba_scale * l_nba


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


SPEC_PAPER = BaselineSpec(
    name="danp",
    source="arXiv:2512.14333v1（2025-12-16）；無官方程式碼",
    value_range=DANP_RANGE,
    eps=PAPER_GAMMA,                      # §V-A，γ = 0.03
    eps_pixel01=PAPER_GAMMA,              # 值域即 [0,1]，見 DANP_RANGE
    norm="linf",                          # §IV-D「‖x_imu − x₀‖∞ ≤ γ」
    steps=PAPER_ITERS,                    # §V-A，N = 100
    step_size=RECONSTRUCTED_STEP_SIZE,    # **論文未給**，見該常數的註解
    step_schedule="constant",             # Algorithm 1 的 α 不隨 n 改變
    update_rule="sign",                   # Algorithm 1 第 13 行 `sign(g_total)`
    objective="minimize",                 # Algorithm 1 第 13 行是減號
    init_rule="none",                     # Algorithm 1 第 1 行 `δ ← 0`
    grad_reps=PAPER_NUM_TIMESTEPS,        # Algorithm 1 第 5–12 行：|𝒯| 次梯度取平均
    needs_target_image=False,
    needs_mask=False,                     # 遮罩由 Eq. 10 自己算，不由外部提供
    modified_from_paper=True,
    modification_note=(
        "全檔為依論文重建（arXiv:2512.14333v1 無官方程式碼、無附錄、arXiv 無 "
        "ancillary files），下列項目論文未給、由本檔決定："
        "(1) 步長 α = 1/255（論文只在 Algorithm 1 的 Input 列出符號；"
        "取本 repo 五篇 L∞ baseline 的慣用值，γ/N 或 2γ/N 同樣說得通）；"
        "(2) 值域 [0,1]（論文只寫 γ=0.03，由 Table IV 的 PSNR 34.76 dB 反推排除 [-1,1]）；"
        "(3) Eq. 3 的 A_l 沒有 head 軸，此處對 head 取平均；"
        "(4) Eq. 4 的 Upsample 目標解析度取觀察到的最大 cross-attention 網格、雙線性插值；"
        "(5) Eq. 10/11 的 token 軸依維度字面讀為逐元素，涵蓋全部 77 個 token；"
        "(6) 加噪在 latent 上進行、z₀ = E(x) 取後驗均值（論文通篇未提 VAE）；"
        "(7) 𝒯 取 [0,T−1] 的等距固定格點；(8) 不用 CFG；"
        "(9) NBA 不額外縮放（§V-F 提到縮放但未給係數）。"
        "另：乾淨分支不建圖、遮罩 detach，兩者是同一個梯度而非簡化。"
    ),
    discrepancy_note=(
        "論文與程式不一致無從比對——作者未釋出程式碼，本檔的對照對象只有論文正文。"
        "論文內部的落差：§III-C Eq. 6 寫上升式 δ+α·g，Algorithm 1 第 13 行寫下降式 δ−α·g，"
        "兩者因 Eq. 11/12 已把符號寫進損失而一致，本檔取 Algorithm 1；"
        "§V-F 說 NBA 與 DAA 的量級差很多、消融時把 NBA 縮放到可比的量級，但未給係數，"
        "故 λ_nba=1.0 的主設定實際上作用在未知的尺度上；"
        "Table IV 的 PSNR 34.76 dB 反推 rms 0.018281*（星號：由算術平均 PSNR 反推得到的是 "
        "rms 的幾何平均），低於 γ=0.03 的飽和上界，論文未說明 sign-PGD 為何未走滿預算。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "lambda_daa": PAPER_LAMBDA_DAA,
        "lambda_nba": PAPER_LAMBDA_NBA,
        "nba_scale": RECONSTRUCTED_NBA_SCALE,
        "kapur_bins": PAPER_KAPUR_BINS,
        "num_timesteps": PAPER_NUM_TIMESTEPS,
        "attn_layers": "全部 attn2（cross-attention）；attn1 不在 Eq. 3 的定義內",
        "head_reduction": "mean（論文 Eq. 3 無 head 軸）",
        "upsample": "bilinear 到觀察到的最大 cross-attention 網格（論文未指定）",
        "step_size_provenance": "論文未給；本檔取 1/255，見 RECONSTRUCTED_STEP_SIZE",
        "models_in_paper": (
            "CompVis/stable-diffusion-v1-4、HQ-Edit、timbrooks/instruct-pix2pix"
        ),
        "dataset_in_paper": "InstructPix2Pix-clip-filtered 抽 200 張，逐張原生指令",
        "metrics_in_paper": "PSNR / SSIM / FSIM / VIFp / LPIPS",
        "baselines_in_paper": "ACE、EditShield、Mist、PGD、PGE、SDS、SA",
    },
)
