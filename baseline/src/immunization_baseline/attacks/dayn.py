r"""DAYN —— Lo, Yeo, Shuai, Cheng，*Distraction is All You Need:
Memory-Efficient Image Immunization against Diffusion-Based Image Editing*
（**CVPR 2024**，正文 pp. 24462–24471；論文自稱這個方法為 **semantic attack**）。

**本檔為依論文重建，無官方程式。** CVPR open access 頁面只有正文 PDF 與
補充材料 PDF，沒有 code／supplementary code 連結；作者未在論文任何一處給出
repo 位址。因此本檔的每一行只能追溯到論文的某一節或某一式，凡論文未寫、
也無法由論文推得的項目，一律列進下方「論文未給的項目」表，並在
`BaselineSpec.modification_note`、`discrepancy_note` 與
`docs/reference/AUDIT_DAYN.md` 各重複一次。**不得把這些項目讀成論文設定。**

這一篇在本專案裡的兩個身分
──────────────────────────────────────────────────────────────────────

1. **`data/lo_aligned` 的結構對齊的就是它的補充材料 §A。** §A 的原文是：
   150 張由擴散模型生成的影像、3 類物件、每類 2 個編輯 prompt
   （prompt 1 改掉指定內容、prompt 2 改動其他區域），編輯用 SD v1.4、
   每個設定平均 20 個隨機種子。`data/lo_aligned/prompts.yaml` 的
   `prompts[0]`／`prompts[1]` 與 `content`（即 `c_a`）就是這個結構。
   §A 逐字寫出的三組 prompt 見 `docs/reference/AUDIT_DAYN.md` §7。
   **§A 沒有提到任何遮罩**——`data/lo_aligned/masks/` 的產生方式是本專案
   自己的決定，不是對齊 §A 的結果。
   `data/dayn_testset/` 是規劃向作者索取的原始測試集，**至今未取得**
   （`data/README.md`）。
2. **`src/baselines/{sifm,danp,tdae}.py` 三篇對照表裡的 `SA`
   （Semantic Attack）欄就是這一篇。** 例如 `AUDIT_SIFM.md` §2 表 II 的
   `SA [32]` 一列（PSNR 17.85、SSIM 0.5583、LPIPS 0.4225）引的即是本篇。
   要對回那三篇的數字，DAYN 是必須有的那一欄——但那些數字是**那三篇自己
   重跑的 baseline 欄**，協定與本專案不同，不可用來判定本檔實作的對錯
   （`docs/reference/BASELINE_PROVENANCE.md` 規則 1、規則 2）。

論文的方法（§3.2 Semantic Attack、§3.3 Timestep Universal Gradient Updating）
──────────────────────────────────────────────────────────────────────

1. 注意力（Eq. 2）

       A_l(x_adv, c_a) = Softmax( (W_q^l ε_θ^l(x_adv)) (W_k^l c_a)ᵀ / √d )

   `ε_θ^l(x_adv)` 是 U-Net 第 `l` 個中間 block 的深層特徵，`c_a` 是
   **要保護的內容**（focal content）的文字嵌入。故 `A_l` 是 softmax 之後、
   乘上 V 之前的 cross-attention **機率**。

2. 聚合（Eq. 3）

       Att(x_adv, c_a) = Σ_{l=1..L} upsample(A_l(x_adv, c_a))

   論文明寫用 **bicubic** 插值上採樣到「initial feature map」的大小，
   且是 **逐像素相加**（"We sum the attention maps pixel by pixel"）——
   不是平均。這一點與 DANP 的 Eq. 4（`(1/L)Σ`）不同。

3. 內容遮罩（Eq. 4）

       M = 𝟙( Att(x, c_a) > τ )

   **由乾淨影像 `x` 算**，不是由 `x_adv` 算，故在整個 PGD 過程中固定。

4. 注意力壓抑損失（Eq. 5）

       L = ‖ Att(M ⊙ x_adv, c_a) ‖₁

   §3.2 的說明是「`Att(M ⊙ x_adv, c_a)` 指免疫圖 `x_adv` 上**含該內容的
   區域**的注意力回應」。**最小化**它 = 壓低 `c_a` 在其自身區域的注意力，
   softmax 的質量因此被推往其他 token——這就是標題的 "distraction"。

5. Timestep universal gradient updating（§3.3、Algorithm 1）

       δ ← 0,  x_adv ← x
       for n = 1..N:
           all_grad ← 0
           for t in 𝒯:                        # |𝒯| = 10
               x_t_adv ← forward-noise(x_adv, t)
               L ← ‖Att(M(x_t_adv), c_a)‖₁
               all_grad ← all_grad + ∇_{x_adv} L
           all_grad ← mean(all_grad)
           δ ← δ + s · sign(all_grad)
           δ ← clip(δ, −κ, κ)
           x_adv ← x_adv − δ                  # 見「對不上的地方」第 3 點

   記憶體的來源就在這裡：反向傳播只穿過**單一個** timestep 的 U-Net
   前向，而不是整條擴散鏈（§3.3、§4.4 Figure 6）。

論文給的超參數與其出處
──────────────────────────────────────────────────────────────────────

| 項目 | 值 | 出處 |
|---|---|---|
| 預算 κ | 0.06 | §4.1「we set the perturbation budget κ = 0.06」 |
| 步數 N | 100 | 同上（「number of iterations N = 100 for all attacks」） |
| timestep 數 \|𝒯\| | 10 | 同上（「the number of diffusion timesteps T is set to 10」） |
| 上採樣方式 | bicubic | §3.2 Eq. 3 前一段 |
| 聚合方式 | 逐像素**相加** | §3.2 Eq. 3 |
| 受害模型 | Stable Diffusion **v1.4**（HuggingFace） | §4.1；§4.3 另用 v2.0 做評測 |
| 資料 | 生成的 150 張、3 類物件、每類 2 個 prompt | §4.3 ＋ 補充材料 §A |
| 評測 | 20 個隨機種子平均 | 補充材料 §A |
| 指標 | PSNR↓ SSIM↓ VIFp↓ FSIM↓ LPIPS↑ | §4.3 Table 1 |
| 對照 | PhotoGuard 的 encoder attack、diffusion attack | §4.1 |

論文未給的項目（本檔的決定，全部標 `modified_from_paper`）
──────────────────────────────────────────────────────────────────────

以下每一項都查過：正文 §1–§6 全文、Eq. 1–5、Algorithm 1 的逐行（含 layout
模式重抽，確認第 6 行原文就沒有公式）、Figure 2–6 的說明文字、補充材料
§A–§D 全文。CVPR open access 頁面無 code 連結，作者亦無公開 repo。

| 項目 | 論文 | 本檔 | 理由 |
|---|---|---|---|
| **步長 s** | **未找到**（只在 Algorithm 1 的 Input 列出符號） | `2/255`（該篇值域，等於 `[0,1]` 的 `1/255`） | 見 `RECONSTRUCTED_STEP_SIZE` |
| **κ = 0.06 的值域** | **未找到** | `[-1,1]`，即 `[0,1]` 的 0.03 | 見 `DAYN_RANGE` |
| **門檻 τ**（Eq. 4） | **未找到**（只寫「a threshold τ」） | `Att(x, c_a)` 的**空間平均** | 見 `default_tau`。`Att` 是 L 層 softmax 機率之和，其絕對尺度隨 L 與 token 數改變，故任何寫死的絕對值都不可攜 |
| **算 M 用哪個 timestep** | **未找到**（Eq. 4 的 `Att(x, c_a)` 沒有 t） | `𝒯` 上各算一次取平均後再門檻化 | M 在 Algorithm 1 是 Input（固定），而 Eq. 2 的 `A_l` 必然依賴加噪後的輸入；取同一組 `𝒯` 不引入演算法以外的新時刻 |
| **𝒯 的取值** | **未找到**（只給 \|𝒯\| = 10） | `linspace(0, T−1, 10)` 的等距固定格點 | Algorithm 1 把 𝒯 當 Input，迴圈內不重抽 |
| **注意力取自哪些層** | Eq. 3 寫「L 個 U-Net 中間 block」，未再縮小 | 全部 `attn2`（cross-attention），不含 `attn1` | Eq. 2 的 K 來自 `c_a`，self-attention 不在定義內 |
| **head 怎麼處理** | **未找到**（Eq. 2 的 `A_l` 沒有 head 軸） | 對 head 取平均 | 平均是唯一不引入新權重的化約 |
| **token 軸怎麼處理** | **未找到** | softmax 在**全部 77 個 token** 上，再取 `c_a` 對應 token 的欄相加 | 見「`c_a` 怎麼定位」 |
| **latent 還是像素** | 通篇寫 `x`，未提 VAE | 在 **latent** 上加噪，`z₀ = E(x)` 取後驗均值 | 受害模型是 latent diffusion，Eq. 2 的 `Q_l` 來自 U-Net 的影像特徵 |
| **forward noise 的式子** | **未找到**：Algorithm 1 第 6 行只寫 `x_t_adv ← x_adv`，沒有公式 | DDPM 的 `√ᾱ_t z₀ + √(1−ᾱ_t) ε` | §3.1 Eq. 1 的訓練目標即以此定義 `z_t` |
| **CFG** | **未找到** | 不用 CFG，單次條件前向 | Eq. 2 只有一條 `c_a` 條件分支 |
| **δ 的初始化** | Algorithm 1 第 2 行 `δ ← 0` | `init_rule="none"` | 論文明寫 |

`c_a` 怎麼定位，以及為什麼不能從攻擊 prompt 推
──────────────────────────────────────────────────────────────────────

Eq. 2 的 `c_a` 是「要保護的內容的文字嵌入」。**它由防禦方選、攻擊 prompt
由攻擊方寫，在威脅模型裡屬於不同的人**（`data/lo_aligned/prompts.yaml`
的說明同此）。論文 §4.3 明說「即使攻擊 prompt 裡沒有出現該內容，攻擊
仍然有效」——這句話只有在 `c_a` 與攻擊 prompt 相互獨立時才有意義。

因此 `prepare` 的 `content` 是**必填**，且本檔**不接受** `prompt` 參數：
傳入 `prompt` 會直接拋錯，而不是拿它去猜 `c_a`。

實作上，U-Net 的 cross-attention 條件就是 `c_a` 自己的 CLIP 編碼
（`sd.encode_text(content)`，padding 到 77）。softmax 在 77 個 token 上，
再取 `c_a` 的內容 token（`[BOS]`／`[EOS]`／`[PAD]` 除外）對應的欄相加，
得到一張空間圖。**不可以只把 `c_a` 的那幾個 token 餵進 K**：那樣 softmax
的分母只剩那幾欄，每一列恆為 1，損失變成常數、梯度恆為零——而這件事在
結果上沒有症狀（輸出仍是一張合理的防禦圖，只是等於沒攻擊）。
定位由 `locate_content_tokens` 做，找不到就拋錯。

實作上與論文對不上的地方
──────────────────────────────────────────────────────────────────────

1. **Eq. 5 的 `M ⊙ x_adv` 有兩種讀法，本檔取「遮罩作用在注意力圖上」。**
   字面讀是「先把影像乘上遮罩、再送進 U-Net」，但 `M` 由 Eq. 4 對
   `Att(x, c_a)` 門檻化而來，住在**注意力網格**上（SD v1.x／512² 下是
   64×64），而 `x_adv` 是 512×512×3，逐元素相乘沒有定義；且 §3.2 自己的
   說明是「`Att(M ⊙ x_adv, c_a)` 指免疫圖上**含該內容的區域**的注意力
   回應」。故本檔實作

       L = ‖ M ⊙ Att(x_adv, c_a) ‖₁

   字面讀法（把影像挖黑後前向）會得到另一個目標，此處**沒有實作**。
2. **`‖·‖₁` 在此等於求和。** `Att` 是 softmax 機率之和，恆為非負；
   本檔照定義寫 `.abs().sum()`，bicubic 上採樣的過衝（overshoot）若讓
   個別元素變負，行為仍是逐元素 ℓ1。
3. **Algorithm 1 第 12–13 行照字面會發散。** 第 12 行 `δ ← δ + s·sign(g)`
   接 `δ ← clip(δ, −κ, κ)`，第 13 行卻寫 `x_adv ← x_adv − δ`——`x_adv`
   是**累加**的，N=100 步後與 `x` 的距離可達 `100κ`，與第 2 行的
   `‖δ‖ < κ` 矛盾。可與 `‖δ‖ ≤ κ` 相容的讀法是 `x_adv ← x − δ`，
   即標準的 sign-PGD 下降。本檔走骨幹的 `objective="minimize"`
   （`x_adv ← x_adv − s·sign(g)` 後投影回 `x ± κ`），與該讀法等價。
4. **遮罩不回傳梯度。** `M` 是指示函數、且由乾淨影像算得，與 δ 無關，
   此處明確 `detach()`。論文未討論。
5. **梯度累積以骨幹的 `grad_reps` 實作。** Algorithm 1 第 5–11 行在一次
   PGD 迭代內對 `𝒯` 的每個元素各求一次梯度再取平均，骨幹的 `grad_reps`
   做的正是這件事，故 `grad_reps = |𝒯| = 10`，`loss_fn` 每次呼叫取 `𝒯`
   的下一個元素。
6. **9 通道（inpainting）權重下後 5 個通道取全 1 遮罩**，與 `mist.loss_fn`、
   `danp.loss_fn` 同一處置：Eq. 2 的 `ε_θ` 沒有影像條件。
7. **注意力 processor 是自己寫的一份。** `src/baselines/danp.py` 的
   `DANPAttnProcessor` 做同一件事（diffusers 的 SDPA 融合核不會實體化
   softmax 後的 `A`），但那是另一篇的檔案，本檔不共用、也不修改它——
   兩篇的聚合規則不同（本篇 Eq. 3 是相加＋bicubic，DANP Eq. 4 是平均），
   共用一份會讓其中一篇量到的不是它自己的方法。

κ 的值域（`DAYN_RANGE` 的依據）
──────────────────────────────────────────────────────────────────────

論文只寫 κ = 0.06，未說在 `[0,1]` 還是 `[-1,1]` 上量，而兩者差一倍。
**Table 1 幫不上忙**：它報的是「免疫後的編輯結果 vs 原始編輯結果」的
PSNR，不是免疫圖對原圖的失真，無法像 DANP 那樣反推上界。

本檔取 `[-1,1]`（即 `[0,1]` 的 0.03 = 7.65/255），理由是 §4.1 把同一個
κ 同時套在 encoder attack、diffusion attack 與 semantic attack 上
（「for a fair comparison」），而那兩個對照攻擊出自 PhotoGuard，其官方
實作的影像張量在 `[-1,1]`（`docs/reference/BASELINE_PROVENANCE.md` §預算
總表）；共用一個 κ 只有在共用值域時才成立。同一個 0.06 在 AdvPaint 也是
`[-1,1]` 上的值（`src/baselines/advpaint.py`）。

**這是本檔挑的，不是論文的。** 另一種讀法（`[0,1]` 的 0.06 = 15.3/255）
同樣說得通，且會給出強度差一倍的解。

要接進 `scripts/baseline_run.py` 的 `CONDITIONS` 需要加的幾行
──────────────────────────────────────────────────────────────────────

**本檔不修改 `scripts/baseline_run.py`，也不註冊進 `src/baselines/__init__.py`
的 `REGISTRY`。** 後者是刻意的：`tests/test_baselines.py` 以
`AUDIT == REGISTRY` 稽核，且逐一檢查 `REGISTRY` 的值域，加進去會打到既有
測試；`danp`／`sifm`／`tdae` 也都刻意沒註冊。

接線要加的是：

    # src/baselines/__init__.py
    from src.baselines import advpaint, dayn, dia, mist, photoguard, promptflare
    _SPECS = (..., dayn.SPEC_PAPER)

    # scripts/baseline_run.py 第 36 行
    from src.baselines import dayn, dia, mist, photoguard  # noqa: E402

    # scripts/baseline_run.py 第 60 行
    CONDITIONS = ["photoguard_c", "photoguard_linf", "mist", "dia_r", "dayn"]

    # scripts/baseline_run.py `run_additive` 的 spec 表（第 90-93 行）
    spec = {"photoguard_c": photoguard.SPEC,
            "photoguard_linf": photoguard.SPEC_PAPER_LINF,
            "mist": mist.SPEC,
            "dia_r": dia.SPEC_R,
            "dayn": dayn.SPEC_PAPER}[name]

    # scripts/baseline_run.py `run_additive` 的 kw 分支（第 94-105 行之後）
    elif name == "dayn":
        # c_a 是防禦方選的那個詞，來自資料集的 `content` 欄；
        # **不可以傳 item["prompt"]**，那是攻擊方寫的，prepare 會拒絕。
        # use_ckpt 讓每個 U-Net 前向各成一個 checkpoint 區塊。
        kw = {"content": item["content"], "use_ckpt": True}
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from src.baselines.pgd import BaselineSpec, ValueRange

# ---------------------------------------------------------------------------
# 論文給的常數
# ---------------------------------------------------------------------------

DAYN_RANGE = ValueRange(
    -1.0,
    1.0,
    "論文 §4.1 只寫 κ=0.06、未述值域；Table 1 量的是編輯結果之間的 PSNR，"
    "無法反推免疫圖的失真上界。取 [-1,1] 的理由是 §4.1 把同一個 κ 套在 "
    "PhotoGuard 的 encoder／diffusion attack 上，而那兩者的官方實作在 "
    "[-1,1]；見模組 docstring「κ 的值域」。這是本檔的決定，不是論文的",
)

PAPER_KAPPA = 0.06              # §4.1，perturbation budget
PAPER_ITERS = 100               # §4.1，number of iterations N
PAPER_NUM_TIMESTEPS = 10        # §4.1，|𝒯|

# 論文全文未給步長 s（只在 Algorithm 1 的 Input 列出符號）。查過：§3.3 全段、
# §4.1 的訓練設定、Algorithm 1 逐行（含 layout 模式重抽）、Table 1 的註腳、
# 補充材料 §A–§D。
#
# 取 `2/255`（該篇值域），等於 `[0,1]` 的 `1/255`：本 repo 已查證的 L∞
# baseline（Mist 的 `1/255·2`、PromptFlare 的 `2/255` 皆施加於 [-1,1]、
# DIA、DiffusionGuard）步長換算到 `[0,1]` 全部是 `1/255`。
# 在 κ=0.06、N=100 下 `100 × 2/255 = 0.78 ≫ 0.06`，預算可達且有餘裕。
#
# **這是本檔挑的，不是論文的。** 換成 κ/N 或 2κ/N 同樣說得通，而三者會給出
# 不同的解。
RECONSTRUCTED_STEP_SIZE = 2.0 / 255.0


# ---------------------------------------------------------------------------
# c_a 的 token 定位
# ---------------------------------------------------------------------------


def _input_ids(encoded) -> List[int]:
    """把 tokenizer 的回傳值化約成一串 token id。

    HuggingFace 的 `BatchEncoding` 與純 dict 都支援 `["input_ids"]`；
    未給 `return_tensors` 時回傳的是巢狀或非巢狀的 list，兩種都處理。
    """
    ids = encoded["input_ids"]
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]


def locate_content_tokens(
    tokenizer, content: str, prompt: Optional[str] = None
) -> List[int]:
    """回傳 `c_a` 的內容 token 在 77 格 padding 序列中的欄索引。

    `prompt` 省略時即以 `content` 自己作為送進文字編碼器的序列——那是本檔
    的預設條件（Eq. 2 的條件就是 `c_a`）。給 `prompt` 只用於把 `content`
    定位在一段較長的**防禦方自己的**描述裡，**不得傳入攻擊 prompt**
    （見模組 docstring「`c_a` 怎麼定位」）。

    做法是把 `content` 單獨編碼、去掉 `[BOS]`／`[EOS]`，在 padding 後的
    序列裡找**第一個**連續相符的位置。找不到就拋 `ValueError`：
    定位失敗時若退回「取全部 token」或「取第 1 格」，攻擊會壓到別的東西，
    而那在輸出上沒有任何症狀。
    """
    if not content.strip():
        raise ValueError("c_a 不得為空字串：Eq. 2 的 K 由 c_a 的嵌入產生")
    text = content if prompt is None else prompt

    full = _input_ids(
        tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    )
    core = _input_ids(tokenizer(content))
    if len(core) < 3:
        raise ValueError(
            f"c_a={content!r} 去掉 [BOS]/[EOS] 之後沒有任何 token"
        )
    core = core[1:-1]

    for start in range(0, len(full) - len(core) + 1):
        if full[start : start + len(core)] == core:
            return list(range(start, start + len(core)))
    raise ValueError(
        f"在序列 {text!r} 中找不到 c_a={content!r} 的 token（id {core}）。"
        "c_a 必須逐 token 出現在送進文字編碼器的序列裡，否則 Eq. 2 的欄"
        "選不到東西"
    )


# ---------------------------------------------------------------------------
# 注意力擷取：必須拿到 softmax 之後的機率，不能用 SDPA
# ---------------------------------------------------------------------------


class DAYNAttnController:
    """收集各 cross-attention 層的**後 softmax 機率** `A_l`，再依 Eq. 3 聚合。

    與 `src/baselines/promptflare.py::AttnController` 的差別：那一支記的是
    `attn2` 經 `to_out` 之後的輸出（`A·V·W_out`），走 SDPA 融合核，`A` 從未
    被實體化。Eq. 2／Eq. 5 的被優化量正是 `A` 本身，故不能沿用那條路。
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
        # head 平均：Eq. 2 的 A_l 沒有 head 軸，見模組 docstring。
        self.maps.append(probs.view(b, heads, hw, tok).mean(1))

    def reset(self) -> None:
        self.maps = []

    def aggregate(self) -> torch.Tensor:
        """Eq. 3：`Att = Σ_l upsample(A_l)`，回傳 `(B, G·G, tokens)`。

        兩件事照論文字面做，**與 DANP 的 Eq. 4 不同**：

        - **相加不平均**（"We sum the attention maps pixel by pixel"）。
        - **bicubic** 插值（§3.2 明寫 "using bicubic interpolation"）。
          bicubic 會過衝，上採樣後的個別元素可能落在 `[0,1]` 之外；
          此處**不夾回**，夾回就不是論文那個量了。

        目標解析度論文寫「initial feature map」的大小，未給數字。此處取
        **本次前向觀察到的最大** cross-attention 網格（SD v1.x／512² 下是
        64×64，即 latent 解析度）——那是不丟失任何一層空間資訊的最小共同
        網格。

        只支援正方形網格：`HW` 不是完全平方數時直接中止，因為長寬比不為 1
        時無法由 token 數還原網格形狀，而猜錯會靜默算出另一張圖。
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
                    "Eq. 3 的 upsample 目標形狀必須另行決定"
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
                    x, size=(grid, grid), mode="bicubic", align_corners=False
                )
                up = x.reshape(b, tok, grid * grid).transpose(1, 2)
            out = up if out is None else out + up
        return out


class DAYNAttnProcessor:
    """明確算 softmax 的 cross-attention processor，供 `A_l` 被實體化。

    這段是 diffusers 內建 `AttnProcessor`（非 2.0 版）的逐項展開，唯一的
    增加是把 `attn.get_attention_scores` 的回傳值交給 controller。
    **不能改用 `AttnProcessor2_0`／SDPA**：融合核不回傳注意力機率。

    `attn1`（self-attention）也會被裝上，但 controller 只在 `attn2` 上被
    呼叫——Eq. 2 的 `K_l` 來自 `c_a` 的文字嵌入。
    """

    def __init__(self, controller: DAYNAttnController, module_name: str):
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
            # 完全沒有症狀。與 promptflare、danp 的同一處置。
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

        # 這一行就是 Eq. 2 的 Softmax(QKᵀ/√d)。
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
# 遮罩（Eq. 4）與損失（Eq. 5）
# ---------------------------------------------------------------------------


def content_attention(att: torch.Tensor, token_indices: Sequence[int]) -> torch.Tensor:
    """把 `(B, HW, tokens)` 化約成 `c_a` 的空間圖 `(B, HW)`。

    多個 token 的情形（`c_a` 不只一個詞）取**相加**，與 Eq. 5 的 ℓ1 一致
    ——ℓ1 對 `c_a` 涵蓋的整塊注意力求和，先沿 token 軸加起來不改變總和。
    """
    idx = list(token_indices)
    if not idx:
        raise ValueError("token_indices 為空：c_a 沒有對應到任何欄")
    if max(idx) >= att.shape[-1] or min(idx) < 0:
        raise IndexError(
            f"token 索引 {idx} 超出注意力的 token 軸長度 {att.shape[-1]}"
        )
    return att[..., idx].sum(-1)


def default_tau(map2d: torch.Tensor) -> torch.Tensor:
    """Eq. 4 的 τ。**論文未給值**，此處取該張圖的空間平均。

    `Att` 是 L 層 softmax 機率之和，其絕對尺度隨層數、token 數與解析度
    改變，故寫死一個絕對值不可攜；平均是不引入新參數的統計量。
    要掃描 τ 時由 `prepare(tau=...)` 覆寫。
    """
    return map2d.detach().mean()


def content_mask(map2d: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Eq. 4：`M = 𝟙(Att(x, c_a) > τ)`，回傳與 `map2d` 同形的 0/1 張量。

    **已 detach**：指示函數幾乎處處梯度為零，且 `M` 由乾淨影像算得、與 δ
    無關。

    全 0 的遮罩直接拋錯：那時 Eq. 5 的損失恆為 0、梯度恆為零，PGD 會原地
    不動並輸出一張與原圖逐位元相同的「防禦圖」——在報表上看不出來。
    """
    mask = (map2d.detach() > tau).to(map2d.dtype)
    if float(mask.sum()) == 0.0:
        raise RuntimeError(
            f"Eq. 4 的遮罩是空的（τ={float(tau):.6g}，"
            f"Att 的全距 {float(map2d.min()):.6g}–{float(map2d.max()):.6g}）。"
            "空遮罩下 Eq. 5 的損失恆為 0、梯度恆為零，攻擊等於沒有執行"
        )
    return mask


def attention_suppressing_loss(
    content_map: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Eq. 5：`L = ‖M ⊙ Att(x_adv, c_a)‖₁`，**最小化**。

    梯度是 `∂L/∂Att = M ⊙ sign(Att)`，在 `Att > 0` 處為正——沿負梯度走即
    **壓低** `c_a` 在其區域的注意力，softmax 的質量被推往其他 token。
    符號寫反不會有症狀（輸出仍是一張合理的防禦圖），只是變成把注意力
    **加強**在該區域，方向與論文相反。
    """
    if content_map.shape != mask.shape:
        raise ValueError(
            f"注意力圖 {tuple(content_map.shape)} 與遮罩 {tuple(mask.shape)} 不同形"
        )
    return (content_map * mask).abs().sum()


def paper_timesteps(sd, count: int = PAPER_NUM_TIMESTEPS) -> torch.Tensor:
    """`𝒯 = {t₁..t_j}`，`|𝒯| = 10`（§4.1）。取值論文未給。

    取 `[0, T−1]` 的等距固定格點：Algorithm 1 把 `𝒯` 當 Input，迴圈內
    不重抽。
    """
    if count < 1:
        raise ValueError(f"|𝒯| 必須 ≥ 1，收到 {count}")
    return torch.linspace(0, sd.num_train_timesteps - 1, count).round().long()


# ---------------------------------------------------------------------------
# prepare / loss_fn
# ---------------------------------------------------------------------------


class DAYNContext:
    """安裝／還原 attention processor 的責任在這裡。`run_pgd` 結束時呼叫 `close()`。"""

    def __init__(
        self,
        sd,
        spec: BaselineSpec,
        x01: torch.Tensor,
        emb,
        token_indices: Sequence[int],
        timesteps: torch.Tensor,
        generator: torch.Generator,
        *,
        content: str,
        tau: Optional[float] = None,
        use_ckpt: bool = False,
    ):
        self.spec = spec
        self.vr = spec.value_range
        self.x01 = x01.detach()
        self.emb = emb
        self.token_indices = list(token_indices)
        self.timesteps = timesteps
        self.generator = generator
        self.content = content
        self.use_ckpt = bool(use_ckpt)
        self.controller = DAYNAttnController()
        self.abar = sd.alphas_cumprod(x01.device)
        self._cursor = 0

        self._saved = []
        for name, module in sd.unet.named_modules():
            if name.endswith("attn2"):
                self._saved.append((module, module.get_processor()))
                module.set_processor(DAYNAttnProcessor(self.controller, name))
        if not self._saved:
            raise RuntimeError("UNet 中找不到任何 attn2 層")

        # Eq. 4 的 M 由**乾淨影像**算，在整個 PGD 過程中固定。
        with torch.no_grad():
            self.z0_clean = sd.encode_image(self.x01).detach()
            clean_map = self._clean_content_map(sd)
        self.tau = (
            default_tau(clean_map)
            if tau is None
            else torch.as_tensor(tau, dtype=clean_map.dtype, device=clean_map.device)
        )
        self.mask = content_mask(clean_map, self.tau)
        self.clean_content_map = clean_map.detach()
        self.mask_fraction = float(self.mask.mean())

    # ---- Eq. 4 的 Att(x, c_a) ----

    def _clean_content_map(self, sd) -> torch.Tensor:
        """`𝒯` 上各算一次 `Att(x_t, c_a)` 再平均（時刻的選擇見模組 docstring）。"""
        total = None
        ones = torch.ones_like(self.x01[:, :1])
        for t in self.timesteps:
            z_t = self.add_noise(self.z0_clean, int(t))
            self.controller.reset()
            with sd.conditioning_for(self.x01, mask=ones):
                sd.unet_forward(z_t, t, self.emb)
                att = self.controller.aggregate()
            self.controller.reset()
            cmap = content_attention(att, self.token_indices)
            total = cmap if total is None else total + cmap
        return total / len(self.timesteps)

    def add_noise(self, z0: torch.Tensor, t: int) -> torch.Tensor:
        """Algorithm 1 第 6 行。式子論文未給，取 §3.1 Eq. 1 的 DDPM 前向。"""
        abar = self.abar[int(t)].to(z0.dtype)
        noise = torch.randn(
            z0.shape, generator=self.generator, device=z0.device, dtype=z0.dtype
        )
        return abar.sqrt() * z0 + (1.0 - abar).sqrt() * noise

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
    content: Optional[str] = None,
    seed: int = 0,
    num_timesteps: int = PAPER_NUM_TIMESTEPS,
    tau: Optional[float] = None,
    use_ckpt: bool = False,
    prompt: Optional[str] = None,
    **_,
) -> DAYNContext:
    """`content` 就是 Eq. 2 的 `c_a`，**必填且必須由呼叫端給**。

    `prompt` 只是一道擋板：本檔不接受編輯 prompt。`c_a` 由防禦方選、
    編輯 prompt 由攻擊方寫，從 prompt 推 `c_a` 會違反威脅模型
    （見模組 docstring「`c_a` 怎麼定位」）。
    """
    if prompt is not None:
        raise RuntimeError(
            "DAYN 不接受編輯 prompt：Eq. 2 的條件是防禦方選定的 c_a，"
            "與攻擊方寫的 prompt 屬於不同的人。請改傳 content=<c_a>"
        )
    if content is None:
        raise NotImplementedError(
            "DAYN 需要 c_a（要保護的內容）：Eq. 2 的 K 由它的嵌入產生、"
            "Eq. 4 的遮罩由它的注意力圖門檻化而來。論文 §4.3 的 c_a 是逐張"
            "由防禦方指定的詞（補充材料 §A 的 content 欄），沒有可沿用的"
            "預設值，請由呼叫端指定"
        )
    if spec.grad_reps != num_timesteps:
        raise ValueError(
            f"grad_reps={spec.grad_reps} 與 |𝒯|={num_timesteps} 不一致。"
            "Algorithm 1 第 5–11 行在一次迭代內對 𝒯 的每個元素各求一次梯度"
            "再平均，本專案以 grad_reps 實現，兩者必須相等，否則有的 t 會被"
            "走訪兩次、有的一次也沒有，而這件事在結果上沒有症狀"
        )

    token_indices = locate_content_tokens(sd.tokenizer, content)
    emb = sd.encode_text(content).detach()
    ts = paper_timesteps(sd, num_timesteps).to(x01.device)
    generator = torch.Generator(device=x01.device).manual_seed(seed)
    return DAYNContext(
        sd,
        spec,
        x01,
        emb,
        token_indices,
        ts,
        generator,
        content=content,
        tau=tau,
        use_ckpt=use_ckpt,
    )


def _unet(sd, ctx: DAYNContext, z, t):
    if ctx.use_ckpt:
        return ckpt.checkpoint(
            lambda a: sd.unet_forward(a, t, ctx.emb), z, use_reentrant=False
        )
    return sd.unet_forward(z, t, ctx.emb)


def loss_fn(sd, x_adv: torch.Tensor, ctx: DAYNContext) -> torch.Tensor:
    """Algorithm 1 第 6–9 行的一個 `t`，回傳 Eq. 5 的 `‖M ⊙ Att(x_adv, c_a)‖₁`。

    `x_adv` 在該篇的值域 `[-1,1]`（`DAYN_RANGE`），先換回 `[0,1]` 再交給
    `sd.encode_image`（本專案的 VAE 介面固定在 `[0,1]`，且取後驗均值而非
    抽樣，故 `z₀` 對同一張圖是決定性的）。
    """
    t = ctx.next_timestep()
    x01 = ctx.vr.to01(x_adv)
    z0 = sd.encode_image(x01)
    z_t = ctx.add_noise(z0, int(t))

    ctx.controller.reset()
    ones = torch.ones_like(x01[:, :1])
    with sd.conditioning_for(x01, mask=ones):
        _unet(sd, ctx, z_t, t)
        att = ctx.controller.aggregate()
    ctx.controller.reset()

    cmap = content_attention(att, ctx.token_indices)
    return attention_suppressing_loss(cmap, ctx.mask)


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


SPEC_PAPER = BaselineSpec(
    name="dayn",
    source=(
        "Lo et al., CVPR 2024, *Distraction is All You Need*"
        "（CVPR open access pp. 24462–24471，含補充材料）；無官方程式碼"
    ),
    value_range=DAYN_RANGE,
    eps=PAPER_KAPPA,                      # §4.1，κ = 0.06
    eps_pixel01=PAPER_KAPPA / 2.0,        # [-1,1] → [0,1]，見 DAYN_RANGE
    norm="linf",                          # Algorithm 1 第 12 行 clip(δ, −κ, κ)
    steps=PAPER_ITERS,                    # §4.1，N = 100
    step_size=RECONSTRUCTED_STEP_SIZE,    # **論文未給**，見該常數的註解
    step_schedule="constant",             # Algorithm 1 的 s 不隨 n 改變
    update_rule="sign",                   # Algorithm 1 第 12 行 sign(all_grad)
    objective="minimize",                 # Eq. 5 是要被壓低的注意力
    init_rule="none",                     # Algorithm 1 第 2 行 δ ← 0
    grad_reps=PAPER_NUM_TIMESTEPS,        # Algorithm 1 第 5–11 行：|𝒯| 次梯度取平均
    needs_target_image=False,
    needs_mask=False,                     # 遮罩由 Eq. 4 自己算，不由外部提供
    modified_from_paper=True,
    modification_note=(
        "全檔為依論文重建（CVPR 2024 open access 正文＋補充材料，無官方程式碼、"
        "無 code 連結），下列項目論文未給、由本檔決定："
        "(1) 步長 s = 2/255（該篇值域；論文只在 Algorithm 1 的 Input 列出符號，"
        "κ/N 或 2κ/N 同樣說得通）；"
        "(2) 值域 [-1,1]（論文只寫 κ=0.06；Table 1 量的是編輯結果之間的 PSNR，"
        "無法反推，取值域的理由見模組 docstring）；"
        "(3) Eq. 4 的門檻 τ 取 Att(x,c_a) 的空間平均（論文只寫「a threshold τ」）；"
        "(4) 算 M 的時刻取 𝒯 上平均（Eq. 4 沒有 t）；"
        "(5) 𝒯 取 [0,T−1] 的等距固定格點（論文只給 |𝒯|=10）；"
        "(6) Eq. 2 的 A_l 沒有 head 軸，此處對 head 取平均；"
        "(7) Eq. 3 的 upsample 目標取觀察到的最大 cross-attention 網格"
        "（插值方式照論文用 bicubic、聚合照論文用相加）；"
        "(8) token 軸：softmax 在全部 77 個 token 上，再取 c_a 的內容 token 相加；"
        "(9) 加噪在 latent 上進行、z₀ = E(x) 取後驗均值（論文通篇未提 VAE，"
        "Algorithm 1 第 6 行也沒有給前向加噪的式子）；(10) 不用 CFG。"
        "另：Eq. 5 的 M ⊙ x_adv 依 §3.2 的說明讀成「遮罩作用在注意力圖上」，"
        "字面讀法（把影像挖黑後前向）沒有實作。"
    ),
    discrepancy_note=(
        "論文與程式不一致無從比對——作者未釋出程式碼，本檔的對照對象只有論文。"
        "論文內部的落差："
        "(a) Algorithm 1 第 12–13 行照字面會發散（δ 夾在 ±κ，但第 13 行是 "
        "x_adv ← x_adv − δ 的累加，N=100 步後可達 100κ，與第 2 行的 ‖δ‖<κ 矛盾）；"
        "本檔取與 ‖δ‖≤κ 相容的讀法 x_adv ← x − δ，即骨幹的 minimize + L∞ 投影。"
        "(b) Eq. 5 的 M ⊙ x_adv 與 §3.2 的文字說明不一致：M 由 Eq. 4 對注意力圖"
        "門檻化而來，住在注意力網格上，與 512×512×3 的 x_adv 無法逐元素相乘。"
        "(c) Algorithm 1 第 6 行「Inject the forward noise」沒有附式子"
        "（已以 layout 模式重抽 PDF 確認不是抽取遺漏）。"
        "(d) §4.1 只說 κ=0.06 對三個攻擊「for a fair comparison」都一樣，"
        "未說 PhotoGuard 的 encoder／diffusion attack 各自用什麼步長與步數。"
        "對照數字：Table 1（SD v1.4）Ours 的 PSNR 15.1487／SSIM 0.4470／"
        "VIFp 0.1462／FSIM 0.6584／LPIPS 0.5901，量的是**免疫後的編輯結果 vs "
        "原始編輯結果**（不是免疫圖對原圖），協定為自建 150 張生成影像、"
        "3 類物件、每類 2 個 prompt、20 個種子平均；引用必須連協定一起引。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "kappa": PAPER_KAPPA,
        "num_timesteps": PAPER_NUM_TIMESTEPS,
        "tau_rule": "Att(x, c_a) 的空間平均（論文未給 τ）",
        "attn_layers": "全部 attn2（cross-attention）；attn1 不在 Eq. 2 的定義內",
        "head_reduction": "mean（論文 Eq. 2 無 head 軸）",
        "aggregation": "Σ_l（相加，§3.2 明寫 sum pixel by pixel）",
        "upsample": "bicubic（§3.2 明寫）到觀察到的最大 cross-attention 網格",
        "step_size_provenance": "論文未給；本檔取 2/255，見 RECONSTRUCTED_STEP_SIZE",
        "models_in_paper": "CompVis/stable-diffusion-v1-4（§4.1）；§4.3 另評 SD v2.0",
        "dataset_in_paper": (
            "生成的 150 張影像、3 類物件、每類 2 個編輯 prompt、20 個種子平均"
            "（§4.3 ＋ 補充材料 §A）"
        ),
        "metrics_in_paper": "PSNR↓ / SSIM↓ / VIFp↓ / FSIM↓ / LPIPS↑（Table 1）",
        "baselines_in_paper": "PhotoGuard 的 encoder attack、diffusion attack",
        "editing_pipelines_in_paper": (
            "SDEdit（§4.3）、inpainting（§4.2）、null-text inversion／EDICT／"
            "DiffEdit（§4.5）"
        ),
    },
)
