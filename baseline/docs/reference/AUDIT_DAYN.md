# DAYN（Lo et al., CVPR 2024）查證報告

*Distraction is All You Need: Memory-Efficient Image Immunization against
Diffusion-Based Image Editing*，正文 pp. 24462–24471。

**無官方程式**，故本檔不是「論文 vs 原始碼」的逐行對照，而是
「論文寫什麼 vs 本專案實作什麼 vs 論文沒寫什麼」三欄。查證的範圍：正文
§1–§6 全文、Eq. 1–5、Algorithm 1 逐行（含 layout 模式重抽，確認第 6 行原文
就沒有公式）、Figure 2–6 的說明文字、補充材料 §A–§D 全文；CVPR open access
頁面無 code 連結，作者亦無公開 repo。

實作在 `src/immunization_baseline/attacks/dayn.py`，測試在 `archive/anti-purification/tests/test_dayn.py`（49 項）。
以下內容與該模組的 docstring 是同一份，**改一邊要改兩邊**。

---

DAYN —— Lo, Yeo, Shuai, Cheng，*Distraction is All You Need:
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
2. **`src/immunization_baseline/attacks/{sifm,danp}.py、archive/anti-purification/src/baselines/tdae.py` 三篇對照表裡的 `SA`
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
7. **注意力 processor 是自己寫的一份。** `src/immunization_baseline/attacks/danp.py` 的
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
`[-1,1]` 上的值（`src/immunization_baseline/attacks/advpaint.py`）。

**這是本檔挑的，不是論文的。** 另一種讀法（`[0,1]` 的 0.06 = 15.3/255）
同樣說得通，且會給出強度差一倍的解。

原接入提案：`archive/anti-purification/scripts/baseline_run.py` 的 `CONDITIONS`
──────────────────────────────────────────────────────────────────────

> 本節為查證當時對 `baseline_run.py` 的接入提案，保留作為歷史紀錄。本專案的實際入口為
> `immunization_baseline.cli.generate_defenses` 的 `dayn` 分支，條件設定為
> `configs/conditions.yaml` 的 `dayn`（`spec: dayn.SPEC_PAPER`）。

**原提案不修改 `archive/anti-purification/scripts/baseline_run.py`，也不註冊進 `src/immunization_baseline/attacks/__init__.py`
的 `REGISTRY`。** 後者是刻意的：`archive/anti-purification/tests/test_baselines.py` 以
`AUDIT == REGISTRY` 稽核，且逐一檢查 `REGISTRY` 的值域，加進去會打到既有
測試；`danp`／`sifm`／`tdae` 也都刻意沒註冊。

接線要加的是：

    # src/immunization_baseline/attacks/__init__.py
    from src.baselines import advpaint, dayn, dia, mist, photoguard, promptflare
    _SPECS = (..., dayn.SPEC_PAPER)

    # archive/anti-purification/scripts/baseline_run.py 第 36 行
    from src.baselines import dayn, dia, mist, photoguard  # noqa: E402

    # archive/anti-purification/scripts/baseline_run.py 第 60 行
    CONDITIONS = ["photoguard_c", "photoguard_linf", "mist", "dia_r", "dayn"]

    # archive/anti-purification/scripts/baseline_run.py `run_additive` 的 spec 表（第 90-93 行）
    spec = {"photoguard_c": photoguard.SPEC,
            "photoguard_linf": photoguard.SPEC_PAPER_LINF,
            "mist": mist.SPEC,
            "dia_r": dia.SPEC_R,
            "dayn": dayn.SPEC_PAPER}[name]

    # archive/anti-purification/scripts/baseline_run.py `run_additive` 的 kw 分支（第 94-105 行之後）
    elif name == "dayn":
        # c_a 是防禦方選的那個詞，來自資料集的 `content` 欄；
        # **不可以傳 item["prompt"]**，那是攻擊方寫的，prepare 會拒絕。
        # use_ckpt 讓每個 U-Net 前向各成一個 checkpoint 區塊。
        kw = {"content": item["content"], "use_ckpt": True}
