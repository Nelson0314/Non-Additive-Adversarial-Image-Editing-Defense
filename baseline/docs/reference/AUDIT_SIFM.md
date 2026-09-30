# SIFM 論文查證（arXiv:2512.14320）

用途：供 baseline 重建使用。本檔所有數值與公式均標註節次或式號；**查不到的
一律列在「未找到的項目」一節**，不作推斷、不補值。依 `BASELINE_PROVENANCE.md`
§規則 1，本檔引用的每個數字都附帶它的協定。

實作在 `src/immunization_baseline/attacks/sifm.py`，驗收在 `archive/anti-purification/tests/test_sifm.py`。

## 查證所用來源

| 代號 | 內容 | 位置 |
|---|---|---|
| `[SIFM-HTML]` | arXiv:2512.14320v1 的 HTML 全文（摘要、§I–§VIII、Algorithm 1、表 I–VII、參考文獻） | https://arxiv.org/html/2512.14320v1 |
| `[SIFM-ABS]` | arXiv 摘要頁（檢查是否掛 code / ancillary files） | https://arxiv.org/abs/2512.14320 |

作者：Shuai Dong、Jie Zhang、Guoying Zhao、Shiguang Shan、Xilin Chen。
版式為 IEEE 期刊模板（PubID 欄），v1 日期 2025-12-16。

---

## 0. 官方程式碼：沒有

| 查了哪裡 | 結果 |
|---|---|
| `[SIFM-HTML]` 全文搜尋 `github`、`code`、`available at`、`anonymous` | 只命中 arXiv HTML 檢視器自身的「Report GitHub Issue」介面字串，正文與結論**無任何 repo 連結** |
| `[SIFM-HTML]` 章節結構 | §I–§VIII ＋ References，**沒有 implementation details 一節、沒有附錄、沒有補充資料** |
| `[SIFM-ABS]` 的 Code, Data and Media 區塊 | 無關聯的程式碼或 ancillary files |
| 網路搜尋 `"Synergistic Intermediate Feature Manipulation" SIFM code github` | 無對應 repo（命中的 `cchen-cc/SIFA` 是 2019 年的醫學影像領域自適應，不同工作） |

**後果**：`src/immunization_baseline/attacks/sifm.py` 是**依論文重建**，不是官方路徑的重現。
`SPEC_PAPER.modified_from_paper = True`，全部自訂項目寫在 `modification_note`
與 `extras` 內。這與 `photoguard_linf`（依論文 Table 9 重建）同一性質，
而與 `mist`／`dia_r`（有官方程式可逐行對照）不同。

---

## 1. 論文寫什麼

### 1.1 中間特徵的聚合（§V 式 (3)）

> The aggregated feature representation φ_t(x,c) is obtained by averaging the
> features from a predefined set of M such targeted layers:
>
> `φ_t(x,c) = (1/M) Σ_{j=1..M} L_j( E(x,t)_noisy , c , t )`

`L_j` 是噪聲預測網路（§V 原文：U-Net or Diffusion Transformer）內第 j 個目標
層的輸出，`E(x,t)_noisy` 是「取得 x 對應 t 的帶噪 latent 的過程」。

### 1.2 語意發散項（§V-A 式 (4)）

`L_dist(δ;t) = Dist( φ_t(x₀+δ,c), φ_t(x₀,c) )`

> Herein, Dist(·,·) represents a distance metric (e.g., MSE) applied to feature
> representations.

方向：**最大化**（§V-A 原文 "we maximize the semantic divergence"）。

### 1.3 知覺劣化項（§V-B 式 (5)）

`L_norm(δ;t) = ‖ φ_t(x₀+δ,c) ‖₁`

§V-B 的理由是 L1 作為 sparsity-inducing regularizer，「compels the model to
discard less critical feature activations」。方向：**最小化**。

### 1.4 合成與更新（§V-C 式 (6)(7)、Algorithm 1）

```
L_SIFM(δ;t)      = L_norm(δ;t) − λ · L_dist(δ;t)              式 (6)，λ>0
L_SIFM_total(δ)  = (1/|T|) Σ_{t∈T} L_SIFM(δ;t)                式 (7)
```

Algorithm 1 逐行：

| 行 | 內容 |
|---|---|
| 1 | `δ ← 0` |
| 2 | `x_imu ← x_orig` |
| 3 | `Φ_orig ← { φ_t(x_orig,c) | t∈T }`（註解：Pre-compute original features） |
| 5–7 | `for n=1..N`，`g_total ← 0`，`for t in T` |
| 8 | `φ_t^imu ← φ_t(x_imu,c)` |
| 9 | `L_t ← ‖φ_t^imu‖₁¹ − λ·Dist(φ_t^imu, Φ_orig[t])` |
| 10 | `g_total ← g_total + ∇_{x_imu} L_t` |
| 12 | `g ← g_total / |T|` |
| 13 | `δ ← clip( δ − α·sign(g), −ε, ε )` |
| 14 | `x_imu ← clip_{0,1}( x_orig + δ )` |

第 13 行是**減號**，故整體是對式 (7) 做**最小化**的 sign-PGD。
第 14 行的 `clip_{0,1}` 指明論文的最佳化值域是 `[0,1]`。

### 1.5 ISR（§VI 式 (8)）

`ISR = N_success / N_total`。判定由 Gemini 2.5 Pro 與 Gemini 2.5 Flash 各自
做出，**兩者都判成功才算成功**（§VI：strict agreement policy，任一判失敗即
計為失敗）。成功的定義是編輯結果「語意偏離 prompt」**或**「有與 prompt 無關
的明顯知覺劣化」（§IV-B）。

### 1.6 實驗設定（§VII、§VII-A、§VII-C）

| 項目 | 值 | 出處 |
|---|---|---|
| 預算 | ε = 0.03 | §VII 首段，"All methods were constrained to a perturbation budget of ε=0.03" |
| 迭代 | 100 | 同上，"limited to 100 optimization iterations" |
| λ | 0.1 | §VII-C 表 VII：λ∈{0.001,0.01,0.1,1.0} 中 ISR 最高（79%），該節明言全文其餘實驗沿用 |
| 受害模型 | StableDiffusion-3 [49]、InstructPix2Pix [36]、HQ-Edit [48] | §VII-A |
| 資料 | 自建 100 張（人像 35／風景 35／畫作 30），取自 `timbrooks/instructpix2pix-clip-filtered` [51] | §VII-A |
| 求解用的 prompt | **該圖在該資料集中原本的 prompt** | §VII-A："Each image's original prompt from this collection was used for perturbation generation" |
| 評測用的 prompt | 另外生成 5 條語意不同的未見過 prompt，每個模型 500 個情境 | §VII-A |
| 傳統指標 | PSNR、SSIM、VIFp、FSIM、LPIPS，量的是**防禦圖的編輯結果**對**原圖的編輯結果** | §VII-A 評測指標段 |

**ε=0.03 與 100 步是「all methods」共用的對齊預算**，不是 SIFM 專屬的設定；
引用時必須連這句一起引。

---

## 2. 論文的對照數字（連協定一起引）

以下皆為 `[SIFM-HTML]` 表 II／VI／VII，協定：自建 100 張資料集、ε=0.03、
100 步、指標量的是「防禦圖的編輯結果 vs 原圖的編輯結果」、ISR 由兩個
Gemini 2.5 模型嚴格一致判定。**這些數字不可用來判定本專案實作的對錯**
（不同資料、不同受害模型、不同指標實作）。

表 II（StableDiffusion-3，Original Prompt ／ Unseen Prompts 的 ISR）：

| 方法 | PSNR↓ | SSIM↓ | VIFp↓ | FSIM↓ | LPIPS↑ | ISR↑ | ISR↑（unseen） |
|---|---|---|---|---|---|---|---|
| PhotoGuard-E [27] | 18.14 | 0.5940 | 0.2076 | 0.7921 | 0.3796 | 63% | 54% |
| PhotoGuard-D [27] | 18.43 | 0.5809 | 0.2150 | 0.7890 | 0.3921 | 67% | 48% |
| EditShield [28] | 22.88 | 0.7600 | 0.3894 | 0.8834 | 0.2155 | 50% | 26% |
| SA [32] | 17.85 | 0.5583 | 0.1894 | 0.7643 | 0.4225 | 70% | 48% |
| SIFM | 15.79 | 0.4747 | 0.1237 | 0.7162 | 0.5046 | 79% | 65% |

表 VI（目標函數消融，SD3）：

| L_dist | L_norm | ISR |
|---|---|---|
| ✓ | — | 66% |
| — | ✓ | 73% |
| ✓ | ✓ | 79% |

表 VII（λ 掃描，SD3）：λ=0.001→73%、0.01→76%、0.1→79%、1.0→71%。
**表 VII 只有四個點，且 PSNR 在四點間的全距是 0.08 dB（15.77–15.85）。**

`[32]`（表中的 SA）是 Lo et al., CVPR 2024,
*Distraction is all you need: Memory-efficient image immunization against
diffusion-based image editing*；論文正文未展開 SA 這個縮寫的全稱。

---

## 3. 本專案實作了什麼

| 論文的東西 | `src/immunization_baseline/attacks/sifm.py` 的對應 |
|---|---|
| 式 (3) 對 M 層平均 | `FeatureRecorder.aggregate`：forward hook 取層輸出後 `stack().mean(0)`，形狀不符即拋錯 |
| 式 (4) `Dist` = MSE | `_dist(..., "mse")` = `((a-b)**2).mean()` |
| 式 (5) `‖φ‖₁` | `sifm_terms`：`a.abs().sum()`，先轉 fp32（fp16 的 65504 上限會溢位成 inf） |
| 式 (6) `L_norm − λ·L_dist` | `sifm_terms` 回傳的第三項 |
| 式 (7) 對 T 平均 | `loss_fn`：逐 t 求損失後除以 `|T|` |
| Algorithm 1 第 1 行 δ←0 | `SPEC_PAPER.init_rule = "none"` |
| Algorithm 1 第 3 行 Φ_orig | `prepare` 內 `torch.no_grad()` 下算完，迴圈中不再更新 |
| Algorithm 1 第 13 行 `δ − α·sign(g)` 與 `clip(±ε)` | `objective="minimize"`、`update_rule="sign"`、`norm="linf"` |
| Algorithm 1 第 14 行 `clip_{0,1}` | `value_range = ValueRange(0,1,...)`，骨幹的 `project` 收尾即是 |
| §VII ε=0.03、100 步 | `eps = eps_pixel01 = 0.03`、`steps = 100` |
| §VII-C λ=0.1 | `PAPER_LAMBDA = 0.1` |
| §VII-A「用該圖原本的 prompt」 | `prepare(prompt=...)` **無預設值**，未給即拋錯 |
| §VI ISR | **未實作**（見 §4.8） |

Algorithm 1 第 10 行是逐 t 求梯度再平均，本實作回傳逐 t 損失的平均再由骨幹
一次反向。線性運算下梯度相同，差別只在 `|T|` 條計算圖同時留存。

**不提供 `use_ckpt`**（Mist 與 DIA 有）：`torch.utils.checkpoint` 的區塊前向在
`no_grad` 下執行，forward hook 攔到的特徵不在計算圖上，梯度會靜默變成零。
要省記憶體只能縮短 `timesteps`，而那會改變方法本身。

---

## 4. 未找到的項目

以下每一項都**在 `[SIFM-HTML]` 全文查過**（含摘要、§I–§VIII、Algorithm 1、
表 I–VII 的標題與註腳），確認論文沒有給。本專案各自做了決定，決定與理由
一併列出；這些值**不是論文的值**，報表上不得寫成原論文設定。

### 4.1 目標層的身分與 M —— 未找到

論文只有兩句定性敘述：§I「deeper diffusion layers encode semantically rich」、
§V-A「deeper layers of the noise predictor exhibit stronger alignment between
textual instructions and intermediate image features」，以及 §I 的「critical
semantic bottlenecks」。**沒有任何層名、沒有 M 的值、沒有選層的規則。**

本專案的決定：`M = 1`，層取 U-Net 的 `mid_block` 輸出（`DEFAULT_LAYERS`）。
理由：那是 U-Net 空間解析度最低、通道數最多的一點，是上述敘述唯一能指認的
對象；取 M>1 還必須再自行決定另外 M−1 層的身分，那是第二個沒有依據的決定。
`FeatureRecorder` 本身支援任意層名清單，換層只需傳 `layers=`。

### 4.2 時間步集合 T —— 未找到

§V 只寫 `T={t₁,…,t_k}`，全文沒有 k、沒有取值、沒有取法（等距？低 t 偏置？
與受害模型的取樣格點對齊？）。

本專案的決定：`T = (200, 400, 600, 800)`（`DEFAULT_TIMESTEPS`），訓練排程
`[0,1000)` 上的四個等距內點。k=4 的理由是成本：`|T|` 條計算圖同時留存。

### 4.3 步長 α —— 未找到

Algorithm 1 的 Input 行列了 `step size α`，**正文與 §VII 都沒有給值**。

本專案的決定：`α = 1/255`（`DEFAULT_STEP_SIZE`）。理由：本專案已查證的七篇
baseline 中，Mist、DIA、PromptFlare、DiffusionGuard 的步長在各自值域下都是
1/255 或 2/255；ε=0.03 ≈ 7.65/255、步數 100，此步長足以讓投影生效。

### 4.4 帶噪 latent 的取法 —— 未找到

式 (3) 只寫 `E(x,t)_noisy` 是「the process of obtaining the noisy latent
representation」。**沒有加噪式子、沒有說 VAE 取 sample 還是 mode、沒有說噪聲
是每個 iteration 重抽還是固定。**

本專案的決定：`z_t = √ᾱ_t·z₀ + √(1−ᾱ_t)·ε`（DDPM 前向式，與
`mist._semantic_loss` 同一條），`z₀` 取 VAE 的 mode（`sd.encode_image`），
**噪聲每個 t 在 `prepare` 抽一次後固定**。固定的理由來自 Algorithm 1 第 3 行：
`Φ_orig` 預先算完且迴圈中不更新，若 `φ_t^imu` 每次換噪聲，第 9 行的
`Dist(φ_t^imu, Φ_orig[t])` 比的是兩個不同噪聲實現下的特徵，該距離的大部分
來自噪聲而不是 δ。

### 4.5 CFG 與 prompt 分支 —— 未找到

式 (3)(4) 只寫一個條件 c。**沒有說 φ_t 取自有條件前向、無條件前向、還是
CFG 合成後的結果，也沒有 guidance scale。**

本專案的決定：單次有條件前向，不做 CFG（與 DIA 的 `cfg=1` 單分支同形狀）。

### 4.6 `Dist` 的確切選擇與 reduction —— 部分未找到

§V-A 寫「a distance metric (e.g., MSE)」——MSE 是**舉例**，不是指定。
另外，式 (5) 的 `‖·‖₁` 是絕對值之和，而 MSE 是平均；兩者的 reduction 差了
元素個數量級（SD v1.4、512²、`mid_block` 的特徵有 1280×8×8 = 81,920 個元素），
**而 λ=0.1 的意義完全取決於這兩個 reduction**。論文兩者都沒有寫明。

本專案的決定：`Dist` 取 MSE、`L_norm` 取絕對值之和，兩者都照式子逐字實作，
不另外做尺度對齊。

### 4.7 L1 還是 Frobenius —— 論文自相矛盾

- §I：「SIFM minimizes the **Frobenius norm** of targeted intermediate features」
- §V-B 標題與內文：「minimizing their **L1 norm**, which acts as a powerful
  sparsity-inducing regularizer」
- 式 (5)：`‖φ‖₁`
- Algorithm 1 第 9 行：`‖φ_t^imu‖₁¹`

三比一，且 §V-B 整段的論證（enforcing sparsity、zeroing out activations）只在
L1 下成立。本專案實作 **L1**，落差記在 `SPEC_PAPER.discrepancy_note`。

### 4.8 ISR —— 未實作

§VI 的 ISR 需要 Gemini 2.5 Pro 與 Gemini 2.5 Flash 兩個 MLLM 各自判定再取嚴格
一致。本專案的評測管線（`archive/anti-purification/scripts/baseline_run.py` 的 `evaluate`）沒有這條路，
`sifm.py` 只產生防禦圖。**因此本專案跑出來的任何數字都不能與論文表 II–VII 的
ISR 欄對照。**

§VI 另外沒有給：送進 MLLM 的 prompt 模板、「significant」劣化的門檻、是否提供
原圖作為對照（表 I 的人類研究寫每個樣本是 (Original Image, Edited Immunized
Image, Edit Prompt) 三元組，MLLM 端未明言是否相同）。

### 4.9 其他未找到的項目

| 項目 | 狀態 |
|---|---|
| 影像解析度 | 未找到 |
| 精度（fp16／fp32） | 未找到 |
| 隨機種子 | 未找到 |
| 受害模型的取樣步數、guidance scale、img2img strength | 未找到 |
| 對照方法（PhotoGuard-E/D、EditShield、ACE、SDS、Mist、SA）各自的實作與超參數 | 未找到；只知全部被對齊到 ε=0.03、100 步 |
| SD3（MMDiT）上對應的「中間層」是哪幾層 | 未找到 |
| 100 張圖的清單／索引 | 未找到 |
| 5 條未見過的 prompt 由什麼生成 | 未找到（只說 "we generated five novel, semantically distinct editing prompts per image"） |

---

## 5. 與本專案設定的落差（不是論文的錯，是移植的代價）

| 項目 | 論文 | 本專案 |
|---|---|---|
| 受害模型 | SD3（MMDiT）、HQ-Edit、InstructPix2Pix | `CompVis/stable-diffusion-v1-4`（U-Net），與其餘 baseline 共用 |
| 「中間層」 | 隨架構而定，論文未指名 | U-Net 的 `mid_block`（§4.1） |
| 編輯管線 | 指令式編輯（IP2P／HQ-Edit）與 SD3 | `sd.sdedit`，strength 0.8（`baseline_run.py`） |
| 主指標 | ISR（MLLM 嚴格一致） | 未實作（§4.8） |
| 資料 | 自建 100 張 | 本專案的資料集 |

**「哪一層是語意瓶頸」是隨架構改變的問題。** U-Net 的 `mid_block` 與 MMDiT 的
某一層並沒有對應關係，本檔沒有處理 MMDiT。

## 6. 接進 `archive/anti-purification/scripts/baseline_run.py`

`sifm.py` 的模組 docstring 末尾寫了需要加的三處（import、`CONDITIONS`、
`run_additive` 的 spec 表與 `kw = {"prompt": item["prompt"]}`）。查證時未修改
該檔。本專案的整合入口為 `immunization_baseline.cli.generate_defenses` 的 `sifm` 分支，
條件設定為 `configs/conditions.yaml` 的 `sifm`（`spec: sifm.SPEC_PAPER`）。
