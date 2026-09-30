# DANP 論文全文查證

用途：供 baseline 嚴格重現使用。本文件所有數值與公式均標註來源；查不到者一律列於
「未找到的項目」一節，**不作推斷、不補值**。

實作在 `src/immunization_baseline/attacks/danp.py`（`SPEC_PAPER`），測試在 `archive/anti-purification/tests/test_danp.py`。

## 查證所用來源

| 代號 | 內容 | 位置 |
|---|---|---|
| `[DANP-HTML]` | arXiv:2512.14333v1 的 HTML 全文（含 Algorithm 1、Eq. 1–13、Table I–VI 的數值） | https://arxiv.org/html/2512.14333v1 |
| `[DANP-ABS]` | arXiv abs 頁 | https://arxiv.org/abs/2512.14333 |

引用格式為「§節次／式編號／表編號」。

---

## 0. 官方程式碼查證

查證範圍為下表所列來源（arXiv:2512.14333v1 全文、arXiv abs 頁及網路搜尋）；在這些來源中未找到官方程式。

| 查過的地方 | 結果 |
|---|---|
| 論文全文出現 `github`、`code is available`、`our code`、`release the code`、`anonymous.4open` | 只有 arXiv HTML 介面自己的「Report GitHub Issue」按鈕，正文一次都沒有 |
| 論文有無附錄／Supplementary | **無**。全文止於 §VI Conclusion 與參考文獻 |
| arXiv abs 頁的 ancillary files | **無**。「Code, Data and Media」區塊只有 alphaXiv／CatalyzeX 等第三方 widget，沒有作者提供的連結 |
| 網路搜尋 `DANP "Dual Attention Guided Defense Against Malicious Edits" code github` | 只回到 arXiv 本身與幾個論文彙整站，無 repo |

作者：Jie Zhang、Shiguang Shan、Xilin Chen（中科院計算所）、Shuai Dong（中國地質大學）。

**本 baseline 依論文 arXiv:2512.14333v1 重建；上述查證範圍內沒有可逐行比對的原始碼。** 因此
`BaselineSpec.discrepancy_note` 裡「論文與程式不一致」那一類的落差在本篇不存在，
取而代之的是「論文內部的落差」與「論文未給而由本檔決定」兩類。

---

## 1. 方法：論文寫什麼／本檔實作什麼

### 1.1 注意力的定義與聚合（§III-B，Eq. 3、Eq. 4）

論文：

> `A_l(x, φ(c)) = softmax(Q_l K_lᵀ / √d_k)`
> `Att(x, φ(c)) = (1/L) Σ_{l=1}^{L} Upsample(A_l(x, φ(c)))`
>
> where `L` is the number of U-Net blocks.

本檔：`DANPAttnProcessor` 以 `attn.get_attention_scores(query, key, attention_mask)`
明確計算 softmax 機率並交給 `DANPAttnController`，聚合在
`DANPAttnController.aggregate`。

- **不能沿用 `src/immunization_baseline/attacks/promptflare.py` 的擷取層。** 那個 processor 走 SDPA
  融合核，注意力機率從未被實體化，它記的是 `attn2` 模組經 `to_out` 之後的輸出
  `A·V·W_out`。Eq. 11 的被優化量是 `A` 本身，兩者不是同一個東西。
  `archive/anti-purification/tests/test_danp.py::test_擷取到的是後softmax機率而不是注意力輸出` 以
  「每一列對 token 軸加總為 1」釘住這一點。
- 只在 `attn2`（cross-attention）上記錄。Eq. 3 的 `K_l` 來自文字嵌入，
  self-attention 不在定義內。

### 1.2 動態門檻（§IV-B，Eq. 7–10）

論文：

> First, we normalize the aggregated attention map to a range of `[0,1]`.
> Second, we apply a dynamic thresholding technique, specifically Kapur's entropy method.
> …given a single-channel image with `L` intensity levels, its histogram is normalized
> to form a probability distribution `p_0 … p_{L-1}`. For a threshold `τ`, pixels are
> split into Class 0 `[0..τ]` and Class 1 `[τ+1..L-1]`…
>
> `H₀(τ) = −Σ_{i=0}^{τ} (p_i/P₀(τ))·log(p_i/P₀(τ))`（Eq. 7）
> `H₁(τ) = −Σ_{i=τ+1}^{L-1} (p_i/P₁(τ))·log(p_i/P₁(τ))`（Eq. 8）
> `τ* = argmax_{0 ≤ τ < L} [H₀(τ) + H₁(τ)]`（Eq. 9）
> `M_t = 𝟙( N(Att(x_t^imu, φ(c))) > τ_t )`（Eq. 10）
>
> In our experiment, this procedure is applied to the normalized attention map at
> each timestep `t` to compute the adaptive threshold `τ_t`.

本檔：`normalise_attention`（min-max）＋ `kapur_threshold_index`（Eq. 7–9）＋
`kapur_mask`（Eq. 10）。`L` = `PAPER_KAPUR_BINS` = 128。門檻每次 `loss_fn`
（即每個 timestep）重算，與「at each timestep `t`」一致。

三件論文沒有處理、本檔必須定義的邊界：

| 情形 | 論文 | 本檔 | 釘住它的測試 |
|---|---|---|---|
| `max == min`（常數圖），min-max 的分母為 0 | 未處理 | 回傳全 0，而不是 nan | `test_常數輸入的正規化回傳全零而不是除以零` |
| `P₀(τ) = 0` 或 `P₁(τ) = 0`，Eq. 7／8 的分母為 0 | 未處理（Eq. 9 的範圍寫 `0 ≤ τ < L`，含 `τ = L−1`，該處 class 1 必為空） | 這些 τ 不合法，argmax 跳過 | `test_質量全落在最高一格時同樣不切` |
| 所有 τ 都不合法（質量集中在同一格） | 未處理 | `τ* = L−1`，遮罩全 0，DAA 退化成「整張抬高」 | `test_常數輸入的門檻讓遮罩全為零`、`test_遮罩全為零時DAA退化成整張抬高` |

另有一處刻意與 Eq. 10 的字面不同：比較以**級距索引**進行（`bin(v) > τ*`），
而非 `N(Att) > τ_t`。兩者只在恰好落在級距邊界 `τ_t = (τ*+1)/L` 的元素上不同
（字面讀法歸 class 0，直方圖歸 class 1）。本檔取與直方圖分割一致的那一種，
並仍回傳 `τ_t` 供報表引用（`test_門檻值等於級距的上邊界`）。

### 1.3 DAA 損失（§IV-B，Eq. 11）

論文：

> `L_DAA = ‖Att(x_t^imu, φ(c)) ⊙ M_t‖_F²  −  λ_daa·‖Att(x_t^imu, φ(c)) ⊙ (1 − M_t)‖_F²`
>
> By minimizing this loss, we actively divert the model's attention from relevant
> regions to erroneous ones.

本檔：`daa_loss(att, mask, lambda_daa)`，逐字。

- **注意 Eq. 10 用的是正規化後的 `N(Att)`，Eq. 11 用的是原始的 `Att`。** 本檔照此
  實作：遮罩由正規化圖決定，損失作用在原始圖上。
- 兩個方向的符號：`∂L/∂Att = 2·Att⊙M − 2λ·Att⊙(1−M)`，最小化即壓低 `M=1` 處、
  抬高 `M=0` 處。由 `test_DAA對相關區的梯度為正對不相關區為負` 與
  `test_沿負梯度走一步會壓低相關區並抬高不相關區` 釘住。符號寫反不會有症狀
  （輸出仍是一張合理的防禦圖），只是方法變成 SA 的反面。
- `M_t` 是指示函數，本檔明確 `detach()`。論文未討論（幾乎處處梯度為零，
  故這不是選擇）。

### 1.4 NBA 損失（§IV-C，Eq. 12）

論文：

> `L_NBA = −‖ε_θ(x_t, t, c) − ε_θ(x_t^imu, t, c)‖₂²`

本檔：`nba_loss(eps_clean, eps_imu)`，逐字。

`ε_θ(x_t, t, c)` 那一支對 `x_imu` 的梯度恆為零（Algorithm 1 第 7 行的 `x_t` 只依賴
`x₀`），故本檔在 `torch.no_grad()` 下算。**這不是簡化，是同一個梯度。**

### 1.5 總目標與最佳化（§IV-D，Eq. 13、Algorithm 1）

論文 Eq. 13：`L_total = L_DAA + λ_nba·L_NBA`。

Algorithm 1 逐行對應到 `BaselineSpec` 與 `src/immunization_baseline/attacks/pgd.py::run_pgd` 的哪裡：

| Algorithm 1 | 論文的字 | 本檔 |
|---|---|---|
| 1 | `δ ← 0` | `init_rule="none"` |
| 2 | `for n = 1 to N` | `steps=100`（§V-A） |
| 5 | `for t in 𝒯` | `grad_reps=10`，`loss_fn` 每次取 `ctx.next_timestep()` |
| 6 | `ε ~ N(0, I)` | 每次 `loss_fn` 重抽，兩條分支共用同一個 ε |
| 7 | `x_t ← √ᾱ_t·x₀ + √(1−ᾱ_t)·ε` | `z_clean`（latent，見 §2） |
| 8 | `x_t^imu ← √ᾱ_t·x_imu + √(1−ᾱ_t)·ε` | `z_imu` |
| 9 | `L_total ← L_DAA + λ_nba·L_NBA` | `loss_fn` 的回傳值 |
| 10–12 | `g ← Σ ∇ / \|𝒯\|` | `run_pgd` 對 `grad_reps` 個梯度取平均 |
| 13 | `δ ← δ − α·sign(g_total)` | `objective="minimize"`、`update_rule="sign"` |
| 14 | `δ ← clip(δ, −γ, γ)` | `norm="linf"`、`eps=0.03` |

`test_spec逐欄對應Algorithm1`、`test_一次迭代走訪每個timestep恰好一次`、
`test_run_pgd在替身上跑得完且擾動落在預算內` 釘住這張表。

---

## 2. 論文給的超參數

| 項目 | 值 | 出處（逐字） |
|---|---|---|
| 約束 | `L∞`、γ = 0.03 | §V-A General Settings：「all methods are constrained by an `L∞` perturbation budget of γ=0.03」 |
| 步數 N | 100 | 同上：「optimized for N=100 iterations」 |
| \|𝒯\| | 10 | 同上：「we uniformly sample a set of \|𝒯\|=10 timesteps to attack」 |
| λ_daa | 1.0 | §V-A Settings of DANP |
| λ_nba | 1.0 | 同上 |
| Kapur 級距數 L | 128 | 同上；§V-F Table VI 掃過 32／64／128／256／512，128 的 LPIPS 0.4861 最佳 |
| 受害模型 | SD v1-4、HQ-Edit、InstructPix2pix | §V-A Target Models and Dataset |
| 資料 | InstructPix2Pix-clip-filtered 抽 200 張，逐張用資料集原本的編輯指令 | 同上 |
| 未見 prompt 測試 | 每張手寫 5 個新指令，共 1000 組 | §V-A Evaluation on Unseen Prompts |
| 讀數 | PSNR／SSIM／FSIM／VIFp（越低越好）、LPIPS（越高越好） | §V-B |
| 對照 | ACE、EditShield、Mist、PGD、PGE、SDS、SA | §V-C／表 I–III |

---

## 3. 未找到的項目

以下每一項都查過 `[DANP-HTML]` 全文（含 Algorithm 1 的 Input 列、圖 1–3 的說明
文字、表 I–VI 的欄位與註腳）與 `[DANP-ABS]`。**論文沒寫，且無官方程式可查。**
本檔各給了一個值，全部記在 `SPEC_PAPER.modification_note` 與
`SPEC_PAPER.extras` 裡，**不得讀成論文設定**。

### 3.1 步長 α —— 未找到

Algorithm 1 的 Input 列出 `step size α`，§III-C Eq. 6 說明 α 是步長，
但**全文沒有任何一處給出它的數值**。§V-A 的三段設定（Target Models、General
Settings、Settings of DANP）都沒有，§V-F 的超參數分析只掃 λ 與 L。

本檔取 `1/255`（`RECONSTRUCTED_STEP_SIZE`）。理由：本 repo 已逐行查證過的
`L∞` baseline（Mist、DIA、PromptFlare 在其 `[-1,1]` 尺度上、DiffusionGuard）
步長全部是 `1/255`。在 γ=0.03、N=100 下 `100 × 1/255 = 0.392 ≫ 0.03`，預算可達
（`test_一百步的sign更新走得滿預算`）。

**這是本檔挑的。** `γ/N = 3e-4`（剛好走滿一次）與 `2γ/N = 6e-4`（PGD 慣例）
同樣說得通，而三者給出不同的解。要換的話改這一個常數即可。

### 3.2 γ = 0.03 的值域 —— 未找到，但可由 Table IV 反推

論文只寫 γ = 0.03，未說是在 `[0,1]` 還是 `[-1,1]` 上量的，而兩者差一倍
（`docs/reference/BASELINE_PROVENANCE.md` 的預算總表記過同一個陷阱在 Mist、
DIA、PromptFlare 上各以不同形式出現）。

§V-D Table IV 給 DANP 在 InstructPix2pix 上免疫圖對原圖的 **PSNR = 34.76 dB**。
反推 rms = `10^(−34.76/20)` = **0.018281**（`[0,1]` 尺度）。

- `[0,1]` 讀法：上界 0.03，`0.018281 < 0.03`，相容。
- `[-1,1]` 讀法：`[0,1]` 的等價上界是 0.015，而任何滿足 `‖δ‖∞ ≤ 0.015` 的擾動
  其 rms 必 ≤ 0.015，`0.018281 > 0.015` 矛盾。

故 `DANP_RANGE = ValueRange(0.0, 1.0, …)`。`test_值域的反推與論文的PSNR一致`
釘住這個算術。

**0.018281 是反推的數，不是量到的數**（`BASELINE_PROVENANCE.md` 規則 4）：
它由一批影像的**算術平均 PSNR** 反推，得到的是 rms 的**幾何平均**，引用時標星號。
另：論文未說明 sign-PGD 為何未走滿預算（飽和時 rms 應接近 0.03，對應 30.46 dB）。

### 3.3 注意力取自哪些層、哪些 head —— 部分未找到

- **層**：Eq. 4 只寫「`L` is the number of U-Net blocks」，沒有子集、沒有解析度
  白名單（對比 PromptFlare 寫死的 `loss_depth = [1024, 256, 64]`）。本檔取**全部**
  `attn2`。Eq. 3 的 `K_l` 來自文字嵌入，故 `attn1` 不在定義內，這一點不是選擇。
- **head**：**未找到**。Eq. 3 把 `A_l` 寫成單一矩陣，沒有 head 軸；論文全文未出現
  `head`。本檔對 head 取平均——那是唯一不引入新權重的化約。

### 3.4 Eq. 4 的 `Upsample` 目標解析度與插值方式 —— 未找到

論文只寫 `Upsample(A_l)`。本檔取**該次前向觀察到的最大** cross-attention 網格
（SD v1.x／512² 下是 64×64，即 latent 解析度），雙線性、`align_corners=False`。
理由：那是不丟失任何一層空間資訊的最小共同網格。
`test_聚合上採樣到最大網格並對層取平均` 釘住聚合的算術。

非正方形的 token 數直接 `NotImplementedError`（`test_非正方形的token數直接拋出`）：
猜錯網格形狀不會有症狀。

### 3.5 token 軸怎麼處理 —— 未找到

Eq. 10 的 `𝟙(·)` 與 Eq. 11 的 `⊙` 都是逐元素運算，故 `M_t` 與 `Att` 必須同形，
而 `Att` 的形狀是 `(HW, 77)`。本檔照這個維度約束逐元素處理，**涵蓋全部 77 個
token（含 BOS、EOT、PAD）**。

論文沒有任何一處提到要挑 token（圖 2／圖 3 展示的是單一詞 `sunset`／`dragon` 的
空間切片，那與逐元素遮罩的某一行相容，但不足以判定作者是否只用了那些 token）。
`N(·)` 的 min-max 同理取整張聚合圖的全域 min/max——論文寫「normalize the
aggregated attention map to a range of [0,1]」，未寫逐 token 或逐層。

### 3.6 latent 還是像素空間 —— 未找到

Algorithm 1 與 Eq. 1／Eq. 12 通篇寫 `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`，**全文未提 VAE、
未提 latent**。但三個受害模型（SD v1-4、HQ-Edit、InstructPix2pix）都是 latent
diffusion，`ε_θ` 的輸入只能是 latent，Eq. 3 的 `Q_l` 也來自 UNet 的影像特徵。

本檔在 latent 上加噪，`z₀ = E(x)` 取後驗均值（`sd.encode_image`，確定性）。
論文未說是取 `sample()` 還是 `mode()`／`mean()`；取確定性的那一個是中性選擇。

### 3.7 𝒯 是固定格點還是每步重抽 —— 論文可判定，但寫法含糊

§V-A 寫「we uniformly sample a set of \|𝒯\|=10 timesteps」，字面像是抽樣；但
Algorithm 1 把 `𝒯` 列在 **Input**，迴圈內沒有重抽。本檔取固定等距格點
`linspace(0, T−1, 10).round()`（`paper_timesteps`）。

論文未給格點的端點（是否含 t=0、是否含 t=999）。本檔取閉區間 `[0, T−1]`。

### 3.8 CFG —— 未找到

Eq. 12 只寫 `ε_θ(·, t, c)`，全文未出現 guidance scale。本檔不用 CFG，
每個分支一次條件前向。

### 3.9 NBA 的尺度 —— 未找到係數

§V-F 寫：

> Notably, the DAA and NBA loss terms differ significantly in their order of magnitude.
> To ensure a fair and balanced ablation, we scale the NBA loss to a comparable level
> with the DAA loss.

**沒有給縮放係數。** 而且這句描述的是 §V-F 的 λ 掃描（0.5–8），不是主設定。
本檔取 `nba_scale = 1.0`，即照 Eq. 13 字面。**後果是：λ_nba = 1.0 這個主設定
實際上作用在一個未知的尺度上**，兩個項的相對權重不可能與論文一致。
這一項記在 `discrepancy_note`，`test_NBA的縮放係數是本檔的決定且為一` 釘住它與
`modification_note` 的連動。

---

## 4. 論文內部的落差

| 落差 | 位置 | 本檔取哪一個 |
|---|---|---|
| 更新式的符號：§III-C Eq. 6 是 `δ_{k+1} = Π(δ_k + α·g_k)`（上升），Algorithm 1 第 13 行是 `δ ← δ − α·sign(g)`（下降） | §III-C vs Algorithm 1 | Algorithm 1。Eq. 6 描述的是 §III-C 那個一般化的**最大化**問題（Eq. 5），而 Eq. 11／Eq. 12 已把符號寫進損失，兩者方向一致，不是矛盾 |
| Eq. 9 的 τ 範圍 `0 ≤ τ < L` 包含 `τ = L−1`，但該處 class 1 必為空、Eq. 8 的分母為 0 | Eq. 8／Eq. 9 | 排除 `τ = L−1` 與所有使任一類為空的 τ，見 §1.2 |
| Table IV 的 PSNR 34.76 dB 對應 rms 0.018281*，遠低於 γ=0.03 飽和時的 0.03（30.46 dB）。sign-PGD 在 N=100 下通常會走滿預算 | §V-D Table IV vs §V-A | 兩者並存記於 `discrepancy_note`。可能的解釋（步長很小、或 γ 另有值域讀法）論文都沒寫，不推斷 |

---

## 5. 本專案這一側的移植

| 項目 | 內容 |
|---|---|
| 值域 | `[0,1]`，與本專案張量介面相同，`run_pgd` 的頭尾轉換是恆等 |
| 9 通道（inpainting）權重 | 後 5 個通道取全 1 遮罩，與 `mist.loss_fn`、`dia.loss_fn` 同一處置：論文的 `ε_θ` 沒有影像條件 |
| prompt | **必填**。§V-A 用的是資料集逐張附帶的編輯指令，本檔沒有預設值，`prepare(prompt=None)` 直接 `NotImplementedError`（`test_沒有給prompt就拒絕執行`） |
| processor 還原 | `DANPContext.close()` 把 `attn2` 的 processor 換回去，`run_pgd` 結束時呼叫。不還原會污染共用同一個 `SDWrapper` 的後續實驗（`test_close還原原本的processor`） |
| 隨機性 | `prepare(seed=...)` 建一個顯式 generator，ε 由它抽。論文未提 seed |
| 未接進 `REGISTRY` | `archive/anti-purification/tests/test_baselines.py::test_五篇的值域全部是負一到一` 會逐一檢查 `REGISTRY` 的值域，而 DANP 是 `[0,1]`。接線要加哪幾行寫在 `src/immunization_baseline/attacks/danp.py` 模組 docstring 末段 |

---

## 6. 可引用的對照數字，以及它們的協定

引用任何一個都要連協定一起引用（`BASELINE_PROVENANCE.md` 規則 1）。

| 數字 | 量的是什麼 | 出處 | 協定 |
|---|---|---|---|
| PSNR 34.76、SSIM 0.8903 | **免疫圖對原圖**（失真，不是防禦效果） | `[DANP-HTML]` §V-D Table IV，InstructPix2pix 欄 | γ=0.03、N=100、\|𝒯\|=10；200 張 InstructPix2Pix-clip-filtered；作者自跑 |
| LPIPS 0.4861 | 免疫圖的編輯結果 對 原圖的編輯結果（防禦效果，越高越好） | §V-F、§V-E | 同上，原生 prompt，InstructPix2pix |
| LPIPS 0.4651 / 0.4551 | 同上，未見 prompt（DANP／w-o-NBA） | §V-E | 每張 5 個手寫新指令，共 1000 組 |
| LPIPS 0.4508 / 0.4817 | w/o DAA（只有 NBA）／w/o NBA（只有 DAA），原生 prompt | §V-E Table V | 同上 |
| LPIPS 0.7165 / 0.6849 | DANP 在 **HQ-Edit** 上，原生／未見 prompt | §V-C Table III | 同上但受害模型是 HQ-Edit |
| PSNR 47.02（ED）、34.90（ACE）、34.44（SDS） | 免疫圖對原圖 | §V-D Table IV | **這些是 DANP 作者重跑的 baseline 欄，不是那幾篇自己的論文值** |

最後一列是 `docs/reference/BASELINE_PROVENANCE.md` 規則 2 的同一個陷阱：
一張表裡的「論文值」可能指七篇不同的論文，而實際上全部是 DANP 作者在
γ=0.03／N=100 下重跑的結果。
