# 指標用法 —— 每個量在問什麼、對齊哪一篇、怎麼讀

> 常設參考，與 `DESIGN.md` 同類。**要引用某個指標之前先讀這份。**
>
> 存在理由：本專案的指標欄位有 74 個，而其中至少三組在問**互不相同的問題**。
> 2026-08-08 實測到「同一批資料，三個判準給出三個相反的結論」
> （`RESULTS_2026-08-08` §11.4），此後任何單一數字的引用都必須說明它屬於哪一層。
>
> 實作在 `scripts/eval_protocols.py`（分層彙總）與 `scripts/class_margin.py`
> （語意判定）。判準的事前宣告寫在各自的 docstring。

---

## 0. 先確認你在問哪一個問題

| 問題 | 用哪一層 | 不能用哪一層 |
|---|---|---|
| 防禦把編輯輸出推開多遠？ | 第 1 層 位移 | — |
| 編輯有沒有失敗？ | 第 2 層 語意（**判定**，不是連續量） | 第 1 層：位移大不等於編輯失敗 |
| 輸出是不是只是變糟了？ | 第 3 層 感知劣化 | — |
| 防禦有沒有用？ | **三層一起看** | 任何單一層 |

**「防禦成功」這個詞在文獻裡指第 1 層，在本專案的問題設定裡指第 2 層。**
這不是用詞不精確，是兩個不同的研究問題共用同一個詞。引用時必須指明。

---

## 1. 第 1 層 · 位移

**比的是**：未防禦的編輯輸出 `y_ref` 對防禦後的編輯輸出 `y_def`。
**在程式裡**：`executors._run_eval` 的 `res.suite.full(y_ctrl, y_def)`，
欄位前綴 `edit_`。

| 欄位 | 免疫成功的方向 | 對齊 |
|---|---|---|
| `edit_psnr` | ↓ | DAYN Table 1、DIA、DiffVax |
| `edit_ssim` | ↓ | DAYN Table 1、DIA、DiffVax |
| `edit_vif_p` | ↓ | DAYN Table 1、PhotoGuard-c |
| `edit_fsim` | ↓ | DAYN Table 1、PhotoGuard-c、DiffVax |
| `edit_lpips` | ↑ | DAYN Table 1、DIA、AdvPaint、PromptFlare |
| `edit_mse` | ↑ | DIA（該篇報 MSE 而非 PSNR，兩者一一對應但尺度不同） |
| `edit_dists` | ↑ | 本專案補充 |

**DAYN（Lo et al., CVPR 2024）的 Table 1 就是前五項**，協定為 L∞ ≤ 0.06、
N=100、150 張 × 20 個隨機種子平均。本專案的 `scripts/run_lo_baseline.py`
重現該協定，`TABLE1` 常數逐字對應。

> **查證狀態**：以上 DAYN 的細節出自本專案讀論文時的紀錄，並由
> `run_lo_baseline.py` 與 `tests/test_lo_protocol.py` 對齊。2026-08-08 嘗試
> 重新對照論文原文失敗（CVF 的 PDF 與 Bytez 皆回 403／取不到內文），故此處
> 標明為專案紀錄而非當次核對。

### 1.1 這一層的陷阱

**位移大不等於編輯失敗。** 2026-08-08 實測：`photoguard_c` 在 `edit_lpips`
上是全部條件最高的 0.391（τ=0.20），但同一批影像的類別判定顯示它把編輯
成功率由 20% 推到 100%（`RESULTS_2026-08-08` §9.6）。**輸出被推得很遠，
而且是往「更像目標類」的方向推。**

**FID 與 Precision／Recall 不要在小樣本上算。** 它們是分布層級指標，
PhotoGuard-c、Mist、AdvPaint 以它們為主指標，但需要數百張才穩定。本專案
現有 3–24 張，算出來的數字沒有意義（`reference/SURVEY` §4.2）。

---

## 2. 第 2 層 · 語意

### 2.1 連續量（既有，變異與訊號同量級）

| 欄位 | 定義 | 方向 | 對齊 |
|---|---|---|---|
| `effect_clip` | `CLIP(y_ref, prompt) − CLIP(y_def, prompt)` | ↑ | PhotoGuard-c、DIA、PromptFlare、DiffVax 的 CLIP 欄 |
| `effect_siglip` | 同式，換 SigLIP | ↑ | 本專案原本的主判定 |
| `effect_abs` | `= effect_siglip` | ↑ | — |

**這一層的連續量在本專案不可用作主判定**，三個獨立的理由：

1. **它比的是兩張不同的圖對同一個 prompt**，所以畫質與風格的劇烈變化會移動
   分數，而那與類別有沒有被改掉無關。實測 dog_03 的未防禦輸出在 seed 1／3
   變成塑膠與木雕質感——分數大幅移動，而那隻狗根本沒變成貓
   （`RESULTS_2026-08-08` §9.2）。
2. **它在平均一個雙峰變數。** 攻擊本身逐 seed 成敗不定，故其變異主要來自
   兩堆（成功／失敗）的混合比例，不是來自防禦（§9.1）。
3. **判準 `mean ≥ 3σ` 在它上面不可達。** 防禦效果的上限就是攻擊本身的效果量，
   而 3σ 在三張圖中有兩張超過該上限，即完美防禦也通不過（§8.2）。
   增加 seed 數無效：n 縮小的是 SEM，判準用的是 σ。

### 2.2 判定：類別 margin（現行）

```
margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)
編輯成功 := margin(y) > 0
```

**同一張圖對兩個 prompt**，故畫質與風格的變化同時影響兩項而抵消，剩下的是
類別訊息。實作在 `scripts/class_margin.py`，原語是
`MetricSuite.semantic_multi`（`semantic` 委派給它，兩者共用同一份前處理，
數字不可能分歧，有測試釘住）。

| 來源 | 取法 |
|---|---|
| 目標類 | 逐格 `metrics_*.json` 的 `prompt` |
| 原類 | 同檔的 `group`（bird／cat／dog），組成 `"a {group}"` |

**校準已驗證**：三張原圖的 margin 是 −0.0713（bird_03）、−0.0575（cat_02）、
−0.0613（dog_03），全部明顯為負，即原圖都被正確判成原類，**決策邊界在零是
對的**。與人眼判定在 dog_03 的四個 seed 上逐格一致。

對齊 SIFM（arXiv:2512.14320）ISR 的**語意分支**。

### 2.3 這一層的陷阱

**成功率飽和的影像不提供訊息。** cat_02 在全部條件（含未防禦對照）都是 5/5
成功，該影像對任何比較都沒有鑑別力（§9.6）。

**對照側是分母，必須逐算子重取。** 淨化算子同時作用於兩側，故未防禦的失敗率
逐算子不同（identity 5/15、blur_2 9/15）。拿 identity 的對照去比 blur 的
防禦側會把「淨化本身的效果」算進防禦的帳上。

**`purify_kind` 在對照側不存在。** `control/` 不寫 `metrics_*.json`，其
`purify_kind` 只能落回目錄名。以 `purify_kind == "identity"` 篩選會**靜默漏掉
整個對照側**，而對照側是分母。一律用目錄名 `purify_dir`。

---

## 3. 第 3 層 · 感知劣化

| 欄位 | 定義 | 讀法 | 對齊 |
|---|---|---|---|
| `edit_niqe_a` / `edit_niqe_b` | 未防禦／防禦後輸出的 NIQE | 越低越好 | — |
| ΔNIQE（導出） | `edit_niqe_b − edit_niqe_a` | **正值＝防禦後更糟** | SIFM 的 ISR 劣化分支 |
| `edit_acutance_ratio` | 銳利度比 | < 1 代表變模糊 | 本專案 |
| `edit_rms` | 兩側輸出的 RMS 差 | — | 本專案 |
| Aesthetic Score、PickScore | — | **本專案算不出來**（缺權重） | PromptFlare |

**這一層是代價，不是成果。** 它存在的理由是讓「靠劣化撐起來的免疫」暴露出來
而不是被計入成績。

---

## 4. 聯集判準（ISR）與它的門檻敏感性

SIFM 的 ISR 定義為

> the proportion of edits where immunization induces **either** semantic failure
> relative to the prompt **or** significant perceptual degradations

原文由 **MLLM** 判定。本專案沒有那個 judge，故：

```
語意失敗 := margin(y_def) ≤ 0
劣化     := ΔNIQE ≥ θ
ISR(θ)   := 語意失敗 ∪ 劣化
```

**這不是原文的重現，而且必須掃描 θ 而不是取一點。** 2026-08-08 實測
（τ=0.35，`RESULTS_2026-08-08` §11.3）：`mist` 的 ISR 由 θ=0.5 的 **13/15**
一路掉到 θ=3 的 5/15，而未防禦對照恆為 5/15。**那 9 格增量全部來自劣化分支。**
取單一門檻等於事後選一個對自己有利的值。

**對照側的 ISR 逐 θ 不變是正確的**：它就是 `y_ref` 本身，ΔNIQE 依定義為零。

---

## 5. 保真／隱蔽（防禦圖對原圖，不是編輯輸出）

前綴 `fid_`（訓練與段 2 用）與 `defimg_`（段 3 記錄）。兩者比的都是
**防禦圖對原圖**，與上面三層（比編輯輸出）不同軸，不可混用。

| 欄位 | 用途 |
|---|---|
| `fid_lpips` | 共同失真約束，段 2 把每個條件縮放到 `τ` |
| `fid_lpips_rel` | 相對 `x_base = G(x; φ=0)` 的 LPIPS。**四道 hinge 用的是這個**，不是絕對值 |
| `fid_acut` / `fid_chroma` | 兩道由 `τ` 依比例導出的 hinge（`0.8 τ`、`16 τ`） |
| `fid_psnr`／`ssim`／`linf`／`dists` | 逐條件報全部，不挑選（第三層主張的要求） |
| `tau_achieved`／`scale_k` | 段 2 的達成值與縮放倍率 |
| `disp_mean_px`／`disp_max_px` | 位移場的幅度。`11.31 = 8 × √2` 即 `--warp-max-disp 8.0` 的對角界 |

**hinge 的對象是 `x_base` 而非原圖**，理由與 2026-08-08 的人眼判讀一致：
N3 的絕對 LPIPS 是 0.197–0.205，但其中大部分是 VAE 來回的重建下限，
那是 `G(x; φ=0)` 本來就有的模糊，不是防禦加上去的。使用者判讀 N3「非常不
明顯」貼合相對量 0.060–0.083 而不是絕對量 0.20（`RESULTS_2026-08-08` §6.6）。

---

## 6. 抗淨化：`retention` 與它為什麼一直不可用

```
retention        = effect(某淨化) / effect(identity)
retention_usable = (mean ≥ 3σ)        # 分母跨 seed 的均值與標準差
```

實作在 `executors._fill_retention`；判定腳本 `scripts/purify_advantage.py`
排除 `retention_usable` 為 false 的列。

**它在本專案至今五次拒絕出表**，原因是分母（identity 那一格的效果）落在
雜訊裡，而 §2.1 第 3 點說明該閘在兩張圖上不可達。**分母為零或負時，比值不可
解讀**——先驗批次曾出現 −43、−98 這種值，`retention_usable` 這一欄的存在就是
為了在資料層擋下它。

要談抗淨化，在 identity 上先有可測的效果是**前提**。目前沒有，故不是
「抗淨化不好」，是**這個問題還問不出來**。

---

## 7. 算不出來的指標，以及為什麼

`runs/v14r_protocols/unavailable.md` 是逐批的版本；此處是通則。
**不算不等於忽略——省略一個欄位會被讀成疏漏。**

| 指標 | 出自 | 為什麼 |
|---|---|---|
| FID、Precision／Recall | PhotoGuard-c、Mist、AdvPaint 的主指標 | 分布層級指標，需數百張 |
| Aesthetic Score、PickScore | PromptFlare | 需額外模型權重 |
| 背景保留（mask 隔離的四項） | DIA | 依賴 PIE-Bench 的編輯 mask；img2img 無對應物 |
| ISR（原文形式） | SIFM | 判定由 MLLM 做 |
| 人類排名 | DiffVax（67 名受試者） | 需受試者招募 |
| `cnn_denoise_substitute` 的全部格 | 本專案的淨化算子之一 | **缺權重**，五個算子中有一個從頭到尾沒有資料 |

---

## 8. 引用時的三條規則

1. **講清楚是哪一層。** 「防禦有效」在第 1 層與第 2 層是兩個不同的陳述。
2. **三層一起報。** 挑一層等於挑一個對自己有利的定義；第 3 層特別不可省，
   否則「靠劣化撐起來的免疫」會被計入成績。
3. **代理量要標明。** ISR 的劣化分支是 ΔNIQE 代理而非 MLLM；DAYN 的 Table 1
   細節是專案紀錄而非當次核對。標明來源狀態，不要讓讀者以為每一項都等價。
