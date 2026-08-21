# 對照組

移植別人的方法一律**逐行對照公開原始碼**，不照摘要寫。查不到的參數設為必填
或標 `modified_from_paper`，不填看起來合理的預設值。每一個改動都要能在報表上
被看見。

## 頻域方法（現行比較對象）

| 方法 | 是什麼 | 程式 | 重現狀態 |
|---|---|---|---|
| **DCT-Shield** | 在**量化後**的 JPEG DCT 係數上加 δ，`ε ≥ 1` 個量化階 | `src/baselines/dct_shield.py` | **成功**，見 [RESULTS.md](RESULTS.md#dct-shield-的重現) |
| **AdvDrop** | 對 DCT 係數施加**可學習的量化**，用丟資訊來攻擊 | `src/baselines/advdrop.py` | **部分**，九量中六 |
| **DJSMA** | DCT 反對角帶上的貪婪 JSMA，一次改一個係數 ±1 | `src/baselines/dct_watermark.py` | 診斷通過，正式批未跑 |
| **BlurGuard** | 對已求得的對抗雜訊做逐區域自適應高斯模糊，重塑頻譜 | `src/baselines/blurguard.py` | **未跑**（需要 SAM） |
| **DiffusionGuard** | 抗淨化訴求的防護 | `src/baselines/diffusionguard.py` | **未跑**，且是移植非重現 |

### DCT-Shield

Bala et al., ICCV 2025 Highlight（arXiv:2504.17894）。官方 repo 是空的，
本檔由論文與補充材料 Algorithm 1 逐行重寫。

定案超參數（正文 §5.4）：`Q_alg = 0.95`、`ε = 1`、`γ = 0.1`、`N = 1000`。
損失是 `‖E(x')‖₂`——**跑 baseline 時必須用它自己的損失**，換成本專案的
encoder-target 就不是那一篇了。

四件必須寫進報表的事：

1. **`ε ≥ 1` 是抗 JPEG 的必要條件，不是調參建議。** 擾動必須至少造成一個
   量化級的改變，否則攻擊方以相同品質重壓時會被四捨五入回原值。
2. **抗 JPEG 的保證是單向的。** `Q_alg = q` 只擋得住品質 `q' ≥ q` 的壓縮。
   要擋更重的壓縮就得先把自己的圖壓糊。
3. **`δ = 0` 時輸出不是原圖**，而是 `Q_alg` 品質的壓縮圖（有失真地板）。
4. **免疫影像不能存成 JPEG**，δ 是連續值加在整數上，存 JPEG 會被捨掉。

**Y-only 變體**（`dct_shield_y`，論文 §6.3 的抗 JPEG 配置）在 JPEG 與
GrIDPure 上遠強於 base 變體。頭對頭表上**不可只放 base 變體**。

### AdvDrop

Duan et al., ICCV 2021（arXiv:2108.09034）。與本方法同屬非加性頻域，但方向
相反：它**移除**資訊，本方法一點都不移除。

**論文與官方程式碼不一致**，兩種模式都保留在 `AdvDropSpec.bounds()`：

| | 論文正文 §3.1 | 官方 `infod_sample.py` |
|---|---|---|
| 初始量化表 | `q_init = 1` | `q_size = 10` |
| 可動區間 | `[1, 1+ε]`，ε ∈ {20,60,100} | `[5, 10]` |
| 方向 | 由低往高 | 由高往低 |

**論文寫的更新規則達不到它報的數字**：式 (7) 隱含步長 1，但 50 步 × 1 走不到
ε=100 的區間。步長 4–8 才對得上。必有一個未載的細節。

### DJSMA

Chen et al., The Imaging Science Journal 2026
（doi:10.1080/13682199.2026.2644653）。無公開程式碼，由掃描 PDF 逐頁判讀後
依 Algorithm 1 與式 (7)–(9) 實作。

論文把 8×8 DCT 的反對角帶切成兩個互不重疊的用途（編號 1-based，位置 (i,j)
屬於第 `i+j+1` 條）：

| 頻帶 | 用途 | 理由 |
|---|---|---|
| 第 6–8 條 | 隱形浮水印 | 中偏高頻，對量化穩健、視覺影響小 |
| 第 3–5 條 | 對抗擾動 | 對網路預測影響強，比低頻不顯眼 |

演算法是**貪婪 JSMA 而非 PGD**：一次只改一個係數 ±1，`τ` 限制 l0、`μ` 限制
l∞，定向成功即停。定案 `τ=1500`、`μ=1`，評測時再壓一次 JPEG Q=75。

**浮水印那兩個階段未實作**（J-UNIWARD ＋ STC ＋ RS 是一整套隱寫工具鏈，本專案
不需要復原訊息）。後果是我們的 PSNR/SSIM 參照原圖，論文參照已嵌浮水印的影像，
**參照點不同**。

## 像素加性方法（已實作，本輪擱置）

程式在 `src/baselines/`，驅動是 `scripts/apa_baseline.py`。結果資料保留在
`runs/external_baselines_24img`、`runs/human_threshold_comparison`、
`runs/sdedit_mainline`、`runs/baseline_apa`。

| 方法 | 場景 | 程式 | 備註 |
|---|---|---|---|
| **PhotoGuard**（`photoguard_c`） | img2img ＋ inpainting | `photoguard.py` | 原生預算下與本方法打平 |
| **Mist** | 風格模仿 | `mist.py` | **預算未對齊，待重測** |
| **DIA**（`dia_r`） | inversion-based editing | `dia.py` | PT／R 兩變體 |
| **AdvPaint** | inpainting | `advpaint.py` | 改寫為全圖 mask |
| **PromptFlare** | inpainting | `promptflare.py` | 改寫為全圖 mask |
| **APA**（`apa_weak`） | 本專案的弱 baseline | `apa_stage1.py`／`apa_native_stage2.py` | 完全原生，只把 reward 換成 targeted output |

擱置的理由是研究主軸已收斂到頻域方法；程式與資料都保留，隨時可重啟。

## 移植時的規則

1. **損失不可換。** 跑 baseline 時用它自己的損失。要做參數化消融時才換成
   本專案共用的 encoder-target，並在報表上分開列。
2. **不可微的步驟走直通估計，但前向值必須逐位元等於真實算子。** 否則量到的
   失真與最終存檔的不一致。
3. **每個未載的參數都要出現在 CSV 的欄位裡。** 不是註解，是欄位。
4. **改寫必須標 `modified_from_paper` 並寫明改了什麼**，`__post_init__` 應該
   在沒寫明時直接拒絕建構。
