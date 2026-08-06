# 步驟一：文獻與原始碼調查

> 2026-08-05。範圍：使用者於本日重定義的研究範圍——保留 APA 移植與位移場兩組非加性方法，
> 加性方法由 6 篇 baseline 擔任對照，威脅模型為全圖 SDEdit，SD 升級至仍具 cross-attention
> 的最高版本。

---

## 1. 研究主張（本輪定案）

| 項 | 內容 |
|---|---|
| **主要主張** | 非加性方法的**抗淨化能力勝過**加性方法 |
| 次要條件 | 抗編輯能力與加性方法**持平或小輸**即可 |
| 第三條件 | 保真度受控 |
| 非加性方法 A | **APA 移植**（diffusion-based，生成路徑） |
| 非加性方法 B | **位移場**（site S，像素路徑） |
| 加性對照 | 6 篇 baseline，不再自行實作加性方法 |
| 威脅模型 | 全圖 SDEdit（inpainting 列為保留方向，不進本輪） |
| 攻擊方模型 | 仍具 cross-attention 的最高 SD 版本 |

---

## 2. Baseline 調查表

### 2.1 逐篇規格

| # | 方法 | 出處 | 威脅模型 | 攻擊方模型 | 失真預算 | 最佳化步數 |
|---|---|---|---|---|---|---|
| 1 | **PhotoGuard-c** | ICML 2023, [arXiv:2302.06588](https://arxiv.org/abs/2302.06588) | img2img + inpainting | SD v1.5 | ℓ∞ ≤ 16/255 | 200 |
| 2 | **Mist** | [arXiv:2305.12683](https://arxiv.org/abs/2305.12683) | 風格模仿（微調） | SD v1.4 | ℓ∞ 17/255，步長 1/255 | 100 |
| 3 | **DIA** | [arXiv:2510.00778](https://arxiv.org/abs/2510.00778) | inversion-based editing | SD v1.4 | ε = 0.05，掃 {0.025, 0.05, 0.075, 0.1} | 20（PhotoGuard/AdvDM 為 60） |
| 4 | **AdvPaint** | ICLR 2025, [arXiv:2503.10081](https://arxiv.org/abs/2503.10081) | inpainting | runwayml/stable-diffusion-inpainting | ℓ∞ η = 0.06 | PGD 250，初始步長 0.03 遞減 |
| 5 | **PromptFlare** | ACM MM 2025, [arXiv:2508.16217](https://arxiv.org/abs/2508.16217) | inpainting | runwayml SD inpainting | ℓ∞ ε = 12/255，步長 2/255 | 400 |
| 6 | **DiffVax** | ICLR 2026, [arXiv:2411.17957](https://arxiv.org/abs/2411.17957) | inpainting（宣稱可推廣） | SD v1.5，轉移測 v2 | L₁ 有界，clamp [0,1] | **無**（前饋 UNet++，optimization-free） |

`PhotoGuard-c` 的 `-c`：官方 repo [MadryLab/photoguard](https://github.com/MadryLab/photoguard) 的 README 將兩個變體命名為
「simple photo-guarding (Encoder Attack)」與「**complex** photo-guarding (Diffusion attack)」，
故 `-c` = complex = diffusion attack，等同其他論文寫的 PhotoGuard-D。

### 2.2 判準指標

| 方法 | 抗編輯指標 | 保真／隱蔽指標 | 語意指標 |
|---|---|---|---|
| PhotoGuard-c | FID, Precision/Recall, SSIM, PSNR, VIFp, FSIM（n = 60） | — | CLIP 影像—文字相似度 |
| Mist | FID, Precision | — | — |
| DIA | PSNR↓, LPIPS↑, MSE↑, SSIM↓（**背景保留，用 PIE-Bench 的 mask 隔離**） | 免疫圖對原圖的 PSNR（700 對） | CLIP Similarity↓（比對未遮罩區與編輯文字嵌入） |
| AdvPaint | FID↑, Precision↓, LPIPS↑ | — | — |
| PromptFlare | LPIPS↑, SSIM↓, PSNR↓（分 `all` 與 `mask` 兩種區域） | — | CLIP Score↓, Aesthetic Score↓, PickScore↓ |
| DiffVax | SSIM, PSNR, FSIM（編輯免疫圖 vs 編輯原圖） | SSIM(Noise)（原圖 vs 免疫圖） | CLIP-T |

### 2.3 抗淨化評測（**本專案賣點所在的軸**）

| 方法 | 是否評測 | 使用的算子 |
|---|---|---|
| PhotoGuard-c | 否 | — |
| Mist | 部分 | 提及對 preprocessing 的強健性，未系統化 |
| **DIA** | **是** | JPEG 壓縮、Crop & Resize、**Adverse Cleaner** |
| AdvPaint | 是 | Appendix A.5，算子清單待補 |
| PromptFlare | 有 robustness 章節 | 非淨化算子 |
| **DiffVax** | **是** | **CNN 去噪（NTIRE 2023 冠軍模型）**、JPEG（ratio 0.75）、**IMPRESS** |
| APA（非 baseline） | 是 | 對 12 種防禦，含 JPEG 與 **DiffPure**，平均 ASR 70.20% |

### 2.4 資料集與樣本數

| 方法 | 資料集 | 樣本數 |
|---|---|---|
| PhotoGuard-c | 自選 | n = 60 |
| Mist | WikiArt（Van Gogh） | — |
| DIA | **PIE-Bench** | **700 張 × 9 子任務 = 6,300 次評測** |
| AdvPaint | 自選公開影像 | 100 張 512²，50 個 ChatGPT 生成 prompt，Grounded SAM 產 mask（ρ = 1.2） |
| PromptFlare | **EditBench** | 240 張（120 真實 + 120 生成） |
| DiffVax | CCP（人像） | 875 張：800 訓練（含 75 已見測試）+ 75 未見 |

DiffVax 另有**人類評估**：67 名 Prolific 受試者、20 組影像對、平均排名 1.64（PhotoGuard-D 為 2.63）。

---

## 3. APA 的移植規格

| 項目 | 原論文（[arXiv:2506.01511](https://arxiv.org/abs/2506.01511)） |
|---|---|
| 階段一目標 | `R_s(Δθ) = E_{t,ε}[ −‖ε − ε_{θ+Δθ}(z_t, t, c)‖² ]`，以 LoRA 微調換取視覺一致性 |
| 階段二目標 | `R_a(z_T) = L(f_φ(x_adv), y)`，**替代分類器的 cross-entropy** |
| 攻擊點 | latent `z_T`（主）或 prompt embedding `τ_θ(c)`；黑盒 ASR 88.02% vs 62.08%，**latent 較優** |
| 模型 | Stable Diffusion v1.5，DDIM T = 50（APA-SG）／T = 10（APA-GC） |
| 失真 | LPIPS 0.23、SSIM 0.69（APA-GC） |
| 資料 | ImageNet-compatible，1,000 張 |

**移植的落差（必須處理）**：階段二的 reward 是**分類器標籤的 cross-entropy**。抗編輯場景沒有
分類器，故「照原論文的 loss」在字面上不可執行。可保留的是**兩階段結構**與**攻擊點選擇
（latent > prompt）**，階段二的 reward 必須換成抗編輯目標。

**本專案的既有對應物**：`site_weight.py`（LoRA，1,591,296 參數）對應階段一；
`site_latent.py`（逐步 ε̂ 注入）與 `site_embedding.py` 對應階段二的兩個攻擊點。三者皆已實作且有測試。

**已量測的風險**：APA 走生成路徑，`decode(encode(x))` 本身即 LPIPS 0.1434 / PSNR 27.51 dB。
在 ℓ∞ = 16/255（本專案實測 ≈ LPIPS 0.582）的預算下佔 25%，可行；預算降到 0.05 以下則不可行。

---

## 4. 四個必須在步驟二解決的衝突

### 4.1 威脅模型三分裂

使用者選定全圖 SDEdit。6 篇 baseline 的原生威脅模型：

| 原生威脅模型 | 方法 | 與 SDEdit 的關係 |
|---|---|---|
| img2img（含 SDEdit） | PhotoGuard-c | **直接可比** |
| inversion-based editing（P2P/PnP/MasaCtrl/DDIM） | DIA | **同族，可比**，但編輯程序不同 |
| 風格模仿（攻擊方要微調） | Mist | **任務不同，結論不互通**（PRIOR_FINDINGS §7.8） |
| inpainting | AdvPaint、PromptFlare、DiffVax | 需改寫成無 mask 版本才能比，**改寫即不忠實** |

### 4.2 指標沒有交集

五組互不相同的判準，且各自的統計前提不同：

- **FID 與 Precision 是分布層級指標**，需要數百張才穩定。本專案現有 `data/lo_aligned/` 只有 24 張，
  不足以支撐 PhotoGuard-c、Mist、AdvPaint 的判準。
- **DIA 的背景保留指標依賴 PIE-Bench 的編輯 mask**。SDEdit 無 mask，該隔離程序無對應物。
- PromptFlare 的 Aesthetic Score 與 PickScore 需要額外模型權重。

### 4.3 失真預算不一致

16/255、17/255、12/255、0.06、0.05 五個值。DIA 已提供掃描先例（0.025–0.1 四點），
可作為第三軸的格點來源。

### 4.4 SD 版本升級的三個後果

SDXL 1.0 是仍使用 UNet + cross-attention 的最高版本（SD 3.x 起為 MMDiT 聯合注意力，無 cross-attention）。
升級後：

1. **全部 baseline 必須自行在 SDXL 上重跑**，不可引用原論文數字——所有 baseline 都跑在 SD v1.4/v1.5/sd-inpainting 上。
2. **SDXL 的 cross-attention 只存在於 2× 與 4× 下採樣層**（SD 2 適用範圍較廣）。
   `src/models/attention.py` 掃 `attn2` 仍能找到層，但層數與解析度分布改變，
   cross-attn 損失的既有校準值全部作廢。
3. **SDXL 原生 1024²**，本機 RTX 2050（4 GB）連 512² 含梯度訓練都 OOM，
   **本輪全部訓練必須在雲端進行**。

---

## 5. 原始碼盤點

### 5.1 保留（不改）

| 檔案 | 理由 |
|---|---|
| `src/residual/site_warp.py` | 非加性方法 B 本體 |
| `src/residual/base.py`、`src/defense/generator.py` | 能力分派介面，新增位置不需改 |
| `src/purify/ops.py` | 抗淨化核心，但需擴充（見 5.4） |
| `src/models/sd.py` | SD wrapper 與 BDIA 精確反演 |
| `src/metrics/suite.py`、`ray_scale.py`、`acutance.py`、`chroma.py`、`local_acutance.py` | 評測 |
| `src/utils/` | 裝置與產物落盤 |

### 5.2 保留並改寫

| 檔案 | 改寫內容 |
|---|---|
| `src/defense/linf_attack.py` | **原計劃列為刪除，此判定修正為保留。** 它含 PGD 迴圈與三個文獻損失，而本輪要自行重跑 6 篇 baseline，PGD 是其共同骨幹。應擴充為 baseline 實作層 |
| `src/models/attention.py` | SDXL 的 cross-attention 層分布不同，掃描與正規化要改 |
| `src/defense/objective.py` | 現行 untargeted hinge 在起點梯度精確為零，涵蓋 59 個有紀錄批次的 100%，必須重寫 |
| `src/defense/optimize.py` | 944 行，隨目標函數一併重寫 |

### 5.3 刪除

`src/residual/site_pixel.py`、`site_pixel_full.py`、`site_color.py`——加性方法與未使用的第三種非加性參數化。

### 5.4 必須新增

| 項目 | 用途 | 依據 |
|---|---|---|
| **淨化算子**：Crop & Resize、Adverse Cleaner、CNN 去噪、IMPRESS、DiffPure | 現有只有 blur/noise/jpeg/quantize，缺文獻共識的五個 | §2.3 |
| **指標**：FID、Precision/Recall、MSE、Aesthetic Score、PickScore、CLIP-T | 現有 suite 缺這六項 | §2.2 |
| **APA 移植模組** | 兩階段結構 + 抗編輯 reward | §3 |
| **6 篇 baseline 的實作** | 加性對照 | §2.1 |
| **資料集擴充** | 24 張不足以支撐 FID／Precision | §4.2 |

### 5.5 現有資料

`data/lo_aligned/`（24 張 CC0 真實照片、六類）、`data/dayn_testset/`、`data/targets/`。
候選外部資料集：**PIE-Bench**（DIA 用，700 張，附 source/target prompt 與編輯 mask）、
**EditBench**（PromptFlare 用，240 張）。

---

## 6. 步驟一結論

1. 6 篇 baseline 中，只有 **PhotoGuard-c 與 DIA** 的威脅模型與全圖 SDEdit 直接相容；
   Mist 是不同任務；AdvPaint／PromptFlare／DiffVax 是 inpainting 專用。
2. 判準指標**沒有共同交集**，且 FID／Precision 的樣本數需求遠超現有資料集。
3. 抗淨化的文獻算子共識為 **JPEG、Crop & Resize、Adverse Cleaner、CNN 去噪、IMPRESS、DiffPure**，
   本專案現有四個算子中只有 JPEG 在此清單內。這是賣點所在的軸，必須優先補齊。
4. SDXL 升級使**所有 baseline 數字必須自行重跑**，且**本機不再具備訓練能力**。
5. APA 的階段二 reward 依賴替代分類器，抗編輯場景無對應物，「照原論文 loss」需要替換。

---

## 附：來源

- [PhotoGuard, arXiv:2302.06588](https://arxiv.org/abs/2302.06588) ／ [官方 repo](https://github.com/MadryLab/photoguard)
- [Mist, arXiv:2305.12683](https://arxiv.org/abs/2305.12683)
- [DIA, arXiv:2510.00778](https://arxiv.org/abs/2510.00778)
- [AdvPaint, arXiv:2503.10081](https://arxiv.org/abs/2503.10081) ／ [ICLR 2025](https://openreview.net/forum?id=m73tETvFkX)
- [PromptFlare, arXiv:2508.16217](https://arxiv.org/abs/2508.16217) ／ [ACM MM 2025](https://dl.acm.org/doi/10.1145/3746027.3755763)
- [DiffVax, arXiv:2411.17957](https://arxiv.org/abs/2411.17957) ／ [官方 repo](https://github.com/ozdentarikcan/DiffVax) ／ [專案頁](https://diffvax.github.io/)
- [APA, arXiv:2506.01511](https://arxiv.org/abs/2506.01511)
- [SDXL vs SD3 架構演進](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)
