# 抗淨化的檢定方法：三份外部協定

2026-08-13 建立。本檔只記**別人怎麼測**，不含本專案的結論。
本專案的判定式在 `docs/PLAN.md` §2，測得的事實在 `docs/FINDINGS.md`。

三份協定各自解決一個本專案現行淨化軸的缺口：

| 缺口 | 協定 |
|---|---|
| 七個算子全部單獨施加，未測串接 | §1 真實世界變換串接 |
| 未面對通用淨化器 | §2 PDM-Pure |
| 未面對針對**非加性**設計的淨化器 | §3 NAPPure |

---

## 1. 真實世界變換串接

**出處**：Do Protective Perturbations Really Protect Portrait Privacy under
Real-world Image Transformations?（arXiv:2604.23688，2026-04）

### 1.1 變換設定

| 記號 | 內容 |
|---|---|
| C | JPEG 壓縮，**quality factor 75** |
| R | **0.5× 降採樣，Lanczos 插值** |
| **C&R** | **先 C 再 R**（順序是協定的一部分） |

核心協定只有這三格；附錄的個別變換沿用同一組參數。無模糊、無噪聲、
無截圖模擬。權重參數 λ 固定 0.2。

### 1.2 評測

- **資料**：CelebA-HQ 前 100 張
- **影像編輯側指標**：SSIM、PSNR、FID、LPIPS、BRISQUE、ID-S
- **talking face 側指標**：SSIM、PSNR、FID、Sync-C、M-LMD、ID-S
- **受測方法**：DF-RAP、Anti-Forgery、FaceLock、FaceShield（編輯）；
  AdvDM(−)、Mist、PhotoGuard、Silencer-I（talking face）

### 1.3 結果與該論文的結論

C&R 之下全部方法大幅失效：SSIM 由 0.54–0.87 升到 0.88–0.89、
PSNR 由 12.93–23.94 升到 27.37–28.25 dB、DF-RAP 的 FID 由 139.6 降到 32.54。

> 作者結論：只以單獨變換評測會**嚴重高估** robustness。

### 1.4 移植到本專案的成本

`src/purify/ops.py` 已有 `jpeg`（真實實作 + 直通代理）與 `crop_resize`。
串接只需一個複合 `Purifier`，不需新算子。Lanczos 降採樣需確認現有
`crop_resize` 的插值核，若不同須另加 `resize_lanczos`。

---

## 2. PDM-Pure：像素空間 diffusion 當通用淨化器

**出處**：Pixel is a Barrier: Diffusion Models Are More Adversarially Robust
Than We Think（arXiv:2404.13320）

### 2.1 程序

| 項目 | 設定 |
|---|---|
| 淨化器 | **DeepFloyd-IF**（LAION 1.2B 文圖對訓練）的 **Stage II** |
| 解析度 | 輸入縮到 64² 與 256²；核心淨化在 **256²** 上做 |
| 可選 | Stage III 上採樣到 1024² 再降回目標尺寸 |
| 程序 | **像素空間 SDEdit**：前向擴散到 **t\* = 0.1T**，排程 respace 成 100 步 |
| 步數 | **10 步去噪** |
| 成本 | 單張 A6000 約 **30 秒** |

### 2.2 破解對象與數字

AdvDM、AdvDM(−)、SDS(−)、SDS(+)、PhotoGuard、Mist、Mist-v2、Glaze，
擾動預算 δ ∈ {4, 8, 16}/255 全部被移除。

δ=16 時 FID 159–179，未防護基線 166；對照 GrIDPure 210、LDM-Pure 300+。

### 2.3 它為什麼是本專案的威脅模型邊界問題

該論文的核心論點是**擾動住在 latent diffusion 的弱點上，而 pixel-space
diffusion 對白盒 PGD 遠為 robust**。攻擊方改用 PDM 當淨化器，等於離開了
「攻擊方使用 stock Stable Diffusion」這個前提——PDM-Pure 需要額外的
DeepFloyd-IF 權重。

**處置**：列為評測端的上界對手並誠實報告，不放進訓練內層。若不納入，
需在論文中明寫威脅模型排除它的理由；若納入而失敗，那是可報的限制。

---

## 3. NAPPure：針對非加性擾動的淨化

**出處**：Adversarial Purification for Robust Image Classification under
Non-Additive Perturbations（arXiv:2510.14025）

**威脅模型是影像分類，不是 diffusion 編輯。** 但它的 flow-field 分支與
本專案的空間變形參數化是同一個東西，是「非加性更抗淨化」這個主張最直接
的反例來源。

### 3.1 三類非加性擾動的生成模型

| 類型 | 形式 | 約束 |
|---|---|---|
| 模糊 | `f_blur(x, ε) = x * ε`，ε 為 k×k 卷積核 | `‖ε − ε₀‖∞ ≤ 0.025`，ε₀ 為 Dirac delta |
| 遮擋 | `f_occl(x, ε) = x·(1−m) + p·m`，ε = (p, a, b, s) | GTSRB 固定 7×7 置中；CIFAR-10 用 50×50 |
| **形變** | `f_dist(x, ε) = x'`，雙線性插值 | **ε ∈ [0,1]^(2hw) 為 2D flow field**，`‖ε‖∞ ≤ 1.2`（GTSRB）／`≤ 3`（CIFAR-10） |

### 3.2 淨化目標

同時還原乾淨影像與擾動參數：

```
x*, ε* = argmax_{x, ε}  log p(x) + log p(ε) + log p(x_adv | x, ε)
```

實際最小化的損失：

```
min_{x, ε} L(x, ε; x_adv)
    = E_{σ, n} ‖D_θ(x_σ, σ) − x‖²      # EDM 去噪先驗（資料似然）
    + λ₁ · φ(ε)                        # 擾動先驗
    + λ₂ · ‖x_adv − f(x, ε)‖²          # 重建一致性
```

`φ(ε)`：加性與 flow 取 L² 範數；模糊與 patch 取「與恆等變換的距離」。

### 3.3 演算法

交替最佳化。初始化 `x⁽⁰⁾ ← x_adv`、`ε⁽⁰⁾ ← ε₀`。每次迭代：

1. 抽 `σ ~ U(0.4, 0.6)`
2. `x ← Adam(∇_x L, η₁ = 0.1)`
3. `ε ← Adam(∇_ε L, η₂ = 0.05)`

GTSRB 的迭代數與權重：

| 攻擊 | 迭代 | λ₁ | λ₂ |
|---|---|---|---|
| 加性 | 100 | 0.1 | 3 |
| 模糊 | 500 | 0.001 | 3 |
| **flow** | **500** | **0.01** | **1** |
| patch | 500 | 0.01 | 5 |

### 3.4 攻擊端設定（被淨化的對象）

- **GTSRB（32²）**：模糊 5×5 uniform kernel；patch 7×7 置中；
  **flow `‖ε‖∞ ≤ 1.2`，Gaussian σ = 1.5、9×9 核平滑**；加性 `‖ε‖∞ ≤ 24/255`
- **CIFAR-10**：加性 `‖ε‖∞ ≤ 8/255`；flow 用 5×5 核、`‖ε‖∞ ≤ 3`
- 全部攻擊用 **APGD-CE**，每個資料集測 512 張

### 3.5 結果

| | GTSRB | CIFAR-10 |
|---|---|---|
| NAPPure | **73.93%** | **68.55%** |
| 標準淨化 | 38.71% | — |
| 對抗訓練 | 37.30% | — |

（robust accuracy，對非加性攻擊的平均）

### 3.6 移植到本專案要注意什麼

1. **解析度差三個數量級的問題**：原設定在 32² 上，本專案是 512²。
   flow 的 `‖ε‖∞` 約束與平滑核大小都要重新定，不能照抄數值。
2. **它需要一個 EDM 去噪器當資料先驗。** 本專案已有 DiffPure 的權重路徑
   （`src/purify/diffpure.py`），可否直接充當需查證。
3. **它是白盒淨化**：假設攻擊方知道防禦用的是哪一類非加性變換。
   這比本專案現行的七個算子強得多，是一個刻意設計的最壞情況。

---

## 4. 三份協定與本專案淨化算子集合的對應

| 本專案 `Purifier.kind` | 對應外部協定 | 狀態 |
|---|---|---|
| `jpeg`(75) | §1 的 C | 已有 |
| `crop_resize` | 近似 §1 的 R（插值核待查證） | 已有 |
| **`jpeg75_then_resize`** | **§1 的 C&R** | **待新增** |
| `diffpure`(t=150) | 近似 §2 但走 latent／較弱 | 已有，權重狀態見 `METRICS.md` §7 |
| **`pdm_pure`** | **§2** | **待評估，需 DeepFloyd-IF 權重** |
| **`nappure_flow`** | **§3 的 flow 分支** | **待新增，本專案最需要的一項** |
| `blur`／`noise`／`quantize`／`adverse_cleaner`／`impress`／`cnn_denoise_substitute` | 無外部對應，本專案既有 | `cnn_denoise_substitute` 缺權重 |

## 5. 引用來源

- [Do Protective Perturbations Really Protect Portrait Privacy under Real-world Image Transformations?](https://arxiv.org/html/2604.23688)
- [Pixel is a Barrier: Diffusion Models Are More Adversarially Robust Than We Think](https://arxiv.org/abs/2404.13320)（PDM-Pure）
- [Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](https://arxiv.org/html/2510.14025)（NAPPure）
