# 文獻查證：2026-08-18 — 頻域／相位方法與抗淨化

本輪查證對應 2026-08-18 的研究目標改版：主軸由「非加性 vs 加性」改為
**頻域／相位方法及其抗淨化能力**。查證範圍分三類：

1. 頻域（DCT／FFT／小波）的防護性擾動與對抗擾動
2. 抗淨化（robustness against purification）的攻防兩側
3. 相位／幅度譜的理論與影像合成背景

前一輪（`SURVEY_2026-08-16.md`）已查證的 DCT-Shield、Perturbing the Phase、
Interpreting Structured Perturbations 在本檔重新整理並補上方法細節，不重複
論述其對新穎性主張的影響。

**查證方式**：全部以 WebSearch 檢索、WebFetch 讀取 arXiv 摘要頁或 HTML 全文。
凡未實際讀到的欄位一律標「未查證」，不以推測填補。

---

## 第一節 頻域／傅立葉／DCT／小波的防護性與對抗擾動

### 1.1 DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing

| 項目 | 內容 |
|---|---|
| 作者 | Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal, Siddharth Roheda（Samsung R&D Institute India, Bangalore） |
| 發表 | ICCV 2025（Highlight） |
| arXiv | [arXiv:2504.17894](https://arxiv.org/abs/2504.17894)／[CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Bala_DCT-Shield_A_Robust_Frequency_Domain_Defense_against_Malicious_Image_Editing_ICCV_2025_paper.pdf)／[專案頁](https://dct-shield.github.io/project-page/) |
| 程式碼 | **未找到公開程式碼**。專案頁存在，但檢索未見 repo 連結，論文正文亦未列出 |

**頻域上的具體運算**：影像先進入標準 JPEG 管線（RGB→YCbCr、8×8 分塊 DCT、
以品質因子 q 的量化表量化）。擾動**加**在**量化後的 DCT 係數**上，約束為
`ε ≥ 1`（即至少變動一個量化階），再走 JPEG 解碼回像素域。三個變體：base
DCT-Shield、mask-based（inpainting 專用）、Y-channel（只動亮度通道，
換取更高的 JPEG 壓縮強健度與更低可見度）。

- 加性與否：**加性**，只是加在 DCT 係數而非像素上。
- 保留幅度：**否**。DCT 是實數變換，沒有幅度／相位分解；擾動直接改變係數值，
  局部能量分布隨之改變。

**威脅模型**：off-the-shelf latent diffusion 編輯模型，防禦方不知道攻擊方的
UNet 架構與編輯 prompt。評測用 InstructPix2Pix（instruction-based editing）與
Stable Diffusion 1.0 inpainting。

**抗淨化測試**：JPEG 品質 65／75／85／95、crop-and-resize（邊界裁 64 像素）、
AdvClean 濾波；補充材料另有 Gaussian noise、IMPRESS、noisy upscaling。

**與本專案的關係**：**最直接的競爭者，應納入 baseline**。它與本方法同一威脅
場景、同樣主張「頻域＋抗淨化＋低可見度」。關鍵差異是它改係數的**值**、
本方法只改係數的**相位角**；把它放進同一協定即可讓「保不保留幅度」成為
可判定的對照，而不只是一個好聽的性質。

---

### 1.2 DDAP: Dual-Domain Anti-Personalization against Text-to-Image Diffusion Models

| 項目 | 內容 |
|---|---|
| 作者 | Jing Yang, Runping Xi, Yingxin Lai, Xun Lin, Zitong Yu |
| 發表 | IJCB 2024 |
| arXiv | [arXiv:2407.20141](https://arxiv.org/abs/2407.20141) |
| 程式碼 | **未找到公開程式碼**（arXiv 摘要頁未列 repo） |

**頻域上的具體運算**：提出 Frequency Perturbation Learning（FPL），宣稱利用
擴散模型在頻域上的特性、聚焦影像細節，與 Spatial Perturbation Learning 交替
進行。**採用哪一個變換、作用在哪些係數、是否加性，摘要頁均未說明——未查證。**

**威脅模型**：個人化生成（personalization／DreamBooth 類），非本專案的
SDEdit 編輯。

**抗淨化測試**：摘要頁未提及——未查證。

**與本專案的關係**：**只是背景**。任務為 anti-personalization 而非編輯防護，
且頻域細節不足以重現。可作為「雙域（空間＋頻率）交替最佳化」這個設計模式的
引用來源。

---

### 1.3 MetaCloak-JPEG: JPEG-Robust Adversarial Perturbation for Preventing Unauthorized DreamBooth-Based Deepfake Generation

| 項目 | 內容 |
|---|---|
| 作者 | Tanjim Rahaman Fardin, S M Zunaid Alam, Mahadi Hasan Fahim, Md Faysal Mahfuz |
| 發表 | arXiv 預印本，2026-04-20 |
| arXiv | [arXiv:2604.18537](https://arxiv.org/abs/2604.18537)／[HTML](https://arxiv.org/html/2604.18537) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：不直接參數化頻域，而是把**可微分 JPEG 層**
（DiffJPEG，以 Straight-Through Estimator 處理 `round()`：前向做真實 JPEG
量化、反向以恆等函數代替）放進最佳化迴圈，並以品質因子由 95 降到 50 的
curriculum 排程包在 MetaCloak 的雙層 meta-learning 之外。效果是**梯度自動把
對抗能量推向 JPEG 量化表保留的低／中頻帶**。

- 加性與否：**加性**，像素域 `ℓ∞ ≤ 8/255`。
- 保留幅度：不適用。

**威脅模型**：攻擊方取得 4–8 張公開人臉照片，微調 DreamBooth 生成 deepfake。

**抗淨化測試**：9 個 JPEG 品質因子。報 PSNR 32.7 dB、JPEG survival rate 91.3%，
宣稱在全部 9 個品質因子上勝過 PhotoGuard。

**與本專案的關係**：**方法論上可直接借用**。本專案目前的抗淨化是「先做防護、
事後量測淨化後位移」，屬於未針對淨化最佳化的設定；把可微分 JPEG 放進
`param_pgd.py` 的前向即是一個現成、低風險的強化路徑。任務（DreamBooth
個人化）與本專案（SDEdit 編輯）不同，故不作 baseline。

---

### 1.4 Disruptive Attacks on Face Swapping via Low-Frequency Perceptual Perturbations

| 項目 | 內容 |
|---|---|
| 作者 | Mengxiao Huang, Minglei Shu, Shuwang Zhou, Zhaoyang Liu |
| 發表 | IEEE IJCNN 2025 |
| arXiv | [arXiv:2508.20595](https://arxiv.org/abs/2508.20595)／[HTML](https://arxiv.org/html/2508.20595) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：以 **DWT（離散小波轉換）** 取出低頻分量，由
encoder–perturbation generator–decoder 的前饋架構產生擾動，只擾動低頻、
**保留高頻細節**，並結合頻域與空間域特徵。**具體擾動哪些子帶（LL／LH／HL／HH）
未在摘要層級說明——未查證。**

- 加性與否：摘要描述為在頻域引入 artifact，形式上為加性——但**未查證**確切算式。
- 保留幅度：不適用（DWT 為實數變換）。

**威脅模型**：GAN-based face swapping（SimSwap／FaceShifter 類），非擴散編輯。

**抗淨化測試**：摘要未提及——未查證。資料集為 CelebA-HQ 與 LFW。

**與本專案的關係**：**只是背景**，但有一個可引用的立場：它與多數方法相反，
主張把擾動放在**低頻**、保住高頻，理由是低頻擾動不易被壓縮抹除且視覺上仍
可接受。這與本方法的徑向頻率閘（限制相位旋轉的頻帶）是同一個設計問題的
兩種答案。

---

### 1.5 Low-Mid Adversarial Perturbation against Unauthorized Face Recognition System

| 項目 | 內容 |
|---|---|
| 作者 | Jiaming Zhang, Qi Yi, Dongyuan Lu, Jitao Sang |
| 發表 | Information Sciences（期刊）；arXiv 2022-06-19，2023-09-03 修訂 |
| arXiv | [arXiv:2206.09410](https://arxiv.org/abs/2206.09410) |
| 程式碼 | **未找到公開程式碼**（arXiv 摘要頁未列 repo） |

**頻域上的具體運算**：提出 LFAP（low frequency adversarial perturbation）
與改良的 LMFAP（low-mid frequency adversarial perturbation）。作法是以
**對抗訓練調節 surrogate model**，使其主要依賴低頻資訊，於是最佳化出來的
擾動能量自然集中在低頻；LMFAP 再納入中頻分量。**採用的變換（FFT 或 DCT）
在摘要層級未指明——未查證。**

- 加性與否：**加性**（像素域擾動，只是能量分布被引導到低／中頻）。
- 保留幅度：不適用。

**威脅模型**：未授權的人臉辨識系統（隱私保護），黑箱轉移，含商用 API Face++。

**抗淨化測試**：JPEG 壓縮是其核心訴求；另測跨 backbone、跨 supervisory head、
跨資料集的轉移性。

**與本專案的關係**：**只是背景**，但它提供了「為什麼頻帶選擇會決定抗壓縮性」
最清楚的一個實證，可用於 related work 中鋪陳動機。

---

### 1.6 Low Frequency Adversarial Perturbation

| 項目 | 內容 |
|---|---|
| 作者 | Chuan Guo, Jared S. Frank, Kilian Q. Weinberger |
| 發表 | UAI 2019 |
| arXiv | [arXiv:1809.08758](https://arxiv.org/abs/1809.08758)／[PMLR](http://proceedings.mlr.press/v115/guo20a/guo20a.pdf) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：把黑箱攻擊的搜尋空間**限制在低頻子空間**——擾動由
低頻 DCT 基底張成，再逆變換回像素域。效果是查詢成本降為 2–4 分之一，並可
繞過輸入變換型防禦。

- 加性與否：**加性**（低頻基底的線性組合仍是加到原圖上）。
- 保留幅度：不適用。

**威脅模型**：黑箱影像分類（Google Cloud Vision API）。

**抗淨化測試**：測「image transformation defenses」，宣稱可繞過；**確切算子
清單未查證**。

**與本專案的關係**：**背景兼理論依據**。它是「限制擾動所在頻帶」這條線的
起點文獻，本方法的徑向頻率閘屬同一族設計，引用時應歸功於此。

---

### 1.7 AdvDrop: Adversarial Attack to DNNs by Dropping Information

| 項目 | 內容 |
|---|---|
| 作者 | Ranjie Duan, Yuefeng Chen, Dantong Niu, Yun Yang, A. K. Qin, Yuan He |
| 發表 | ICCV 2021 |
| arXiv | [arXiv:2108.09034](https://arxiv.org/abs/2108.09034) |
| 程式碼 | [github.com/RjDuan/AdvDrop](https://github.com/RjDuan/AdvDrop) |

**頻域上的具體運算**：DCT 把影像轉到頻域，對頻域係數施加**可學習的量化**
（量化表本身是最佳化變數），量化造成資訊丟失，再 IDCT 回像素域。與一般攻擊
相反，它**不加入任何東西，而是移除既有資訊**。

- 加性與否：**非加性**。這是文獻中少數明確的非加性頻域攻擊。
- 保留幅度：**否**（量化直接壓掉係數）。

**威脅模型**：影像分類（ImageNet），白盒與黑盒皆有。

**抗淨化測試**：論文主張此類樣本對「現有防禦」較有抵抗力；**確切測了哪些
防禦未查證**。

**與本專案的關係**：**重要的對照組候選**。它與本方法同屬「非加性的頻域操作」，
但方向相反——AdvDrop 移除資訊，紋理重相位保留全部能量只重排相位。若要主張
「非加性頻域操作」這個類別，AdvDrop 是必須引用的前例，且其量化操作
不可逆、對 JPEG 的關係值得在 related work 中辨明。

---

### 1.8 Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity（SSAH）

| 項目 | 內容 |
|---|---|
| 作者 | Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen |
| 發表 | CVPR 2022 |
| arXiv | [arXiv:2203.05151](https://arxiv.org/abs/2203.05151) |
| 程式碼 | [github.com/LinQinLiang/SSAH-adversarial-attack](https://github.com/LinQinLiang/SSAH-adversarial-attack)（依賴 `pywavelets`，確認使用 DWT） |

**頻域上的具體運算**：攻擊目標是特徵空間的語意相似度而非分類 logit；
另加一條 **low-frequency constraint**，以 DWT 分解後懲罰低頻子帶的變動，
把擾動逼到高頻分量，以維持不可見性。

- 加性與否：**加性**（頻域約束只是正則項）。
- 保留幅度：**否**。

**威脅模型**：影像分類與檢索的轉移性攻擊。

**抗淨化測試**：**未查證**。

**與本專案的關係**：**只是背景**，但它是「把擾動限制在高頻以求不可見」這個
主流做法的代表，正是本專案動機段落要反對的對象——高頻正是壓縮與模糊最先
抹掉的位置。引用它可讓「不可見性與抗淨化互相衝突」這個張力有具體出處。

---

### 1.9 Perturbing the Phase: Analyzing Adversarial Robustness of Complex-Valued Neural Networks

| 項目 | 內容 |
|---|---|
| 作者 | 未查證（arXiv 頁面作者欄前一輪未記錄） |
| 發表 | 2026，已投稿 IEEE |
| arXiv | [arXiv:2602.06577](https://arxiv.org/abs/2602.06577)／[HTML](https://arxiv.org/html/2602.06577) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：對複數值輸入（PolSAR、MRI）**只旋轉相位、幅度逐點
保留**（論文原文 `|Z| = p.w. |X|`）。相位偏移的上限與局部幅度反比相關：
幅度夠大處限制為 `2·arcsin(ε/(2|X|))`，`2|X| ≤ ε` 之處相位自由。攻擊算法為
PIFGSM／PMIFGSM。

- 加性與否：**非加性**（相位旋轉）。
- 保留幅度：**是，逐點保留**。

**威脅模型**：CVNN 與 RVNN 的分類（ResNet／ConvNeXt），資料為 S1SLC_CVDL
PolSAR 與 FastMRI Prostate。

**抗淨化測試**：**未測**。

**與本專案的關係**：**新穎性的最大威脅，同時是最有用的技術來源**。它與本方法
共用「只轉相位、幅度逐點保留」這個構造，並已獨立測到「相位攻擊比幅度攻擊
有效」。差異在於資料（天然複數 vs 實數自然影像）、變換（逐像素複數相位 vs
重疊區塊加窗 FFT）、任務（分類 vs 擴散編輯防護）、以及是否測抗淨化。
其幅度相依相位上限 `2·arcsin(ε/(2|X|))` 可直接取代本專案現行的固定 `θ_max`，
解決「固定相位角不等於固定失真」的問題。

---

### 1.10 Black box phase-based adversarial attacks on image classifiers

| 項目 | 內容 |
|---|---|
| 作者 | 未查證 |
| 發表 | Journal of Electronic Imaging 34(1):013041, 2025 |
| DOI | [10.1117/1.JEI.34.1.013041](https://doi.org/10.1117/1.JEI.34.1.013041) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：黑箱條件下對影像的傅立葉**相位**施加擾動。**確切的
相位參數化與是否保留幅度未查證**（僅由標題與檢索摘要確認為相位域攻擊）。

**威脅模型**：影像分類，黑箱。

**抗淨化測試**：**未查證**。

**與本專案的關係**：與 1.9 同屬「相位攻擊已有前例」的證據，用於收窄新穎性
主張。不作 baseline。

---

### 1.11 BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing

| 項目 | 內容 |
|---|---|
| 作者 | Jinsu Kim, Yunhun Nam, Minseon Kim, Sangpil Kim, Jongheon Jeong |
| 發表 | NeurIPS 2025 |
| arXiv | [arXiv:2511.00143](https://arxiv.org/abs/2511.00143) |
| 程式碼 | [github.com/jsu-kim/BlurGuard](https://github.com/jsu-kim/BlurGuard) |

**頻域上的具體運算**：不在頻域參數化，而是對已求得的**對抗噪聲**施加
**自適應的逐區域高斯模糊**，藉此重塑噪聲的整體頻譜（壓掉最高頻、把能量
留在較低頻）。主張防護噪聲除了 imperceptible 之外還必須 **irreversible**——
在看不到原圖的前提下難以被辨識為噪聲。

- 加性與否：**加性**（模糊後的噪聲仍是加到原圖）。
- 保留幅度：**否**。

**威脅模型**：AI 影像編輯（擴散編輯／個人化）。

**抗淨化測試**：宣稱測「a wide range of reversal techniques」，明確點名 JPEG
壓縮；**完整算子清單未查證**。

**與本專案的關係**：**可作對照組，也是概念上的競爭者**。它證明「調整擾動的
頻譜形狀」本身就能換到抗淨化能力，且做法極簡（後處理一個模糊）。若本方法
的優勢僅來自「能量落在較低頻」，BlurGuard 會是打平的對照；因此它應與
DCT-Shield 一併納入，用來分離「頻譜形狀」與「相位重排」兩個變因。
有公開程式碼，實作成本低。

---

### 1.12 Interpreting Structured Perturbations in Image Protection Methods for Diffusion Models

| 項目 | 內容 |
|---|---|
| 作者 | Michael R. Martin, Garrick Chan, Kwan-Liu Ma |
| 發表 | arXiv 預印本，2025-12-09 |
| arXiv | [arXiv:2512.08329](https://arxiv.org/abs/2512.08329) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：本身不提出擾動，而是以可解釋性方法分析既有防護。
頻域結論：Glaze 與 Nightshade **不是**擴散的隨機噪聲，而是把能量
**沿影像本身的主頻率軸重新分配**。整體性質為「structured, low-entropy，
與影像內容緊密耦合」。

**與本專案的關係**：**支撐本方法框架的分析文獻**。既有強防護在頻域上做的本質
是結構化的頻譜重分配，紋理重相位是同一件事的極端形式（重分配到只改相位、
完全不改幅度）。同時是一個警告：這類擾動因低熵而**始終可被偵測**，報告不應
暗示不可偵測性。

---

### 1.13 Robust cross-image adversarial watermark with JPEG resistance for defending against Deepfake models

| 項目 | 內容 |
|---|---|
| 作者 | Zhiyu Lin, Hanbin Lin, Liqiang Lin, Shuwu Chen, Xiaolong Liu（福建農林大學） |
| 發表 | Computer Vision and Image Understanding 260 (2025) 104459 |
| 程式碼 | [github.com/fishlin20/JPEG-Watermark](http://github.com/fishlin20/JPEG-Watermark)（論文列出，內容未查證） |
| 全文 | `paper_pdfs/jpeg_watermark_lin_cviu.pdf`（本地留存，逐節讀過） |

**頻域上的具體運算**：擾動不是加在像素上，而是加在 **Y 通道的 DCT 係數**上
（cb、cr 明確保持不變）。可微分 JPEG 編碼器 `En(·)` 把影像轉成中間 DCT 係數
`y = En(I)`，擾動 `W` 直接加在 `y` 上，再由 `De(y + W)` 逆轉回像素域得到交付影像
`I_inv`。`round()` 用的近似是 `x_approx = round(x) + (x − round(x))³`，出處是
DiffJPEG（Shin & Song, 2017，見 1.15）。

**損失（Eq. 3）**：

    L = L_MSE(G(I_inv), G(I)) + λ · 1/L_D ,   L_D = PSNR(I_inv, I)

第一項是 Deepfake 模型 `G` 在防禦圖與原圖上的輸出 MSE（要拉大），第二項是把
PSNR 當可見度項。更新是 DCT 係數上的 PGD **上升**（Eq. 4）：
`y_{t+1} = clip_{y,ε}{ y_t + α · sign(∇_y L) }`。

**這條式子有一個號誤，移植前必須決定怎麼處理**：Eq. 4 在最大化 `L`，而 `L` 內含
`+λ/PSNR`；最大化 `1/PSNR` 等於**最小化 PSNR**，也就是這一項會主動把浮水印推得
更明顯，與它自稱的 imperceptibility 約束相反。可見度實際上只由 `clip_{y,ε}` 撐著。
若實作本方法，該項必須標 `modified_from_paper`，且要逐行比對公開程式碼確認
真正的號。

**跨影像融合（Eq. 5、6）**：批內對 `n` 張影像的 sign 梯度取平均，批間用指數移動
平均把各批的浮水印接起來：`W^{m+1} = β·W^m + (1−β)·w^m`。產物是**通用擾動**——
一張浮水印貼到任意新臉上，不重訓。

**超參數**：β = 0.50、λ = 0.05、α = 0.03；ε 依目標模型為 0.20（StarGAN）／
0.18（AttGAN）／0.30（HiSD），迭代 T = 15／15／20。訓練資料 128 張分 16 批
（batch 8），單一 epoch。影像統一 256×256。

**訓練用的 JPEG 品質 q = 35**（Fig. 7 的取捨曲線選出），而評測的攻擊壓縮是
**Q = 75**。也就是**訓練壓得比攻擊狠**。消融給出兩邊的代價：q < 30 防禦更強但
防禦圖 PSNR < 30 dB；q > 50 則 PSNR > 32 dB 而抗壓縮變弱。

**抗淨化測試**：只有 JPEG，掃 Q 值曲線（Fig. 4），主表固定 Q = 75。**沒有測
blur、crop-resize 或任何幾何淨化。**

**報的數字**（CelebA，StarGAN，`SR_mask` 為擋下率）：

| 方法 | L2_mask（無壓縮） | SR_mask（無壓縮） | L2_mask（Q=75） | SR_mask（Q=75） |
|---|---|---|---|---|
| DISRUPTING (Ruiz 2020) | 1.236 | 100.0 | 0.007 | 1.8 |
| CMUA | 0.215 | 100.0 | 0.008 | 0.4 |
| SUA | 0.371 | 100.0 | 0.008 | 0.2 |
| DF-RAP | 0.258 | 100.0 | 0.064 | 45.4 |
| 本篇 | 0.489 | 100.0 | **0.097** | **76.4** |

融合模組的消融：關掉融合則防禦圖 PSNR 22.76 dB、L2_mask 0.101、SR_mask 84.0%；
開啟則 PSNR 30.10 dB、L2_mask 0.515、SR_mask 100.0%。

**與本專案的關係**：這是**與量化交付同一族、但參數化不同**的一支。相同處：
擾動只動 Y 通道、走 8×8 DCT、把可微分 JPEG 放進最佳化迴圈。相異處有三，都直接
可測：

1. **它把擾動參數化在 DCT 係數上**，我方是把擾動長在像素上、再用可微分 JPEG
   往返把它壓回格點（`--deliver-jpeg`）。前者的 ε 球在係數域，後者在像素域。
2. **訓練品質遠低於攻擊品質**（q = 35 訓練 / Q = 75 攻擊）。我方目前
   `--deliver-jpeg 0.85` 是訓練品質**高於**多數攻擊品質，方向相反。這是現成旗標
   就能掃的一刀。
3. **通用擾動**（跨影像 EMA 融合）。本專案是逐圖最佳化，此路目前不在主線上。

同時它也是一個誠實的參照點：即便是專門為抗 JPEG 設計的方法，Q = 75 就把擋下率
從 100% 打到 76.4%，且**完全沒有 blur／crop 的數字**——與本專案量到的
「JPEG 以外沒有防禦」一致，不是本專案獨有的失敗。

---

### 1.14 Improving the JPEG-resistance of Adversarial Attacks on Face Recognition by Interpolation Smoothing（IAM）

| 項目 | 內容 |
|---|---|
| 作者 | Kefu Guo, Fengfan Zhou, Hefei Ling, Ping Li, Hui Liu（華中科技大學） |
| arXiv | [arXiv:2402.16586](https://arxiv.org/abs/2402.16586) |
| 程式碼 | 論文未列出 |

**沒有可借的損失。** Eq. 1 把目標寫成含 JPEG 的形式
`argmin_x' D(F(J(x_adv)), F(x_t))`，但真正拿去求導的 Eq. 2 是

    L(x_adv, x_t) = ‖φ(F(x_adv)) − φ(F(x_t))‖₂²

——單純的人臉特徵距離（φ 是正規化），**沒有 JPEG 項、沒有頻域項、沒有 TV 或
平滑正則**。全部機制在 Algorithm 1。

**具體運算**：每一次迭代
1. 雙線性**降取樣** `x̃_{t−1} = I(x_{t−1}, f_inter)`，f_inter < 1；
2. **在半解析度上**算 sign 梯度並走一步 `x̃_t = x̃_{t−1} − β · sign(∇_x̃ L)`；
3. 雙線性**升取樣**回原尺寸 `x_t = I(x̃_t, 1/f_inter)`。

擾動因此被限制在「半解析度可表示」的子空間裡，天生沒有高頻——而高頻正是 JPEG
量化丟掉的部分。

**超參數**：f_inter = 1/2、N_max = 10、β = 1.0、ε = 10（0–255 的 L∞）。f_inter
的掃描峰值就在 1/2；`1/f_inter > 2` 之後 ASR 不穩且只是增加計算量。

**兩個瑕疵，移植時必須寫進 docstring**：
- Algorithm 1 **沒有投影回 ε 球、也沒有 clamp 到值域**，儘管 Eq. 1 寫了該約束。
- 結論段寫「decrease the **low**-frequency signals」，與摘要與方法的 high-frequency
  相反，是論文自身的筆誤。

**抗淨化測試**：只有 JPEG，QF = 25／50／75（另有 10–90 的曲線）。無 blur、無 crop。

**報的數字**（CelebA-HQ，IRSE50 為代理模型，黑箱 ASR，「基線 / 基線+IAM」）：
QF = 50 時 BIM 對 IR152 是 31.1 / 49.0、DI 是 19.2 / 59.6；QF = 25 時 DI 對
FaceNet 是 4.2 / 32.0。不壓縮時也有提升（BIM 對 IR152 40.3 / 56.3），即它並非
只在壓縮下有效。

**與本專案的關係**：這是「在 upsampling 上下功夫」的具體形式，是目前唯一指向
**blur 欄**的未試機制——blur 是純低通，只用半解析度表示的擾動照理穿得過去。
但它**救不了 crop**：crop-resize 的失效來自空間不對齊，不是頻率內容。

移植不能照抄：本方法的擾動不是加性的，是窗化 FFT 上的相位旋轉，「把影像降解析度
再走一步」對不上參數化。對應的作法是**把 θ 參數化在半解析度網格上、雙線性升取樣
成全解析度的 θ**——動的是參數化，不是損失。

---

### 1.15 JPEG-resistant Adversarial Images（Shin & Song）

| 項目 | 內容 |
|---|---|
| 作者 | Richard Shin, Dawn Song（UC Berkeley） |
| 發表 | NIPS 2017 Workshop on Machine Learning and Computer Security |
| 全文 | [mlsec17_paper_54.pdf](https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf) |
| 狀態 | **核心已實作**：`src/baselines/jpeg_codec.py` 的 `quantize_ste`／`jpeg_roundtrip_ste` |

**可微分 JPEG**：整條管線（色彩空間轉換、4:2:0 次取樣以 2×2 平均池化實作、
8×8 DCT、量化表、解碼）都用可微分算子重寫，唯一不可微的 `round()` 用

    ⌊x⌉_approx = ⌊x⌉ + (x − ⌊x⌉)³

取代——處處有非零導數，最大誤差 0.125，發生在 n + 0.5。

**與本專案實作的差別**：本專案用的是**直通估計**
（`q + (round(q) − q).detach()`），前向值逐位元等於真的 `round`，反向當恆等。
兩者都能通梯度，但直通估計在**前向值上更忠實**——這對本專案是必要的，因為
`--deliver-jpeg` **交付的就是真的量化影像**，前向若與真實 JPEG 有 0.125 的偏差，
最佳化看到的與交付出去的就不是同一張圖。故此處不改採論文的三次式。

**本專案尚未採用的部分——多品質集成**。論文明說單一品質的最佳化會過度特化
（Table 1 第 16 列：只對 q = 25 最佳化的攻擊不轉移）。它的作法是對
q ∈ {25, 50, 75, ∞}（∞ 表示不壓縮）四個模型集成，梯度以損失大小加權：

    Σᵢ ( 1 − exp(Lᵢ) / Σⱼ exp(Lⱼ) ) · ∇_{x'} Lᵢ ,   Lᵢ = ℓ(C(x), C(JPEG_diff(x', qᵢ)))

損失小的那一項拿到較大的權重。標題的 691× 就是集成相對單一品質的差距
（ε = 7/255，對 q = 25 的防禦，成功率 0.1% → 69.1%）。

**設定**：ResNet-50、ILSVRC 2012 前 1000 張、224×224 單裁切；防禦端用真的 JPEG
（非近似）評測。

**與本專案的關係**：`--deliver-jpeg 0.85` 正是**單一品質**，論文預言的過度特化在
本專案的資料上看得到——十張的淨增益從 jpeg75 的 0.4370 一路掉到 jpeg30 的 0.2087。
可移植的一刀是把集成放在**損失**上（多個 q 加上一個不壓縮），交付仍只能落在單一
格點（本方法交付的是真的量化影像，沒辦法同時落在多個格點）。

---

## 第二節 抗淨化：淨化側與防禦側的攻防

### 2.1 DiffPure: Diffusion Models for Adversarial Purification

| 項目 | 內容 |
|---|---|
| 作者 | Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar |
| 發表 | ICML 2022 |
| arXiv | [arXiv:2205.07460](https://arxiv.org/abs/2205.07460) |
| 程式碼 | [github.com/NVlabs/DiffPure](https://github.com/NVlabs/DiffPure)／[專案頁](https://diffpure.github.io/) |

**運算**：把受擾動影像加噪至時間 `t*`，再以預訓練擴散模型逆向去噪還原。
非頻域方法，但其效果在頻域上是**優先抹除高頻**（見 2.4）。

**與本專案的關係**：**已實作**，是 `src/purify/ops.py` 的 `diffpure` 算子。

---

### 2.2 Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable Diffusion?（GrIDPure）

| 項目 | 內容 |
|---|---|
| 作者 | Zhengyue Zhao, Jinhao Duan, Kaidi Xu, Chenan Wang, Rui Zhang, Zidong Du, Qi Guo, Xing Hu |
| 發表 | CVPR 2024 |
| arXiv | [arXiv:2312.00084](https://arxiv.org/abs/2312.00084)／[CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Can_Protective_Perturbation_Safeguard_Personal_Data_from_Being_Exploited_by_CVPR_2024_paper.pdf) |
| 程式碼 | **未找到公開程式碼**（arXiv 摘要頁未列 repo） |

**運算**：提出 **GrIDPure**——把影像切成**多個重疊網格**，每格以較小的擴散
時間步單獨跑 DiffPure，再把重疊區平均合回。相較整圖 DiffPure，它保住解析度
與結構一致性，抑制擴散的生成性。結論是 Stable Diffusion 能從淨化後影像
有效學習，既有防護在真實情境下不足。

**攻破的防護**：Glaze、AdvDM、Anti-DreamBooth。

**與本專案的關係**：**必須納入淨化算子清單**。GrIDPure 的重疊網格結構與本
方法的重疊區塊 FFT 在空間尺度上高度相似，這是一個尚未被檢驗的針對性風險：
若淨化與防護的區塊格點對齊，重疊相加的平均可能特別有效地抹平區塊間的相位
不一致。目前 `src/purify/ops.py` 只有整圖 `diffpure`，**沒有 GrIDPure**。

---

### 2.3 FreqPure: a High-frequency Preservation Diffusion-based Purification Method for Protective Perturbation

| 項目 | 內容 |
|---|---|
| 作者 | Yan Ju, Hongfei Xue, Siwei Lyu |
| 發表 | ICCV 2025 Workshop（APAI），pp. 1533–1542 |
| 連結 | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025W/APAI/papers/Ju_FreqPure_a_High-frequency_Preservation_Diffusion-based_Purification_Method_for_Protective_Perturbation_ICCVW_2025_paper.pdf)（直接抓取回 403，內容由檢索摘要確認） |
| 程式碼 | **未找到公開程式碼** |

> **2026-08-19 更正（本節以下的描述有誤）。** 本條目自己註明內容「由檢索
> 摘要確認」，即撰寫時未讀到原文。2026-08-19 取得 CVF PDF 全文後確認：
> **FreqPure 不是逆向過程中的頻域介入，而是一條兩階段的訓練式管線**——
> (1) 一個重建模組移除保護擾動造成的瑕疵；(2) 一個以低頻影像為條件的擴散
> 模型合成高頻。它需要在 FFHQ 人臉資料上訓練，無公開程式碼，**無法在不
> 訓練的前提下重現**。
>
> 下面描述的機制（逐時間步替換低頻幅度、投影低頻相位）實際上屬於 §2.4 的
> **arXiv:2505.01267（FD-Pure）**，該篇訓練自由、Algorithm 1 完整。本專案
> 實作的是後者，見 `src/purify/freq_grid.py`。

**頻域上的具體運算**：在擴散逆向過程的**每一個時間步**介入：

- **幅度譜**：把當前估計影像幅度譜的**低頻部分**替換成**對抗影像**的對應部分。
- **相位譜**：把當前估計影像的相位**投影到對抗影像低頻相位的指定範圍內**。

亦即它刻意保留對抗影像的低頻幅度與低頻相位，只讓擴散模型重建高頻。

**攻擊對象**：protective perturbation（具體清單未查證）。

**與本專案的關係**：**對本方法最尖銳的針對性威脅，必須測**。理由是它的設計
恰好與本方法互補：本方法把訊息編碼在**相位**上，FreqPure 卻**刻意保留對抗
影像的低頻相位**。兩種可能的結果都有研究價值——

- 若本方法的相位擾動落在低頻，FreqPure 會**保留**它，本方法可能異常抗此淨化；
- 若落在中高頻，FreqPure 會由擴散模型重建掉，本方法可能特別脆弱。

這使 FreqPure 成為檢驗徑向頻率閘設定的最佳診斷工具。**目前專案未實作。**

---

### 2.4 Diffusion-based Adversarial Purification from the Perspective of the Frequency Domain

| 項目 | 內容 |
|---|---|
| 作者 | Gaozheng Pei, Ke Ma, Yingfei Sun, Qianqian Xu, Qingming Huang |
| 發表 | arXiv 預印本，2025-05-02（v4 2025-12-08） |
| arXiv | [arXiv:2505.01267](https://arxiv.org/abs/2505.01267)／[HTML](https://arxiv.org/html/2505.01267) |
| 程式碼 | **未查證**（arXiv 頁未見 repo 連結） |

**頻域上的具體運算**：先把影像分解為幅度譜與相位譜，測到**兩者的破壞程度
都隨頻率單調遞增**——低頻相對完好、高頻受損嚴重。據此在逆向擴散中
(1) 用受擾動影像的低頻幅度替換估計影像的低頻幅度，(2) 約束相位估計對齊
受擾動影像的低頻相位。與 FreqPure 的機制幾乎相同，屬於同期獨立工作。

**與本專案的關係**：**理論上最重要的一篇**。「破壞程度隨頻率單調遞增」這個
量化結論，正是本專案「高頻擾動易被淨化」這句動機的直接文獻依據，比引用
DiffPure 更精確。同時它也界定了本方法的機會窗口：**若相位擾動能被推到低頻，
就落在淨化最不敢動的區域**。

---

### 2.5 IMPRESS: Evaluating the Resilience of Imperceptible Perturbations against Unauthorized Data Usage in Diffusion-Based Generative AI

| 項目 | 內容 |
|---|---|
| 作者 | Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen |
| 發表 | NeurIPS 2023 |
| arXiv | [arXiv:2310.19248](https://arxiv.org/abs/2310.19248) |
| 程式碼 | [github.com/AAAAAAsuka/Impress](https://github.com/AAAAAAsuka/Impress) |

**運算**：以「受保護影像經 VAE 編碼—解碼後與自身不一致」為訊號，最佳化出
一張與原圖接近但重建一致的影像，藉此剝除擾動。非頻域方法。

**與本專案的關係**：**已實作**，`impress` 算子。

---

### 2.6 AntiPure: Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification

| 項目 | 內容 |
|---|---|
| 作者 | Wenkui Yang, Jie Cao, Junxian Duan, Ran He |
| 發表 | ICCV 2025 |
| arXiv | [arXiv:2509.13922](https://arxiv.org/abs/2509.13922)／[HTML](https://arxiv.org/html/2509.13922) |
| 程式碼 | **未找到公開程式碼**（論文未載明；通訊作者 rhe@nlpr.ia.ac.cn） |

**頻域上的具體運算**：兩個引導項疊加在 PGD 上——

- **Patch-wise Frequency Guidance（PFG）**：對預測的去噪影像 `x̂₀` 切成
  `s×s` 區塊做 **DCT**，取每塊**右下四分之一**（即高頻）的係數，經 sigmoid
  正規化後構成損失，目的是**削弱淨化模型對高頻的支配力**。
- **Erroneous Timestep Guidance（ETG）**：讓 UNet 在**錯誤時間步**與正確時間步
  的噪聲預測趨於一致，破壞去噪策略的跨步一致性。

- 加性與否：**加性**，像素域 `ℓ∞`，`η = 16/255`。頻域只出現在損失函數裡。
- 保留幅度：**否**。

**威脅模型**：purification–customization 工作流（先淨化、再個人化微調）。

**抗淨化測試**：**GrIDPure**，2 輪各 20 次迭代、`t_p = 10`（共 40 次）；另做
10–40 次迭代的收斂分析。作者**未**單獨評測整圖 DiffPure。

**與本專案的關係**：**方法論對照**。它是「明確為抗淨化而設計」的加性代表作，
且其頻域引導同樣是分塊 DCT——與本方法的分塊 FFT 在空間結構上同構，差別在
它把頻域放進**損失**、本方法把頻域放進**參數化**。這個對比是 related work
中值得寫的一段。另可注意其預算 `16/255` 遠大於一般的 `8/255`，比較時
必須對齊失真而非對齊 `ε`。

---

### 2.7 DiffusionGuard: A Robust Defense Against Malicious Diffusion-based Image Editing

| 項目 | 內容 |
|---|---|
| 作者 | June Suk Choi, Kyungmin Lee, Jongheon Jeong, Saining Xie, Jinwoo Shin, Kimin Lee |
| 發表 | arXiv 2024-10-08（2025-09-29 修訂）；ICLR 2025 投稿紀錄見 arXiv 頁 |
| arXiv | [arXiv:2410.05694](https://arxiv.org/abs/2410.05694) |
| 程式碼 | [github.com/choi403/DiffusionGuard](https://github.com/choi403/DiffusionGuard) |

**運算**：損失針對擴散過程的**最早期階段**（高噪聲時間步）產生對抗噪聲；
另有 mask-augmentation，使防護對測試時未知的 mask 也有效。非頻域方法。

- 加性與否：**加性**。

**抗淨化測試**：宣稱對 JPEG 壓縮、crop-and-resize、專用 adversarial cleaning
一致優於 baseline；**arXiv 摘要未逐項列出設定**，需查全文。

**與本專案的關係**：**強 baseline 候選**（有公開程式碼、任務為擴散編輯防護、
且明確以抗淨化為賣點）。目前專案的強 baseline 只有 `photoguard_c`／`mist`／
`dia_r`，三者都不是為抗淨化設計的；補上 DiffusionGuard 可讓「抗淨化」的
比較有意義。

---

### 2.8 MetaCloak: Preventing Unauthorized Subject-driven Text-to-image Diffusion-based Synthesis via Meta-learning

| 項目 | 內容 |
|---|---|
| 作者 | Yixin Liu, Chenrui Fan, Yutong Dai, Xun Chen, Pan Zhou, Lichao Sun |
| 發表 | CVPR 2024（Oral） |
| arXiv | [arXiv:2311.13127](https://arxiv.org/abs/2311.13127) |
| 程式碼 | [github.com/liuyixin-louis/MetaCloak](https://github.com/liuyixin-louis/MetaCloak) |

**運算**：以 meta-learning 在一池代理模型上求解擾動，並在最佳化中
**取樣資料變換**（transformation sampling），得到 transformation-robust 的
語意扭曲。針對既有方法在 Gaussian filtering 下失效的問題。

- 加性與否：**加性**。非頻域。

**與本專案的關係**：**背景**（任務為 DreamBooth 個人化）。但它是
「把變換取樣放進最佳化」＝ EOT-over-purification 這條路線的代表作，
與 1.3 的 MetaCloak-JPEG 是同一支的延伸。若本專案要做抗淨化最佳化，
此為主要引用來源。

---

### 2.9 Rethinking and Red-Teaming Protective Perturbation in Personalized Diffusion Models

| 項目 | 內容 |
|---|---|
| 作者 | Yixin Liu, Ruoxi Chen, Xun Chen, Lichao Sun |
| 發表 | arXiv 2024-06-27（最新版 2026-05-14） |
| arXiv | [arXiv:2406.18944](https://arxiv.org/abs/2406.18944) |
| 程式碼 | [github.com/liuyixin-louis/DiffShortcut](https://github.com/liuyixin-louis/DiffShortcut) |

**運算**：把防護擾動的有效性歸因於 **shortcut learning**——擾動造成影像與
文字在 CLIP 空間的錯位。紅隊框架兩部分：(1) 以現成影像修復技術做資料淨化，
(2) Contrastive Decoupling Learning，引入 noise token 讓個人化概念與噪聲樣式
解耦。

**與本專案的關係**：**背景兼威脅**。它指出防護的有效性可能來自可被繞過的
捷徑而非本質困難；本專案報告的 limitation 段落應納入。有公開程式碼。

---

### 2.10 Fragile by Design: On the Limits of Adversarial Defenses in Personalized Generation

| 項目 | 內容 |
|---|---|
| 作者 | Zhen Chen, Yi Zhang, Xiangyu Yin, Chengxuan Qin, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan |
| 發表 | arXiv 預印本，2025-11-13 |
| arXiv | [arXiv:2511.10382](https://arxiv.org/abs/2511.10382) |
| 程式碼 | **未找到公開程式碼** |

**運算**：不提出新防護，而是論證既有對抗防護同時**可偵測**（有可見 artifact）
且**脆弱**（簡單濾波即可移除，模型恢復記憶身分的能力）。檢索摘要提到脆弱性
的根源與「擾動在頻域上的集中」及其與影像語意的耦合有關，但**arXiv 摘要頁
未展開頻域分析的細節——未查證**，需讀全文。

**與本專案的關係**：**動機文獻**。「擾動在頻域上過度集中導致脆弱」若在全文中
確有量化支撐，即是本專案改採頻域／相位路線最直接的立論依據。**建議取全文
查證後再引用**，目前不可據以下結論。

---

### 2.11 Protective Perturbations against Unauthorized Data Usage in Diffusion-based Image Generation（綜述）

| 項目 | 內容 |
|---|---|
| 作者 | Sen Peng, Jijia Yang, Mingyue Wang, Jianfei He, Xiaohua Jia |
| 發表 | arXiv 預印本，2024-12-25 |
| arXiv | [arXiv:2412.18791](https://arxiv.org/abs/2412.18791) |
| 程式碼 | **未找到公開程式碼** |

**內容**：防護性擾動的系統性綜述，建立威脅模型並依下游任務分類，附評測框架。
**是否涵蓋頻域方法與抗淨化章節，摘要頁未說明——未查證。**

**與本專案的關係**：**背景**，可作為 related work 的分類骨架來源，並用以確認
本專案的威脅模型敘述與領域慣例一致。

---

## 第三節 相位／幅度譜的理論與影像合成背景

### 3.1 The Importance of Phase in Signals

| 項目 | 內容 |
|---|---|
| 作者 | Alan V. Oppenheim, Jae S. Lim |
| 發表 | Proceedings of the IEEE 69(5):529–541, 1981 |
| 連結 | [MIT PDF](https://dsp-group.mit.edu/wp-content/uploads/2024/11/ImportancePhaseSignals_1981.pdf) |
| 程式碼 | 不適用 |

**內容**：只保留相位重建的影像仍可辨識，只保留幅度的則不可辨識。相位承載
可辨識內容。

**與本專案的關係**：**本方法必須正面處理的矛盾**。若相位決定可辨識內容，
大幅旋轉相位理應嚴重破壞影像；本方法的答案是紋理閘與徑向頻率閘把旋轉限制在
「相位對人眼語意貢獻低」的位置。此矛盾應在論文中明寫，不能迴避。

---

### 3.2 Random Phase Textures: Theory and Synthesis

| 項目 | 內容 |
|---|---|
| 作者 | Bruno Galerne, Yann Gousseau, Jean-Michel Morel |
| 發表 | IEEE TIP 20(1):257–267, 2011 |
| 連結 | [DOI 10.1109/TIP.2010.2052822](https://doi.org/10.1109/TIP.2010.2052822)／[Télécom PDF](https://perso.telecom-paristech.fr/gousseau/random_phase.pdf)／[HAL](https://hal.science/hal-00418389) |
| 程式碼 | 未查證（有第三方 IPOL 實作，未逐一確認） |

**內容**：定義並分析兩種 micro-texture 模型——Random Phase Noise（RPN）與
Asymptotic Discrete Spot Noise（ADSN）。RPN 把影像傅立葉係數的相位換成
均勻隨機、**幅度譜完整保留**，對 micro-texture 可得到視覺上等價的樣本。

- 加性與否：**非加性**（相位替換）。
- 保留幅度：**是**。

**與本專案的關係**：**紋理重相位的構造來源**。本專案的 `phase_rand` 對照組
即 RPN 本身。差別在 Galerne 不切塊，因此不會遇到區塊間相位不一致與
`fx=0`／`fx=N/2` 的共軛對稱處理；本方法把 RPN 的隨機換成最佳化，並套上
STFT 的重疊區塊框架。

---

### 3.3 STFT 的分析／合成骨架

| 項目 | 內容 |
|---|---|
| 論文一 | Jonathan Allen, Lawrence Rabiner. A Unified Approach to Short-Time Fourier Analysis and Synthesis. Proc. IEEE 65(11):1558–1564, 1977 |
| 論文二 | Daniel Griffin, Jae Lim. Signal Estimation from Modified Short-Time Fourier Transform. IEEE TASSP 32(2):236–243, 1984 |
| 連結 | 未查證（DOI 未於本輪重新確認） |
| 程式碼 | 不適用 |

**內容**：切塊、加窗、FFT、重疊相加的分析／合成框架（Allen & Rabiner）；
由**被修改過的** STFT 還原訊號的最小平方最佳解 `OLA(w²·x)/OLA(w²)`
（Griffin & Lim）。

**與本專案的關係**：本方法逐區塊旋轉相位後的係數集合一般**不是**任何訊號的
合法 STFT，Griffin–Lim 的重建式即把它投影回一致集合，`amplitude_deviation`
就是這個投影誤差。這是本方法「幅度保留」為何不是逐位元精確的根本原因，
必須引用。

---

### 3.4 Amplitude-Phase Recombination: Rethinking Robustness of Convolutional Neural Networks in Frequency Domain（APR）

| 項目 | 內容 |
|---|---|
| 作者 | Guangyao Chen, Peixi Peng, Li Ma, Jia Li, Lin Du, Yonghong Tian |
| 發表 | ICCV 2021 |
| arXiv | [arXiv:2108.08487](https://arxiv.org/abs/2108.08487) |
| 程式碼 | [github.com/iCGY96/APR](https://github.com/iCGY96/APR) |

**頻域上的具體運算**：資料增強——取當前影像的**相位**與另一張影像的**幅度**
重組成訓練樣本，迫使網路依賴相位、對幅度變化不變。

**核心主張**：CNN 過度依賴高頻與幅度譜，幅度譜對噪聲與 corruption 敏感；
人類視覺依賴相位。

**與本專案的關係**：**背景兼張力**。若模型天然對幅度敏感、對相位不變（APR
的訓練目標即為此），本方法只動相位可能較難影響模型；但 1.9 的實測結論相反。
兩者的差異值得在 related work 中辨明——APR 談的是**經 APR 訓練後**的網路，
不是 stock Stable Diffusion。

---

### 3.5 A Fourier-based Framework for Domain Generalization（FACT）

| 項目 | 內容 |
|---|---|
| 作者 | Qinwei Xu, Ruipeng Zhang, Ya Zhang, Yanfeng Wang, Qi Tian |
| 發表 | CVPR 2021 |
| arXiv | [arXiv:2105.11120](https://arxiv.org/abs/2105.11120)／[CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html) |
| 程式碼 | 未查證 |

**頻域上的具體運算**：**amplitude mix（AM）**——在兩張影像的幅度譜之間做
線性內插，**相位譜維持不變**，隱含地迫使模型聚焦相位；另加 co-teacher
一致性正則。

**核心假設**：傅立葉**相位含高階語意、不易受 domain shift 影響**；幅度譜對
domain shift 敏感。

**與本專案的關係**：**背景**。它是「相位=語意、幅度=樣式」這個廣泛假設的
代表引用，同時是本方法的一個潛在反證來源：若擴散模型的視覺編碼也遵循此
分工，動相位就是動語意，失真代價會很高。

---

### 3.6 High-Frequency Component Helps Explain the Generalization of Convolutional Neural Networks

| 項目 | 內容 |
|---|---|
| 作者 | Haohan Wang, Xindi Wu, Zeyi Huang, Eric P. Xing |
| 發表 | CVPR 2020（Oral） |
| arXiv | [arXiv:1905.13545](https://arxiv.org/abs/1905.13545)／[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Frequency_Component_Helps_Explain_the_Generalization_of_Convolutional_Neural_Networks_CVPR_2020_paper.html) |
| 程式碼 | [github.com/HaohanWang/HFC](https://github.com/HaohanWang/HFC) |

**內容**：CNN 能利用人眼幾乎不可見的高頻分量，這解釋了對抗樣本的存在、
強健性與準確率的取捨，以及若干訓練啟發式的效果。

**與本專案的關係**：**背景**。它是「對抗擾動天然偏高頻」這個現象的經典出處，
也就是本專案動機（高頻擾動易被淨化抹平）的上游前提。

---

### 3.7 Adversarial amplitude swap towards robust image classifiers

| 項目 | 內容 |
|---|---|
| 作者 | Chun Yang Tan, Kazuhiko Kawamoto, Hiroshi Kera |
| 發表 | arXiv 2022-03-14（後續期刊狀態未查證） |
| arXiv | [arXiv:2203.07138](https://arxiv.org/abs/2203.07138) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：把**對抗影像的幅度譜**與**乾淨影像的相位譜**重組，
產生訓練樣本（「adversarial amplitude image」）。對稱地也可構造
「adversarial phase image」。

**發現**：檢索摘要指出對抗**相位**影像比對抗**幅度**影像更能誤導分類器；
而論文本身的訓練方案是保留乾淨相位、換掉幅度，以避開 catastrophic 與
robust overfitting。**兩個敘述的精確關係需讀全文確認——部分未查證。**

**與本專案的關係**：**支持本方法的證據之一**（相位擾動的破壞力大於幅度擾動），
但任務是分類，且需以全文確認上述兩項敘述不互相矛盾後才可引用。

---

### 3.8 Phase-aware Adversarial Defense for Improving Adversarial Robustness

| 項目 | 內容 |
|---|---|
| 作者 | Dawei Zhou, Nannan Wang, Heng Yang, Xinbo Gao, Tongliang Liu |
| 發表 | ICML 2023（PMLR v202） |
| 連結 | [PMLR PDF](https://proceedings.mlr.press/v202/zhou23m/zhou23m.pdf)／[OpenReview](https://openreview.net/forum?id=EX3gxKQOoO) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：由影像相位的角度分析對抗噪聲的干擾，發現一般訓練的
模型**對相位級擾動缺乏足夠強健度**。方法為聯合防禦：(1) 相位級對抗訓練
強化相位樣式上的強健度，(2) 幅度上的前處理操作抑制幅度樣式中的擾動。

**與本專案的關係**：**支持本方法動機的最直接證據**——「一般訓練的模型對相位
擾動特別脆弱」正是本方法賭的那件事，且此結論來自防禦側（無誇大攻擊效果的
動機）。同時它也是潛在的**反制**：若攻擊方對編碼器做相位級對抗訓練，
本方法即失效；此點應寫入 limitation。

---

### 3.9 Improving Adversarial Robustness via Phase and Amplitude-aware Prompting（PAP）

| 項目 | 內容 |
|---|---|
| 作者 | Yibo Xu, Dawei Zhou, Decheng Liu, Nannan Wang |
| 發表 | arXiv 2025-02（v2 2025-05） |
| arXiv | [arXiv:2502.03758](https://arxiv.org/abs/2502.03758) |
| 程式碼 | **未找到公開程式碼** |

**頻域上的具體運算**：為每個類別建立**相位級**與**幅度級**兩種 prompt，
訓練時依模型強健度表現調整 prompt 權重，測試時依預測標籤選 prompt 生成
防禦後輸入。

**與本專案的關係**：**背景**，與 3.8 同一研究群的延伸，佐證「相位與幅度譜
分別對應不同語意樣式」在強健性文獻中已是操作性假設。

---

### 3.10 Phase Matching for Out-of-Distribution Generalization

| 項目 | 內容 |
|---|---|
| 作者 | 未查證 |
| 發表 | 未查證 |
| arXiv | [arXiv:2307.12622](https://arxiv.org/abs/2307.12622) |
| 程式碼 | 未查證 |

**內容**：以傅立葉相位為 OOD 泛化的對齊目標。**本輪僅由檢索結果確認其存在
與主題，未讀摘要或全文——方法細節、威脅模型、抗淨化測試皆未查證。**

**與本專案的關係**：**背景**，列出以便後續補查。**在查證前不得引用其結論。**

---

### 3.11 Shaping Inductive Bias in Diffusion Models through Frequency-Based Noise Control

| 項目 | 內容 |
|---|---|
| 作者 | 未查證 |
| 發表 | arXiv 2025-02 |
| arXiv | [arXiv:2502.10236](https://arxiv.org/abs/2502.10236)／[HTML](https://arxiv.org/html/2502.10236v1) |
| 程式碼 | 未查證 |

**內容**（由檢索摘要，未讀全文）：擴散模型的前向過程中**高頻的衰減遠快於
低頻**，因此逆向去噪呈現由粗到細、**先重建低頻再重建高頻**的順序；亦即
擴散模型具有 spectral bias／frequency principle。

**與本專案的關係**：**背景兼機制假設**。它解釋了為什麼落在高頻的防護擾動
會被 DiffPure 一類的擴散淨化優先抹除（與 2.4 的實測一致），也解釋了為什麼
攻擊方的 SDEdit 在低 strength 下仍保留低頻結構——即本方法要影響的目標。
**細節須讀全文後才可作為論述依據。**

---

## 第四節 與本專案的定位比較

「保幅度?」一欄指方法是否在構造上保證傅立葉幅度譜逐位（或逐點）不變；
DCT／DWT 等實數變換無幅度／相位分解，記為「不適用」。
「抗淨化測試」欄只填論文自身有做的測試。

| 方法 | 變換域 | 加性? | 保幅度? | 抗淨化測試 | 可否作為 baseline |
|---|---|---|---|---|---|
| **紋理重相位**（本專案） | 重疊區塊加窗 FFT | 否（相位旋轉） | 是（投影誤差 0.0065–0.0653） | 十個算子 | — |
| DCT-Shield（ICCV 2025） | JPEG 量化 DCT | 是 | 不適用 | JPEG 65/75/85/95、crop-resize、AdvClean、Gaussian noise、IMPRESS、noisy upscaling | **可，首選**（無公開程式碼，需自行實作） |
| BlurGuard（NeurIPS 2025） | 像素域，後處理重塑頻譜 | 是 | 否 | JPEG 等多種 reversal（清單未查證） | **可**（有公開程式碼） |
| DiffusionGuard（2024） | 無（時間步損失） | 是 | 不適用 | JPEG、crop-resize、adversarial cleaning | **可**（有公開程式碼） |
| AntiPure（ICCV 2025） | 分塊 DCT（在損失中） | 是（ℓ∞ 16/255） | 否 | GrIDPure（2×20 步，t_p=10） | 可作對照（無公開程式碼；預算需對齊） |
| AdvDrop（ICCV 2021） | 全域 DCT，可學量化 | **否**（丟資訊） | 否 | 未查證 | 可作**非加性頻域**對照（有公開程式碼；任務為分類，需移植） |
| Perturbing the Phase（2026） | 逐像素複數相位 | **否**（相位旋轉） | **是**（逐點） | 未測 | 否（複數值資料、分類任務）；但**約束式可借用** |
| Black-box phase attacks（JEI 2025） | 傅立葉相位 | 否 | 未查證 | 未查證 | 否 |
| DDAP（IJCB 2024） | 未查證 | 未查證 | 未查證 | 未查證 | 否（anti-personalization） |
| 跨影像抗 JPEG 浮水印（CVIU 2025） | JPEG 量化 DCT（僅 Y 通道） | 是（係數域 ε 球） | 不適用 | **只有 JPEG**（Q 曲線，主表 Q=75） | 否（GAN 人臉屬性編輯）；**訓練品質低於攻擊品質這一招可借用** |
| IAM 內插平滑（arXiv:2402.16586） | 無（半解析度上做 PGD） | 是 | 不適用 | **只有 JPEG**（QF 25/50/75） | 否（人臉辨識）；**半解析度參數化可借用，是唯一指向 blur 的機制** |
| JPEG-resistant Adversarial Images（MLSec 2017） | 可微分 JPEG（DCT 量化） | 是（ℓ∞） | 不適用 | **只有 JPEG**（q 25/50/75 與集成） | 否（ImageNet 分類）；**多品質集成尚未採用，可微分往返已實作** |
| MetaCloak-JPEG（2026） | 可微分 JPEG（DCT 量化） | 是（ℓ∞ 8/255） | 不適用 | 9 個 JPEG 品質因子 | 否（DreamBooth 任務）；**最佳化技巧可借用** |
| MetaCloak（CVPR 2024） | 無（變換取樣） | 是 | 不適用 | 變換取樣、Gaussian filtering | 否（DreamBooth 任務） |
| 低頻臉部交換擾動（IJCNN 2025） | DWT 低頻 | 是（未完全查證） | 不適用 | 未查證 | 否（GAN face swap） |
| LFAP／LMFAP（Inf. Sci.） | 低／中頻（變換未查證） | 是 | 不適用 | JPEG | 否（人臉辨識） |
| Low Frequency Adv. Pert.（UAI 2019） | 低頻 DCT 子空間 | 是 | 不適用 | 輸入變換型防禦（清單未查證） | 否（黑箱分類） |
| SSAH（CVPR 2022） | DWT（低頻約束正則） | 是 | 不適用 | 未查證 | 否（分類／檢索） |
| RPN（Galerne, TIP 2011） | 全域 FFT | **否**（相位替換） | **是** | 不適用 | **內部對照組**（即 `phase_rand`） |
| APR（ICCV 2021） | 全域 FFT（幅度／相位重組） | 否 | 否（換幅度） | 不適用 | 否（資料增強） |
| FACT（CVPR 2021） | 全域 FFT（幅度內插） | 否 | 否（換幅度） | 不適用 | 否（domain generalization） |
| Adversarial amplitude swap（2022） | 全域 FFT（幅度／相位重組） | 否 | 否 | 不適用 | 否（訓練方案） |
| Phase-aware Adv. Defense（ICML 2023） | 全域 FFT（相位級對抗訓練） | 否 | 不適用 | 不適用 | 否（防禦側，但為**反制威脅**） |
| PAP（2025） | 全域 FFT（prompt） | 否 | 不適用 | 不適用 | 否 |
| FreqPure（ICCVW 2025） | 全域 FFT，逐步替換低頻幅度＋投影低頻相位 | — | — | 本身是**淨化器** | **應納入淨化算子** |
| Freq.-domain Purification（2505.01267） | 全域 FFT，同上機制 | — | — | 本身是**淨化器** | **應納入淨化算子** |
| GrIDPure（CVPR 2024） | 重疊網格 ＋ 小步 DiffPure | — | — | 本身是**淨化器** | **應納入淨化算子（優先）** |
| DiffPure（ICML 2022） | 像素／latent 擴散 | — | — | 本身是**淨化器** | 已實作 |
| IMPRESS（NeurIPS 2023） | VAE 重建一致性 | — | — | 本身是**淨化器** | 已實作 |

---

## 第五節 本輪查證對專案的四項結論

以下為查證結果的整理，不構成裁決（裁決以 `DECISIONS.md` 為準）。

1. **baseline 清單應擴充三筆**：DCT-Shield（頻域加性、同一威脅場景、
   ICCV 2025 Highlight）、BlurGuard（頻譜重塑、有程式碼）、DiffusionGuard
   （抗淨化訴求、有程式碼）。目前的強 baseline（`photoguard_c`／`mist`／
   `dia_r`）沒有一個是為抗淨化設計的，在新研究目標下不足以支撐比較。

2. **淨化算子清單漏了三個針對性最強的**：GrIDPure（重疊網格，與本方法的
   重疊區塊結構同尺度）、FreqPure 與 arXiv:2505.01267（兩者都刻意保留
   對抗影像的**低頻相位**，與本方法的作用位置直接相關）。這三個是目前
   最可能推翻或最可能佐證本方法的檢定。

3. **「高頻擾動易被淨化」這句動機現在有精確出處**：arXiv:2505.01267 測到
   幅度譜與相位譜的破壞程度**都隨頻率單調遞增**。引用它比引用 DiffPure
   更貼近本專案的論點。

4. **新穎性主張的邊界維持 2026-08-16 的收窄結論不變，並新增一項威脅**：
   AdvDrop（ICCV 2021）是明確的**非加性頻域**攻擊前例，因此「非加性頻域
   操作」本身也不是首創。本專案可主張的仍是
   **加窗重疊區塊的頻譜相位旋轉 ＋ 兩個由原圖決定的閘 ＋ 擴散編輯防護 ＋
   抗淨化評測** 這個組合。另外，Phase-aware Adversarial Defense（ICML 2023）
   提供了一個現成的反制（相位級對抗訓練），應寫入 limitation。
