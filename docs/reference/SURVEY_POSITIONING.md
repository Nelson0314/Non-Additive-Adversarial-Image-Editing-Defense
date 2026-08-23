# 文獻查證：論文定位的候選空間

本檔回答一個問題：**在現有量測不變的前提下，這份工作還有哪一塊地是空的。**

不重複 `SURVEY_FREQUENCY.md`（頻域／相位方法清單）與 `SURVEY_PHASE_PRIORART.md`
（相位擾動的前例）已收錄的內容，只補三件事：

1. 數位浮水印處理**幾何去同步**（desynchronisation）的二十年成果，以及它們
   能不能搬進保護擾動；
2. 保護擾動側有沒有人做過同一件事；
3. 由此推出的候選定位，逐個標明**支持它的既有量測**、**證實或否證它要跑的
   實驗**、以及**失敗長什麼樣**。

查證深度一律照實標：**讀原文**（取得全文並逐段讀）、**讀原文（工具摘要）**
（取得全文但由摘要工具轉述，未逐行核對）、**只讀摘要**、**二手引述**（未取得
原文，內容由另一篇論文轉述）、**查不到**（付費牆或連線失敗）。

---

## 第零節 一個決定全局的結構性差異

先講一件會反覆用到的事，因為它決定了浮水印的解法哪些能搬、哪些不能。

**浮水印有一個合作的解碼器，保護擾動沒有。**

浮水印的幾何強健性幾乎全部建立在同一個前提上：接收端是**自己人**，可以
先估計影像被做了什麼幾何變換、把它反轉回去，再解讀訊號。Fourier–Mellin
不變域、模板嵌入、自相關週期 tiling、特徵點對齊、SyncSeal 的變換預測網路，
全部是這個前提的不同實作。

保護擾動的「接收端」是**攻擊方的擴散模型**。它不會反轉任何東西，也不會去找
模板。因此：

| 浮水印的作法 | 能不能搬進保護擾動 | 理由 |
|---|---|---|
| 模板嵌入 ＋ 反轉幾何（Pereira & Pun 一系） | **不能** | 需要合作解碼器 |
| 自相關／週期 tiling 的自同步 | **不能**（作為同步手段） | 同上；tiling 本身作為冗餘機制另計 |
| 特徵點對齊（SIFT／質心／主方向） | **不能**（作為同步手段） | 同上 |
| SyncSeal 式的變換預測網路 | **不能** | 同上 |
| **RST 不變域嵌入**（訊號本身在幾何群下不變） | **理論上可以** | 不需要解碼器，但代價見 §1.1 |
| **冗餘／重複嵌入**（每個子窗都自帶完整訊號） | **可以** | 不需要解碼器，只需要「局部即足夠」 |
| **訓練時對變換族取期望**（EOT／noise layer） | **可以** | 這正是保護擾動側已在做的事 |

也就是說：**「把浮水印的 resynchronisation 帶進保護擾動」這句話在字面上是
不可執行的。** 可執行的只剩兩支——不變域嵌入與冗餘，加上第三支（對變換族
取期望）本來就不屬於浮水印。這一點必須在提案裡講清楚，否則整條路會被審稿人
一句話問死。

---

## 第一節 已被佔走的地

### 1.1 Fourier–Mellin／log-polar 的 RST 不變域嵌入

**主張**：把浮水印嵌在傅立葉幅度譜的 log-polar 映射（LPM）上。空間域的旋轉與
縮放在 LPM 域裡變成兩個軸上的平移，平移在傅立葉幅度上又消失，因此構造上對
rotation／scale／translation 不變。

**與本方法重疊在哪**：這是「不靠解碼器、只靠訊號本身的幾何不變性」的原型，
也是候選定位 B 唯一有原則性前例的分支。

**實際內容與限制**（**這一段是本輪最有價值的查證結果**）：

- O'Ruanaidh & Pun (1998) 的原始構造要求原圖同時通過 LPM 與**逆 LPM**
  （ILPM）。Zheng, Zhao & El-Saddik (2003) 自行重現後寫道：作者本人
  「noted very severe implementation difficulties which might have hampered
  further work in this area」，而他們的重現得到「great ringing effect caused
  mainly by LPM and ILPM, the quality is definitely unacceptable」。
- O'Ruanaidh & Pun 自己提出的補救是只讓**浮水印訊號**過 ILPM、再加進影像的
  幅度譜。Zheng et al. 對此的評語是：浮水印訊號本身被 ILPM 扭曲，且
  「it is extremely hard to achieve the tradeoff between the invisibility of
  watermark and the robustness of the watermark」。
- Zheng et al. 自己的方案改用 LPM ＋ **phase correlation** 求位移，用近似
  ILPM 避開插值誤差，量到 PSNR **44.21 dB**（Barbara，強度取「剛好不可見」）；
  JPEG 品質降到 10% 仍可偵測、5% 失效；100 張自然影像在
  「JPEG 50% ＋ 縮放 0.7793 ＋ 旋轉 20°」下全部偵測成功。
  **但它的位移是靠與原圖的 LPM 做相位相關求出來的——仍然是合作解碼器**，
  而且論文結尾自陳未來工作是「better deal with cropping」，裁切在 2003 年
  就是這一支的弱項。

**對本專案的意義**：不變域嵌入這一支的歷史紀錄是**用失真換不變性**，而本專案
的比較全部在匹配失真下做，且目前已經在等效果上多付 1.38 倍失真
（`ip2p_axis_necessity/equal_effect_anchor.csv`）。把 ILPM 級的失真加上去，
等失真比較會直接出界。這不是調參可以解決的。

- 連結：[Zheng et al., IEEE TCSVT 13(9):753–765, 2003（PDF）](https://home.cis.rit.edu/~cnspci/references/zheng2003.pdf)
- 查證深度：**讀原文**（PDF 全文取出，逐段讀 §I、§III、§V、§VI）
- O'Ruanaidh & Pun (1998), *Rotation, scale and translation invariant spread
  spectrum digital image watermarking*, Signal Processing 66(3):303–317：
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165168498000127)
  ——查證深度：**二手引述**（僅透過 Zheng et al. 2003 的重現與評語，未取得原文）

### 1.2 模板嵌入式的 resynchronisation（Pereira & Pun）

**主張**：同時嵌兩個浮水印——一個不帶資訊的**模板**用來估計幾何變換，一個
帶 payload 的展頻訊號。偵測時先用模板反轉幾何，再解 payload。

**限制**（Zheng et al. 2003 §I 轉引 Lin et al. 的評語，原文引號內）：

> "because it requires the insertion of a registration watermark in addition
> to the data-carrying watermark, this approach is likely to reduce the image
> quality. A second problem arises because all image watermarked with this
> method will share a common registration watermark. This fact may improve
> collusion attempts to discern the registration pattern and, once found, the
> registration pattern could be removed from all watermarked images."

**與本方法重疊在哪**：不重疊，且**不可搬**（§0）。列在這裡是因為它是「把
resynchronisation 帶進保護擾動」這個想法最直觀的形態，必須明確排除。第二點
限制對白盒威脅模型尤其致命：本專案假設攻擊方知道防禦方的一切，共用模板等於
免費送給攻擊方一個可移除的目標。

- 連結：模板法原文未取得。可循 Zheng et al. 2003 的參考文獻 [6]。
- 查證深度：**二手引述**

### 1.3 自相關／週期 tiling 的自同步

**主張**：把浮水印以週期方式鋪滿全圖。浮水印的自相關函數會出現格狀峰，峰距
給出縮放、峰的方向給出旋轉，解碼端據此反轉幾何。

**限制**：搜尋結果中的浮水印綜述指出，自同步浮水印「susceptible to removal or
estimation attacks in much the same way as template-based methods, because an
attacker can use knowledge about the watermark's periodic tiling to remove
it」。週期性同時是同步機制與攻擊面。

**與本方法重疊在哪**：**作為同步手段不可搬**（§0）。但它的另一半——**冗餘
嵌入**——可搬，且是候選定位 C 的核心：多區域重複嵌入把系統從「單點失效」
變成「冗餘容錯」，這是浮水印領域對裁切的標準答案。

**但這一半與本專案的既有量測直接衝突**：冗餘要求擾動在空間上均勻鋪開，而
`allowed_budget_gini.csv` 量到本方法對 DCT-Shield 的主要構造差異就是**空間
集中度**（可用預算 Gini 0.531／合計 0.163 對 DCT-Shield 的 0）。往冗餘走等於
往 DCT-Shield 走。這個衝突不是可以繞過的，它是候選定位 C 的全部內容（見 §2.3）。

- 連結：本輪未取得原始文獻（Kutter 1998、Voloshynovskiy 一系）；上述評語來自
  搜尋結果中的浮水印綜述轉述。
- 查證深度：**二手引述**

### 1.4 SSyncOA：物件對齊的自同步浮水印（ICME 2024）

**主張**：裁切—貼上攻擊會同時引入平移、旋轉、縮放三種去同步。SSyncOA 把
浮水印限制在**被保護物件**的區域內，再用物件的不變特徵（質心、主方向、
最小外接正方形）把三種去同步各自正規化掉；編碼器—雜訊層—解碼器端到端訓練，
雜訊層裡放真實的裁切—貼上。

**與本方法重疊在哪**：兩點。

1. 它是本輪找到**唯一**把「裁切造成的去同步」拆成疊加的多個幾何分量、並分別
   處理的論文。本專案 `band_transfer.csv` 的裁切那一欄（對原網格餘弦 0.000、
   對算子搬過的同一擾動 0.995–0.996）是同一個現象的另一種量法。
2. 它的引言裡有一句對候選定位 B 直接不利的話：純靠訓練時加入去同步失真
   （HiDDeN 的 cropping、SSL 的資料增強）「various superimposed
   desynchronization distortions would place a heavy burden on model
   optimization, resulting in severe visual quality degradation」。
   **這正是隨機化幾何 EOT 最可能的死法**，而且有人先寫下來了。

**不重疊的部分**：它需要語意分割（物件遮罩）、需要解碼器重算不變特徵，兩者
在保護擾動裡都沒有對應物。

- 連結：[arXiv:2405.03458](https://arxiv.org/abs/2405.03458)
- 查證深度：**讀原文**（PDF 前六頁全文，摘要與 §1 逐段讀；實驗數字未讀）

### 1.5 SyncSeal：深度學習的幾何同步（2025）

**主張**：一個可疊加在任何既有浮水印之上的「同步浮水印」。embedder 嵌入不可見
訊號，extractor **直接預測**變換後影像的四個角點映射回原座標系的位置，藉此
還原幾何。

**內容與限制**：處理旋轉 ±135°、裁切 0.3–1.0、水平翻轉、透視 0.3、identity；
**不能預測 resize／尺度變化**（論文自陳）。PSNR 43.9、SSIM 0.994。
限制：同步模組與浮水印萃取模組之間的誤差會疊加；計算成本高於直接萃取。
程式碼開源（facebookresearch/wmar）。

**與本方法重疊在哪**：不重疊，**不可搬**（§0：它是解碼器側的東西）。列出的
理由是它代表這一支的現況（2025 年最新、Meta ＋ ETH），可用來說明「幾何同步
在浮水印裡是活的問題、有成熟解法、而那些解法在保護擾動裡結構上不適用」。

- 連結：[arXiv:2509.15208](https://arxiv.org/abs/2509.15208) ／
  [程式碼](https://github.com/facebookresearch/wmar/tree/main/syncseal)
- 查證深度：**讀原文（工具摘要）**——取得 HTML 全文，由摘要工具轉述方法、
  變換範圍、PSNR、限制與 related work；未逐行核對。

### 1.6 幾何攻擊的綜述

- Licks & Jordan, *Geometric attacks on image watermarking systems*,
  IEEE MultiMedia 12(3):68–78, 2005。同步誤差造成的效能損失、Stirmark、
  各類解法的綜覽。
  [ACM](https://dl.acm.org/doi/10.1109/MMUL.2005.46)。查證深度：**只讀摘要**
  （付費牆）。
- Zheng, Liu, Zhao & El-Saddik, *A survey of RST invariant image watermarking
  algorithms*, ACM Computing Surveys 39(2), 2007。把這一支分成兩類：
  **先矯正再偵測** 與 **在（半）不變域嵌入與偵測**。
  [ACM](https://dl.acm.org/doi/10.1145/1242471.1242473)。查證深度：**只讀摘要**
  （付費牆）。**這兩篇是 related work 的必引，但目前只有摘要級的查證。**

### 1.7 保護擾動側：有沒有人處理過幾何去同步

這是任務指定要查清的 (b)(c) 兩項。結論：**沒有人在擴散編輯防護上做過，
但相鄰場景已經有人佔位。**

| 論文 | 做了什麼 | 與本方法的重疊 | 缺什麼 | 查證深度 |
|---|---|---|---|---|
| **EOLT**（Towards Robust Protective Perturbation against DeepFake Face Swapping，[arXiv:2512.07228](https://arxiv.org/abs/2512.07228)） | 系統性評測 6 類 30 種變換（含 geometric：affine／crop／hflip／vflip／swirl，各 9 個強度）；把**變換分布本身當可學元件**，用 RL policy 學出 instance-specific 的分布（EOT 的推廣）。加性 L∞ ε=0.05、PGD 150 步。任務是換臉（SimSwap 與 ReFace，含擴散式） | **這是「對幾何族取期望」這塊地最強的佔位者**，且它比單純 EOT 更進一步 | 任務是換臉不是文字引導的擴散編輯；加性參數化；論文沒有給 crop／affine 的**逐變換**數字（只給類別層的 0.261 與 12.7%／13.0% 的相對改善）；未見程式碼 | **讀原文（工具摘要）**（HTML 全文，工具轉述） |
| **MetaCloak**（CVPR 2024 Oral，[arXiv:2311.13127](https://arxiv.org/abs/2311.13127)） | meta-learning ＋ transformation sampling。取樣的變換是 Gaussian filtering（kernel 7）、水平翻轉（p=0.5）、**center crop**、resize 到 512² | 有幾何，但**是固定的中心裁切**，與本專案 `make_eot_ops_transform` 的固定 0.10 同性質 | 任務是 DreamBooth 個人化不是編輯；沒有隨機化裁切比例與偏移 | **只讀摘要**（含專案頁與 repo README） |
| **DiffusionGuard**（[arXiv:2410.05694](https://arxiv.org/abs/2410.05694)） | 早期時間步損失 ＋ **mask augmentation**（輪廓內縮）。評測 JPEG、crop-and-resize、AdverseCleaner | 宣稱對 crop-and-resize 有韌性，是擴散編輯場景裡與我們最接近的抗裁切主張 | **最佳化時不做幾何 EOT**（工具讀出的原文如此），只對 mask 取樣；加性 L∞ 4–16/255；inpainting（需要 mask） | **讀原文（工具摘要）** |
| **DF-RAP**（IEEE TIFS 2024） | 用 ComGAN 明確建模社群平台（OSN）的壓縮，讓擾動在上傳後仍存活；另釋出 CelebA 的 OSN 傳輸資料集 | **「威脅模型收窄到部署管線」這塊地的佔位者**（見候選 E） | 是壓縮不是幾何；任務是 Deepfake 換臉 | **只讀摘要**（[IEEE](https://ieeexplore.ieee.org/document/10458678/)、[程式碼](https://github.com/ZOMIN28/DF_RAP)） |
| **Do Protective Perturbations Really Protect Portrait Privacy under Real-world Image Transformations?**（[arXiv:2604.23688](https://arxiv.org/abs/2604.23688)） | C（JPEG 75）、R（0.5× Lanczos）、**C&R 串接**三格協定；結論是只用單獨變換評測會嚴重高估 robustness | 本專案 `ROBUSTNESS_TESTS.md` §1 已收錄 | 是評測不是方法；且它的結論對我方不利（C&R 之下全部方法大幅失效） | 專案既有紀錄；本輪**只讀摘要**複驗 |
| **Compositional Adversarial Training for Robust Visual Watermarking**（[arXiv:2605.16720](https://arxiv.org/html/2605.16720v1)） | 標題與搜尋摘要顯示它把 synchronisation 當成「讓對抗壓力轉成幾何錯位後仍可解碼的浮水印」的必要環節 | 仍是浮水印（有解碼器） | — | **只讀摘要** |

**(b) 的結論**：把 resynchronisation 帶進保護擾動——**查不到任何人做過**，
但如 §0 所述，這是因為它結構上不可執行，不是因為沒人想到。真正可執行的替代
物（對幾何族取期望）**已經有人做了**：EOLT 在換臉場景，MetaCloak 在
DreamBooth 場景。

**(c) 的結論**：擴散模型防護脈絡下處理幾何去同步——**DiffusionGuard 宣稱對
crop-and-resize 有韌性但不做幾何 EOT**；除此之外查不到。**文字引導的擴散編輯
防護 ＋ 隨機化幾何族 ＋ 匹配失真比較，這個交集是空的。**

### 1.8 對抗攻擊側的幾何不變性前例（不是保護擾動，但是同一個技術）

| 論文 | 做了什麼 | 與本方法的關係 | 查證深度 |
|---|---|---|---|
| **Athalye et al., Synthesizing Robust Adversarial Examples**（ICML 2018，[arXiv:1707.07397](https://arxiv.org/abs/1707.07397)） | EOT 本身。對旋轉、光照、感測器雜訊、壓縮取期望 | 幾何 EOT 的出處，必引 | **只讀摘要** |
| **Brown et al., Adversarial Patch**（2017） | 對 patch 的位置、尺度、旋轉取期望，做出位置無關的 patch | 「對幾何族取期望可換到位置不變性」的最早示範 | **只讀摘要**（經 Athalye 與 Niederhut 轉述） |
| **TI-FGSM**（CVPR 2019，[arXiv:1904.02884](https://arxiv.org/abs/1904.02884)） | 對平移取期望，並以「梯度與核卷積」做閉式近似，省掉多次前向 | **技術上可直接借用**：平移那一半的 EOT 有免費的近似式 | **只讀摘要** |
| **SI-NI-FGSM**（ICLR 2020，[arXiv:1908.06281](https://arxiv.org/abs/1908.06281)） | 對多個尺度副本取期望 | 尺度那一半的對應物 | **只讀摘要** |
| **Area is all you need**（[arXiv:2306.07768](https://arxiv.org/abs/2306.07768)） | 把很小的 patch **鋪滿**（tiling）整個物件再一起最佳化，宣稱之前 ASR 的提升主要來自面積而非技巧 | **冗餘／tiling 在對抗攻擊側的前例**，支持候選定位 C 的機制假設 | **讀原文**（PDF 前 8 頁） |

**必須寫進 limitation 的一件事**：TI-FGSM 與 SI-NI-FGSM 的動機是**黑盒轉移**，
而 `GOAL.md` 明定黑盒轉移不在範圍內。引用它們時只能引用「對變換族取期望可以
換到對該族的不變性」這個技術事實，不能借用它們的實驗結論。

### 1.9 淨化失效的機制分析：現況

| 論文 | 分析到什麼程度 | 缺什麼 | 查證深度 |
|---|---|---|---|
| **Diffusion-based Adversarial Purification from the Perspective of the Frequency Domain**（[arXiv:2505.01267](https://arxiv.org/abs/2505.01267)） | 量到擾動對**幅度譜與相位譜的破壞程度都隨頻率單調遞增** | 是「隨頻率」的一維剖面，沒有把破壞拆成不同**種類** | 專案既有紀錄（`SURVEY_FREQUENCY.md` §2.4） |
| **MANI-Pure**（[arXiv:2509.25082](https://arxiv.org/abs/2509.25082)） | 指出既有擴散淨化假設擾動在頻域均勻分布，而實際上「集中在高頻、且各頻帶的幅度強度不均」，據此做幅度自適應注入 | 同上，是頻帶剖面 | 專案既有紀錄；本輪**只讀摘要**複驗 |
| **AntiPure**（ICCV 2025，[arXiv:2509.13922](https://arxiv.org/abs/2509.13922)） | 形式化 anti-purification；Patch-wise Frequency Guidance ＋ Erroneous Timestep Guidance | 是方法不是診斷；仍以頻率為唯一軸 | 專案既有紀錄 |
| **The Purification Paradox: Dissecting and Exploiting Generative Vulnerability Bands**（CVPRW 2026） | 標題指向「頻帶級的脆弱性剖析」 | — | **查不到**（CVF 連結回 HTTP 403，未取得內容。**不要在論文裡引用它的內容**，先確認能不能拿到） |

**結論**：既有的機制分析全部把「淨化把擾動破壞了多少」當成**一個沿頻率變化
的純量**。本專案 `band_transfer.csv` 的三分法（能量／方向／對位）與那一欄
「對算子自己搬過的同一擾動」的控制組，**查不到前例**。這是候選定位 A 的
全部依據。

### 1.10 非加性／頻域重參數化：AdvDrop 之外

| 論文 | 做了什麼 | 查證深度 |
|---|---|---|
| **AdvDrop**（ICCV 2021） | 全域 DCT ＋ 可學量化，**丟掉**資訊而非加上噪聲 | 專案既有紀錄，已實作 |
| **AdvWave**（Pattern Analysis and Applications, 2025，[Springer](https://link.springer.com/article/10.1007/s10044-025-01458-1)） | AdvDrop 的小波版：以 DWT 取代 DCT，宣稱 AdvDrop 因分塊 DCT 而有 blocking artifact、頻率解析不足；在低頻加擾動、在量化時丟棄判別資訊 | **只讀摘要** |
| **Perturbing the Phase**（[arXiv:2602.06577](https://arxiv.org/html/2602.06577)） | 只動相位、幅度逐點保留 | 專案既有紀錄，`SURVEY_PHASE_PRIORART.md` §2.1 |

AdvWave 是本輪唯一的新增。它的存在讓「非加性頻域重參數化」這塊地更擁擠，
但它是分類任務、且與本方法的重疊只在「非加性」這個大類上。**不足以支撐一個
新定位，但必須引用**，否則 AdvDrop 那條收窄過的新穎性宣告仍會被追問。

---

## 第二節 候選定位

每一個候選都對照第零步的五個硬事實檢查過。凡與 `GOAL.md`「明確不做的事」或
`RESULTS.md`「已否決的方向」衝突者，衝突處直接寫出來。

---

### 候選 A — 淨化失效的機制三分法（診斷型貢獻）

**一句話主張**：保護擾動的「抗淨化」不是一個軸。淨化算子以三種互不相同的
機制破壞擾動——**拿走能量**、**打散方向**、**只是搬走**——每一種需要不同的
對策，而既有文獻把三者混成一個沿頻率變化的保留率。

**支持它的既有量測**（全部來自 `runs/ip2p_residual_signature/band_transfer.csv`，
13 張，不跑 GPU）：

| 機制 | 算子 | 讀數 |
|---|---|---|
| 能量 | blur σ=1 | 方向存活率每帶 **0.88–1.00**（活下來的部分方向完全對得上），但半 Nyquist 以上能量只剩 **0.3%** |
| 方向 | jpeg75 | 能量比 **1.4–2.8**（大於 1），方向由 **0.68 掉到 0.05** |
| 對位 | crop_resize 0.1 | 能量留 **51–99%**，對原網格方向 **0.000**，對**算子自己搬過的同一擾動** **0.995–0.996** |

第三列的控制組是這個貢獻的核心：把同一個擾動放在中性灰上跑同一個算子，量到
的是算子對擾動本身做了什麼、與影像內容無關。它把「裁切等於去同步」從推論變成
量測。

**誰做過相近的、還缺什麼**：見 §1.9。2505.01267 與 MANI-Pure 都只給頻率剖面；
AntiPure 是方法不是診斷。**沒有人把破壞拆成種類，也沒有人用「算子搬過的同一
擾動」當控制組。**

**要跑什麼實驗**：

1. 把 `band_transfer.csv` 的算子集合由 3 個擴到全部（加 `gridpure`、
   `adverse_cleaner`、`jpeg_then_resize` 串接、`resize_only`）。不跑 GPU 的
   算子可以直接補；`gridpure`／`diffpure` 需要權重。
2. 把它跑在**別人的殘差**上（DCT-Shield、PhotoGuard-c、Mist、AdvDrop 的
   `*__def.png` 都已存在），證明三分法是算子的性質而不是本方法的性質。
3. **關鍵驗證**：三分法要能**預測**淨增益的排序。也就是：對某個算子而言
   方向存活率高、能量存活率高的方法，其淨增益應該高。

**失敗長什麼樣**：

- 第 3 步失敗——三個量與淨增益沒有單調關係，或關係被第四個沒量到的因素支配。
  那麼三分法只是一組描述性統計，不是機制。
- 三個軸塌成一個。既有 7 個條件裡 `block_gini` 與 `hf_share` 幾乎完全反相關
  （0.65／0.278 對 0.098／0.954，`signature.csv`），若三個機制軸也彼此共變，
  「三分」就只是「頻率」換個講法。
- 樣本數：13 張影像、抗淨化那一線只有 5 張且空白地板只齊 6/13。要撐一個
  「機制」層級的主張，這個規模會被質疑。

**與已否決清單的關係**：**無衝突**。這是量測，不動方法，不碰任何已否決的旋鈕。

**與五個硬事實的相容性**：完全相容，因為它不主張本方法比較強。**代價是它與
`GOAL.md` 的主張階層衝突**——`GOAL.md` 把「抗淨化勝出」列為主主張，而這個
定位把方法降級為量測工具。這是一次論文重心的改寫，不是補一節。

---

### 候選 B — 幾何去同步是保護擾動的獨立失效模式，對策是隨機化幾何族

**一句話主張**：裁切縮放打敗保護擾動的方式與模糊、壓縮不同——它不移除擾動，
只把擾動搬走；因此對策不是選頻帶、不是加預算，而是對一族幾何取期望。

**支持它的既有量測**：

- `band_transfer.csv` 的裁切欄（上表第三列）。**沒有任何一帶的方向存活率高於
  0.02**，故選頻帶救不了它——這是排除法，不是猜測。
- `RESULTS.md` 三、IP2P 線抗淨化：crop_resize 0.1 的空白地板 0.4987，
  DCT-Shield **+0.1083**、本方法 **+0.0360**，逐圖 **0/5**，輸 3.0 倍。
- DCT-Shield 單獨的保留率（69 張）：裁切縮放 **0.982**，而本方法同格只留
  **13%**。同一個算子對兩種參數化的效果差七倍以上——這本身就是「參數化決定
  幾何脆弱性」的直接證據。

**誰做過相近的、還缺什麼**：見 §1.7。EOLT 佔了「學出來的變換分布」這塊地
（換臉、加性）；MetaCloak 佔了「固定中心裁切」（DreamBooth）；DiffusionGuard
宣稱抗 crop-and-resize 但**最佳化時不做幾何 EOT**。**文字引導擴散編輯 ＋
隨機化幾何族 ＋ 匹配失真，這個交集是空的。**

**要跑什麼實驗**：

1. **先跑那個決定性的拆解，成本最低、資訊量最大。** 目前「裁切等於去同步」
   是由殘差的餘弦推出來的，而 `crop_resize 0.1` **同時**做了兩件事：
   中心裁掉每邊 10%（平移原點）與 1.25× 的重取樣（改變尺度）。把它拆成
   **純平移**（同尺度、只 roll 或偏移取樣格）與**純重取樣**（同構圖、只換
   尺度）兩個算子，各自量淨增益。這一步不需要重跑防禦，`phase_retention.py`
   讀已存的防禦圖即可。
   - 若**純平移幾乎不掉、純重取樣掉光**：問題不是「對位」而是**尺度**，
     「desynchronisation」這個詞用錯了，正確的對策是尺度多樣性（多尺度或
     自相似的擾動），而隨機化裁切偏移是白做的。
   - 若**純平移就掉光**：對位是真的，隨機化幾何 EOT 是對的方向。
2. `scripts/eot_geometry_sweep.sh` 已寫好、`make_eot_geometry_transform` 已
   實作（每步抽裁切比例**與位置**，族內含 identity），三個半徑與
   `ip2p_axis_necessity/b_ph_*` 對齊以便做等失真對照。
3. **必須加一組對照：DCT-Shield ＋ 同一個幾何 EOT。** 沒有這組，任何改善都
   無法歸因給參數化。

**失敗長什麼樣**（三種，都有前例）：

- **強度換抗性**：裁切淨增益上升，但未淨化效果掉 10–25%，等失真下總帳沒有
  變好。這正是既有三個 purify-aware 變體發生的事
  （`RESULTS.md` 三、「針對淨化最佳化沒有改善抗淨化」）。
- **失真爆掉**：SSyncOA 引言明寫，疊加的去同步失真會「place a heavy burden on
  model optimization, resulting in severe visual quality degradation」。本專案
  在等效果上已多付 1.38 倍失真，沒有餘裕。
- **歸因失敗**：DCT-Shield ＋ 同一個幾何 EOT 改善得比我們多。那麼貢獻是 EOT
  本身、與參數化無關，落回硬事實 1 的同一個陷阱——**讓方法有效的東西正是讓它
  不獨特的東西**。

**與已否決清單的關係**：**必須主動澄清，否則會被當成重試已否決項。**
`RESULTS.md` 三否決的是「把可微分淨化算子放進 PGD 前向」的**三個變體**
（固定 JPEG75／課程排程／多算子 EOT），其否決理由寫得很清楚：JPEG 那格保留率
已達 97.5%、沒有優化空間。**隨機化幾何 EOT 是第四個變體，且 `PENDING.md`
二把它列為待裁定而非已否決**；`make_eot_geometry_transform` 的 docstring 也
明確區分了「固定的幾何可被 co-adapt」與「對一族取期望」。提案時要引這兩處。

**與五個硬事實的相容性**：與 1、3 相容且由 3 直接推出。**與 2 有張力**：
未淨化那條已經輸，EOT 只會讓它更輸；所以這個定位只能搭配「主主張是抗淨化」
的敘事，而抗淨化目前唯一明確贏的軸（JPEG）還在待驗（硬事實 5）。

---

### 候選 C — 空間選擇性與幾何脆弱性的取捨（把硬事實 1 由弱點翻成貢獻）

**一句話主張**：保護擾動有一個可量的構造軸——**預算被允許花在哪裡**的空間
集中度。集中度買到感知效率，付出的是幾何脆弱性；DCT-Shield 與本方法是這條
取捨曲線的兩端，而加性下限是把本方法沿曲線往中間推的旋鈕。

這是本輪唯一一個**把硬事實 1 從「我們不獨特」翻成「我們量出了一條規律」**的
定位。加性下限吃掉 67.6% 的預算、把 Gini 由 0.531 壓到 0.163——在候選 A、B 的
敘事下這是失去獨特性；在這個敘事下這是**方法沿著自己發現的取捨曲線移動**。

**支持它的既有量測**：

- `allowed_budget_gini.csv`：可用預算的逐區塊 Gini——乘法半邊單獨 **0.5311**、
  `uniform` 合計 **0.1625**、`complement` **0.0783**、`complement_rank`
  **0.1144**、`watson` **0.2674**、DCT-Shield 由構造 **0**。
  **五個點跨 0 到 0.27，而且是同一個方法的旋鈕產生的。**
- `signature.csv`：實際殘差的 `block_gini` 0.428–0.654 對 DCT-Shield
  0.097–0.111，差四到六倍。
- 曲線的兩個端點已經有對應的幾何讀數：DCT-Shield（Gini 0）裁切保留 **0.982**；
  本方法（Gini 高）裁切只留 **13%**。
- 浮水印領域對這條規律有獨立的、二十年的定性版本：多區域重複嵌入把「單點失效」
  變成「冗餘容錯」（§1.3）；`Area is all you need` 是對抗攻擊側的對應物（§1.8）。

**誰做過相近的、還缺什麼**：**查不到任何人量過保護擾動的空間集中度。**
`Interpreting Structured Perturbations`（[arXiv:2512.08329](https://arxiv.org/abs/2512.08329)）
分析 Glaze／Nightshade 的**頻譜**結構（沿主頻率軸重分配能量），沒有碰空間分布。
浮水印側的冗餘嵌入是定性的工程慣例，沒有把「集中度」量成一個連續軸再與幾何
強健性掛勾。

**要跑什麼實驗**：

1. 拿 `--floor-gate` 的四個變體（uniform／complement／complement_rank／watson）
   ＋ DCT-Shield，五個 Gini 點（0、0.078、0.114、0.163、0.267），在**匹配失真**
   下量 `crop_resize` 的淨增益。假設是單調遞減。
2. **這個設計的價值在於它能解開混淆。** 既有 7 個條件裡 `block_gini` 與
   `hf_share` 幾乎完全反相關，所以「空間集中度」與「頻帶」分不開。而四個
   `floor_gate` 變體**只改預算的空間分配、不改徑向帶通**，正好把兩者拆開。
   這是目前唯一能做到這件事的實驗設計。
3. 若第 1 步成立，再加第三個端點：把加性下限的比例當連續旋鈕
   （`--spectral-floor` 0 / 0.02 / 0.04 / 0.08）掃出曲線。

**失敗長什麼樣**：

- **沒有單調關係。** 五個點跨的 Gini 範圍是 0–0.27，而方法之間的差距是
  0.10 對 0.65（殘差層）——**floor_gate 變體撐開的範圍可能太窄，量不出趨勢**。
  這是最可能的死法。
- **混淆沒解開。** `watson` 是亮度×對比遮蔽，它同時改了頻率分配；若四個變體
  的 `hf_share` 也跟著動，拆解就失敗。這一點**跑之前就可以用
  `residual_signature.py` 檢查**，不需要 GPU。
- **效果差異被失真對齊吃掉。** `PENDING.md` 已記 `watson` 在等 DISTS 下位移
  掉 4–6%、`complement` 幾乎免費。若五個點的未淨化效果本來就不同，淨增益的
  差就無法歸因給 Gini。
- 樣本數同候選 A（抗淨化 5 張、地板 6/13）。

**與已否決清單的關係**：需要澄清一處。`PENDING.md` 一把 `--floor-gate watson`
列為「證據完整、可以裁定否決」（等失真下位移掉 4–6%）。**本提案不是要採用
`watson` 當操作點，是要把它當成取捨曲線上的一個探針點。** 這個區別必須寫明，
否則會被讀成重試已否決項。`complement` 與 `complement_rank` 仍在待裁定。

**與五個硬事實的相容性**：這是唯一一個把硬事實 1 用起來而不是繞開的定位。
與 2 相容（不主張強度勝出）。與 3、4 相容且互補——三分法給機制，這條曲線給
**是什麼構造性質決定了落在哪個機制上**。與 5 無關（不依賴 JPEG 那一格）。

---

### 候選 D — 非加性參數化買的是感知代價，不是防禦強度

**一句話主張**：在相同的防禦效果下，不同的參數化在**不同的失真指標上**收費
不同；非加性的頻譜重參數化在 PSNR 上便宜 2.4–3.2 dB，而 PSNR 是失真帶的另一半。

**支持它的既有量測**（`RESULTS.md`「相位那一半買的是可見度，不是效果」）：
等 DISTS 0.1430／0.1409 下，純加性下限 PSNR **21.50**、相位＋下限 **24.65**，
擋下率 12/13 對 11/13。純加性那一點因 PSNR 21.50 直接出失真帶。

**誰做過相近的、還缺什麼**：AdvDrop 與 Perturbing the Phase 都是非加性前例，
但都沒有做「等效果下的多指標定價」比較。**查不到前例。**

**要跑什麼實驗**：把 `distortion_axis_analysis` 的迴歸擴成「每個方法在每個
失真指標上的定價」矩陣（LPIPS／DISTS／PSNR／SSIM／GMSD／HaarPSI），證明
定價差異是系統性的、不是單一指標的巧合。

**失敗長什麼樣**：**這個定位有一個已量到的內部矛盾。**
`RESULTS.md` 四已經測到 DISTS 對不同方法的定價差 3.40 倍、而 LPIPS 只差 1.27 倍，
並據此裁定「強度比較的主軸取 LPIPS」。**但上面那個 PSNR 優勢是在等 DISTS 上
成立的。** 若改用 LPIPS 錨點，兩條曲線在 LPIPS 軸上**完全不重疊**
（DCT-Shield 0.441–0.623，本方法 0.046–0.262），錨點全部 `out_of_range`。
也就是：**這個主張賴以成立的錨點，正是本專案自己判定不可靠的那一個。**
不先解決這件事，這個定位站不住。

**與已否決清單的關係**：無衝突。

**與五個硬事實的相容性**：與 2 相容（承認強度輸）。**與 1 有張力**：等 DISTS
下純加性擋下率一樣好，而加性下限已佔 67.6% 預算，所以「非加性」實際貢獻的
比例本身就是要被追問的。

---

### 候選 E — 威脅模型收窄到部署管線

**一句話主張**：現實的部署情境是「受保護的影像先經過平台的重壓縮與重取樣，
攻擊方才拿去編輯」，所以評測算子集合應該是平台管線而不是任意的淨化器動物園。

**支持它的既有量測**：JPEG 家族是目前唯一明確贏的軸（但硬事實 5 待驗）。

**誰做過相近的、還缺什麼**：**這塊地大致上被佔走了。** DF-RAP（TIFS 2024）
用 ComGAN 明確建模 OSN 壓縮並釋出傳輸資料集；arXiv:2604.23688 給了 C／R／C&R
的三格協定；另有 `Compression as an Adversarial Amplifier`
（[arXiv:2604.06954](https://arxiv.org/html/2604.06954v1)，**只讀摘要**）與
`Unlearnable Faces: Privacy Protection Surviving Extraction Pipeline`
（[arXiv:2607.05996](https://arxiv.org/pdf/2607.05996)，**只讀摘要**）。

**失敗長什麼樣**：占位者的結論對我方不利——2604.23688 測到 C&R 串接之下全部
方法大幅失效。把威脅模型收窄到這裡，等於自願走進一個已知會輸的格子。

**建議**：**不作為定位，作為評測協定採用**（`ROBUSTNESS_TESTS.md` §1 已規劃）。

---

## 第三節 排序與建議

### 排序

| 序 | 定位 | 為什麼在這個位置 |
|---|---|---|
| **1** | **C（空間選擇性 ↔ 幾何脆弱性）＋ A（機制三分法）合併** | 唯一把硬事實 1 用起來而不是繞開的路。A 提供機制、C 提供構造原因，兩者互為對方缺的那一半 |
| 2 | B（隨機化幾何 EOT） | 方向由量測直接推出、程式已備妥，但三種死法都有前例，且成功了也可能歸因不到參數化 |
| 3 | D（感知定價） | 主張本身乾淨，但賴以成立的錨點是本專案自己判定不可靠的那一個 |
| 4 | E（威脅模型收窄） | 地被佔走，且占位者的結論不利。降為評測協定 |

### 建議

**最強的定位是 C＋A 的合併，一句話是：**

> 保護擾動的抗淨化不是單一軸。淨化以三種機制破壞擾動（拿走能量／打散方向／
> 只是搬走），而落在哪一種由一個可量的構造性質決定——**預算被允許花在哪裡的
> 空間集中度**。集中度買到感知效率，付出的是幾何脆弱性。

這個敘事的三個好處：

1. **它不需要贏。** 硬事實 2（未淨化輸）、4（模糊是結構性的輸）、5（JPEG 待驗）
   都不會殺死它。DCT-Shield 在這個敘事裡不是被打敗的對手，是曲線的另一個端點。
2. **它把加性下限由弱點變成證據。** 67.6% 的預算佔比、Gini 0.531 → 0.163，
   在這個敘事下是「同一個方法可以沿曲線移動」的示範，而不是「方法失去了獨特性」。
3. **兩件工具都已經寫好且不跑 GPU**（`residual_signature.py`、
   `purifier_band_transfer.py`），`allowed_budget_gini.csv` 的五個 Gini 點也
   已經在手上。

### 取捨要講清楚的三件事

- **這是一次重心改寫，不是補一節。** `GOAL.md` 現行的主張階層是
  「主＝抗淨化勝出、並列＝未淨化不低於對照」。C＋A 把方法降級成量測的工具，
  兩條主張都要重寫。這需要使用者裁定，不是可以順手做的事。
- **B 應該保留為 C＋A 的驗證步驟，不是獨立定位。** 若 C 的取捨曲線成立，
  那麼幾何 EOT 就是「沿曲線往低集中度移動的代價」的直接檢定：EOT 應該讓
  裁切變好、讓感知效率變差。這比把 B 當成獨立的方法改良更值錢，也把 B 的
  三種死法轉成了 C 的三個預測。
- **樣本數是共同的死穴。** 13 張（指紋、band_transfer）、5 張且地板只齊 6/13
  （抗淨化）。任何「機制」或「規律」層級的主張都要面對這個規模。擴樣本的成本
  必須先估，不然三個候選都會在同一個地方被問倒。

### 建議的下一步（由便宜到貴）

1. **不跑 GPU**：對四個 `floor_gate` 變體跑 `residual_signature.py`，檢查
   `block_gini` 動的時候 `hf_share` 有沒有跟著動。**若跟著動，候選 C 當場
   否決**，省下所有後續成本。
2. **不重跑防禦**：把 `crop_resize` 拆成純平移與純重取樣兩個算子，用
   `phase_retention.py` 讀已存的防禦圖量淨增益。這一步決定「desynchronisation」
   這個詞用得對不對，也決定候選 B 該做隨機偏移還是多尺度。
3. **要 GPU**：`PENDING.md` 三那一格（JPEG 重測，`dct_antijpeg_configs.sh`
   ＋ `purify_antijpeg.sh`）仍然優先於以上一切——**在跑完之前不要用任何
   JPEG 的數字做規劃**，這一點本檔的所有排序都遵守（沒有任何一個候選依賴
   JPEG 那一格）。
4. 才是 `eot_geometry_sweep.sh`。

---

## 第四節 明確查不到的

以下項目本輪未能查證，**不要用摘要腦補**：

- **O'Ruanaidh & Pun (1998) 原文**：未取得。全部內容經 Zheng et al. (2003) 轉述。
- **Pereira & Pun 的模板法原文**：未取得。評語經 Zheng et al. (2003) 轉引
  Lin et al.。
- **Licks & Jordan (2005) 幾何攻擊綜述**：付費牆，只讀摘要。
- **Zheng et al. (2007) ACM CSUR 的 RST 不變浮水印綜述**：付費牆，只讀摘要。
  **這是 related work 的必引，需要另尋管道取得。**
- **Kutter (1998) 一系的自相關／tiling 自同步原文**：未取得。§1.3 的限制
  評語來自搜尋結果中的綜述轉述，**未核對原始出處**。
- **The Purification Paradox（CVPRW 2026）**：CVF 連結回 HTTP 403，內容
  完全未取得。這是候選 A 唯一可能的直接競爭者，**必須在提案定案前拿到**。
- **EOLT（arXiv:2512.07228）的逐變換數字**：論文只給類別層結果，crop 與
  affine 的個別數字在取得的內容中沒有。是否有附錄未確認。
- **EOLT 的程式碼**：未見發布。
- **本輪未查的方向**：`GrIDPure` 的重疊網格與本方法 32×32／hop 8 重疊區塊
  是否有尺度上的交互作用（`SURVEY_FREQUENCY.md` §5 第 2 點提過，本輪未推進）；
  DT-CWT 這一支（近似平移不變的變換，`BIBLIOGRAPHY.md` §3b 列為值得挖）
  本輪**未查**。
