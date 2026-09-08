# 除了色彩之外，還有哪些「不同的進入方式」

**這一份不是判準來源。** 判準見 [../DECISIONS.md](../DECISIONS.md)，測得的事實見
[../RESULTS.md](../RESULTS.md)。本頁回答一個問題：

> `docs/DIRECTION.md` §2.2 的色彩載體已被否定（§6.0）。文獻上還有哪些載體
> 滿足「防禦以不同的方式進入影像」，而且本專案還沒走過？

本專案已走過的載體先列出來，避免重複：加性 δ、相位／DCT 旋轉（已凍結）、
色彩映射（已否定）、補丁（現行主線）、位移場（`ip2p_warp`／`disp_*`，
WaNet 式）、明暗場（`ip2p_shading`）、散射（`ip2p_dispersion`）、
語意誘餌（已結案）、量化表（AdvDrop，對照組）。

---

## 〇 一個必須先記下的命名衝突

**「STP-Diff」有兩篇完全不同的論文。**

| | 本專案要的那一篇 | 另一篇（同縮寫） |
|---|---|---|
| 全名 | *Synergistic fusion of **S**patial **T**ransformation **P**erturbations and diffusion models for robust face privacy protection* | *Differentiable Adversarial Discovery of Regulatory Vulnerabilities in Gene Circuits via **S**emi-**T**ensor **P**roduct Mapping* |
| 領域 | 人臉隱私保護 | 布林基因調控網路、計算腫瘤學 |
| 出處 | Information Fusion, 2025-12（[ScienceDirect S1566253525011315](https://www.sciencedirect.com/science/article/abs/pii/S1566253525011315)） | Khalil, Virginia Tech（github.com/yaskhalil/STP-Diff） |
| 驗證 | — | 全文 11 頁逐字檢查：diffusion／image／privacy／purification／watermark 各出現 **0 次**，Boolean 32 次、gene 24 次、CRISPR 6 次 |

**取得這一族文獻時要以全名比對，不可用縮寫。**

---

## 一 STP-Diff（人臉隱私那一篇）：定位上與本專案最接近

**全文取不到，已停止嘗試。** 付費牆後，且無 arXiv 預印本、無公開程式碼。
以下每一條都是**摘要層級**，不可當成逐行查證過的內容——`CLAUDE.md` 的
「移植他人的方法」要求逐行對照公開原始碼，這一篇達不到那個標準，
**故本專案不移植它的方法，只用它定位**。

投稿時要寫成「據其摘要」，不可寫成「該方法如何如何」。若日後取得全文，
要抽的是三件事：**空間變換的具體參數化**、**非顯著區域怎麼定義**、
**有沒有報 JPEG／模糊／重取樣**。

| 項目 | 內容 | 來源層級 |
|---|---|---|
| 主張 | 首度把**空間變換擾動（STP）**用於黑箱**目標式**人臉隱私保護 | 摘要 |
| 機制 | 非加性的空間擾動放在**非顯著區域**當「前置擾動」，用來抵抗擴散的淨化效應；再把擴散模型的生成能力集中到**身分關鍵區域**，在那裡產生加性擾動 | 摘要 |
| 讀數 | PSR 81.09%、FID 8.79 | 摘要 |

### 與本專案的三個結構性差異

1. **它動臉，本專案不動。** STP-Diff 的加性擾動明確放在「身分關鍵區域」；
   本專案的整個前提是**受保護主體逐位元不動**（`PatchParam.reset` 由構造保證）。
2. **判準不同。** 它打人臉辨識（PSR＝protection success rate）；本專案打的是
   **指令編輯模型**，判準是編輯輸出還認不認得出來。
3. **它把非加性當「前置」，本專案把非加性當「載體本身」。** 它的主要擾動仍是
   加性的，空間變換只是為了抗淨化而先鋪一層。

**後果**：兩者不是同一個主張，但**「非加性 ＋ 區域限制 ＋ 抗淨化」這個組合已經
有人做了**，投稿時必須引用並說清楚差異。

**對照已經用本專案自己的資料做完了**，不需要它的全文——見
[`runs/readout_parallel/README.md`](../../runs/readout_parallel/README.md) 的附表：
本專案的 `warp` 單獨當載體時，要付兩倍的失真（DISTS 0.280 對補丁的 0.187）
才拿到相當的擋下率（60% 對 73%）。**那與 STP-Diff 把空間變換降級為「前置層」
而不是主載體的作法方向一致**，這一條不靠它的全文也站得住。

---

## 二 候選載體：七類，含本專案的適配判讀

### 2.1 重取樣格點載體（image-scaling attack 反用）——**在本專案的評測協定下無效**

**機制**：Xiao et al.（USENIX Sec'19）、Quiring et al.（USENIX Sec'20）。降取樣時
只有特定像素進入輸出，少數像素即可控制縮圖。擾動因此是**潛伏的**：全解析度
看不到，降取樣後才浮現。概念上直接反轉本專案最弱的那一欄（`resize_only` 1%）。

**為什麼不成立**：Quiring et al. 指出**均勻核或動態核寬的縮放算法天生免疫**，
area scaling 是標準解。而本專案 `src/purify/ops.py:201-202` 用的是
`bicubic + antialias=True`，**antialias 在 PyTorch 就是動態核寬**。

**判讀**：這是威脅模型的觀察（真實縮圖器常不開 antialias），不是研究方向。
要用就得把評測協定改弱。**不走。**

### 2.2 摩爾紋／差頻週期——**與 2.1 共用同一個死因**

**機制**：Moiré-Watermark（IEEE 2025）把**失真本身當載體**——摩爾紋是螢幕像素
格與相機感光元件的頻率干涉產物。相關的還有 JND-guided neural watermarking
（[arXiv:2603.26766](https://arxiv.org/html/2603.26766)），模擬 LCD 次像素重取樣
→ 透視變換 → Bayer CFA 內插的完整物理鏈。

吸引力在於摩爾紋是**兩個高頻的差頻**，落在低頻，而低頻是重取樣抹不掉的。

**但差頻要產生，必須有混疊。** 螢幕翻拍那條鏈沒有抗混疊濾波（相機的光學低通
很弱），所以摩爾紋才會出現；本專案的重取樣算子開著 antialias，**混疊在取樣前
就被濾掉了**。實測見 [`runs/moire_probe/README.md`](../../runs/moire_probe/README.md)。

**判讀**：與 2.1 同一個死因。**除非改威脅模型（納入螢幕翻拍），否則不走。**

### 2.3 結構化圖樣 ＋ 固定調色盤——**可做，而且與現行程式接得上**

*Structured Adversarial Camouflage via Voronoi Diagrams*
（[arXiv:2606.17711](https://arxiv.org/pdf/2606.17711)）：最佳化 Voronoi 種子點
位置與每格顏色，**胞的結構初始化後固定、調色盤固定且可印**。給的理由是
「離散受限的色彩空間降低對數位渲染的過擬合，提升實體世界的轉移性」，
跨 YOLOv9–12 黑箱轉移。

**與本專案的關係**：`PatchPaletteParam`（`--patch-palette K`）已經做了「固定
基數的色彩」那一半；Voronoi 多的是**把幾何也參數化**——種子點位置是可學的，
而胞邊界由 Voronoi 圖決定，天生是規則結構。那是本專案還沒有的旋鈕，
且直接對應 `DIRECTION.md` 證據清單第 5 項（產物像標記不像壞掉）。

**判讀**：**值得做**，實作見 `src/defense/patch_param.py`。

### 2.4 空間變換擾動（STP）——本專案已有資料，要做的是對照不是重跑

`runs/ip2p_warp`／`ip2p_warp_hard`／`disp_*`／`runs/displacement_decomposition`
已有 WaNet 式位移場的完整批次。**先把既有資料與 §1 的 STP-Diff 對照**，
不要重跑。

### 2.5 半色調／誤差擴散——**文獻站在對面**

*Error Diffusion Halftoning Against Adversarial Examples*
（[arXiv:2101.09451](https://arxiv.org/abs/2101.09451)）與 *Dithering Defense*
（CVPR 2025）都是拿半色調來**清掉**擾動。當載體用的只有 Adversarial Halftone
QR Code。**有人證明它是好的淨化器**，這對當載體不利。**不走。**

### 2.6 RAW／相機管線域——威脅模型不合

*Adversarial RAW*（[arXiv:2206.01733](https://arxiv.org/pdf/2206.01733)）在 ISP
之前進入影像。本專案的場景是使用者上傳既有 JPEG，**沒有 RAW**。**不走。**

### 2.7 後門觸發（GuardDoor）——威脅模型不合

[arXiv:2503.03944](https://arxiv.org/pdf/2503.03944)：模型提供方微調編碼器植入
後門，影像擁有者掛觸發器。**需要模型方合作**，本專案的威脅模型裡沒有這個對象。
**不走。**

---

## 三 一篇會影響定位的負面結果

*Do Protective Perturbations Really Protect Portrait Privacy under Real-world
Image Transformations?*（[arXiv:2604.23688](https://arxiv.org/html/2604.23688)）
測 JPEG75、resize 0.5× Lanczos、以及兩者串接（C&R），評測 Anti-Forgery／
DF-RAP／FaceShield／FaceLock／PhotoGuard／Mist／AdvDM／Silencer。結論是
**像素級擾動撐不住常見變換的串接**；連專門為抗壓縮設計的 DF-RAP 在串接下
也失效。

**對本專案**：診斷被獨立驗證，但「指出這件事」不再是貢獻。可以保留的差異是
**它只測三個變換、沒有等幾何隨機對照**，而本專案有十二個算子與完整的地板。

---

## 四 結論

| 載體 | 主張適配 | 判讀 |
|---|---|---|
| 結構化圖樣 ＋ 固定調色盤（Voronoi） | 中高 | **做** |
| STP（對照既有資料） | 中 | **做，零機時** |
| 重取樣格點 | 高 | 不走：自家 antialias 就擋掉 |
| 摩爾紋／差頻 | 高 | 不走：同一個死因 |
| 半色調 | 中 | 不走：文獻證明它是淨化器 |
| RAW／後門 | 低 | 不走：威脅模型不合 |

**這一輪查證的淨結果是把候選由七個收斂到兩個**，而且兩個都不需要換威脅模型。
