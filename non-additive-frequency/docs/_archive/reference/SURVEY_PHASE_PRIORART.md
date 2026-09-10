# 文獻查證：2026-08-16

本輪查證的動機有兩個：

1. `BIBLIOGRAPHY.md` 自稱收錄「本專案引用過的**全部**文獻」，但紋理重相位的
   兩篇構造來源（Galerne 的 RPN、Ding 的 DISTS）都不在裡面。
2. 專案至今沒有查過「**相位擾動當對抗攻擊**」這件事本身有沒有前例。規格
   §1 把「相位擾動能換到比加性更大的位移」寫成一句待否證的假設，但沒有
   確認這句話是不是已經有人測過。

結論先講：**第 2 項有前例，而且不只一篇。** 本方法的新穎性主張必須收窄。
細節見 §2。

---

## 1. DCT-Shield（ICCV 2025 Highlight）—— 目前最直接的競爭者

| 項目 | 內容 |
|---|---|
| 出處 | Bala et al., ICCV 2025 Highlight（Samsung R&D Bangalore）。[arXiv:2504.17894](https://arxiv.org/abs/2504.17894)／[CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Bala_DCT-Shield_A_Robust_Frequency_Domain_Defense_against_Malicious_Image_Editing_ICCV_2025_paper.pdf)／[專案頁](https://dct-shield.github.io/project-page/) |
| 原生解決的問題 | 與本專案完全相同：擴散模型的惡意文字編輯防護 |
| 方法 | 在 **DCT 係數**上做最佳化，走 JPEG 管線轉回像素 |
| 加性與否 | **加性**——對量化後的 DCT 係數加擾動，不保留任何頻譜分量 |
| 主張 | 參數量比像素空間少 67%、視覺瑕疵更少、**對 JPEG 壓縮強健** |
| 特別的保證 | 在品質 q 上免疫的影像，對所有**不高於** q 的壓縮等級都保持防護 |

### 為什麼它必須進 baseline 清單

它的兩個主張與本專案的兩個主張逐條重疊：

| | DCT-Shield | 紋理重相位 |
|---|---|---|
| 失真更小 | 「更少視覺瑕疵」 | 人眼門檻上 DISTS 便宜 4.5 倍 |
| 抗淨化 | 對 JPEG 有**構造上的**保證 | 淨化後絕對位移量勝加性 10/10 |
| 參數量 | 比像素空間少 67%（約 2.6e5） | 592,416，**比像素加性的 7.9e5 只少 25%** |

第三列對本專案不利，必須主動寫出來——參數量不是紋理重相位的優勢。

### 與本方法的實質差別（可寫進 related work）

DCT-Shield 動的是**係數的值**，紋理重相位動的是**係數的相位角**。前者改變
局部頻譜的能量分布，後者由代數恆等式 `|X·e^{iθ}| = |X|` 保證能量分布不變。
這個差別是可量的：本專案的 `amplitude_deviation()` 實測 0.0065–0.0653，
DCT-Shield 沒有對應的量。

**但這個差別目前沒有被證明有用。** 見 §4 的第 2 點。

---

## 2. 相位擾動作為對抗攻擊：前例存在

這是本輪最重要的發現。

### 2.1 Perturbing the Phase（2026）

[arXiv:2602.06577](https://arxiv.org/html/2602.06577)。查證方式：讀論文全文。

| 項目 | 內容 |
|---|---|
| 對象 | 複數值神經網路（CVNN）與實數值網路（RVNN），ResNet／ConvNeXt |
| 資料 | S1SLC_CVDL PolSAR（合成孔徑雷達）、FastMRI Prostate（磁振造影） |
| 擾動 | **只動相位，幅度由構造逐點保留**——論文原文 `|Z| = p.w. |X|` |
| 約束 | 幅度夠大的位置，相位偏移限制在 `2·arcsin(ε/(2|X|))`；`2|X| ≤ ε` 的位置相位自由 |
| 結論 | **相位攻擊（PIFGSM／PMIFGSM）比幅度攻擊有效**；CVNN 的強健度與 RVNN 相當或更好 |

**與本方法重疊的部分**：「只轉相位、幅度逐點保留」這個構造，以及「相位比
幅度更能造成模型錯誤」這個結論。本專案規格 §1 把後者寫成待否證的假設，
但它在 2026 年已經被獨立測到。

**沒有重疊的部分**：

| 面向 | Perturbing the Phase | 紋理重相位 |
|---|---|---|
| 資料 | 天然複數（雷達、磁振造影） | 實數自然影像 |
| 變換 | 逐像素複數相位 | **重疊區塊加窗 FFT** 的頻譜相位 |
| 任務 | 分類 | **擴散編輯防護** |
| 閘 | 無 | **紋理閘＋徑向頻率閘** |
| 抗淨化 | 未測 | 十個算子 |

### 2.2 相位約束：一個可以直接借用的東西

上面那條 `2·arcsin(ε/(2|X|))` 值得單獨講。

它讓**相位的可動範圍隨局部幅度反比縮放**，使得像素域的位移被 ε 界住。也就是
它用一個閉式的公式解決了「固定相位角不等於固定失真」這個問題。

本專案現在的做法是固定 `θ_max = 1.30` 給所有位置、所有影像，實測 24 張圖的
PSNR 從 23.15 漂到 39.54（16.4 dB）。這正是那條約束要處理的事。

**這是本輪查證最有操作價值的一項**：文獻已經有一個原則性的解法，不必自創。

### 2.3 其他前例

| 論文 | 內容 | 與本方法的關係 |
|---|---|---|
| Black box phase-based adversarial attacks on image classifiers（JEI 34(1):013041, 2025） | 黑盒、相位域的分類攻擊 | 同樣是相位攻擊，但黑盒、分類任務 |
| PPD: Permutation Phase Defense（[arXiv:1812.10049](https://arxiv.org/pdf/1812.10049)） | 把相位置換當**防禦**（讓分類器學置換過相位的輸入） | 用途相反，但確認相位置換保留可學習的內容 |
| Phase-aware Adversarial Defense（ICML 2023, Zhou et al.） | 以相位資訊提升對抗強健度 | 同上，防禦側 |
| NeuralRemaster: Phase-Preserving Diffusion（[arXiv:2512.05106](https://arxiv.org/html/2512.05106v2)） | 擴散生成時保留相位以對齊結構 | 確認「相位承載結構」在擴散脈絡下成立 |

### 2.4 新穎性主張要怎麼改寫

**不能再寫**：「本文首次把相位擾動用於對抗攻擊」。

**可以寫**：本文首次把**加窗重疊區塊的頻譜相位旋轉**用於**擴散編輯防護**，
並以紋理度與徑向頻率兩個由原圖決定的閘限制其作用範圍；相位攻擊在分類與
複數值網路上的前例（[arXiv:2602.06577](https://arxiv.org/html/2602.06577)、
JEI 2025）不涉及自然影像的區塊頻譜、不涉及擴散模型、也未測抗淨化。

---

## 3. 保護擾動的頻譜結構：一篇支持本方法框架的分析

[arXiv:2512.08329](https://arxiv.org/abs/2512.08329)，Interpreting Structured
Perturbations in Image Protection Methods for Diffusion Models。

原先列在 `BIBLIOGRAPHY.md` §10「曾查證但未納入」。本輪重讀後認為應該升到
related work，理由是它的頻域結論直接支撐本方法的框架：

> Glaze 與 Nightshade **不是**加隨機噪聲。它們把能量**沿著影像本身的主頻率軸
> 重新分配**，而不是把干擾均勻散布到整個頻譜。

也就是：既有的強保護方法在頻域上做的事，本質上就是**結構化的頻譜重分配**。
紋理重相位是同一件事的極端形式——重分配到只改相位、完全不改幅度。

這篇同時提供一個警告：這類擾動「structured, low-entropy，與影像內容緊密耦合」
因而**始終可被偵測**。本專案沒有主張不可偵測，但報告裡不應該暗示這一點。

---

## 4. 對本專案主張的三個影響

### 4.1 主張要收窄（§2）

相位擾動＋幅度保留這個構造有前例。新穎性落在區塊頻譜、兩個閘、擴散編輯、
抗淨化這四項的組合上。

### 4.2 「幅度保留」目前是一個沒有被證明有用的性質

專案花了很多篇幅論證幅度保留由構造保證（測試釘住、`amplitude_deviation()`
量測）。但**沒有任何實驗顯示幅度保留本身帶來好處**：

- 對照組 `add` 是全圖無限制的加性，位置沒對齊
- 對照組 `phase_rand` 是隨機相位，也保留幅度

兩者都不隔離「保留幅度」這個變因。DCT-Shield 的存在讓這件事更尖銳：它是
**不保留幅度的頻域方法**，如果它在同一個協定下打平或勝出，那「幅度保留」就
只是一個好聽的性質而非機制。

處置建議：把 DCT-Shield 納入 baseline，讓「保不保留幅度」變成可判定的對照。

### 4.3 有一個現成的約束可以修掉最大的方法學弱點（§2.2）

`2·arcsin(ε/(2|X|))` 型的幅度相依相位上限，可以讓固定的失真預算落在逐圖
一致的水位上。目前 24 張圖的 PSNR 漂 16.4 dB，而該漂移與「相位在哪張圖上贏」
的相關是 r = +0.776。

---

## 5. 要補進 `BIBLIOGRAPHY.md` 的條目

以下在本專案的程式或文件中被實際引用，但索引檔漏收：

| 論文 | 在本專案的角色 | 連結 |
|---|---|---|
| Galerne, Gousseau, Morel. Random Phase Textures: Theory and Synthesis. IEEE TIP 20(1):257–267, 2011 | **紋理重相位的構造來源**；`phase_rand` 即 RPN 本身 | — |
| Ding, Ma, Wang, Simoncelli. Image Quality Assessment: Unifying Structure and Texture Similarity. TPAMI 2021 | DISTS，預算軸與「對紋理重取樣寬容」的依據 | [arXiv:2004.07728](https://arxiv.org/abs/2004.07728) |
| Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018 | **PGD 本身**。`param_pgd.py` 的 sign 更新式沒有引用出處 | [arXiv:1706.06083](https://arxiv.org/abs/1706.06083) |
| Oppenheim & Lim. The Importance of Phase in Signals. Proc. IEEE 69(5):529–541, 1981 | 相位決定可辨識內容。**本方法必須處理的矛盾**：兩個閘就是它的答案 | [PDF](https://dsp-group.mit.edu/wp-content/uploads/2024/11/ImportancePhaseSignals_1981.pdf) |
| 結構張量（Förstner／Harris／Bigün 一系） | 紋理閘的 coherence | — |

本輪新增：

| 論文 | 角色 | 連結 |
|---|---|---|
| DCT-Shield（ICCV 2025 Highlight） | **待納入的 baseline**，頻域加性 | [arXiv:2504.17894](https://arxiv.org/abs/2504.17894) |
| Perturbing the Phase（2026） | 相位攻擊的前例，幅度逐點保留 | [arXiv:2602.06577](https://arxiv.org/html/2602.06577) |
| Black box phase-based adversarial attacks（JEI 2025） | 相位攻擊的前例，黑盒分類 | [doi:10.1117/1.JEI.34.1.013041](https://doi.org/10.1117/1.JEI.34.1.013041) |
| Interpreting Structured Perturbations（2025） | 保護擾動的頻譜結構分析 | [arXiv:2512.08329](https://arxiv.org/abs/2512.08329) |
| NeuralRemaster: Phase-Preserving Diffusion | 相位承載結構，擴散脈絡 | [arXiv:2512.05106](https://arxiv.org/html/2512.05106v2) |

---

## 6. 查過但判定不納入的

| 論文 | 為什麼不納入 |
|---|---|
| STP-Diff（Information Fusion 2025） | 非加性空間變換擾動＋非顯著區域，形式接近，但任務是**人臉辨識隱私**不是擴散編輯，判準與威脅模型都不通用 |
| Enhancing Facial Privacy Protection via Weakening Diffusion Purification（[arXiv:2503.10350](https://ar5iv.labs.arxiv.org/html/2503.10350)） | 同上，人臉辨識任務 |
| CAT: Contrastive Adversarial Training（[arXiv:2502.07225](https://arxiv.org/pdf/2502.07225)） | 是**攻擊防護的訓練式攻擊**，不是淨化算子；納入需要重訓，成本與本輪範圍不符 |
| Empirical Robustness of Pixel Diffusion（OpenReview） | 結論是「幾乎所有針對 latent diffusion 的攻擊對 pixel-space diffusion 無效」。本專案的威脅模型明定為 stock SD（latent），故不在範圍內——**但這是一個要在 limitation 寫明的外部威脅** |
| DDAP（[arXiv:2407.20141](https://arxiv.org/pdf/2407.20141)） | 雙域 anti-personalization，任務是個人化微調不是編輯 |
| Vid-Freeze（[arXiv:2509.23279](https://arxiv.org/pdf/2509.23279)） | image-to-video，威脅模型不同 |

---

## 7. 建議的處置順序

1. 把 §5 的兩張表補進 `BIBLIOGRAPHY.md`（缺漏五筆 ＋ 新增五筆）
2. 依 §2.4 改寫新穎性主張，並在 related work 加一段處理 Oppenheim & Lim 的矛盾
3. 把 DCT-Shield 納入 baseline 清單——它同時是競爭者與「幅度保不保留」的對照
4. 依 §2.2 評估幅度相依的相位上限，取代現行的固定 `θ_max`
5. 在 limitation 寫明 pixel-space diffusion 的威脅（§6 最後一列）
