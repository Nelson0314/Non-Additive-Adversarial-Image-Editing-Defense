# 工作點對齊協定：如何讓 DCT-Shield 成為可比較的 baseline

2026-08-19 建立。裁決見 `DECISIONS.md` 的 DEC-028／DEC-029，本檔是它們的
完整依據與操作說明，自足可獨立讀完。

---

## 1. 問題：三個不對齊的軸

把 DCT-Shield（Bala et al., ICCV 2025 Highlight，[arXiv:2504.17894](https://arxiv.org/abs/2504.17894)）
與紋理重相位放進同一張表時，**沒有任何一個軸是天然對齊的**。

| 軸 | DCT-Shield | 紋理重相位 | 為什麼不能直接比 |
|---|---|---|---|
| 強度參數 | `ε` = 量化階數（`ε ≥ 1`） | `θ` = 相位旋轉半徑（rad） | 兩者的單位無關，且 `ε` 的像素效果隨 `Q_alg` 的量化表逐係數不同 |
| 論文的 baseline 預算 | 全部 baseline `L∞ = 16/255` | 本專案走**原生預算** | 我們的 `photoguard_c` rms 0.0091、論文的同名方法 rms 0.0384，**差 4.2 倍** |
| 資料與模型 | OmniEdit 150 張＋InstructPix2Pix | set0817 7 張＋SD 1.4 SDEdit(0.7) | Edit Protection 的數值不可跨協定比較 |

`ε` 不是 `L∞` 球這件事，與 StyleGuard 記過的 Glaze 問題同型：**當一個方法的
原生約束不是像素範數時，統一 `ε` 的慣例就失效**（Glaze 用 LPIPS 當約束，
同樣不約束 `L∞`）。

### 1.1 實測到的落差（2026-08-19，本機）

論文 Table 1 的 Noise Perception 半張表對上本專案的量測：

| 方法 | 論文 LPIPS | 我們 LPIPS(VGG) | 我們 LPIPS(Alex) | 論文 PSNR | 我們 PSNR | 論文 rms* | 我們 rms |
|---|---|---|---|---|---|---|---|
| DCT-Shield（原生 ε=1） | 0.267 | 0.5532 | — | 27.61 | 29.75 | 0.0416 | 0.0326 |
| MIST | 0.362 | 0.6234 | 0.4718 | 26.62 | **26.66** | 0.0467 | **0.0465** |
| PhotoGuard | 0.284 | 0.3718 | 0.1114 | 28.32 | 40.83 | 0.0384 | 0.0091 |

\* 由論文的 PSNR 反推：`rms = 10^(−PSNR/20)`。

三個讀數：

1. **Mist 逐項吻合**（PSNR 差 0.04 dB、rms 差 0.4%），證明我們的量測管線與
   論文的量級是可比的——落差不是量測錯誤。
2. **PhotoGuard 差 12 dB**，純粹是預算選擇不同（他們 16/255、我們原生）。
3. **DCT-Shield 的擾動我們反而更小（rms 0.0326 < 0.0416）卻更醜（LPIPS 2.07 倍）**。
   這是唯一無法用預算解釋的落差，指向擾動的**空間分布**不同。

### 1.2 LPIPS 的 backbone 已排除為單一解釋

2026-08-19 本機實測（7 張 set0817 的防禦圖，`runs/s0817/merged`）：

| 條件 | `piq.LPIPS` | 官方 `lpips(net='vgg')` | 官方 `lpips(net='alex')` | VGG/Alex |
|---|---|---|---|---|
| 紋理重相位 | 0.1884 | 0.1884 | 0.0285 | 6.61 |
| photoguard_c | 0.3718 | 0.3718 | 0.1114 | 3.34 |
| mist | 0.6234 | 0.6234 | 0.4718 | 1.32 |
| dia_r | 0.2769 | 0.2769 | 0.0903 | 3.06 |
| add | 0.1585 | 0.1585 | 0.0088 | 17.98 |

- **`piq.LPIPS` 與官方 VGG 版逐位相同**，故本專案用的是標準的 VGG16 變體，
  不是模組 docstring 原本寫的 AlexNet（該筆錯誤已於同日更正）。
- 若論文用的是更常見的 AlexNet 預設，Mist 的落差會由 1.72 倍縮到 1.30 倍，
  **縮小但不消失**。VGG/Alex 的比值本身逐條件由 1.32 變動到 17.98，
  **與擾動的結構高度相關**，故不能用單一係數換算兩篇的數字。
- **後果**：報表必須標明 LPIPS 的 backbone。跨論文引用他人的 LPIPS 數字時，
  未標 backbone 者不得直接並列。

---

## 2. 文獻上處理不對齊的四種做法

均為 2026-08-19 查證，附精確設定。

### 2.1 統一像素預算，逐預算各報一張表

**Evaluating Adversarial Protections for Diffusion Personalization**
（[arXiv:2507.03953](https://arxiv.org/abs/2507.03953)）：全部方法統一走 PGD、
步長固定 `1/255`，其餘超參數依各自論文預設；預算掃 `ε ∈ {4, 8, 12, 16}/255`，
**低預算與高預算各報一張表**。資料為 VGGFace2（50 身分 × 8 張）與 WikiArt
（50 藝術家 × 8 張），全部 center-crop 到 512²。

**限制**：該篇**沒有**討論原生約束不是 `L∞` 的方法要怎麼併進來——正是我們
遇到的情形。

### 2.2 對齊防禦效果，再比失真（反向單點對齊）

**IMPASTO / Imperceptible Protection against Style Imitation**
（[arXiv:2403.19254](https://arxiv.org/abs/2403.19254)）§IV-A：

> "we adaptively adjust the perturbation strength for each method *to achieve
> comparable protection levels*"

理由是各方法的最佳化目標不同、取捨形狀不同。它以 **FID 當防禦效果的錨**，
對齊之後才比 DISTS／PieAPP／TOPIQ 等品質指標，並附人眼評估。

### 2.3 掃強度、畫取捨曲線

同一篇 §IV-A「Varying strengths」（Fig. 8）：

> "we manipulate the protection strengths (budget) to delineate an
> imperceptibility-protection trade-off curve"

主張因此可寫成「**同一防禦水準下畫質更好，或同一感知預算下防禦更強**」，
而不是單點勝負。

### 2.4 Pareto 前緣上自動選工作點

**VETO**（[arXiv:2607.27292](https://arxiv.org/html/2607.27292v1)）：對每一個
「方法 × 資料集 × 編輯模型」組合掃 `ε ∈ {0, 4, 8, 12, 16, 32}`，在
（LPIPS，`ESR/ESR_base`）的二維平面上算 Pareto 前緣，取**與理想點 (0, 0)
歐氏距離最小**的那個點當該組合的工作點。`ESR/ESR_base = 1` 表示毫無防禦效果、
`0` 表示擋掉全部原本會成功的編輯。資料 EditBench／AnyEdit／VetoBench 各 300 張。

### 2.5 重現落差的處理慣例

跨資料集／跨模型的重現落差，慣例是**兩組數字都報，且任何一個數字都不得脫離
它的量測協定被引用**（"as reported" 與 "our reimplementation" 分列）。本專案
既有的 `modified_from_paper` 欄位即為此設計。

---

## 3. 本專案採用的協定（DEC-029）

使用者 2026-08-19 裁定：**採 §2.3 的掃描曲線為主，並在曲線上標出 §2.2 與
既有單點對齊的兩個錨點**；資料集擴到 150 張，另加一次論文協定的驗證批。

### 3.1 掃描網格

| 條件 | 掃描參數 | 網格 | 出處 |
|---|---|---|---|
| DCT-Shield base | `ε` | 0.8 / 1.0 / 1.2 / 1.4 | 論文 §6.1 圖 5 的取捨曲線用的就是這四點 |
| DCT-Shield base | `ε` 補點 | 0.4 / 0.6 | **本專案指定**，為了往下延伸到我們的失真區間（`ε < 1` 時論文的抗 JPEG 保證失效，必須標 `modified_from_paper`） |
| 紋理重相位 | `θ` | 0.8 / 1.0 / 1.30 / 1.6 / 2.0 | 1.30 是既有定案值（`MAINLINE.md` §4），其餘為對稱延伸 |

兩邊各 6 與 5 點。**兩條曲線都必須跨過對方的失真區間**，否則交點要靠外插，
那正是 FND-062 單點對齊落入的困境。

### 3.2 兩個軸與兩個錨點

- **橫軸（失真）**：五項成對指標一起報（LPIPS／SSIM／PSNR／VIFp／DISTS），
  曲線本身以 **DISTS** 為橫軸畫（它是既有的預算軸，且 FND-062 已證實它與
  PSNR 同組）。**LPIPS 另畫一條**——四個指標二比二分裂時，兩張圖都要有。
- **縱軸（防禦效果）**：`LPIPS(edit(def), edit(orig))`，即既有的 `edit_lpips`；
  另報 FID／SSIM／PSNR／VIFp／CLIP／SigLIP 的對應欄位。
- **錨點 A（等失真）**：在對方曲線上取 DISTS 相同處，比防禦效果。這是 FND-062
  的做法，保留下來當曲線上的一個點。
- **錨點 B（等效果）**：取 `edit_lpips` 相同處，比失真。這是 §2.2 的做法。
- **兩個錨點一律用線性內插**求得，不再另跑二分搜尋；內插所在的區間端點必須
  同時報出，區間外的一律標「外插，不可解讀」。

### 3.3 資料與模型

**2026-08-19 改版（DEC-030）：主線也改用 OmniEdit，「只收 CC0」的規定撤銷。**

| 批次 | 影像 | 攻擊模型 | 目的 |
|---|---|---|---|
| **主線** | `data/omniedit150`：OmniEdit dev split 的 `src_img` 150 張（五類 × 30） | SD 1.4 SDEdit strength 0.7 | 本專案的正式比較；FID 在此才可報 |
| **驗證** | 同一批的子集（50 張足夠） | InstructPix2Pix | **只有一個目的：確認我們的 DCT-Shield 實作能重出論文 Table 1 的量級**。成功之後主線的數字才有可信度 |

兩批落在同一個影像域，對帳最乾淨。`src_img` 是真實照片（來源 LAION-5B 與
OpenImagesV6，最低 1 MP），生成的是 `edited_img`、本專案不用，故
**「不得使用生成影像」的規定仍然成立**；**「只收 CC0」那一條已由使用者明示
撤銷**（底層照片的權利狀態未知，使用者知悉並接受）。

**論文用的那 150 張取不到**（無 seed、無索引清單、無釋出 split，repo 為空），
故對齊的是**來源與任務分布**而非同一批影像——`n=150` 的抽樣誤差是驗證批本身
的精度上限，必須寫進 limitation。我們自己抽的 id 全部寫進
`data/omniedit150/provenance.json`，可重現。

### 3.4 一個還沒解決的相容性問題：prompt 的形態

`edited_prompt_list` 是**指令式**（"Add a hat to the cat"），因為 OmniEdit 與
DCT-Shield 的攻擊模型是 InstructPix2Pix；本專案主線的 SDEdit 吃的是**描述式**
caption（"a cat wearing a hat"）。兩者不等價，而 `fetch_omniedit.py`
**原樣存下指令、不自行改寫**——改寫等於捏造資料。

兩條出路，二選一（尚未裁定）：

1. **主線攻擊模型改成 InstructPix2Pix**，與論文完全一致，指令直接可用。
   代價是本專案既有的全部 SDEdit 讀數（FND-035…062）換了攻擊模型就不能與
   新批次並列。
2. **另行產生描述式 caption** 並記錄其來源與產生方式，SDEdit 協定不變。
   代價是 caption 的品質成為一個新的變因。

---

## 4. 待跑清單與成本

以單張 RTX 3090 計。DCT-Shield 免疫實測 **150 s/張**（1000 步，`runs/dctshield`
的 `total_seconds`）；紋理重相位的免疫成本遠低於此，主要成本在編輯與淨化。

| 工作 | 格數 | 單卡時數 | 備註 |
|---|---|---|---|
| ~~主線資料集擴到 150 張~~ | — | 0（無 GPU） | **2026-08-19 已完成**：`data/omniedit150`，五類 × 30，全部 512²、150 張互異 |
| **θ 的人眼門檻重新校準** | 150 | 低 | **必做**。θ=1.30 是在 `set0817`（人物／動物特寫）上定的；OmniEdit 是通用場景（風景、室內），而紋理閘的作用面積直接取決於影像的紋理分布。用 `phase_distortion_sweep.py` 重掃 |
| DCT-Shield 掃 6 點 × 150 張 | 900 | 37.5 | 可分 8 卡 → 約 4.7 h |
| 紋理重相位掃 5 點 × 150 張 | 750 | 低 | 免疫成本小 |
| 兩條曲線的編輯與指標（3 seed） | 6600 次 SDEdit | 約 5.5 | 30 步 512²，約 3 s/次 |
| 抗淨化（5 算子 × 3 seed，只在兩個錨點上） | 6600 次 | 約 5.5 | 全曲線都跑淨化沒有必要 |
| 驗證批（50 張 OmniEdit ＋ InstructPix2Pix） | 50 | 2.1 ＋ 編輯 | 一次性 |

**總計約 50 GPU-hr，八卡約 6–7 小時**，不含資料集收集。

---

## 5. 現在（無 GPU）就能做完的部分

**已完成（2026-08-19）：**

1. **FID 補進指標套件** — `MetricSuite.fid`（Inception-V3 pool3、
   `use_fid_inception=True`），`MetricSuite.FID_MIN_TRUSTED = 150`。
   `tests/test_suite_fid.py` 五項釘住。
2. **LPIPS backbone 的文件錯誤已更正** — 見 §1.2 的實測。
3. **兩個半邊的欄位統一** — `src/metrics/standard.py` 的 `standard_row`，
   已接進 `apa_baseline.evaluate`（涵蓋紋理重相位臂）、`dct_shield_run.py`、
   `freq_baselines_run.py`。此前失真半邊缺 VIFp、防禦半邊只有 LPIPS，
   於是與論文 Table 1 無法逐欄對照。**缺欄位會拋錯，不會靜默少報。**
4. **批次 FID** — `scripts/fid_batch.py`，欄名 `frechet`，`n < 150` 預設拒絕
   寫出。既有 7 張批次實測跑得動（`phase` 防禦半邊 219.7、`mist` 332.5，
   兩者都標 `trusted=False`；論文報 MIST 288.6，量級相符但不可據以下結論）。
5. **曲線與錨點** — `scripts/tradeoff_curve.py`，純函式、不需 GPU，
   `tests/test_tradeoff_curve.py` 七項釘住，含「範圍外拒絕外插」。
6. **上機工作表** — `scripts/run_s0820.sh`，六段：`sweep_dct`／`sweep_phase`／
   `curve`／`frechet`／`ret_anchor`／`merge`。合併段的輸出落在被 glob 匹配的
   路徑之外（FND-062 的教訓）。
7. **資料集工具取回** — `scripts/fetch_cc0_images.py` 與
   `scripts/prepare_dataset.py` 在 2026-08-13 的清理中被刪，已由
   `git checkout 9aaf69f7d^ -- <path>` 取回並實測可用（Wikimedia 的 robot
   policy 仍要求 User-Agent 帶聯絡方式；三張候選實抓成功）。

**還沒做完的離線工作：**

8. **資料集由 24 張擴到 150 張**。候選池抓取是網路工作、可離線完成，但
   **必須逐張看過再挑**——CC-Zero 分類混著大量館藏文物照（胸針、標本、
   酒吧招牌），搜尋排序完全不可信。類別組成沿用既有的六類，每類約 25 張。

**無法離線完成的：** 任何需要 SD 前向的東西，以及 `runs/dctshield/` 的
防禦圖——那批**只有 CSV 入庫、PNG 留在遠端**，違反資料保全規定，網路恢復
後第一件事就是把它們拉回來。

---

## 6. 報表要求

1. 每一條曲線的每一點都要標 `modified_from_paper`（`ε < 1` 為真）。
2. 引用他人的 LPIPS 數字時必須標 backbone；未標者不得與本專案的數字並列。
3. 兩個錨點的值一律附上內插區間的兩個端點。
4. FID 只在 n ≥ 150 的批次上報；n = 7 的既有批次一律留空，不得以小樣本 FID
   充數。
