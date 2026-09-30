# 候選 baseline 的篩選

現行六個條件（PhotoGuard、Mist、DIA、DCT-Shield）最新的是 2025 年。本檔把更新的
防護方法過三道篩子，**只做篩選，不決定要不要跑**：

1. **受害模型在 RTX 3090（24 GB）上跑不跑得動。** 白箱防護要對輸入影像反傳，
   權重放不下就整篇不可移植。
2. **有沒有公開程式。** 沒有程式的移植是「依論文重建」，與
   `photoguard_linf` 同一類，`BaselineSpec` 要標 `reconstructed_from_paper`。
3. **數字能不能連協定一起引用**（`BASELINE_PROVENANCE.md` 規則 1）。

孤立的 arXiv 編號一律以 arXiv API 的 `id_list` 查證過標題、日期與版本。

---

## 1. 跑得動的（受害模型都在 SD1.x／SD3 這一帶）

| 方法 | 出處 | 受害模型 | 預算 | 程式 | 對照的 baseline |
|---|---|---|---|---|---|
| **SIFM** | [arXiv:2512.14320](https://arxiv.org/abs/2512.14320) | SD3、HQ-Edit、InstructPix2Pix | `L∞` 0.03、100 步 | **無** | PhotoGuard(E/D)、EditShield、SA、ACE、SDS、Mist |
| **DANP** | [arXiv:2512.14333](https://arxiv.org/abs/2512.14333) | SD v1-4、HQ-Edit、InstructPix2Pix | `L∞` 0.03、100 步、10 個時刻 | **無** | ACE、EditShield、Mist、PGD、PGE、SDS、SA |
| **TDAE** | [arXiv:2512.14341](https://arxiv.org/abs/2512.14341)（IEEE TPAMI） | InstructPix2Pix、SD1.4、**SD3** | 論文未載 | **無** | ACE、Mist、PGD、PGE、SA |
| **Universal Semantic Injection** | [arXiv:2602.14679](https://arxiv.org/abs/2602.14679)（ECCV 2026） | 白箱 SD1.5；黑箱 SD1.4／2.0／IP2P | `L∞` 10/255（比較的逐影像法用 16/255） | **無** | PhotoGuard(EA/DA)、AdvPaint、SA、FastProtect，各有通用化版本 |

四篇都沒有公開程式。三篇（SIFM／DANP／Universal）的受害模型與本專案場景一
**完全重疊**（IP2P），TDAE 另外把 SD3 當受害模型，是這一批裡唯一同時碰到
場景一與場景三的。

**TDAE 已依論文重建（`src/baselines/tdae.py`），但不進本次的外部比較。**
`scripts/defence_run.py` 的 `CONDITIONS` 不含它。理由是論文 Algorithm 1 的
起點 `δ_v ← 0` 在本專案的威脅模型（防禦方看不到指令 → 空 prompt → `y₀` 是
無指令重建＋固定噪聲）下恰好是零梯度點，求解一步都不動；改起點才跑得動，
但那已不是論文的方法。機制、實測讀數與成本見
[`AUDIT_TDAE.md` §8「為什麼不進本次的外部比較」](AUDIT_TDAE.md#8-為什麼不進本次的外部比較)。

**`HQ-Edit` 是什麼**：`UCSC-VLAA/HQ-Edit-ckpt`，IP2P 架構在 HQ-Edit 資料集上
微調的指令編輯器，SD1.5 底，權重規模與 IP2P 同級。SIFM 與 DANP 都用它，
若要對回它們的數字，這是第二個必須封裝的受害模型。

**SIFM 另外帶一個讀數**：ISR（Immunization Success Rate）。它不比「防禦後的
編輯」與「未防禦的編輯」有多不像，而是問編輯結果**有沒有偏離指令語意或明顯
劣化**，由 Gemini 2.5 Pro 與 2.5 Flash 兩個 MLLM 各判一次、兩者都說成功才計成功。
這正對著本專案量到的
[CLIP 對齊增益不能當防禦讀數](../../../HANDOFF.md)那一格——防禦成功時 CLIP 反而上升。
資料是 100 張（人像 35／風景 35／畫作 30）× 5 條指令 = 500 格，每個受害模型一輪。

---

## 2. 硬體擋住的（權重放不下 24 GB）

| 方法 | 出處 | 受害模型 | 程式 |
|---|---|---|---|
| **DeContext** | [arXiv:2512.16625](https://arxiv.org/abs/2512.16625) | FLUX.1-Kontext（12B）、Step1X-Edit（19B） | **有**（[repo](https://github.com/LinghuiiShen/DeContext)，含 `attack/`、`inference/`、`evaluations/`） |
| **NullEdit** | [arXiv:2608.10870](https://arxiv.org/abs/2608.10870) | Step1X-Edit、Qwen-Image-Edit | 無 |
| **VETO** | [arXiv:2607.27292](https://arxiv.org/abs/2607.27292) | FLUX.2 等兩個；附 VetoBench | 無 |
| **CCS** | [arXiv:2607.16898](https://arxiv.org/abs/2607.16898) | BAGEL-7B-MoT（SigLIP2 ViT ＋ FLUX VAE）、InternVL-U | 補充資料內，無公開 repo |

**CCS 的論文自己寫著實驗跑在單張 A100 80 GB。** 其餘三篇的受害模型光權重就
超過 24 GB。這一族唯一有公開程式的是 DeContext，但程式跑得動不等於卡放得下。

CCS 的讀數與本專案的身分欄可以對齊：ISM（身分相似度）、FID、BRISQUE、
Und-Score（指令遵循），資料是 VGGFace2（20 個身分 × 4 張）與 CelebA-HQ。
NullEdit 同樣用 CelebA-HQ ＋ VGGFace2。

---

## 3. 威脅模型不同，放進同一張表會比錯

| 方法 | 出處 | 為什麼不同 |
|---|---|---|
| **MIRAGE** | [arXiv:2606.26199](https://arxiv.org/abs/2606.26199) | 打的不是生成模型，是商用編輯服務**生成前的安全審查分類器**：把影像推向違規概念，讓 GPT-Image／Nano Banana／Grok Imagine 直接拒絕，回報 >88% 成功率。不需要模型權重也不需要指令，但要 API 額度，且沒有白箱受害模型可比 |
| **Blank Canvas** | [arXiv:2511.22237](https://arxiv.org/abs/2511.22237)（AAAI 2026） | 目標是讓 SAM「看不見任何東西」，以便事後**定位竄改區域**。是鑑識不是免疫，沒有「編輯有沒有被擋下」這一欄 |

---

## 4. 場景二（inpainting）：關鍵字掃不到新的，但有一篇已查證未實作的

以 `inpainting`＋`protection`／`adversarial perturbation`＋`diffusion` 掃 arXiv，
時間排序下最新的一篇是本專案已經移植的
**PromptFlare**（[arXiv:2508.16217](https://arxiv.org/abs/2508.16217)，
*Prompt-Generalized Defense via Cross-Attention Decoy in Diffusion-Based Inpainting*）。

**但關鍵字掃不到 DiffVax**，而它是這一格最強的候選：

| 項目 | DiffVax（[arXiv:2411.17957](https://arxiv.org/abs/2411.17957)，ICLR 2026） |
|---|---|
| 機制 | 前饋式 immunizer（UNet++），一次前向就產生擾動，不逐影像最佳化 |
| 程式 | **有官方 repo**，本專案已 clone 並逐行查證（`AUDIT_MIST_DIFFVAX.md`，commit `77fe66a`） |
| 受害模型 | SD inpainting。**跑得動 24 GB** |
| 預算 | **沒有硬性 `L∞` 上界**，只有 `α·L_noise` 軟懲罰，`α=4`。程式裡的 `eps=32/255` 在 DiffVax 路徑上從未被使用 |
| 訓練資料 | `ozdentarikcan/DiffVaxDataset`（HF），800 train／200 val，512² 附遮罩與 prompt |
| 卡住的地方 | immunizer **吃 masked image**，擾動只加在遮罩外，所以它結構上需要編輯遮罩；要當 img2img／全域編輯的 baseline 必須先決定遮罩怎麼定義，而那個設定不在原作的訓練分布內 |

沒有硬預算這一點使它無法直接進 DEC-029 的等失真比較，要比必須自己加約束或掃 `α`。

## 4.1 DAYN：本專案資料集對齊的那一篇，還沒當成 baseline

**DAYN** — *Distraction is All You Need: Memory-Efficient Image Immunization
against Diffusion-Based Image Editing*，Lo et al.，**CVPR 2024**
（[open access PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lo_Distraction_is_All_You_Need_Memory-Efficient_Image_Immunization_against_Diffusion-Based_CVPR_2024_paper.pdf)）。

機制是打 cross-attention 的 semantic attack：分散模型對受保護內容的注意力，
使它無法確定要編輯哪裡；配合 timestep-universal 的梯度更新，記憶體約為前人
方法的一半。

它在本專案裡有兩個身分，兩個都指向同一個缺口：

1. **`data/lo_aligned` 的結構就是對齊它的補充材料 §A**（每個物件兩個編輯
   prompt，`prompts[0]` 改掉指定內容、`prompts[1]` 保留該內容改動其餘）；
   `data/dayn_testset/` 是規劃向作者索取的原始測試集，**至今未取得**。
2. **§1 那三篇（SIFM／DANP／Universal Semantic Injection）對照表裡的 `SA`
   （Semantic Attack）就是它。** 要對回那三篇的數字，DAYN 是必須有的那一欄。

**未找到公開程式。** 搜尋 CVPR open access、作者頁與 GitHub 均無 repo 連結。

---

## 5. 攻擊側：新的破解器都有程式

這三篇決定抗淨化那一欄要打哪些算子。

| 方法 | 出處 | 內容 | 程式 |
|---|---|---|---|
| **img2img denoiser** | [arXiv:2602.22197](https://arxiv.org/abs/2602.22197)（SaTML 2026） | 現成的 image-to-image 生成模型加一句提示詞就是通用去噪器，6 個防護方案、8 個案例全破 | **有**（`mlsecviswanath/img2imgdenoiser`） |
| **TIP-RSR** | [arXiv:2604.23688](https://arxiv.org/abs/2604.23688) | 問的是**連續的**日常變換（縮放＋色彩壓縮）串起來會怎樣，答案是逐像素擾動撐不住；附一個免訓練的區域超解析還原框架 | **有** |
| **Purify Once, Edit Freely** | [arXiv:2603.13028](https://arxiv.org/abs/2603.13028) | 模型不匹配下的淨化，`VAE-Trans` 與 `EditorClean` 兩個算子，六個防禦全破 | 未查 |

`phase_retention.py` 現有的八道算子不含這三類。**要不要加是使用者的裁定**，
本檔只記下它們存在、有程式、且都宣稱能破現有防護。

---

## 6. 三件會影響排程的事

1. **新的防護方法一篇都沒有公開程式**（DeContext 有程式但卡放不下）。移植成本
   是依論文重建，落差處置照 `BASELINE_PROVENANCE.md` 規則 3：論文與程式不一致
   時兩者都列，不挑一個。
2. **SIFM／DANP／Universal 三篇共用一組 baseline**（ACE、SDS、PGE、SA、
   EditShield），本 repo 一個都沒有。其中 `SA` 就是 §4.1 的 DAYN，也就是本專案
   資料集對齊的那一篇。要對回它們表裡的數字，缺口不只是那三篇。
3. **SIFM 與 DANP 的預算（`L∞` 0.03 = 7.65/255）落在現有總表的區間內**
   （6/255 到 16/255），與 AdvPaint 的 0.03 相同，可以直接進 DEC-029 的掃描曲線。
