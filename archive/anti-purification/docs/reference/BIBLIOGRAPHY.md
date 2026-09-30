# 文獻清單

逐篇的查證層級要寫明：讀過全文、只讀摘要、或只有二手引用。
移植他人的方法一律**逐行對照公開原始碼**，查不到原始碼時明確標為「摘要重建」，
偏離要寫進模組 docstring。

## 1. 顏色空間的對抗擾動

| 論文 | 內容 | 連結 |
|---|---|---|
| **Spatial Chroma-Shift** | 只在 YUV 色度通道做空間變形 | [arXiv:2108.02502](https://arxiv.org/abs/2108.02502) |
| **PerC-AL**（CVPR 2020） | 以 CIEDE2000 感知色差取代 `L∞` 當約束 | [arXiv:1911.02466](https://arxiv.org/abs/1911.02466) |
| **Adversarial Perturbations Prevail in the Y-Channel** | 相反結論：對抗能量集中在亮度而非色度 | [arXiv:2003.00883](https://arxiv.org/pdf/2003.00883) |
| **ALA: Adversarial Lightness Attack** | 只改亮度，附自然度正則 | [arXiv:2201.06070](https://arxiv.org/pdf/2201.06070) |
| **Chroma Backdoor** | UV 通道的高頻小波注入 | [doi:10.3390/sym17071014](https://doi.org/10.3390/sym17071014) |

## 2. 無約束顏色攻擊

與上表的差別在約束的形式：上表把顏色改動壓在小的 `L∞` 或色差半徑內，
這一族不設幅度上限，改以「結果看起來自然」當約束，因而可以改寫整張影像的
色彩分布。**自然度約束本身就是方法**，拆掉它再換損失，產出的不是該載體的
極限而是壞掉的圖。

| 論文 | 顏色改動的載體 | 自然度由什麼保證 | 連結 |
|---|---|---|---|
| **NCF**（NeurIPS 2022） | CIELab 上的 3×3 轉移矩陣，**逐語意類別各一個** | ADE20K 逐類別的真實色彩分布庫（150 類 × 20 組） | [arXiv:2210.02041](https://arxiv.org/abs/2210.02041) ／ [repo](https://github.com/VL-Group/Natural-Color-Fool) |
| **SAE**（CVPR-W 2018） | HSV 的 H、S 兩通道整張均勻位移 | 無，只靠位移幅度小 | [arXiv:1804.00499](https://arxiv.org/abs/1804.00499) |
| **ReColorAdv**（NeurIPS 2019） | CIELUV 上的 3D 查表，同色像素同變換 | `lp_bound` 加平滑正則 | [arXiv:1906.00001](https://arxiv.org/abs/1906.00001) |
| **cAdv**（ICLR 2020） | 預訓練上色網路，同時改 hint 與 mask | 低熵色群保持不變 | [arXiv:1904.06347](https://arxiv.org/abs/1904.06347) |
| **ColorFool**（CVPR 2020） | 只動 Lab 的 a、b，**亮度完全不碰** | 逐語意類別的自然色彩範圍 | [arXiv:1911.10891](https://arxiv.org/abs/1911.10891) ／ [repo](https://github.com/smartcameras/ColorFool) |
| **ACE / AdvCF**（BMVC 2020 / TIFS 2023） | 分段線性的顏色濾鏡，三通道獨立 | 濾鏡的分段線性參數化 | [arXiv:2002.01008](https://arxiv.org/abs/2002.01008) ／ [repo](https://github.com/ZhengyuZhao/AdvColorFilter) |

NCF 的兩個核心運算另有出處，移植時要一併對照：

| 出處 | 提供什麼 |
|---|---|
| Pitié & Kokaram, *The linear Monge–Kantorovitch linear colour mapping*（CVMP 2007） | 轉移矩陣的閉式解 |
| Afifi et al., *Image recoloring based on object color distributions*（Eurographics 2019） | 逐物件色彩分布庫的建法 |

**與本專案威脅模型的落差**：這一族的目標都是分類器誤判，損失是 C&W 的
logit 差；本專案沒有 logits，目標是擴散編輯失效。換損失是消融，要分開列。

## 3. 對防護有效性的否定證據

| 論文 | 結論 | 連結 |
|---|---|---|
| **Is Perturbation-Based Image Protection Disruptive to Image Editing?**（ICIP 2025） | 多數情況下受保護影像的編輯仍符合 prompt | [arXiv:2506.04394](https://arxiv.org/abs/2506.04394) |
| **Do Protective Perturbations Really Protect Portrait Privacy...** | 像素級擾動撐不住常見變換的串接 | [arXiv:2604.23688](https://arxiv.org/html/2604.23688) |

## 4. 人眼評測

| 論文 | 內容 | 連結 |
|---|---|---|
| **SCOOTER** | 無約束對抗樣本（含色彩空間攻擊）的人眼評測框架：群測方法、標註工具、分析腳本、Likert 等價界 | [arXiv:2507.07776](https://arxiv.org/pdf/2507.07776) |

## 5. 底層模型

| 項目 | 用途 | 連結 |
|---|---|---|
| **InstructPix2Pix** | 本專案的攻擊方 | [repo](https://github.com/timothybrooks/instruct-pix2pix) |
| **Stable Diffusion v1.4** | IP2P 的底模 | [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) |
| **FaceLock**（CVPR 2025） | 攻擊模型與三個推論參數的對照來源 | [arXiv:2411.16832](https://arxiv.org/abs/2411.16832) |

## 6. 指標

| 項目 | 用途 | 連結 |
|---|---|---|
| **ShiftTolerant-LPIPS** | 位移容忍的感知指標 | [arXiv:2207.13686](https://arxiv.org/abs/2207.13686) |

## 7. 防護方法（外部比較的 baseline）

本節是**防護／免疫**方法，與第 1、2 節的顏色攻擊不是同一件事：那兩節的目標是
讓分類器誤判，這一節的目標是讓擴散編輯失效。逐項的預算、論文與官方程式的落差、
以及可引用數字的出處在 [`BASELINE_PROVENANCE.md`](BASELINE_PROVENANCE.md)。

比現有六個條件更新的方法，連同「跑不跑得動／有沒有程式／數字能不能引用」三道
篩選，在 [`BASELINE_CANDIDATES.md`](BASELINE_CANDIDATES.md)。

### 7.1 已在本 repo 實作並查證過

| 論文 | 場景 | 機制 | 連結 |
|---|---|---|---|
| **PhotoGuard**（ICML 2023） | SD img2img／inpainting | encoder attack ＋ diffusion attack | [arXiv:2302.06588](https://arxiv.org/abs/2302.06588) ／ [repo](https://github.com/MadryLab/photoguard) |
| **Mist**（v1） | SD 微調與編輯 | textural ＋ semantic 融合損失 | [arXiv:2305.12683](https://arxiv.org/abs/2305.12683) ／ [repo](https://github.com/mist-project/mist) |
| **DIA**（ICCV 2025） | SD inpainting | 對 inversion／reconstruction 軌跡攻擊，DIA-PT 與 DIA-R 兩型 | [arXiv:2510.00778](https://arxiv.org/abs/2510.00778) ／ [repo](https://github.com/sohn1029/DIA) |
| **AdvPaint** | SD inpainting | 打自注意力的 QKV 與交叉注意力 | [repo](https://github.com/JoonsungJeon/AdvPaint) |
| **PromptFlare** | SD inpainting | 在交叉注意力注入誘餌 prompt 嵌入 | [arXiv:2508.16217](https://arxiv.org/abs/2508.16217) ／ [repo](https://github.com/NAHOHYUN-SKKU/PromptFlare) |
| **DiffusionGuard** | SD inpainting | 只打噪聲最大的那一個時刻 | [arXiv:2410.05694](https://arxiv.org/abs/2410.05694) |
| **DCT-Shield** | IP2P 指令編輯 | 在 JPEG 量化域放擾動，換取抗壓縮 | [arXiv:2504.17894](https://arxiv.org/abs/2504.17894)；[官方 repo 目前是空的](https://github.com/SamsungLabs/dct-shield) |

### 7.2 場景一（指令式編輯）的 baseline，尚未實作

| 論文 | 內容 | 連結 |
|---|---|---|
| **FaceLock**（CVPR 2025） | 以人臉生物特徵為目標的防護。**它的評測協定可以直接借**：IP2P 為受害模型、CelebA-HQ 篩出 2000 張人像、25 條指令分三類（臉部特徵／配件／背景），baseline 是 PhotoGuard 與 EditShield | [arXiv:2411.16832](https://arxiv.org/abs/2411.16832) |
| **EditShield** | 指令式編輯的防護 | [arXiv:2311.12066](https://arxiv.org/abs/2311.12066) |

### 7.3 場景三（DiT／流匹配編輯）的 baseline

這一族的共同論點是 UNet 時代的防護在 DiT 上不夠。**沒有任何一篇做抗淨化評估。**

| 論文 | 受害模型 | 對照的 baseline | 連結 |
|---|---|---|---|
| **DeContext** | FLUX.1-Kontext、Step1X-Edit | PhotoGuard、FaceLock | [arXiv:2512.16625](https://arxiv.org/abs/2512.16625) ／ [repo](https://github.com/LinghuiiShen/DeContext) |
| **NullEdit** | Step1X-Edit、Qwen-Image-Edit-2511 | DiffPGD、PhotoGuard-Encoder、EditShield、DeContext。讀數是 ArcFace ＋ RetinaFace 偵測失敗率 ＋ CLIP-I ＋ SSIM；12 條指令中 8 條進最佳化、4 條 held-out | [arXiv:2608.10870](https://arxiv.org/html/2608.10870) |
| **VETO** | FLUX.2 | 多個；工作點由 Pareto 前緣自動選 | [arXiv:2607.27292](https://arxiv.org/html/2607.27292v1) |
| **TDAE** | IP2P、SD1.4、**SD3** | ACE、MIST、PGD、PGE、SA。記錄 ACE 與 MIST 因架構與 SD3 的 DiT 不相容而被排除。無程式、無 checkpoint 版本、未說明 SD3 如何當編輯模型用 | [arXiv:2512.14341](https://arxiv.org/abs/2512.14341) |

**硬體限制**：遠端是 RTX 3090（24 GB）。白箱防護要對輸入影像反傳，
FLUX.1-Kontext（12B）、Step1X-Edit（19B）、Qwen-Image-Edit（20B）的權重就放不下。
**SD3-medium（2B MMDiT）是這批卡上唯一能做白箱最佳化的 DiT 編輯器。**

**取得權重的阻塞**：`stabilityai/stable-diffusion-3-medium-diffusers` 與
`stable-diffusion-3.5-medium` 在 HuggingFace 上都是 `gated=auto`。遠端沒有 token，
實測匿名下載回 `GatedRepoError 401`。**要跑場景三，必須先用 HF 帳號接受該模型的
授權條款，並把 token 放到遠端**（`huggingface-cli login`，或在 `~/env.sh` 設
`HF_TOKEN`）。`diffusers 0.39.0` 本身已有
`StableDiffusion3Pipeline`／`StableDiffusion3Img2ImgPipeline`／
`StableDiffusion3InpaintPipeline`，程式端沒有缺口，缺的只有授權。

### 7.4 攻擊側（破解防護）

| 論文 | 結論 | 連結 |
|---|---|---|
| **Purify Once, Edit Freely** | `VAE-Trans` 與 `EditorClean` 兩個淨化器不需接觸受保護影像或防禦內部，在六個防禦方法上都成功；主張該領域須在模型不匹配下重新評測 | [arXiv:2603.13028](https://arxiv.org/abs/2603.13028) |
| **CAT**（ICML 2025） | 以 LoRA adapter 做對比式對抗訓練，指出保護性擾動的有效性主要來自 latent 表徵失真，因而不穩健 | [arXiv:2502.07225](https://arxiv.org/abs/2502.07225) ／ [repo](https://github.com/senp98/CAT) |
| **GuardDoor** | 改走保護性後門（需模型提供者配合），動機是傳統對抗擾動撐不過壓縮與加噪 | [arXiv:2503.03944](https://arxiv.org/abs/2503.03944) |

### 7.5 自然度的人眼評測

| 論文 | 內容 | 連結 |
|---|---|---|
| **SCOOTER** | 無約束對抗樣本的人眼評測框架。**六個攻擊（含三個顏色空間攻擊 SemAdv／cAdv／NCF）全部沒通過人眼的等價檢定**；客觀指標的排序與人眼幾乎相反（Fréchet Distance 把人眼判為最不自然的 NCF 排成最好）。協定：5 點 Likert（−2 確定被改過 → +2 確定是真的）、±0.2 等價界、346 名受測者、34K 筆評分，附瀏覽器作答模板與 Python／R 分析腳本 | [arXiv:2507.07776](https://arxiv.org/abs/2507.07776) |
| **NR-IQA 的對抗穩健性** | 無參考影像品質指標本身可被對抗攻擊 | [arXiv:2310.06958](https://arxiv.org/abs/2310.06958) |
