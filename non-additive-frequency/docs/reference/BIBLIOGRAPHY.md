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
