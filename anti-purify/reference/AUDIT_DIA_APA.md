# DIA 與 APA 原始碼／論文全文查證

日期：2026-08-05
用途：供 baseline 嚴格重現使用。本文件所有數值與公式均標註來源；查不到者一律列於各對象的
「未找到的項目」一節，不作推斷、不補值。

## 查證所用來源

| 代號 | 內容 | 位置 |
|---|---|---|
| `[DIA-CVF]` | DIA 正式版（ICCV 2025 open access，主文 8 頁 + 參考文獻，**不含補充資料**） | https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_DIA_The_Adversarial_Exposure_of_Deterministic_Inversion_in_Diffusion_Models_ICCV_2025_paper.pdf |
| `[DIA-ARX]` | DIA arXiv:2510.00778 v1（18 頁，**含 Supplementary Material A–E**） | https://arxiv.org/pdf/2510.00778 |
| `[DIA-CODE]` | DIA 官方程式碼 | https://github.com/sohn1029/DIA |
| `[DIA-ANON]` | 論文正文所附匿名程式碼連結（內容與 `[DIA-CODE]` 相同） | https://anonymous.4open.science/r/DIA-13419/ |
| `[APA-ARX]` | APA arXiv:2506.01511 v1（11 頁，**無 Appendix**） | https://arxiv.org/pdf/2506.01511 |
| `[APA-CODE]` | APA 官方程式碼 | https://github.com/deep-kaixun/APA |
| `[AC-MIRROR]` | AdverseCleaner 鏡像（原 repo 已刪除） | https://github.com/toyxyz/AdverseCleaner 、 https://huggingface.co/spaces/p1atdev/AdverseCleaner |

引用格式為「檔案:行號」或「論文頁碼／公式編號」。

---

# 對象一：DIA（ICCV 2025，arXiv:2510.00778）

## 0. 官方程式碼

論文正文最後一段（`[DIA-CVF]` p.1 摘要末）寫：

> Our code is available here: `https://anonymous.4open.science/r/DIA-13419/`.

該匿名連結**目前仍可存取**。此外，共同第一作者 Geonho Son 的 GitHub 帳號 `sohn1029` 下有
正式 repo：

- **https://github.com/sohn1029/DIA**
  - description：`DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models`
  - `created_at`: 2025-10-08、`pushed_at`: 2025-11-12、`fork: false`
  - 檔案結構與匿名 repo 逐項相同（`README.md`、`attack_benchmark.py`、`attack_setting.json`、
    `attack/DIA_PT.py`、`attack/DIA_R.py`、`utils/utils_general_H.py`、
    `conda_requirements.yaml`、`requirements.txt`）。

作者 GitHub 帳號對照：`hoo0681` = Seunghoo Hong、`sohn1029` = Geonho Son、`josejhlee` = Juhun Lee。
另兩人帳號下均無 DIA repo。

> **注意**：repo 內**只有攻擊端程式碼**。淨化（purification）、PIE-Bench 評測、CLIP similarity
> 計算、以及 Photoguard／Glaze／AdvDM／SDS／PID 等 baseline 的實作**均未釋出**。

## 1. DIA-PT 與 DIA-R 的完整損失式

### 1.1 論文原文

推導的起點是把 DDIM inversion 的終點 `x_T` 拆解（`[DIA-CVF]` p.4, Eq. 7）：

```
x_T = √ᾱ_T x_0   +   Σ_{i=0..T} (√ᾱ_T / √ᾱ_{i+1}) Δ_i
      └─ bias ─┘       └────────── MT ──────────┘
```

其中 `Δ_t` 為 Eq. 5 定義的 noising part：
`x_{t+1} = √α_{t+1} x_t + √ᾱ_{t+1} λ(t) ε_θ(x_t, t+1)`，底線部分即 `Δ_t`。

改寫成以 `x_0` 為基準（`[DIA-CVF]` p.4, Eq. 8）：

```
x_T = x_0 + (√ᾱ_T − 1) x_0 + Σ_{i=0..t} (√ᾱ_T / √ᾱ_{i+1}) Δ_i
            └──────────────── PT (Process Trajectory) ────────────────┘
```

**DIA-PT（Eq. 9，`[DIA-CVF]` p.4）**：

```
δ_DIA-PT = arg max_{||δ|| ≤ ε}  || x̂_{0:T}(x_0 + δ)  −  E(x_0 + δ) ||²₂
```

> 原文：「where `x̂_{i:j}` represents the process of inversion, specifically the transition from
> timestep i to j during the backward denoising process. Through Eq. 9, we can attack the inversion
> process trajectory to `x_T`. Intuitively, we maximize accumulated changes in the reverse diffusion
> process to corrupt the inverted latent code used for editing.」

`E(·)` 為 VAE encoder。故 DIA-PT 的損失是「反演終點 `z_T`」與「該影像自身的 VAE latent `z_0`」
的距離，即 Eq. 8 中的 PT 項。

**DIA-R（Eq. 10，`[DIA-CVF]` p.5）**：

```
δ_DIA-R = arg max_{||δ|| ≤ ε}  || x̃_{T:0}( x̂_{0:T}(x_0 + δ) )  −  (x_0 + δ) ||²₂
```

> 原文：「where `x̃_{i:j}` refers to the reconstruction process, which involves generating the data at
> timestep j from timestep i by following the reverse diffusion steps. […] Intuitively, maximizing
> Eq. 10 implies a digression of the reconstructed image away from the source image.」

即：完整反演到 `x_T` 再完整重建回 `x_0`，最大化重建影像與輸入影像的**像素空間**距離。

**補充變體 DIA-MT（`[DIA-ARX]` Supplementary A, Eq. 1）**——只攻擊 MT 項，論文明示較弱：

```
δ_DIA-MT = arg max_{||δ|| ≤ ε}  || x̂_T(x_0 + δ)  −  √ᾱ_T ( E(x_0 + δ) ) ||²₂
```

> 「Same as DIA-PT, `x_0 + δ` is detached from the computational graph used to calculate the gradient.」
> （此句同時確認 DIA-PT 的第二項是 detach 的。）

### 1.2 程式碼實作（與論文的差異）

**DIA-PT**（`[DIA-CODE]` `attack/DIA_PT.py:388`）：

```python
loss = (inputs - tmp_latent.detach()).norm(p=2).float()
```

- `inputs` = `x_T`（完整 10 步反演的終點 latent）
- `tmp_latent` = `self.model.vae.encode(x_adv).latent_dist.sample() * 0.18215`，即 `E(x_0+δ)`，且已 `detach()`

**DIA-R**（`[DIA-CODE]` `attack/DIA_R.py`，`loss_type == '20'`）：

```python
assert self.__dict__['backward_point'] == 0
loss = (self.model.decode_image(inputs).float() - x_adv.detach().float()).norm(p=2)
```

- `inputs` = 完整反演後再完整重建得到的 latent
- `decode_image(l)` = `vae.decode((1/0.18215) * l).sample`（`utils/utils_general_H.py:312-318`）
- `x_adv` 為像素空間張量，同樣 `detach()`

> **必須注意的差異**：論文 Eq. 9 / Eq. 10 寫的是 **squared L2**（`||·||²₂`），
> 程式碼用的是 `.norm(p=2)`，即**未平方的 L2 範數**。由於更新規則只取 `grad.sign()`
> （見下節），平方與否僅改變梯度的純量倍率、不改變符號，因此對 PGD 結果無影響；
> 但若要改用非 sign 的更新規則則必須留意。

**兩者差在哪／哪一個較強**

| 面向 | DIA-PT | DIA-R |
|---|---|---|
| 攻擊目標 | 反演終點 `z_T` 對 `z_0` 的偏離（latent 空間） | 反演＋重建往返後對輸入影像的偏離（像素空間） |
| 前向計算量 | 10 步反演 | 10 步反演 + 10 步重建 |
| 初始化 | `norm: "L1"` → 隨機初始化於 L1 球內 | `norm: null` → 從乾淨影像出發，無隨機初始化 |
| 耗時（`[DIA-ARX]` Supp. E） | 約 40 秒 | 約 1 分 50 秒 |
| VRAM（`[DIA-ARX]` Supp. E） | 6–7 GB（兩者共同陳述） | 同左 |

論文的結論（`[DIA-CVF]` p.7, §4.2）：

> 「Notably, DIA-R demonstrates superior performance as it accumulates residual error throughout the
> entire learned diffusion process.」

以 Table 1 的 9 組 inversion-edit 配對逐項比較 CLIP similarity（越低越強）：**DIA-R 在 6/9 組較低**
（DDIM-PnP、DDIM-P2P、NTI-P2P、NTI-ProxGuidance、NPI-P2P、NPI-ProxGuidance），
DIA-PT 在 3/9 組較低（DDIM-DDIM 幾乎持平 23.4614 vs 23.4626、DDIM-MasaCtrl、Direct-P2P）。

但在失真型指標上兩者分工不同（`[DIA-CVF]` p.7, §4.3.2 原文）：

> 「DIA-PT imprints a uniform synthetic artifact spread across the image. This characteristic leads to
> outstanding performance in metrics that evaluate perceptual similarity, such as LPIPS and SSIM.
> However, DIA-R shows strength in metrics that measure pixel-wise differences, such as PSNR and MSE.」

隱蔽性方面（Fig. 5 內嵌表格，700 張 PIE-Bench 影像的原圖／免疫圖平均 PSNR，**越高越隱蔽**；
以 PDF 文字座標逐欄核對後確認對應關係）：

| 方法 | Photoguard | Glaze | AdvDM | SDS | PID | DIA-PT | DIA-R |
|---|---|---|---|---|---|---|---|
| PSNR | 33.7949 | 41.1567 | 34.7445 | 34.0196 | 28.4406 | 36.5023 | 40.2686 |

論文據此陳述「DIA-R showed closest PSNR values to Glaze」，與上表一致。

### 1.3 記憶體友善的可微分軌跡（重現時必須實作）

`[DIA-CVF]` p.5, Eq. 11（引自 FlowGrad）：

```
∇_{h_t} J = ⎧ ∂L/∂h_t                        , t = T
            ⎨ ∇_{h_{t+1}} J · J_VAE(h_t)      , t = 0
            ⎩ ∇_{h_{t+1}} J · J_DDIM(h_t)     , otherwise
```

程式碼以 `torch.autograd.grad` 逐步回推 `lam`（即 `∇_{h_{t+1}}J`），最後一步用
`torch.autograd.functional.vjp` 穿過 VAE encoder（`attack/DIA_PT.py:392-404`）。

## 2. PGD 設定（eps、迭代數、DDIM 步數、step_size）

### 2.1 論文原文（`[DIA-CVF]` p.6, §4.1 "Attack Baselines and Setup"）

> 「All edits are performed on Stable Diffusion v1.4 [20] using default benchmark settings. The
> settings for the immunization methods used in the experiment are as follows: All methods use a PGD
> [14] perturbation epsilon of 0.05. The iterations for Photoguard and AdvDM are set to 60, while
> DIA-PT and DIA-R use 20 iterations to train the adversarial noise. Additionally, the inversion and
> reconstruction process trajectories used in DIA-PT and DIA-R each consist of 10 DDIM steps.」

以上完全確認題目所述的 eps = 0.05、20 iterations、inversion/reconstruction 各 10 DDIM 步。

PGD 的一般式（`[DIA-CVF]` p.4, Eq. 6）：

```
x' = Π_{x+S}( x + α · sign( ∇_x J(θ, x, y) ) )
```

> 「α is the step size」——但**論文正文與補充資料均未給出 α 的數值**。

### 2.2 step_size（由程式碼取得）

`[DIA-CODE]` `attack_setting.json`（DIA_PT 與 DIA_R 兩區塊數值相同）：

```json
{
    "DIA_PT": {
        "method_name": "DIA_PT",
        "num_inference_steps": 10,
        "num_inner_inference_steps": 10,
        "lr": 0.003921568627451,
        "iters": 20,
        "forward_uncond_prompt": "",
        "forward_cond_prompt": "",
        "forward_neg_prompt": "",
        "forward_cfg": 1,
        "backward_uncond_prompt": "",
        "backward_cond_prompt": "",
        "backward_neg_prompt": "",
        "backward_cfg": 1,
        "wramup_interval": 14,
        "preprocess_full_overwrite": false,
        "postprocess_full_overwrite": false,
        "using_neg": true,
        "precision": "float16",
        "norm": "L1",
        "clamp_min": -1,
        "clamp_max": 1,
        "eps": 0.05
    },
    "DIA_R": {
        ...（同上，差異如下）
        "norm": null,
        "loss_type": "20",
        "forward_point": 10000,
        "backward_point": 0
    }
}
```

**`step_size` = `lr` = 0.003921568627451 = 1/255**（以 `[-1,1]` 值域計）。

PGD 更新實作（`attack/DIA_PT.py:102-106`，`DIA_R.py` 相同）：

```python
def grad_normalize(source_latent, X_adv, grad, i, eps=20, step_size=10, clamp_min=-200, clamp_max=200, iters=100):
    adv_image = X_adv + step_size * grad.sign()
    eta = torch.clamp(adv_image - source_latent, -eps, eps)
    X_adv = torch.clamp(source_latent + eta, clamp_min, clamp_max).detach()
    return X_adv, (X_adv - source_latent).detach()
```

呼叫端（`attack/DIA_PT.py:261`）：

```python
x_adv, d_x_norm = grad_normalize(source_latents, x_adv, grad, i_inter,
                                 eps=self.__dict__["eps"], step_size=self.__dict__["lr"],
                                 clamp_min=self.__dict__["clamp_min"], clamp_max=self.__dict__["clamp_max"])
```

即標準的 L∞ PGD **上升**（`+ step_size * sign(grad)`，因為目標是最大化損失），投影到
半徑 `eps` 的 L∞ 球，再夾到值域上下界。函式簽名的預設值（`eps=20`、`step_size=10`…）
是無意義的佔位值，實際一律由 JSON 覆寫。

### 2.3 其他重現必需項（皆來自程式碼）

- **模型**：`CompVis/stable-diffusion-v1-4`（`attack_benchmark.py`），scheduler 為 `DDIMScheduler`。
- **精度**：`float16`。
- **prompt**：`forward_cond_prompt` / `forward_uncond_prompt` / `forward_neg_prompt` **全為空字串**，
  `forward_cfg = backward_cfg = 1`。即攻擊端反演與重建都在**空 prompt、無 CFG** 下進行。
  （`attack_benchmark.py` 另有一條路徑：若傳入 `--target_obj`，則
  `forward_uncond_prompt = f"a photo of {target_obj}"`；預設為 `None`，不啟用。）
- **timestep 排程**：自訂 `custom_set_timesteps`（`attack/DIA_PT.py:165-192`），
  `timesteps = round(linspace(0,1,10) * 998) + 1e-6`，再加 `steps_offset`；反演時不反轉順序。
- **DIA-PT 的隨機初始化**（`attack/DIA_PT.py:241-248`）：

  ```python
  if self.norm == 'L1':
      t = torch.randn_like(source_latents).to(device).detach()
      delta = L1_projection(source_latents, t, self.eps)
      x_adv = source_latents + t + delta
  else:
      x_adv = source_latents.clone()
  x_adv = x_adv.clamp(-1., 1.).to(dtype=source_latents.dtype)
  ```

  DIA-PT 的 `norm` 為 `"L1"` → 有隨機初始化；DIA-R 的 `norm` 為 `null` → **無**隨機初始化。
- **seed**：`attack_benchmark.py` 的 `--seed` 預設 1234。
- **`wramup_interval: 14`**：此鍵存在於 JSON，但在 `DIA_PT.py`／`DIA_R.py`／`utils_general_H.py`
  中**完全未被引用**，為無作用的殘留設定。

## 3.「Crop & Resize」淨化的確切參數

`[DIA-ARX]` Supplementary B.2 全文（該節僅存在於 arXiv 版，CVF 版不含補充資料）：

> 「**Crop & Resize**: A naturally occurring and effective purification technique. We cropped 10% of
> each image and then resized it to match the model's input requirements.」

**可確認**：裁掉 10%，再 resize 回模型輸入尺寸（正文他處為 512×512）。

**未確認（見 §7 未找到清單）**：

- 「10%」指面積的 10%、每邊各 10%、或單邊合計 10%——原文未界定。
- 中心裁切或隨機裁切——原文未界定。
- resize 的插值方式（bilinear／bicubic／lanczos）——原文未提及。
- 官方 repo 未釋出任何淨化程式碼，無法從實作反推。

同節列出的其他淨化方法參數（可確認者）：

- **JPEG Compression**：quality ∈ {70, 80, 90}（Fig. 6 圖中示範用 80）。
- **Adverse Cleaner**：見 §4。
- **Gaussian Noising**：「adds random Gaussian noise on immunized images. We provide results with σ=0.1.」
- **Noisy Upscaling**：「a two-stage purification method proposed by Shan et al. [23], which applies
  Gaussian Noising (σ=0.1) followed by Stable Diffusion Upscaler [20].」

## 4.「Adverse Cleaner」是什麼、出處、如何呼叫

### 4.1 論文中的描述與引用

`[DIA-ARX]` Supplementary B.2：

> 「**Adverse Cleaner [33]**: An algorithmic approach capable of purifying high-frequency noise patterns.」

參考文獻 [33]（`[DIA-CVF]` p.10）：

> [33] Lvmin Zhang. AdverseCleaner. `https://github.com/lllyasviel/AdverseCleaner`, 2023.

### 4.2 出處連結現況

**原始 repo `https://github.com/lllyasviel/AdverseCleaner` 已被刪除（HTTP 404）。**
GitHub API 對該 repo 亦回傳空內容。以下為仍可存取的鏡像／再實作：

- https://github.com/toyxyz/AdverseCleaner （完整保留原 README 與 `clean.py`，僅改為批次處理）
- https://huggingface.co/spaces/p1atdev/AdverseCleaner （Gradio 版，`main.py`）
- https://github.com/gogodr/AdverseCleanerExtension （AUTOMATIC1111 webui 擴充）

### 4.3 演算法與呼叫方式

不使用深度學習：**64 次 bilateral filter 串接，再 4 次 guided filter**。
兩個獨立鏡像的參數完全一致，交叉驗證通過。

`[AC-MIRROR]` `toyxyz/AdverseCleaner` `clean.py`：

```python
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter

def process_image(input_path, output_path):
    img = cv2.imread(input_path).astype(np.float32)
    y = img.copy()

    for _ in range(64):
        y = cv2.bilateralFilter(y, 5, 8, 8)

    for _ in range(4):
        y = guidedFilter(img, y, 4, 16)

    cv2.imwrite(output_path, y.clip(0, 255).astype(np.uint8))
```

HuggingFace Space `p1atdev/AdverseCleaner` `main.py` 的預設參數與之相同：

```python
def clean_image(input_image, diameter=5, sigma_color=8, sigma_space=8, radius=4, eps=16):
    img = np.array(input_image).astype(np.float32)
    y = img.copy()
    for _ in range(64):
        y = cv2.bilateralFilter(y, diameter, sigma_color, sigma_space)
    for _ in range(4):
        y = guidedFilter(img, y, radius, eps)
    return Image.fromarray(y.clip(0, 255).astype(np.uint8))
```

參數對照：

| 步驟 | 函式 | 參數 | 次數 |
|---|---|---|---|
| 1 | `cv2.bilateralFilter(y, d, sigmaColor, sigmaSpace)` | `d=5, sigmaColor=8, sigmaSpace=8` | 64 |
| 2 | `cv2.ximgproc.guidedFilter(guide, src, radius, eps)` | `guide=原圖 img`, `src=y`, `radius=4`, `eps=16` | 4 |

重點：guided filter 的 **guide 影像是未經處理的原圖 `img`**（含對抗噪聲），非上一輪輸出。
原作者在 README 的 2023/03/28 更新中自承這一點是弱點：

> 「Seems that using guided filter is not safe enough because the guidance already has adversarial
> noise in it; the guided filter may bring the adversarial noise back.」

輸入為 `cv2.imread` 的 **BGR uint8 → float32，值域 [0,255]**；輸出 `clip(0,255).astype(np.uint8)`。
依賴 `opencv-contrib-python`（`cv2.ximgproc` 不在主套件中）。

## 5. 淨化章節的完整數字

**先更正題目的定位**：`[DIA-CVF]` 的 **Table 2 不是淨化表**，而是「9 種編輯技術平均的
structure / background preservation」。**Fig. 6 是純質性圖示、不含任何數字**
（單一影像 "a [rabbit→cat] is sitting in a pile of colorful eggs" 在 Immunized／Edited／
JPEG(80)／Crop & Resize／AdverseCleaner 五欄下的視覺比較）。

淨化的完整數字在 **`[DIA-ARX]` Supplementary Table 6**（CVF 版沒有）。
設定：PIE-Bench **700 張**免疫影像，編輯組合為正文預設；箭頭方向為「攻擊方越好」。

### 5.1 Table 6（Supplementary B.2）完整內容

| 淨化方法 | 攻擊方法 | CLIP↓ | Distance↑ | PSNR↓ | LPIPS↑ | MSE↑ | SSIM↓ |
|---|---|---|---|---|---|---|---|
| —（無淨化） | Natural Edit | 25.7100 | 0.02613 | 23.8400 | 0.09933 | 0.00639 | 0.80723 |
| JPEG (90) | Photoguard | 25.6936 | 0.05174 | 21.4936 | 0.22142 | 0.00962 | 0.70957 |
| JPEG (90) | Glaze | 25.8862 | 0.03472 | 22.4310 | 0.15541 | 0.00829 | 0.74462 |
| JPEG (90) | AdvDM | 24.5583 | 0.07191 | 21.0165 | 0.21653 | 0.01233 | 0.67113 |
| JPEG (90) | SDS | 24.1685 | 0.06742 | 20.8744 | 0.21799 | 0.01181 | 0.67212 |
| JPEG (90) | **DIA-PT** | 24.2789 | 0.07655 | 20.0105 | 0.27472 | 0.01610 | 0.63854 |
| JPEG (90) | **DIA-R** | 23.7255 | 0.08374 | 19.2542 | 0.21633 | 0.02637 | 0.67706 |
| JPEG (80) | Photoguard | 26.0247 | 0.04463 | 22.0655 | 0.18531 | 0.00880 | 0.73312 |
| JPEG (80) | Glaze | 26.0196 | 0.03095 | 23.0004 | 0.13806 | 0.00741 | 0.76862 |
| JPEG (80) | AdvDM | 24.4738 | 0.07044 | 21.1526 | 0.21349 | 0.01209 | 0.67737 |
| JPEG (80) | SDS | 24.1725 | 0.06773 | 21.0680 | 0.21292 | 0.01159 | 0.67927 |
| JPEG (80) | **DIA-PT** | 24.8818 | 0.05645 | 20.9928 | 0.23660 | 0.01259 | 0.68420 |
| JPEG (80) | **DIA-R** | 24.2000 | 0.07318 | 19.9076 | 0.20011 | 0.02179 | 0.69623 |
| JPEG (70) | Photoguard | 26.0953 | 0.04212 | 22.3060 | 0.16901 | 0.00836 | 0.74815 |
| JPEG (70) | Glaze | 26.0306 | 0.03010 | 23.1220 | 0.13043 | 0.00712 | 0.77931 |
| JPEG (70) | AdvDM | 24.7252 | 0.06771 | 21.3877 | 0.20781 | 0.01172 | 0.68590 |
| JPEG (70) | SDS | 24.2350 | 0.06743 | 21.1680 | 0.21060 | 0.01171 | 0.68238 |
| JPEG (70) | **DIA-PT** | 25.5055 | 0.04608 | 21.6928 | 0.20653 | 0.01036 | 0.71623 |
| JPEG (70) | **DIA-R** | 25.0112 | 0.06244 | 20.6276 | 0.18408 | 0.01673 | 0.71640 |
| **Crop & Resize** | Photoguard | 25.7733 | 0.08424 | 17.3266 | 0.25859 | 0.02362 | 0.61375 |
| **Crop & Resize** | Glaze | 25.8354 | 0.06285 | 17.3832 | 0.23109 | 0.02411 | 0.61677 |
| **Crop & Resize** | AdvDM | 25.1026 | 0.07810 | 17.1060 | 0.27175 | 0.02563 | 0.57034 |
| **Crop & Resize** | SDS | 24.5399 | 0.07795 | 16.9971 | 0.26720 | 0.02630 | 0.57432 |
| **Crop & Resize** | **DIA-PT** | 24.8340 | 0.07498 | 16.9972 | 0.29035 | 0.02618 | 0.57572 |
| **Crop & Resize** | **DIA-R** | 24.8310 | 0.08518 | 16.5469 | 0.25361 | 0.03056 | 0.59598 |
| **Adverse Cleaner** | Photoguard | 25.3614 | 0.06022 | 21.7390 | 0.19018 | 0.00939 | 0.75646 |
| **Adverse Cleaner** | Glaze | 25.7053 | 0.03406 | 22.8134 | 0.14303 | 0.00768 | 0.78250 |
| **Adverse Cleaner** | AdvDM | 24.6748 | 0.04763 | 22.4885 | 0.16196 | 0.00926 | 0.75834 |
| **Adverse Cleaner** | SDS | 24.1779 | 0.05513 | 22.1588 | 0.16882 | 0.01001 | 0.74709 |
| **Adverse Cleaner** | **DIA-PT** | 25.3572 | 0.03936 | 21.9392 | 0.18543 | 0.00971 | 0.75839 |
| **Adverse Cleaner** | **DIA-R** | 24.6166 | 0.06166 | 20.6104 | 0.18917 | 0.01714 | 0.73489 |
| Gaussian Noising | Photoguard | 26.1351 | 0.0426 | 21.5181 | 0.2908 | 0.0094 | 0.5805 |
| Gaussian Noising | Glaze | 26.1265 | 0.0364 | 22.1301 | 0.2591 | 0.0084 | 0.6048 |
| Gaussian Noising | AdvDM | 25.5851 | 0.0528 | 21.5729 | 0.2772 | 0.0104 | 0.5920 |
| Gaussian Noising | SDS | 25.3543 | 0.0515 | 21.7844 | 0.2720 | 0.0101 | 0.6031 |
| Gaussian Noising | **DIA-PT** | 26.2993 | 0.0423 | 21.4033 | 0.2809 | 0.0099 | 0.5788 |
| Gaussian Noising | **DIA-R** | 25.9461 | 0.0449 | 21.1744 | 0.2750 | 0.0115 | 0.5918 |
| Noisy Upscaling | Photoguard | 25.4812 | 0.0381 | 23.0208 | 0.1561 | 0.0076 | 0.7654 |
| Noisy Upscaling | Glaze | 25.4772 | 0.0351 | 22.9662 | 0.1506 | 0.0077 | 0.7658 |
| Noisy Upscaling | AdvDM | 25.5246 | 0.0344 | 22.8962 | 0.1582 | 0.0076 | 0.7553 |
| Noisy Upscaling | SDS | 25.5639 | 0.0355 | 22.8581 | 0.1609 | 0.0079 | 0.7516 |
| Noisy Upscaling | **DIA-PT** | 25.5119 | 0.0374 | 22.8195 | 0.1568 | 0.0078 | 0.7615 |
| Noisy Upscaling | **DIA-R** | 25.6470 | 0.0351 | 22.8716 | 0.1511 | 0.0079 | 0.7662 |

> **PID 未出現在 Table 6**（正文 Table 1／Table 2 有 PID，淨化表沒有）。

正文對此的結論（`[DIA-CVF]` p.8, §4.4.3）僅一句：

> 「As shown in Fig. 6, our method maintains strong performance, with only minor degradation.」

補充資料 B.2 的結論：

> 「As shown in Table 6, all baselines demonstrate robustness to purification when compared to Natural
> Edit. Notably, our method maintains superior performance while remaining robust to most purification
> methods. In some experiments, SDS shows sub-optimal performance, which appears to be due to its
> low-frequency pattern and higher degradation scale.」

**客觀觀察（供本專案判讀）**：在 Gaussian Noising 與 Noisy Upscaling 兩列，DIA-PT／DIA-R 的
CLIP 值（26.2993／25.9461、25.5119／25.6470）已接近甚至高於 Natural Edit 的 25.7100，
即免疫效果基本被消除；論文正文並未討論這兩列。

### 5.2 正文 Table 2（非淨化，補充列出以免混淆）

`[DIA-CVF]` p.7, Table 2「Average background and structure preservation metric for 9 editing techniques」：

| Method | Distance↑ | PSNR↓ | LPIPS↑ | MSE↑ | SSIM↓ |
|---|---|---|---|---|---|
| Natural Edit | 0.0249 | 24.3767 | 0.0914 | 0.0071 | 0.8124 |
| PhotoGuard | 0.0773 | 19.6509 | 0.2617 | 0.0148 | 0.6584 |
| Glaze | 0.0440 | 21.3841 | 0.1927 | 0.0111 | 0.6958 |
| AdvDM | 0.0940 | 19.6309 | 0.2838 | 0.0167 | 0.5933 |
| SDS | 0.0685 | 20.5587 | 0.2703 | 0.0135 | 0.6232 |
| PID | 0.0630 | 20.0265 | 0.2878 | 0.0151 | 0.6211 |
| **DIA-PT** | 0.1059 | 18.2202 | 0.3410 | 0.0237 | 0.5653 |
| **DIA-R** | 0.1252 | 16.3055 | 0.2940 | 0.0460 | 0.5903 |

### 5.3 超參數消融（`[DIA-ARX]` Supplementary B.3，重現時的收斂依據）

Table 7（attack iteration，DDIM-to-DDIM）：

| Method | Iter | CLIP↓ | Distance↑ | PSNR↓ | LPIPS↑ | MSE↑ | SSIM↓ |
|---|---|---|---|---|---|---|---|
| DIA-PT | 5 | 25.6048 | 0.0482 | 21.5865 | 0.2094 | 0.0103 | 0.6992 |
| DIA-PT | 10 | 24.3525 | 0.0751 | 20.0086 | 0.2693 | 0.0155 | 0.6366 |
| DIA-PT | 15 | 23.7979 | 0.0913 | 19.2879 | 0.2949 | 0.0188 | 0.6078 |
| DIA-PT | **20** | 23.4575 | 0.1006 | 18.7744 | 0.3124 | 0.0208 | 0.5874 |
| DIA-R | 5 | 24.6790 | 0.0547 | 20.8372 | 0.1791 | 0.0133 | 0.7274 |
| DIA-R | 10 | 24.3205 | 0.0670 | 19.9336 | 0.2038 | 0.0186 | 0.6967 |
| DIA-R | 15 | 23.8511 | 0.0796 | 19.3068 | 0.2190 | 0.0239 | 0.6818 |
| DIA-R | **20** | 23.4670 | 0.0882 | 18.7633 | 0.2307 | 0.0288 | 0.6666 |

Table 8（trajectory length）：

| Method | Traj. Len | CLIP↓ | Distance↑ | PSNR↓ | LPIPS↑ | MSE↑ | SSIM↓ |
|---|---|---|---|---|---|---|---|
| DIA-PT | 5 | 25.6181 | 0.0506 | 21.1142 | 0.2163 | 0.0107 | 0.6967 |
| DIA-PT | **10** | 23.4575 | 0.1006 | 18.7744 | 0.3124 | 0.0208 | 0.5874 |
| DIA-PT | 20 | 24.1361 | 0.0782 | 20.0423 | 0.2835 | 0.0154 | 0.6209 |
| DIA-R | 5 | 24.3258 | 0.0676 | 19.7006 | 0.2118 | 0.0179 | 0.6899 |
| DIA-R | 10 | 23.4670 | 0.0882 | 18.7633 | 0.2307 | 0.0288 | 0.6666 |
| DIA-R | **20** | 22.0941 | 0.1101 | 17.5972 | 0.2540 | 0.0432 | 0.6451 |

原文說明：

> 「Table 8 reveals a difference between DIA-PT and DIA-R, with their best values found at trajectory
> lengths of 10 and 20, respectively. […] To ensure a consistent inversion trajectory environment
> across all our experiments, we set the trajectory length to 10.」

亦即 **DIA-R 在論文的主結果中並非其最佳設定**（其最佳為 20）。

## 6. 值域

**論文正文與補充資料均未明述值域。** 由程式碼可完全確認為 **`[-1, 1]`**：

前處理（`[DIA-CODE]` `attack_benchmark.py:16-27`）：

```python
TARGET_SIZE = 512
tform = tf.Compose([
    tf.Resize(TARGET_SIZE),
    tf.CenterCrop(TARGET_SIZE),
    tf.ToTensor(),
])
Image_to_tensor = lambda x: 2.0*tform(x)-1.0
path_to_tensor  = lambda x: Image_to_tensor(Image.open(x).convert('RGB'))
```

`ToTensor()` 給出 `[0,1]`，再 `2x−1` → **`[-1,1]`**，解析度 512×512。

後處理（`attack_benchmark.py:56-70`）：

```python
x = (x/2.0+0.5).clamp(0,1)
x = to_pil(x[0]).convert('RGB')
```

設定檔中 `clamp_min: -1`、`clamp_max: 1` 與之一致（`grad_normalize` 的最後一步夾在此範圍）。

**對本專案的直接後果**：`eps = 0.05` 是在 **`[-1,1]` 尺度**上量測的，換算成 `[0,1]` 尺度為
**0.025**，換算成 8-bit 為 **≈ 6.375/255**。`step_size = 1/255` 同樣以 `[-1,1]` 計，
換算為 `[0,1]` 尺度是 `0.5/255`。若要做「匹配人眼可辨失真」的公平比較，
**不可**直接把 0.05 當成 `[0,1]` 尺度的 L∞ 預算，否則失真預算會高出一倍。
此換算與 Fig. 5 報告的 PSNR（DIA-R 40.27 dB、DIA-PT 36.50 dB）量級一致。

## 7. DIA — 未找到的項目

逐條列出，不省略：

1. **Crop & Resize 的裁切定義**：「cropped 10%」未界定是面積 10%、每邊各 10%、
   或單邊合計 10%。**未找到**。
2. **Crop & Resize 的裁切位置**：中心裁切或隨機裁切，原文未說明。**未找到**。
3. **Crop & Resize 的插值方式**：resize 回 512 時用 bilinear／bicubic／lanczos／area，
   原文未提及，官方 repo 亦無淨化程式碼。**未找到**。
4. **淨化流程的實作程式碼**：JPEG／Crop&Resize／AdverseCleaner／Gaussian Noising／
   Noisy Upscaling 皆未釋出。**未找到**。
5. **評測程式碼**：PIE-Bench 的 CLIP similarity、Distance、mask 套用方式（正文只說
   「we compare the cosine similarity between the unmasked image and the edit text embedding」）
   的具體實作未釋出。**未找到**。
6. **baseline 的實作與超參數**：Photoguard、Glaze、AdvDM、SDS、PID 在本論文中的
   具體設定，除「Photoguard 與 AdvDM 用 60 iterations、全部方法 eps = 0.05」外，
   其餘（step size、attack 的 timestep 取樣、Glaze/PID 的專屬參數）**未找到**。
7. **Fig. 4（epsilon budget 0.025/0.05/0.075/0.1 與 sampling steps 20/50/1000 的
   CLIP similarity 曲線）的數值**：僅為折線圖，論文未附數表。Sampling steps 的部分
   可由 Supplementary Table 5 取得 20/50/1000 三點（但那是 140 張影像的子集，
   非正文 Fig. 4 的同一組數據）。Epsilon budget 的逐點數值**未找到**。
8. **PID 在淨化表（Table 6）中的數字**：該表未列 PID。**未找到**。
9. **Table 6 所用的 inversion-edit 組合**：補充資料未指明是 DDIM-to-DDIM 或 9 組平均。**未找到**。
10. **Eq. 9 / Eq. 10 的平方與否**：論文寫 `||·||²₂`、程式碼寫 `.norm(p=2)`。
    兩者不一致；何者為作者本意**未找到**明確說明（對 sign-based PGD 無影響）。
11. **`wramup_interval: 14`** 的用途：設定檔存在但程式碼未引用。**未找到**用途說明。
12. **`num_inner_inference_steps: 10`** 的用途：設定檔存在，`DIA_PT.py`／`DIA_R.py` 中
    未見引用（只用 `num_inference_steps`）。**未找到**用途說明。

---

# 對象二：APA（arXiv:2506.01511）

## 0. 官方程式碼

論文摘要末（`[APA-ARX]` p.1）：

> 「Code will be available at `https://github.com/deep-kaixun/APA`.」

**該 repo 存在且已釋出程式碼**（`created_at` 2024-12-05、`pushed_at` 2025-11-19、
語言 Python、無 LICENSE）。檔案：

```
README.md
visual_alignment.py      ← 階段一（Visual Consistency Alignment）
attack_alignment.py      ← 階段二驅動
pipe_ours.py             ← 階段二核心（AttackPipeline）
utils.py                 ← 模型載入、可微分資料增強
data.json                ← 資料清單（image_path / class / label / id）
images_un/1..5.png       ← 5 張範例影像
model_ckpt/download.sh
```

> **關鍵警示**：`[APA-ARX]` v1 是**唯一版本**（arXiv 提交紀錄只有 v1，2025-06-02），
> 且**不含 Appendix**。論文正文四處寫「More implementation details in Appendix」、
> 「Appendix further shows that…」、「we extend it to various diffusion models … in Appendix」，
> 但 PDF（11 頁）與 HTML 版的章節結構皆終止於 References，`ltx_appendix` 標記數為 0。
> **因此階段一的所有訓練超參數在論文中完全不存在，只能取自官方程式碼。**

## 1. 階段一（Visual Consistency Alignment）的完整目標式與訓練設定

### 1.1 論文中的目標式

`[APA-ARX]` §3.3（p.3–4）。出發點（Eq. 5）：

```
max_{Δθ}  R_s(Δθ) = S( D(z̄_0), x )
```

> 「where `z̄_0` represents the T-step denoised output, requiring T computations of Eq. 3.」

化簡過程（原文）：

> 「To reduce this, we first shift the similarity metric to the latent space, calculating `S(z̄_0, z_0)`.
> We then approximate trajectory-level similarity by accumulating similarity across all steps.
> Thus, inspired by Eq. 2, `R_s(Δθ)` is reformulated as:」

**最終目標式（Eq. 6）**：

```
R_s(Δθ) = E_{t,ε}  −‖ ε − ε_{θ+Δθ}(z_t, t, c) ‖²,     t ∈ [1, T]
```

更新規則（原文，Eq. 6 下方）：

> 「Since Eq. 6 is differentiable, we can update `Δθ` via the direct backpropagation to maximize the
> reward, as follows: `Δθ = Δθ + α∇_{Δθ}R_s`, where α represents the learning rate. Finally, `Δθ` is
> integrated into `ε_θ`, enabling the model to generate visually consistent outputs whether regular
> noise or adversarial noise is applied to `z_T`.」

亦即：**階段一就是在單張輸入影像上，用標準的 ε-prediction diffusion 損失微調 LoRA**
（最大化負 MSE 等價於最小化 MSE）。這與 DreamBooth／LoRA 個人化的訓練目標相同，
差別在於只用一張影像、且目的是讓模型對 `z_T` 的擾動具備重建魯棒性。

### 1.2 訓練設定（**僅存在於程式碼**）

`[APA-CODE]` `visual_alignment.py:248-258`：

```python
cfg = {
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "seed": 0,
    "rank": 8,
    "n_epochs": 200,
    "checkpointing_steps": 500,
    "noise_offset": 0.1,
    "max_grad_norm": 1.0,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
}
```

**LoRA 設定**（`visual_alignment.py:95-100`）：

```python
unet_lora_config = LoraConfig(
    r=train_cfg['rank'],            # r = 8
    lora_alpha=train_cfg['rank'],   # lora_alpha = 8（與 rank 相同 → scaling = alpha/r = 1.0）
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
unet.add_adapter(unet_lora_config)
```

| 項目 | 值 | 來源 |
|---|---|---|
| LoRA rank `r` | **8** | `visual_alignment.py:251` |
| LoRA `alpha` | **8**（等於 rank，故 scaling = 1.0） | `visual_alignment.py:97` |
| 初始化 | `init_lora_weights="gaussian"` | `visual_alignment.py:98` |
| 掛載模組 | **`["to_k", "to_q", "to_v", "to_out.0"]`** | `visual_alignment.py:99` |
| 掛載範圍 | **僅 UNet**。VAE 與 text_encoder 皆 `requires_grad_(False)` 且未加 adapter | `visual_alignment.py:85-93, 106` |

> **掛載層的精確含義**：PEFT 以模組名稱後綴比對，`to_q/to_k/to_v/to_out.0` 同時涵蓋
> UNet 內的 **self-attention（`attn1`）與 cross-attention（`attn2`）**，
> 不含 ResNet 卷積、不含 FFN（`ff.net.*`）、不含 `proj_in`／`proj_out`。

**optimizer**（`visual_alignment.py:109-126`）：

```python
optimizer_cls = torch.optim.AdamW
optimizer = optimizer_cls(
    lora_layers,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)
lr_scheduler = get_scheduler('constant', optimizer=optimizer)
```

| 項目 | 值 |
|---|---|
| optimizer | **AdamW** |
| lr | **1e-4**（硬編碼，非由 cfg 傳入） |
| betas | (0.9, 0.999) |
| weight_decay | 1e-2 |
| eps | 1e-8 |
| lr scheduler | **constant**（無 warmup、無 decay） |
| 步數 | **200**（`max_train_steps = n_epochs = 200`；迴圈每個 epoch 只做一次 optimizer step，batch=1，故總計 200 次參數更新） |
| grad clipping | `max_grad_norm = 1.0` |
| gradient accumulation | 1 |
| seed | 0 |
| 權重精度 | `weight_dtype = torch.float32` |

**損失與取樣**（`visual_alignment.py:158-192`）：

```python
latents = vae.encode(batch_data["pixel_values"]).latent_dist.sample()
latents = latents * vae.config.scaling_factor          # 0.18215

noise = torch.randn_like(latents)
if train_cfg['noise_offset']:
    noise += train_cfg['noise_offset'] * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device)   # noise_offset = 0.1

timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
encoder_hidden_states = text_encoder(batch_data["input_ids"])[0]

target = noise                                          # prediction_type == "epsilon"
model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
```

- **timestep 取樣**：`randint(0, 1000)` 均勻取樣，每步一個 timestep（對應 Eq. 6 的 `E_{t,ε}`）。
- **noise offset = 0.1**：論文完全未提及的額外項（`add_noise` 前對 noise 加上逐通道的
  常數偏移）。重現時若省略此項，會與官方實作不符。
- scheduler：`DDPMScheduler`（由 SD1.5 的 `scheduler` 子資料夾載入）。

**輸入前處理**（`visual_alignment.py:42-50`）：

```python
train_transforms = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
```

→ 512×512、值域 **`[-1,1]`**（VAE 輸入）。

**條件文字 `c`**（`visual_alignment.py:146, 151`）：`caption = data_cfg['class']`，
即 `data.json` 中該影像的 **ImageNet 類別名稱**（無 "a photo of" 模板）。
tokenizer 以 `padding="max_length"`、`max_length=tokenizer.model_max_length`（77）處理。

**產物**：每張影像各訓練一組 LoRA，存成 `{id}.safetensor`
（`visual_alignment.py:230-235`）。即 **LoRA 是 per-image 的**，不是全資料集共用。

## 2. "rule-based similarity reward" 具體是什麼

摘要原文：

> 「In the first stage, APA fine-tunes LoRA to improve visual consistency using rule-based similarity
> reward.」

其定義即 **Eq. 6 的 `R_s(Δθ) = E_{t,ε} −‖ε − ε_{θ+Δθ}(z_t,t,c)‖²`**，
在程式碼中就是 `-F.mse_loss(model_pred, noise)`。

「rule-based」的含義由 §1（p.1–2）與 §3.3 的脈絡界定：APA 主張對抗偏好無法取得成對偏好資料，
因此無法用 DPO 或學習式 reward model；他們改用**解析式、固定規則的可微分度量**當作 reward。
論文原文：

> 「First, preference data is unavailable. […] This makes it infeasible to obtain stable,
> preference-consistent adversarial examples for pairwise data collection, rendering traditional
> preference optimization techniques like DPO [60] or unified reward modeling inapplicable.」

> 「1) Visual Consistency Alignment: We use a differentiable visual similarity metric as a rule-based
> reward and perform policy updates by fine-tuning the diffusion model's Low-Rank Adaptation (LoRA)
> parameters [25].」

**結論**：所謂 rule-based similarity reward **不是** LPIPS／SSIM／CLIP 之類的感知度量，
而是「denoising MSE 的負值」——一個由 diffusion 訓練目標直接給出的解析規則。
論文明確說明它是從影像空間相似度 `S(D(z̄_0),x)` 經兩步近似（移至 latent 空間、
以逐步相似度累加取代軌跡端點相似度）推導而來。

## 3. 階段二的 latent 最佳化

### 3.1 優化的是 `z_T` 還是每步的 latent

**主要優化變數是 `z_T`（DDIM inversion 的終點），單一變數，不是每步 latent。**

論文原文（§3.4 開頭）：

> 「In this stage, We use `z_T` obtained via DDIM inversion as the optimization variable (optional
> prompt embedding discussed in Section 4.5).」

程式碼（`[APA-CODE]` `pipe_ours.py`，`attack_optimization`）：

```python
la_0 = latents.clone().detach()      # 原始 z_T
adv_latents = latents                # 被優化的變數
momentum = 0
...
l1_grad = la_t_g / torch.norm(la_t_g, p=1)
momentum = momentum + l1_grad
adv_latents = adv_latents + torch.sign(momentum) * alpha
noise = (adv_latents - la_0).clamp(-eps, eps)
adv_latents = la_0 + noise
```

對應論文 Eq. 7：

```
g_tr = ∇_{z_T} R_a( f_φ(x_adv), y )
m^i_tr = m^{i-1}_tr + g_tr / ‖g_tr‖₁
z_T = Π_{z⁰_T + ε_a}( z_T + µ · sgn(m^i_tr) )
```

**但另有「step-wise」機制作用於軌跡內部**（Eq. 8，step-level attack guidance）：

```
ε_{θ+Δθ}(z_t,t,c) = ε_{θ+Δθ}(z_t,t,c) − √(1−ᾱ_t) ∇_{z_t} R_a( f_φ(D(z_t)), y )
```

這是 classifier-guidance 形式的**噪聲預測修正**，不是被持續保存、跨迭代累積的優化變數；
每次 denoise 重新計算。論文並將 `∇_{z_t}R_a` 換成 step-level momentum 的 sign（Eq. 11）：

```
g_st = ∇_{z_t} R_a( f_φ(x^t_in), y ),   m^t_st = m^{t+1}_st + g_st / ‖g_st‖₁
```

程式碼（`pipe_ours.py`，`cond_guidance`）：

```python
pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred.detach()) / alpha_prod_t ** 0.5
fac = torch.sqrt(beta_prod_t)
sample = ori_image_latent * fac + pred_original_sample * (1 - fac)      # 對應 Eq. 10
img = self.decode_la(sample)
img = torch.clamp(img, 0, 1)
img = F.interpolate(img, size=(self.image_size, self.image_size))
out = classfier(img)
loss = torch.nn.CrossEntropyLoss()(out, label)
grads = torch.autograd.grad(loss, latents)[0]
l1_grad = grads / torch.norm(grads, p=1)
self.m = self.m + l1_grad.detach()
...
noise_pred = noise_pred.detach() - torch.sqrt(beta_prod_t) * torch.sign(self.m)
```

其中 Eq. 9 與 Eq. 10 為：

```
z^t_0  = ( z_t − √(1−ᾱ_t) ε_{θ+Δθ}(z_t,t,c) ) / √ᾱ_t
z^t_in = √(1−ᾱ_t) · z_0  +  (1 − √(1−ᾱ_t)) · z^t_0
```

程式碼與 Eq. 10 完全一致（`fac = sqrt(1−ᾱ_t)`，`ori_image_latent = z_0 = E(x)`）。

### 3.2 lr、步數、投影與約束

`[APA-ARX]` §4.1 "Implementation Details" 原文：

> 「We set attack guidance step `T_a = 10`, attack iterations `N = 10`, attack scale `ε_a = 0.4`, and
> attack step size `µ = 0.04`. APA-SG adopts the entire inversion step of `T = 50`. APA-GC adopts
> `T = 10` to improve efficiency. Our work is based on Stable Diffusion V1.5 [51].」

程式碼預設值（`[APA-CODE]` `attack_alignment.py:6-21`）完全吻合：

```python
parser.add_argument('--alpha', type=float, default=0.04, help='alpha')     # µ
parser.add_argument('--niters', type=int, default=10, help='niters')       # N
parser.add_argument('--eps', type=float, default=0.4, help='eps')          # ε_a
parser.add_argument('--index_cond', default=40, type=int)
parser.add_argument('--inversion_step', default=10, type=int)
parser.add_argument('--gradient_back', default='skip-gradient', type=str)
parser.add_argument('--sd_name', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--source_model', type=str, default='ResNet50')
parser.add_argument('--seed', default=0, type=int)
```

| 項目 | 值 | 來源 |
|---|---|---|
| 優化變數 | `z_T`（DDIM inversion 終點） | 論文 §3.4；`pipe_ours.py` |
| 更新規則 | momentum + sign（MI-FGSM 式），非 Adam | 論文 Eq. 7；程式碼 |
| step size `µ` | **0.04** | 論文 §4.1；`--alpha` |
| 迭代數 `N` | **10** | 論文 §4.1；`--niters` |
| **投影／約束** | **L∞ 球，半徑 `ε_a = 0.4`，中心為原始 `z_T`（latent 空間）** | 論文 Eq. 7 的 `Π_{z⁰_T+ε_a}`；程式碼 `(adv_latents - la_0).clamp(-eps, eps)` |
| momentum | `m ← m + g/‖g‖₁`（L1 正規化累加） | 論文 Eq. 7；程式碼 |
| guidance 起點 | `index_cond = 40`（50 步中的第 40 步起 → 最後 10 步 = `T_a`） | `attack_alignment.py:18` |
| CFG | `guidance_scale = 1.0`（等同關閉 CFG） | `attack_alignment.py:148, 152, 156` |
| 反演 CFG | `guidance_scale=1` | `attack_alignment.py:146` |

**沒有**對 latent 的絕對值域做任何 clamp；唯一的約束就是相對於 `z⁰_T` 的 L∞ 球。

### 3.3 梯度回傳的兩種方式

**APA-SG（skip gradient）** — `pipe_ours.py`，`attack_optimization`。
整個 denoise 迴圈在 `@torch.no_grad()` 下執行，只對最終 latent `la_t` 取梯度，
再乘上一個常數換算到 `z_T`：

```python
la_t_g = 14.58 * torch.autograd.grad(reward, la_t, retain_graph=False, create_graph=False)[0].detach()
```

對應論文的 `g_tr ≈ ρ · ∇_{z̄_0} R_a(f_φ(x_adv), y)`。
**`ρ = 14.58` 為硬編碼常數，論文未給出此數值**（論文只寫 `ρ`，引用 ACA [9]）。

**APA-GC（gradient checkpointing）** — `pipe_ours.py`，`attack_optimization_checkpoint`。
以 `torch.utils.checkpoint.checkpoint(self.unet, ...)` 逐步重算，直接對 `adv_latents` 求梯度：

```python
noise_pred = checkpoint.checkpoint(self.unet, latent_model_input, t, prompt_embeds, use_reentrant=False).sample
...
la_t_g = torch.autograd.grad(reward, adv_latents, retain_graph=False, create_graph=False)[0].detach()
```

**無 `14.58` 縮放**（因為是真實梯度）。

## 4. "diffusion augmentation strategy" 的具體內容

### 4.1 論文描述

`[APA-ARX]` §3.4 "Diffusion Augmentation"：

> 「Studies on HPA [41] have shown that direct backpropagation often leads to the diffusion model
> over-optimizing for the reward model. Similarly, in APA, this causes overfitting to the substitute
> classifier, limiting transfer attack performance. To address this, we propose diffusion augmentation
> which uses step-level outputs as data augmentation to enhance the generalization of the
> trajectory-level gradient `g_tr`. Specifically, we collect the step-level `z^t_0` generated during
> the denoising using Eq. 9, and mix them with the trajectory-level final output `z̄_0`:」

**Eq. 12**：

```
x^t_0 = ϱ( ( D(z^t_0) + D(z̄_0) ) / 2 )
```

> 「where `ϱ(·)` denotes differentiable data augmentation including **random padding, resizing, and
> brightness adjustment**. Appendix further shows that stronger data transformations (e.g., [62] used
> in Lp attacks) can further boost performance, underscoring the scalability of our method. Finally,
> the trajectory-level gradient `g_tr` in Eq. 7 is enhanced to
> `g_tr = ∇_{z_T} (1/T) Σ_{t=0..T} R_a( f_φ(x^t_0), y )`.」

### 4.2 程式碼實作

`pipe_ours.py`：

```python
def diffusion_augmentation(self, img0, la_list, p='ori'):
    img_size = img0.shape[-1]
    img_l = torch.zeros((len(la_list), 3, img_size, img_size)).cuda()
    for i in range(len(la_list)):
        img = self.decode_la(la_list[i].detach()).detach()
        img = torch.clamp(img, 0, 1)
        img = F.interpolate(img.detach(), size=(img_size, img_size))
        img = (img0 + img) / 2
        if 'sia' in p:
            out = sia(img)
        else:
            out = ori_trans(img)
        img_l[i] = out
    return img_l
```

`ϱ(·)` = `utils.py` 的 `ori_trans`：

```python
def random_size(x):
    img_size = x.shape[-1]
    rnd = int(np.random.randint(int(img_size*0.9), img_size, size=1)[0])
    return F.interpolate(x.clone(), size=(rnd, rnd))

def padding(x_ori, x_resize):
    ori_size = x_ori.shape[-1]; resize_size = x_resize.shape[-1]
    h_rem = ori_size - resize_size; w_rem = ori_size - resize_size
    pad_top = int(np.random.randint(0, h_rem, size=1)[0]);  pad_bottom = h_rem - pad_top
    pad_left = int(np.random.randint(0, w_rem, size=1)[0]); pad_right = w_rem - pad_left
    return F.pad(x_resize, (pad_left, pad_top, pad_right, pad_bottom), mode='constant', value=0)

def ori_trans(x_in):
    img_size = x_in.shape[-1]
    x_resize = random_size(x_in)
    x_out = padding(x_in, x_resize)
    #x_out=brightness(x_out)
    x_out = F.interpolate(x_out, size=(img_size, img_size))
    return x_out
```

具體步驟：
1. 隨機縮放到 `[0.9·S, S)` 之間的正方形尺寸（`F.interpolate` 預設 `mode='nearest'`）。
2. 隨機四邊零填充回原尺寸 `S`。
3. 再 `interpolate` 回 `S`（第 2 步已是 `S`，此步實質為恆等或重取樣）。

> **與論文不符處**：論文說 `ϱ` 包含 **brightness adjustment**，但釋出程式碼中
> `#x_out=brightness(x_out)` **被註解掉**。`utils.py` 中 `brightness(x, p=0.5)` 定義為
> `return p*x`（乘以固定係數 0.5，非隨機）。重現時須自行決定依論文或依程式碼。

`la_list` 的內容（`attack_optimization` 迴圈內）：

```python
if i >= index_cond and use_da:
    alpha_prod_t = self.scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
    la_list.append(pred_original_sample.detach())
```

即只收集**最後 `T_a = 10` 步**的 `z^t_0`（不是全部 T 步）。因此 Eq. 12 的
`(1/T)Σ_{t=0..T}` 在實作上是「對這 10 個增強樣本取平均的 cross-entropy」：

```python
out = classfier(img_l)
reward = torch.nn.CrossEntropyLoss()(out, label.repeat_interleave(len(la_list)))
```

（`CrossEntropyLoss` 預設 `reduction='mean'`，故等價於平均。）

## 5. APA-SG（T=50）與 APA-GC（T=10）的差異

### 5.1 論文陳述

- §3.4 末：「We introduce APA-SG for skip gradient and APA-GC for gradient checkpointing.」
- §4.1：「APA-SG adopts the entire inversion step of `T = 50`. APA-GC adopts `T = 10` to improve efficiency.」
- §4.6 "Inversion Step T"：

  > 「APA-GC employs gradient checkpointing to save memory at the cost of additional time. To improve
  > efficiency, we investigate the impact of reducing inversion steps `T` on performance. Figure 6(a)
  > shows that setting `T` below `T_a` reduces attack performance due to insufficient guidance, while
  > exceeding `T_a` also degrades attack performance due to bias introduced by overly deep gradient
  > chains. Thus, we set `T = T_a = 10`.」

- §4.2：「APA-SG (with the same gradient backpropagation as ACA) to improve black-box ASR by 12.5%,
  while APA-GC improves black-box performance by 24.4% over ACA across four models.」
  以及「APA-GC achieves an average performance improvement of 11.9% over APA-SG.」

效能對照（Table 1，黑箱平均 ASR %）：

| 替代模型 | ACA | APA-SG | APA-GC |
|---|---|---|---|
| MobViT-s | 56.90 | 65.85 | **77.48** |
| MN-v2 | 54.38 | 72.14 | **87.01** |
| RN-50 | 59.49 | 75.32 | **88.02** |
| ViT-B | 58.68 | 66.21 | **74.82** |

防禦模型（Table 2，ViT-B 為替代模型，Avg. ASR %）：ACA 57.47、APA-SG 63.59、**APA-GC 70.20**。

視覺品質（Table 3，RN-50 為替代模型）：

| 方法 | LPIPS↓ | SSIM↑ | CLIPScore↑ | NIMA-AVA↑ | CNN-IQA↑ | Avg.ASR↑ |
|---|---|---|---|---|---|---|
| Clean | 0.00 | 1.00 | 1.00 | 4.99 | 0.58 | 6.83 |
| VCA（僅階段一） | 0.05 | 0.85 | 0.97 | 5.13 | 0.63 | 7.60 |
| ACA | 0.37 | 0.61 | 0.79 | 5.38 | 0.65 | 59.49 |
| **APA-SG** | 0.25 | 0.67 | 0.86 | 5.29 | 0.62 | 75.32 |
| **APA-GC** | 0.23 | 0.69 | 0.83 | 5.39 | 0.67 | 88.02 |
| **APA-GC-P**（優化 prompt） | 0.09 | 0.82 | 0.91 | 5.22 | 0.63 | 62.08 |

### 5.2 程式碼中的實際差異（**與論文字面敘述不同，重現必讀**）

`attack_alignment.py:145-157`：

```python
if args.gradient_back == 'skip-gradient':
    latent_T = pipeline.inverse(image_path, prompt, 50, guidance_scale=1)
    adv_image_tensor = pipeline.attack_optimization(...)
else:   # gradient checkpointing
    latent_T = pipeline.inverse(image_path, prompt, 50, guidance_scale=1, inversion_step=args.inversion_step)
    adv_image_tensor = pipeline.attack_optimization_checkpoint(..., inversion_step=args.inversion_step)
```

`ddim_inversion` 中的 `inversion_step`（`pipe_ours.py`）：

```python
self.scheduler.set_timesteps(n_steps)      # n_steps = 50（兩者皆然）
...
for i, t in enumerate(tqdm(timesteps, desc="DDIM inversion")):
    ...
    if inversion_step is not None and i == inversion_step:
        break
```

`attack_optimization_checkpoint` 中的 denoise 迴圈：

```python
for i, t in enumerate(timesteps):        # len(timesteps) = 50
    if i < len(timesteps) - inversion_step - 1:   # i < 39 → 跳過
        continue
```

**因此 APA-GC 的 `T = 10` 並非「用 10 步均勻排程」，而是「沿用 50 步的 timestep 排程，
但只執行其中的 11 步」**：反演只跑前 11 步（`i = 0..10`，反演到中等噪聲水準而非完整 `z_T`），
去噪只跑最後 11 步（`i = 39..49`）。`index_cond = 40` 使 attack guidance 覆蓋 `i = 40..49`
共 10 步，與 `T_a = 10` 一致。

而 APA-SG 走完整的 50 步反演與 50 步去噪，guidance 同樣只在 `i ≥ 40` 的最後 10 步。

| 面向 | APA-SG | APA-GC |
|---|---|---|
| `g_tr` 取得方式 | skip gradient，`14.58 × ∇_{z̄_0}` | gradient checkpointing，真實 `∇_{z_T}` |
| 反演步數 | 50 步排程、跑滿 | 50 步排程、只跑前 11 步 |
| 去噪步數 | 50 | 11（`i=39..49`） |
| attack guidance 步數 | 10（`i≥40`） | 10（`i≥40`） |
| 記憶體 | 低（denoise 全程 `no_grad`） | 高（需 checkpoint 重算） |
| 時間 | 較快 | 較慢（§4.6 圖示為 1.2–1.7 倍） |
| 額外 reward 項 | **無** | **有 `−10·MSE(z_0, z̄_0)`**（見 §6） |

## 6. 失真如何約束

**這是本次查證中最重要的發現，論文與程式碼不一致。**

### 6.1 論文層面

論文**沒有**在目標函數中放入任何影像空間的失真約束（LPIPS／SSIM／L_p 皆無）。
視覺一致性完全由兩個機制間接達成：

1. **階段一的 LoRA 過擬合**：把輸入影像的結構寫入模型生成空間。原文（§3.3）：

   > 「To preserve visual consistency during adversarial optimization, we aim to strengthen the
   > diffusion model's retention of the input image `x`. […] This stage encodes the input image's
   > structure into the model's generation space, forming a visually stable foundation for downstream
   > attack optimization.」

2. **latent 空間的 L∞ 球**：`ε_a = 0.4`，中心為 `z⁰_T`（Eq. 7 的 `Π`）。
   這是**唯一的硬約束**，且作用在 latent 而非像素。

Table 3 報告的 LPIPS 0.23–0.25 與 SSIM 0.67–0.69 是**事後量測值，不是被約束的目標**。

論文更進一步在 §4.5 "Two-stage vs. One-stage Alignment" 明確**反對**加入顯式失真項：

> 「To validate the advantages of our two-stage alignment, we adapt APA into a single-stage alignment
> (APA*): replacing LoRA-based visual alignment and incorporating joint optimization in the second
> stage, i.e., `R_a = R_a − λ‖z_0 − z̄_0‖²`. Experimental results in Figure 5(b) demonstrate:
> 1) One-stage alignment (both APA* and ACA) suffer from reward hacking due to conflicting objectives
> during joint optimization (as λ increases, Avg. ASR decreases while SSIM increases).」

### 6.2 程式碼層面（與論文矛盾）

`pipe_ours.py`，**`attack_optimization_checkpoint`（即 APA-GC）** 的 reward：

```python
if use_da:
    reward = torch.nn.CrossEntropyLoss()(out, label.repeat_interleave(len(la_list))) - 10*torch.nn.MSELoss()(ori_latents, la_t)
else:
    reward = torch.nn.CrossEntropyLoss()(out, label) - 10*torch.nn.MSELoss()(ori_latents, la_t)
```

其中 `ori_latents = pipeline.enimg2latent(image_path)` = `E(x)` = `z_0`，`la_t` = 最終 `z̄_0`。

**這正是論文宣稱屬於 one-stage ablation（APA*）、會導致 reward hacking 的
`R_a = R_a − λ‖z_0 − z̄_0‖²` 形式，λ = 10，且被用在主方法 APA-GC 上。**

對照之下，**`attack_optimization`（APA-SG）的 reward 是純 cross-entropy，無此項**：

```python
if use_da:
    reward = torch.nn.CrossEntropyLoss()(out, label.repeat_interleave(len(la_list)))
else:
    reward = torch.nn.CrossEntropyLoss()(out, label)
```

**結論**：
- APA-SG 的失真約束 = 階段一 LoRA + latent L∞ 球（`ε_a = 0.4`）。與論文一致。
- APA-GC 的失真約束 = 階段一 LoRA + latent L∞ 球 + **latent MSE 懲罰項（λ = 10）**。
  **論文未載明此項**，且與論文的 one-stage 批評自相矛盾。
  由於 Table 3 中 APA-GC 的 LPIPS（0.23）與 SSIM（0.69）都優於 APA-SG（0.25／0.67），
  此項很可能正是差異來源。

> 本專案若要移植 APA 的參數化與階段一，此點不影響階段一；但若日後要引用
> 「APA 只靠 LoRA 達成視覺一致性」的說法，必須加註 APA-GC 另含 λ=10 的 latent MSE 項。

## 7. 值域

論文完全未述。由程式碼可完全確認，且**分屬三個不同值域**：

| 位置 | 值域 | 來源 |
|---|---|---|
| VAE 輸入（階段一訓練） | **`[-1,1]`**，512×512 | `visual_alignment.py:42-50`，`Normalize([0.5],[0.5])` |
| VAE 輸入（階段二反演） | **`[-1,1]`**，512×512 | `pipe_ours.py` `get_img()`：`Resize((512,512)) + ToTensor() + Normalize([0.5]*3,[0.5]*3)` |
| VAE 解碼輸出／分類器輸入／最終對抗影像 | **`[0,1]`**，224×224（`mvit` 為 320、`inception_v3` 為 299） | `pipe_ours.py` `decode_la()`：`image = image/2 + 0.5`，隨後 `torch.clamp(img, 0, 1)` |
| 資料載入（評測用） | **`[0,1]`** | `attack_alignment.py:87`：`transforms.Compose([Resize((224,224)), ToTensor()])`（無 Normalize） |
| latent | `× 0.18215`（`vae.config.scaling_factor`） | `pipe_ours.py` `image2latent()`、`visual_alignment.py:160` |

分類器的 ImageNet 標準化在 `WrapperModel` 內部完成（`utils.py:20-28`），
mean/std 依模型而異：ViT 用 `[0.5,0.5,0.5]`／`[0.5,0.5,0.5]`，MobileViT 用 `[0,0,0]`／`[1,1,1]`，
其餘用 ImageNet 的 `[0.485,0.456,0.406]`／`[0.229,0.224,0.225]`。

**注意**：`ε_a = 0.4` 是在 **latent 空間**（已乘 0.18215 的 SD latent）上的 L∞ 半徑，
與像素值域無關，不可與像素 L∞ 預算直接比較。

## 8. APA — 未找到的項目

逐條列出，不省略：

1. **Appendix 本身**：arXiv v1（唯一版本）**不含 Appendix**，但正文四處引用它。
   因此論文承諾的「More implementation details」、「stronger data transformations 的結果」、
   「ControlNet 等其他 diffusion model 的擴展」、「targeted attack／VQA／object detection 的擴展」、
   「Time analysis」——**全部未找到**。
2. **階段一的所有超參數在論文中不存在**：LoRA rank／alpha／target modules／lr／步數／
   optimizer／noise offset 皆**未在論文中出現**，僅能自程式碼取得（本文 §1.2 已完整列出）。
3. **`noise_offset = 0.1`**：論文未提及此項。其動機（crosslabs 的 offset noise）僅由
   程式碼註解的 URL 推知，論文無對應說明。
4. **skip gradient 的縮放常數 `ρ = 14.58`**：論文只寫 `ρ`，未給數值；程式碼硬編碼 14.58。
   其來源（是否引自 ACA）**未找到**論文層級的說明。
5. **APA-GC reward 中的 `−10 · MSE(z_0, z̄_0)`**：論文完全未載，且與論文對 one-stage
   的批評矛盾。λ = 10 的選取依據**未找到**。
6. **`ϱ(·)` 是否含 brightness adjustment**：論文說含，程式碼註解掉。何者為準**未找到**依據。
7. **APA-GC 的 `T = 10` 之精確語意**：論文寫「adopts `T = 10`」，未說明是「重設 10 步排程」
   還是「沿用 50 步排程只跑 11 步」。程式碼為後者，但論文文字**未找到**對應說明。
8. **`index_cond = 40` 的論文對應**：程式碼以絕對索引 40 界定 guidance 起點，
   論文只給 `T_a = 10`。當 `T ≠ 50` 時兩者如何對應**未找到**說明。
9. **文字條件 `c` 的具體形式**：論文未說明 prompt 內容；程式碼用 `data.json` 的
   `class` 欄位（ImageNet 類別名，無模板）。論文層級**未找到**。
10. **LoRA 是 per-image 還是共用**：論文未明說；程式碼為每張影像一組 `.safetensor`。
    論文層級**未找到**。
11. **階段一的訓練耗時／單張影像成本**：論文說在 Appendix，Appendix 不存在。**未找到**。
12. **Figure 5(b) 中 λ 的掃描範圍與逐點數值**：僅為折線圖，無數表。**未找到**。
13. **Figure 6(a) 中 `T` 的掃描範圍與逐點數值**：僅為折線圖；圖中僅可辨識
    Avg ASR 為 87.14／88.02／84.58／83.46／82.70 與時間倍率 ×1.7／×1.4／×1.3／×1.2，
    但**各數值對應哪個 `T`（座標軸刻度）未找到**。
14. **Figure 6(b)(c) 的 `T_a` 掃描**：可辨識 `T_a=5` LPIPS=0.24、`T_a=10` LPIPS=0.25、
    `T_a=20` LPIPS=0.3，以及 Avg ASR 69.66／75.32／88.00 與時間倍率 ×1.3／×1.5，
    但**ASR 與 `T_a` 的逐項對應未找到**明確標註。
15. **`use_lora` 為 `action='store_true'` 但 `default=True`**：此為 argparse 的常見錯誤
    （無法用旗標關閉）。是否為刻意設計**未找到**說明。
16. **評測所用的 LPIPS／SSIM 實作與計算解析度**：Table 3 的度量在 224×224 或 512×512
    上計算**未找到**說明；repo 中無評測程式碼。
17. **`data.json` 與 ImageNet-compatible Dataset 1000 張的對應關係**：repo 只提供
    5 張範例（`images_un/1..5.png`）與 `--test_sample_num` 預設 5。完整 1000 張的
    清單與標籤對應**未找到**（README 指向 Natural-Color-Fool 的 release 壓縮檔）。

---

# 附錄：兩者的關鍵設定並列（供本專案對齊用）

| 項目 | DIA | APA |
|---|---|---|
| 基礎模型 | `CompVis/stable-diffusion-v1-4` | `runwayml/stable-diffusion-v1-5` |
| 影像解析度（擴散端） | 512×512 | 512×512 |
| 像素值域（擴散端） | `[-1,1]` | `[-1,1]` |
| 最終輸出值域 | `[0,1]`（PIL） | `[0,1]`（224×224 張量） |
| 優化空間 | **像素空間** `x_0 + δ` | **latent 空間** `z_T` |
| 約束 | L∞，`ε = 0.05`（`[-1,1]` 尺度） | L∞，`ε_a = 0.4`（SD latent 尺度） |
| step size | `1/255`（`[-1,1]` 尺度） | `µ = 0.04` |
| 迭代數 | 20 | 10 |
| 更新規則 | `sign(grad)`（無 momentum） | `sign(momentum)`，`m ← m + g/‖g‖₁` |
| 精度 | float16 | float32 |
| 反演步數 | 10 | SG: 50／GC: 50 步排程取 11 步 |
| prompt | 空字串，CFG = 1 | ImageNet 類別名，CFG = 1 |
| 顯式失真項 | 無 | SG: 無；**GC: `−10·MSE(z_0, z̄_0)`（程式碼獨有）** |
| 官方程式碼 | github.com/sohn1029/DIA（僅攻擊端） | github.com/deep-kaixun/APA（兩階段皆有） |
