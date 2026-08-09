# 淨化算子原始碼查證

> 2026-08-05。對象為 `DESIGN_2026-08-05.md` §3 表列的五個淨化算子。
> **本文件的規則：只記錄可在原始碼或論文原文中直接讀到的內容。**
> 凡是查不到的參數一律標記「未找到」，不以合理推斷填補。
> 凡屬本文件作者的工程建議（非來源記載），一律加註「（建議，非來源）」。
>
> 每一節的程式碼片段皆為抓取當日（2026-08-05）自該來源取得的原文，
> 未經改寫。原始檔已下載至暫存區比對過完整內容。

## 摘要表

| # | 算子 | 官方 repo | 演算法可完整重現 | 預設超參數齊全 | 原生可微 |
|---|---|---|---|---|---|
| 1 | Adverse Cleaner | 原 repo 已刪除，鏡像存在 | ✅ | ✅ | ❌ |
| 2 | IMPRESS | `AAAAAAsuka/Impress` | ✅ | ✅（三組並存，見 §2.4） | ✅ |
| 3 | DiffPure | `NVlabs/DiffPure` | ✅ | ✅ | ✅（原始碼即以可微 SDE 求解器實作） |
| 4 | CNN 去噪（NTIRE 2023 冠軍） | **未找到** | ❌ | ❌ | 不適用 |
| 5 | Crop & Resize | 無（DIA 論文內描述） | 部分 | 部分（見 §5.3） | ✅ |

---

## 1. Adverse Cleaner

### 1.1 出處

DIA（ICCV 2025）參考文獻 [33] 的原文為：

```
[33] Lvmin Zhang. AdverseCleaner. https://github.com/lllyasviel/AdverseCleaner, 2023. 8, 3
```

（來源：DIA arXiv:2510.00778v1 PDF，參考文獻第 33 條）

**該 URL 目前為 404。** 以 GitHub API 查詢 `lllyasviel/AdverseCleaner` 回傳
`{"message": "Not Found", "status": "404"}`，原 repo 已被刪除。

現存的完整鏡像（fork 網路，GitHub 在原始 repo 刪除後將網路根節點改指
`gogodr/AdverseCleanerExtension`）：

| repo | 內容 | 授權 |
|---|---|---|
| `shidoto/AdverseCleaner` | 與原 repo 檔案結構相同（`clean.py` / `environment.yaml` / `input.png` / `output.png`），README 保留 lllyasviel 第一人稱原文 | Apache-2.0 |
| `Fawzan09/AdverseCleaner` | 同上，另加 `notebooks/` | Apache-2.0 |
| `toyxyz/AdverseCleaner` | `clean.py` 被改為批次處理資料夾 | Apache-2.0 |
| `gogodr/AdverseCleanerExtension` | A1111 webui 擴充（`scripts/denoise.py`） | Apache-2.0 |
| `p1atdev/stable-diffusion-webui-adverse-cleaner-tab` | A1111 webui tab 擴充 | 未查證 |
| HuggingFace Space `p1atdev/AdverseCleaner` | 原 README 列出的實作之一 | 未查證 |

> ⚠️ 這些 repo 的 Apache-2.0 授權是 fork 者所掛。原 repo 的授權狀態**未找到**
> （repo 已刪除，無法查證 lllyasviel 原本是否附授權檔）。

### 1.2 完整演算法（原始版本，未經修改）

來源：`https://raw.githubusercontent.com/shidoto/AdverseCleaner/main/clean.py`
（README 自稱「16 lines of Python codes」，此檔即為該 16 行）

```python
import numpy as np
import cv2

from cv2.ximgproc import guidedFilter

img = cv2.imread('input.png').astype(np.float32)
y = img.copy()

for _ in range(64):
    y = cv2.bilateralFilter(y, 5, 8, 8)

for _ in range(4):
    y = guidedFilter(img, y, 4, 16)

cv2.imwrite('output.png', y.clip(0, 255).astype(np.uint8))
```

步驟：

1. 以 `cv2.imread` 讀入（BGR、uint8），轉 `float32`，值域 **[0, 255]**。
2. 保留原圖 `img` 作為後續 guided filter 的 guide。
3. 對 `y` **連續施加 64 次** bilateral filter，參數 `d=5, sigmaColor=8, sigmaSpace=8`。
4. 對 `y` **連續施加 4 次** guided filter，guide 為 `img`（原圖，非上一步結果），
   參數 `radius=4, eps=16`。
5. `clip(0, 255)` 後轉 `uint8` 寫出。

**兩種濾波器並用，不是二選一。** 先 64 次 bilateral，再 4 次 guided。

### 1.3 預設超參數

| 參數 | 值 | 出處 |
|---|---|---|
| bilateral 重複次數 | **64** | `clean.py` `for _ in range(64)` |
| bilateral `d`（鄰域直徑） | **5** | `cv2.bilateralFilter(y, 5, 8, 8)` 第 2 引數 |
| bilateral `sigmaColor` | **8** | 同上第 3 引數 |
| bilateral `sigmaSpace` | **8** | 同上第 4 引數 |
| guided 重複次數 | **4** | `for _ in range(4)` |
| guided `radius` | **4** | `guidedFilter(img, y, 4, 16)` 第 3 引數 |
| guided `eps` | **16** | 同上第 4 引數 |
| guide 影像 | **原圖 `img`**，每次迭代都用同一張 | `guidedFilter(img, y, ...)` |

A1111 擴充 `gogodr/AdverseCleanerExtension/scripts/denoise.py` 的 UI 滑桿預設值
與上述完全一致，可作為交叉驗證：

```python
bilateral_steps = gr.Slider(minimum=1, maximum=128, step=1,
                            value=64, label="Bilateral Steps")
diameter = gr.Slider(minimum=1, maximum=30, step=1,
                     value=5, label="Diameter")
sigma_color = gr.Slider(minimum=1, maximum=30,
                        step=1, value=8, label="SigmaColor")
sigma_space = gr.Slider(minimum=1, maximum=30,
                        step=1, value=8, label="SigmaSpace")
...
guided_steps = gr.Slider(minimum=1, maximum=64, step=1,
                         value=4, label="Guided Steps")
radius = gr.Slider(minimum=1, maximum=30, step=1,
                   value=4, label="Radius")
eps = gr.Slider(minimum=1, maximum=30, step=1,
                value=16, label="Accuracy")
```

其 `process` 函式的濾波順序也相同：先 `bilateral_steps` 次 bilateral，
再 `guided_steps` 次 `guidedFilter(img, y, radius, eps)`。

### 1.4 輸入值域與影像格式假設

| 項目 | 值 | 依據 |
|---|---|---|
| 值域 | **[0, 255]**，`float32` | `cv2.imread(...).astype(np.float32)` |
| 通道順序 | **BGR** | `cv2.imread` 預設；擴充版顯式做 `cv2.COLOR_RGB2BGR` |
| 通道數 | 3 | `cv2.bilateralFilter` 只支援 1 或 3 通道 |
| 輸出 | `clip(0,255)` → `uint8` | `clean.py` 最後一行 |

`sigmaColor=8` 是在 **[0, 255] 尺度**上定義的。若把影像換算到 [0, 1] 而不同步
把 `sigmaColor` 換成 `8/255`，濾波強度會差 255 倍。

依賴：`guidedFilter` 位於 `cv2.ximgproc`，需 `opencv-contrib-python`
（`environment.yaml` 的 pip 段即列此套件，非 `opencv-python`）。

### 1.5 可微性

**不可微。** 全部運算在 OpenCV C++ 內完成，無 autograd 圖。

可用於訓練的替代方案（**建議，非來源**）：

- Guided filter 本身是 box filter 與逐元素線性運算的組合，有封閉形式，
  可在 PyTorch 中直接寫成可微版本。
- Bilateral filter 的可微近似需另尋實作，且 64 次串接的計算量不小。
- 若採直通估計（straight-through），前向用 OpenCV 真值、反向用恆等或用
  可微近似的梯度，須依 `ARCH_2026-08-05.md` §「淨化算子」的規劃量測 `proxy_gap`。

### 1.6 DIA 對此算子的設定

DIA 補充材料 B.2 對 Adverse Cleaner 只寫一句：

> � Adverse Cleaner [33]: An algorithmic approach capable of purifying high-frequency noise patterns.

**未指定任何參數**，故應理解為採用上游預設值（§1.3）。

---

## 2. IMPRESS

### 2.1 出處

- 論文：Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen.
  *IMPRESS: Evaluating the Resilience of Imperceptible Perturbations Against
  Unauthorized Data Usage in Diffusion-Based Generative AI.* NeurIPS 2023.
  arXiv:2310.19248。
- 官方 repo：**https://github.com/AAAAAAsuka/Impress**（README 自稱 official repository）。
- DiffVax（ICLR 2026）正文第 370–371 行：
  `(iii) applying the IMPRESS defense (Cao et al., 2023), denoted as DiffVax w/ IMPRESS.`

### 2.2 完整演算法（`impress.py` 全文，29 行）

```python
import torch.nn as nn
import torch
from tqdm import tqdm
import lpips


def impress(X_adv, model, eps=0.1, iters=40, clamp_min=0, clamp_max=1, lr=0.001, pur_alpha=0.5, noise=0.1):
    # init purified X
    X_p = X_adv.clone().detach()  + (torch.randn(*X_adv.shape) * noise).to(X_adv.device).half()
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    loss_fn_alex = lpips.LPIPS(net='vgg').to(X_adv.device)
    optimizer = torch.optim.Adam([X_p], lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=1e-5)
    for i in pbar:
        X_p.requires_grad_(True)
        _X_p = model(X_p).sample
        optimizer.zero_grad()
        lnorm_loss = criterion(_X_p, X_p)
        d = loss_fn_alex(X_p, X_adv)
        lpips_loss = max(d - eps, 0)
        loss = lnorm_loss + pur_alpha * lpips_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        X_p.data = torch.clamp(X_p, min=clamp_min, max=clamp_max)
        pbar.set_description(f"[Running purify]: Loss: {loss.item():.5f} | l2 dist: {lnorm_loss.item():.4} | lpips loss: {d.item():.4}")
    X_p.requires_grad_(False)
    return X_p
```

呼叫端（`glaze_pur.py` 第 52–60 行）確認 `model` 就是 **SD 的 VAE**：

```python
x_adv = preprocess(adv_image).to(device).half()
x_purified = impress(x_adv,
                     model=pipe_img2img.vae,
                     clamp_min=-1,
                     clamp_max=1,
                     eps=args.pur_eps,
                     iters=args.pur_iters,
                     lr=args.pur_lr,
                     pur_alpha=args.pur_alpha,
                     noise=args.pur_noise, )

x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
```

VAE 參數在呼叫前被凍結（`glaze_pur.py` 第 41–42 行）：

```python
for name, param in pipe_img2img.vae.named_parameters():
    param.requires_grad = False
```

`pg_mask_pur_helen.py` 的呼叫形式相同，只是 `model=pipe_inpaint.vae`。

### 2.3 損失式

論文 Eq. (4.1)（arXiv PDF 第 351 行）：

```
min ||xpur - D(E(xpur))||²₂ + α · max(LPIPS(xpur, xptb) - L, 0)
```

論文 Algorithm 1（附錄 A）：

```
Input: Image encoder E, image decoder D; hyperparameters α, σ, L.
 1: Initialize xpur = xptb + N(μ, σ²I)
 2: for iter = 1, 2, ..., N do
 3:   Lsim   = ||xpur - D(E(xpur))||²₂
 4:   Llpips = max(LPIPS(xpur, xptb) - L, 0)
 5:   L      = Lsim + Llpips
 6:   xpur  ← xpur - ∇x L
 7:   xpur  = Clip(xpur, min = -1, max = 1)
 8: end for
 9: Return xpur
```

程式與論文的對應：

| 論文符號 | 程式 | 備註 |
|---|---|---|
| `\|\|xpur - D(E(xpur))\|\|²₂` | `criterion(_X_p, X_p)`，`criterion = nn.MSELoss()` | 程式用 **MSE（取平均）**，論文寫 squared L2（取和）。差一個常數倍，等價於 α 的縮放 |
| `L`（LPIPS margin） | 函式引數 `eps` / CLI `--pur_eps` | 命名不同，語意相同 |
| `α` | `pur_alpha` | Algorithm 1 第 5 行漏寫 α，Eq. (4.1) 有 |
| `σ`（初始化雜訊） | `noise` / `--pur_noise` | |
| 梯度下降 | Adam + CosineAnnealingLR | Algorithm 1 寫的是樸素梯度下降，程式用 Adam |

值得記錄的實作細節（照抄會影響結果者）：

- LPIPS 用的是 **VGG backbone**（`lpips.LPIPS(net='vgg')`），變數名 `loss_fn_alex` 是誤導。
- `lpips_loss = max(d - eps, 0)` 用的是 Python 內建 `max`。當 `d <= eps` 時
  回傳 **Python `int` 0**，該項對梯度無貢獻；此為 hinge 的正常行為。
- `model(X_p).sample`：`diffusers.AutoencoderKL.forward` 的 `sample_posterior`
  預設為 `False`，故走 posterior 的 mode，**重建是確定性的**，不含 VAE 取樣隨機性。
- 全流程以 `.half()`（fp16）執行；Adam 的 `eps=1e-5`（非預設 1e-8）應是為配合 fp16。
- 學習率排程：`CosineAnnealingLR(optimizer, iters, eta_min=1e-5)`，
  即在 `iters` 步內從 `lr` 餘弦衰減到 `1e-5`。
- 每步 `optimizer.step()` 後才 clamp，且以 `X_p.data =` 就地寫入。

### 2.4 預設超參數：**三組並存，必須指名採用哪一組**

| 參數 | (A) `impress.py` 函式簽名 | (B) Glaze 情境 | (C) PhotoGuard 情境 |
|---|---|---|---|
| `eps` / `pur_eps`（LPIPS margin） | 0.1 | **0.1** | **0.1** |
| `iters` / `pur_iters` | 40 | **3000** | **1000** |
| `lr` / `pur_lr` | 0.001 | **0.01** | **0.005** |
| `pur_alpha`（α） | 0.5 | **0.1** | **0.01** |
| `noise` / `pur_noise`（σ） | 0.1 | **0.1** | **0.05** |
| `clamp_min` / `clamp_max` | 0 / 1 | **-1 / 1** | **-1 / 1** |

出處：

- (A)：`impress.py` 第 7 行函式簽名。**這組值在任何實驗腳本中都沒被使用**，
  因為兩個呼叫端都顯式傳入所有引數。不應採用。
- (B) Glaze 情境：README「Execute Impress」段落文字說明
  （`pur_eps` 0.1 / `pur_lr` 0.01 / `pur_iters` 3000 / `pur_alpha` 0.1 / `pur_noise` 0.1），
  與論文附錄 B 一致：
  > For our method, we set the perturbation budget to p = 0.1, regularization
  > coefficient α = 0.1, and train for 3,000 steps using the Adam optimizer with
  > a learning rate of 10⁻².

  ⚠️ `glaze_pur.py` 的 argparse 預設 `--pur_noise` 為 **`0.`**（第 95 行），
  與 README／論文寫的 0.1 **不一致**。README 與論文是實驗依據，argparse 是程式預設。
- (C) PhotoGuard 情境：README 文字說明（`pur_lr` 0.005 / `pur_iters` 1000 /
  `pur_alpha` 0.01 / `pur_noise` 0.05）、實驗腳本 `scripts/new/pg_mask_diff_test.sh`
  （`pur_eps=0.1` `pur_iters=1000` `pur_lr=0.005` `pur_alpha=0.01` `pur_noise=0.05`）、
  以及論文附錄 B 三者一致：
  > For our method, we set the perturbation budget to p = 0.1, regularization
  > coefficient α = 10⁻², and train for 1,000 steps using the Adam optimizer with
  > a learning rate of 5 × 10⁻³.

  ⚠️ `pg_mask_pur_helen.py` 的 argparse 預設（`pur_iters=100` `pur_lr=0.01`
  `pur_alpha=0.1` `pur_noise=0.1`，第 97–101 行）與上述三者**全部不同**。
  argparse 預設在此情境下也不是實驗設定。

**本專案的建議取法（建議，非來源）**：本專案的威脅模型是編輯（非風格微調），
情境對應 (C) PhotoGuard，故採 (C) 那一組。

### 2.5 使用的 SD 模型

| 情境 | 模型 | 出處 |
|---|---|---|
| Glaze | `stabilityai/stable-diffusion-2-1-base` | `glaze_pur.py` 第 75 行 argparse 預設 |
| PhotoGuard | `runwayml/stable-diffusion-inpainting` | `pg_mask_pur_helen.py` argparse 預設 |

只取用其 `.vae`，不需 UNet。

### 2.6 輸入值域與影像格式假設

`utils.py` 的 `preprocess`：

```python
def preprocess(image):
    w, h = image.size
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0
```

| 項目 | 值 |
|---|---|
| 輸入值域 | **[-1, 1]**（`clamp_min=-1, clamp_max=1`） |
| 通道順序 | RGB（PIL → numpy） |
| 形狀 | `(1, 3, H, W)`，`float16` |
| 尺寸 | `preprocess` 中的 resize 被註解掉，**不改變解析度**；但 VAE 要求邊長為 8 的倍數 |
| 輸出還原 | `(x_purified / 2 + 0.5).clamp(0, 1)` |

⚠️ LPIPS 官方要求輸入為 [-1, 1]，此處恰好相符，不需額外換算。

### 2.7 可微性

**可微。** 整個流程就是對 `X_p` 做梯度下降，本身即以 autograd 實作。
作為淨化算子放進訓練迴圈時，需對 `iters` 步（1000 步）展開反向傳播，
記憶體與時間成本高；是否需要截斷或代理，屬本專案的工程決策（**建議，非來源**）。

### 2.8 授權

**未找到。** repo 無 `LICENSE` 檔（`raw.githubusercontent.com/.../LICENSE` 回 404），
GitHub API 的 `license` 欄位為 `null`。

repo README 有「Ethical Statement」段落，聲明反對未經授權的仿作，並說明
研究去除方法的目的是促進更好的保護技術。該段落是倫理聲明，不是授權條款。
**若要在論文中使用其程式碼，需另行向作者確認授權。**

---

## 3. DiffPure

### 3.1 出處

- 論文：Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar.
  *Diffusion Models for Adversarial Purification.* ICML 2022。arXiv:2205.07460。
- 官方 repo：**https://github.com/NVlabs/DiffPure**（預設分支 `master`）。
- 專案網站：https://diffpure.github.io/

**關於「被 APA 當作防禦」**：APA = *Enhancing Diffusion-based Unrestricted Adversarial
Attacks via Adversary Preferences Alignment*（arXiv:2506.01511）。該文 §4.3 把
DiffPure [46] 列為六個「preprocessing defenses」之一（HGD、R&P、NIPS-r3、JPEG、
Bit-Red、DiffPure），Table 2 報告攻擊在其上的 ASR。
**APA 全文未給出任何 DiffPure 超參數**（`t`、擴散模型、步數皆無）。
故 DiffPure 的設定必須以 NVlabs 原始碼為準。

### 3.2 完整演算法（guided-diffusion 版本，`runners/diffpure_guided.py`）

```python
def image_editing_sample(self, img, bs_id=0, tag=None):
    with torch.no_grad():
        ...
        x0 = img
        xs = []
        for it in range(self.args.sample_step):
            e = torch.randn_like(x0)
            total_noise_levels = self.args.t
            a = (1 - self.betas).cumprod(dim=0)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

            for i in reversed(range(total_noise_levels)):
                t = torch.tensor([i] * batch_size, device=self.device)

                x = self.diffusion.p_sample(self.model, x, t,
                                            clip_denoised=True,
                                            denoised_fn=None,
                                            cond_fn=None,
                                            model_kwargs=None)["sample"]
            x0 = x
            xs.append(x0)

        return torch.cat(xs, dim=0)
```

逐步驟：

1. **前向加噪一次到底**（非逐步）：
   `x = x0 · sqrt(ᾱ_{t-1}) + e · sqrt(1 - ᾱ_{t-1})`，其中
   `a = (1 - betas).cumprod(dim=0)` 即 `ᾱ`，索引為 `total_noise_levels - 1`。
2. **逆向逐步去噪**：`for i in reversed(range(total_noise_levels))`，
   每步呼叫 `diffusion.p_sample(...)`，`clip_denoised=True`，無條件引導。
   即從 `t-1` 一路走到 `0`，共 `t` 步。
3. 外層 `for it in range(sample_step)` 重複整個「加噪→去噪」流程；
   `sample_step` 預設 **1**。回傳時是 `torch.cat(xs, dim=0)`（各輪結果串接）。

模型載入（同檔）：

```python
model_config = model_and_diffusion_defaults()
model_config.update(vars(self.config.model))
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(self.device)
```

### 3.3 SDE 版本（`runners/diffpure_sde.py`，論文主結果所用）

前向加噪與上式完全相同，逆向改為以 `torchsde` 求解 reverse-VP-SDE：

```python
e = torch.randn_like(x0).to(self.device)
total_noise_levels = self.args.t
if self.args.rand_t:
    total_noise_levels = self.args.t + np.random.randint(-self.args.t_delta, self.args.t_delta)
a = (1 - self.betas).cumprod(dim=0).to(self.device)
x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

epsilon_dt0, epsilon_dt1 = 0, 1e-5
t0, t1 = 1 - self.args.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1
t_size = 2
ts = torch.linspace(t0, t1, t_size).to(self.device)

x_ = x.view(batch_size, -1)  # (batch_size, state_size)
if self.args.use_bm:
    bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
    xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler', bm=bm)
else:
    xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler')
x0 = xs_[-1].view(x.shape)
```

注意連續時間與離散步數的換算：**`t*（論文的連續時間）= args.t / 1000`**。
`RevVPSDE` 的預設為 `beta_min=0.1, beta_max=20, N=1000`。

### 3.4 t* 值（加噪程度）

| 資料集 | 論文 `t*` | 原始碼 `--t` | 出處 |
|---|---|---|---|
| CIFAR-10（AutoAttack L∞, ε=8/255） | **0.1** | **100** | 論文 Table 1 caption：`In our method, the diffusion timestep is t = 0.1.`；`run_scripts/cifar10/run_cifar_rand_inf.sh` 的 `for t in 100` |
| ImageNet（AutoAttack L∞, ε=4/255） | **0.15** | **150** | 論文 Table 3 caption：`the diffusion timestep is t = 0.15.`；`run_scripts/imagenet/run_in_rand_inf.sh` 的 `for t in 150` |
| CelebA-HQ（BPDA+EOT） | 未在正文標明 | **500** | `run_scripts/celebahq/run_celebahq_bpda_smiling.sh` 的 `for t in 500` |
| CIFAR-10（L2, ε=0.5） | **0.075** | 未逐一核對 | 論文 Table 2 caption：`the diffusion timestep is t = 0.075.` |
| CIFAR-10（StAdv） | **0.125** | 未逐一核對 | 論文 caption：`For our method, the diffusion timestep is t = 0.125.` |

`eval_sde_adv.py` 的 argparse 預設為 `--t 400`，但**所有實驗腳本都顯式覆寫**，
故 400 不是任何一組實驗設定。

### 3.5 使用的擴散模型

論文附錄 B.1 原文：

> In experiments, we use different pre-trained models on three datasets: Score SDE
> (Song et al., 2021b) for CIFAR-10, Guided Diffusion (Dhariwal & Nichol, 2021) for
> ImageNet and DDPM (Ho et al., 2020) for CelebA-HQ. In specific, we use the
> `vp/cifar10_ddpmpp_deep_continuous` checkpoint from the score_sde library:
> https://github.com/yang-song/score_sde for the CIFAR-10 experiments. We use the
> 256x256 diffusion (unconditional) checkpoint from the guided-diffusion library:
> https://github.com/openai/guided-diffusion for the ImageNet experiments. Finally,
> for the CelebA-HQ experiments, we use the CelebA-HQ checkpoint from the SDEdit
> library: https://github.com/ermongroup/SDEdit.

| 資料集 | 檢查點 | 解析度 | 上游授權 |
|---|---|---|---|
| CIFAR-10 | `vp/cifar10_ddpmpp_deep_continuous`（score_sde） | 32×32 | `yang-song/score_sde` Apache-2.0 |
| ImageNet | `256x256_diffusion_uncond.pt`（guided-diffusion） | **256×256** | `openai/guided-diffusion` MIT |
| CelebA-HQ | `celeba_hq.ckpt`，`diffpure_ddpm.py` 中硬編 URL `https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt` | 256×256 | SDEdit（`ermongroup/SDEdit`），未逐一核對 |

⚠️ **對本專案的直接影響**：三個檢查點的最高解析度是 256×256。
本專案的 SD 影像為 512×512。DiffPure 官方並未提供 512×512 的擴散模型，
「在 512 上跑 DiffPure」必須自行選擇一個作法（下採樣→淨化→上採樣、
換一個 512 的無條件擴散模型、或改用 SD 自身的 latent 空間），
**這三種都不是 DiffPure 原始設定**，必須在論文中明記為本專案的改動。

### 3.6 其他預設超參數

| 參數 | 預設 | 出處 |
|---|---|---|
| `--sample_step` | 1 | `eval_sde_adv.py` argparse |
| `--t_delta` | 15 | 同上（僅 `--rand_t True` 時生效） |
| `--rand_t` | False | 同上 |
| `--diffusion_type` | `ddpm`（腳本覆寫為 `sde`） | 同上 |
| `--score_type` | `guided_diffusion`（CIFAR-10 腳本覆寫為 `score_sde`） | 同上 |
| `--eot_iter` | 20 | 同上 |
| `--use_bm` | False（`store_true`） | 同上 |
| SDE 求解器 | `torchsde.sdeint_adjoint`，`method='euler'` | `diffpure_sde.py` |
| SDE 步長 | `dt = 1e-3` | 論文附錄 B.1：`both SDEs with a fixed step size dt=10⁻³` |
| SDE 時間區間 | `t0 = 1 - t/1000`，`t1 = 1 - 1e-5`，`t_size = 2` | `diffpure_sde.py` |
| VP-SDE | `beta_min=0.1, beta_max=20, N=1000` | `RevVPSDE.__init__` |
| `p_sample` | `clip_denoised=True`，無 `cond_fn` | `diffpure_guided.py` |

### 3.7 輸入值域與影像格式假設

| 項目 | 值 | 依據 |
|---|---|---|
| 值域 | **[-1, 1]** | 存圖時一律寫 `tvu.save_image((x0 + 1) * 0.5, ...)` |
| 形狀 | `(B, C, H, W)`，`assert img.ndim == 4` | `image_editing_sample` |
| 解析度 | 綁定檢查點（32 或 256） | §3.5 |

### 3.8 可微性

**可微。** SDE 版本使用 `torchsde.sdeint_adjoint`，即 adjoint 法反向傳播，
記憶體不隨步數線性成長——這是 DiffPure 論文能做白盒（含 BPDA+EOT）評估的前提。

guided 版本的 `image_editing_sample` 包在 `with torch.no_grad():` 內，
**該版本不可微**；要可微須用 SDE 版本或移除該 context manager。

### 3.9 授權

**NVIDIA Source Code License for DiffPure**（repo 根目錄 `LICENSE`）。
關鍵條款原文：

> ...non-commercially. Notwithstanding the foregoing, NVIDIA and its affiliates may
> use the Work and any derivative works commercially. As used herein,
> "non-commercially" means for research or evaluation purposes only.

**可用於研究**（研究與評估用途明確被允許），不可商用。

---

## 4. CNN 去噪（NTIRE 2023 冠軍模型）

### 4.1 DiffVax 對此算子的說明

DiffVax（ICLR 2026，arXiv:2411.17957）正文第 368–369 行：

> (i) passing the image through a convolutional neural network (CNN)-based denoiser
> (Li et al., 2023a), denoted as DiffVax w/ D.;

`Li et al., 2023a` 的參考文獻條目：

> Yawei Li, Yulun Zhang, Luc Van Gool, Radu Timofte, et al. Ntire 2023 challenge on
> image denoising: Methods and results. In Proceedings of the IEEE/CVF Conference on
> Computer Vision and Pattern Recognition Workshops, 2023a.

**DiffVax 引用的是挑戰賽報告本身，而不是任何一個具體模型。**
全文未指名使用哪一隊的方法、未給權重來源、未給雜訊等級假設。
其附錄 A.3.3 只有定性描述（去噪針對高頻成分），亦無參數。

DiffVax 官方 repo `ozdentarikcan/DiffVax` 的檔案清單為
`src/diffvax/{__init__.py, attack.py, immunization/, metrics/, model.py, utils.py}` 與
`scripts/{compare_baselines.py, demo.py, download_dataset.py, evaluate.py, train.py}`，
`scripts/evaluate.py` 中 grep 不到 `denois` / `impress` / `jpeg` / `counter` 任一關鍵字。
**counter-attack 的實作未包含在公開 repo 中。**

### 4.2 NTIRE 2023 冠軍是誰

來源：Li et al., *NTIRE 2023 Challenge on Image Denoising: Methods and Results*,
CVPRW 2023，Table 1。

| Team | PSNR [dB] | SSIM | Ranking |
|---|---|---|---|
| **Apply AI** | **29.96** | **0.87** | **1** |
| SRC-B | 29.92 | 0.87 | 2 |
| MiAlgo | 29.87 | 0.86 | 3 |

冠軍方法為 **IPTV2**。報告 §4.1 原文：

> Inspired by [48, 50], we proposed an image processing transformer architecture for
> image restoration, namely IPTV2. As shown in the Fig. 1, IPTV2 is a U-shape
> encoder-decoder network as [36] with 3 times downsampling and upsampling. The basic
> module used in the IPTV2 is the spatial-channel transformer block, which helps fully
> capture both spatial interactions and channel interactions of feature maps. For
> spatial transformer, we split the feature map into small patches and get the
> self-attention map in the fixed window size for efficient computing as [34]. For the
> channel transformer, we calculate the feature similarity of different channels with
> cosine distance. The spatial transformer and channel transformer are serially
> connected in the same stage, and feature maps of the post-downsampling layers are
> concatenated with those of the post-upsampling layers. With the input of 128 × 128 × 3,
> the FLOPs and parameters number of IPTV2 is 41.16 GB and 26.03 M.
>
> During the training phase, we use the flipping, rotating, RGB channel shuffling and
> mix-up strategies to enhance the original input image, and progressively train the
> model with resolutions of [128, 192, 256, 320, 384]. The model is jointly trained with
> L1, MSE, and SOBEL loss. And we only use the DIV2K and LSDIR [27] datasets in the
> training stage. Following Restormer [50], the optimizer and the scheduler of the
> learning rate in the training stage are AdamW and 'CosineAnnealingRestartCyclicLR'.
> During the inference phase, the original high-resolution image are split into patches
> of 384 × 384. For higher performance, model ensemble is also used in the inference phase.

**注意：這是挑戰賽 factsheet 等級的描述，不足以重建模型。**
缺失的必要細節包含：各 stage 的通道數與 block 數、window size 的具體值、
三個損失項的權重、學習率數值、訓練步數、ensemble 的組成。

IPTV2 後續有獨立論文 *IPT-V2: Efficient Image Processing Transformer using
Hierarchical Attentions*（arXiv:2404.00633）。以 GitHub search API 查詢
`IPT-V2` 與 `IPTV2` 皆查無對應 repo（回傳結果全為 IPTV 串流相關的無關 repo）。
Papers with Code 標示為「No code available」。

### 4.3 官方 repo 與權重

挑戰賽官方 repo：**https://github.com/ofsoundof/NTIRE2023_Dn50**
（報告正文列出：`Code: https://github.com/ofsoundof/NTIRE2023_Dn50.`）

報告正文聲稱：

> The code of the submitted solutions and the pre-trained weights are also available in
> this repository.

**此聲明與 repo 現況不符。** 實際內容：

- `models/` 只有一個檔案：`team00_SGN.py`
- `model_zoo/` 只有一個權重：`team00_sgn.ckpt`
- `test_demo.py` 的 `select_model` 只實作 `model_id == 0`：

```python
def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # SGN test
        from models.team00_SGN import SGNDN3
        name, data_range = f"{model_id:02}_RFDN_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team00_sgn.ckpt')
        model = SGNDN3()
        ...
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")
```

`team00` 是**主辦方提供的 SGN baseline**，不是任何參賽隊伍的方法，更不是冠軍。

**結論：NTIRE 2023 冠軍（Apply AI / IPTV2）的程式碼與權重皆未公開，
無法忠實重現。**

### 4.4 輸入輸出規格與雜訊等級假設

挑戰賽層級的規格（可信，來源為 repo README，適用於所有參賽方法）：

```python
def add_noise(image, sigma=50):
    """
    image: input image, numpy array, dtype=uint8, range=[0, 255]
    sigma: default 50
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, sigma / 255, image.shape)
    gauss_noise = image + noise
    return gauss_noise * 255
```

| 項目 | 值 | 依據 |
|---|---|---|
| 雜訊模型 | **加性高斯白雜訊（AWGN）**，零均值 | 上述 `add_noise` |
| 雜訊等級 | **σ = 50**（[0,255] 尺度），固定非盲 | 報告 Table 1 標題 `NTIRE2023 image denoising (σ = 50)` |
| 輸入格式 | `uint8`，`[0, 255]`，RGB | `add_noise` docstring |
| 網路輸入值域 | **每個模型不同，需逐一手動確認** | `select_model` 註解：`Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.` baseline 為 `data_range = 1.0` |
| 尺寸限制 | 寬高需為 8 的倍數（驗證集已如此裁切） | README 的 `crop_image(image, s=8)` |
| 訓練資料 | DIV2K（800）+ LSDIR（84,991） | 報告 §2.1 |
| 測試資料 | DIV2K test 100 + LSDIR test 100 | 報告 Table 1 caption |

⚠️ **σ = 50 這個假設與本專案的用途嚴重不匹配。** 該模型被訓練成移除
σ=50 的 AWGN；防禦擾動（ℓ∞ = 16/255 ≈ σ 尺度上的 16）既非高斯亦非該強度。
把它當淨化算子用，等於在訓練分佈之外操作。DiffVax 沒有說明它如何處理這一點。

### 4.5 可微性

不適用（模型不可得）。若改用 baseline SGN 或其他公開去噪網路，
則為純 PyTorch 前饋網路，**可微**。

### 4.6 授權

`ofsoundof/NTIRE2023_Dn50` 的 GitHub API `license` 欄位為 `null`，
repo 無 `LICENSE` 檔。**未找到授權聲明。**

### 4.7 可行的替代方案（建議，非來源）

既然冠軍模型不可得，選項為：

1. 用挑戰賽 baseline `team00_SGN`（權重公開，但授權未明）。
2. 用第 2–4 名所基於的公開架構（SRC-B 基於 Restormer；HIT-IIL 基於 NAFNet，
   報告 §4.4 給了完整訓練設定：AdamW β₁=0.9 β₂=0.9、125K iterations、
   lr 3×10⁻⁴ 餘弦衰減至 1×10⁻⁷、batch 64、patch 256×256、PSNR loss、
   通道數為 NAFNet 的兩倍）——但這些是**該隊的方法，不是冠軍**。
3. 用任一公開的通用去噪網路，並在論文中明記「NTIRE 2023 冠軍模型不可得，
   本文以 X 代替」。

三者都必須在論文中明寫替代事實，不可寫成「NTIRE 2023 冠軍模型」。

---

## 5. Crop & Resize

### 5.1 出處與 DIA 的確切設定

DIA 補充材料 §B.2 原文（arXiv:2510.00778v1 PDF）：

> In this section, we compare the robustness of different methods against cleaning
> approaches known as 'purification' for adversarial noise. We provide performance
> measurements after applying JPEG Compression, Crop & Resize, and AdverseCleaner to
> 700 immunized images across all methods. Details for each purification method are as
> follows:
>
> • JPEG Compression: The simplest and fastest image compression algorithm for purifying
>   adversarial noise. Compression quality can be selected between 0 and 100, where lower
>   values cause more image degradation. We provide results with quality values of 70, 80, and 90.
>
> • **Crop & Resize: A naturally occurring and effective purification technique. We cropped
>   10% of each image and then resized it to match the model's input requirements.**
>
> • Adverse Cleaner [33]: An algorithmic approach capable of purifying high-frequency
>   noise patterns.
>
> • Gaussian Noising: The purification method that adds random Gaussian noise on immunized
>   images. We provide results with σ=0.1.
>
> • Noisy Upscaling [8]: A two-stage purification method proposed by Shan et al. [23],
>   which applies Gaussian Noising (σ=0.1) followed by Stable Diffusion Upscaler [20].

**DIA 有給裁切比例：10%。** 評測資料為 PIE-Bench 的 700 張影像
（Table 6 caption：`evaluated on 700 images from the PIE-Bench dataset`）。

> ⚠️ 更正一項網路搜尋常見的錯誤說法：有二手來源稱 DIA 的 Crop & Resize 是
> 「crop 20%」。**DIA 原文寫的是 10%**，以原文為準。

### 5.2 完整演算法步驟

1. 對輸入影像裁切 10%。
2. 將裁切結果縮放回模型要求的輸入尺寸（DIA 用 SD，即 512×512）。

### 5.3 未在 DIA 中指定的參數

以下四項 DIA 全文（正文 + 補充材料）皆**未說明**：

| 未指定項目 | 可能的取法 |
|---|---|
| 「10%」是**邊長**的 10% 還是**面積**的 10% | 512 → 460（邊長）或 512 → 486（面積） |
| **中心裁切**還是**隨機裁切** | — |
| **縮放的插值方法**（bilinear / bicubic / lanczos / nearest） | — |
| 是否有 antialias | — |

DIA 未公開 repo（見 `SOURCE_AUDIT_2026-08-05.md` §狀態表第 3 列），無從以程式碼補齊。

### 5.4 對抗淨化文獻中的常見設定（**注意：這不是 DIA 原設定**）

此算子的正典來源為 Guo et al., *Countering Adversarial Images using Input
Transformations*, ICLR 2018（arXiv:1711.00117）。該文 §5.1 原文：

> In the cropping defense, we sample 30 crops of size 90×90 from the 224×224 input
> image, rescale the crops to 224×224, and average the model predictions over all crops.

即：224→90 的**隨機**裁切（面積僅約 16%，遠比 DIA 的 10% 激烈），
取 **30 個** crop，並**對模型預測取平均**。

⚠️ **這個形式不能直接搬到本專案。** Guo et al. 的「平均 30 個 crop 的預測」
是分類器專用的集成防禦；本專案的下游是影像編輯，沒有可平均的 logits。
DIA 的用法（單次裁切 + 縮放回原尺寸 → 送進編輯管線）才是本專案要的形式。

**因此建議（建議，非來源）**：以 DIA 的 10% 為主設定，並把
§5.3 的四個未指定項固定為一組明寫在論文的值，作為本專案的實作選擇說明。

### 5.5 輸入值域與影像格式假設

**DIA 未說明。** 裁切與縮放對值域不敏感（[0,1] 或 [0,255] 皆可），
但**插值方法**與**是否在 uint8 上做**會影響結果，屬 §5.3 的未指定項。

### 5.6 可微性

**可微。** 裁切是索引切片，縮放可用 `torch.nn.functional.interpolate`
（`bilinear` / `bicubic` 皆可微）。若採隨機裁切，隨機位置本身不可微但可視為
資料相依的常數（如 EOT 的處理方式）。

### 5.7 授權

不適用（無程式碼，為基本影像運算）。

---

## 6. 未找到的項目（逐條）

以下每一條都是**確認查不到**，不是尚未查。

### 6.1 Adverse Cleaner

1. **原始 repo `lllyasviel/AdverseCleaner` 的授權**。repo 已刪除（GitHub API 回 404），
   無法確認原作者是否附授權檔。現存 fork 掛 Apache-2.0，但那是 fork 者所掛。
2. **原始 repo 的建立日期與最後提交**。已刪除，無法查證。
3. `p1atdev/stable-diffusion-webui-adverse-cleaner-tab` 的授權（未查證，
   該 repo 的 `scripts/adverse_cleaner_tab.py` 路徑回 404，實際檔名為 `scripts/main.py`，
   本次未展開）。
4. HuggingFace Space `p1atdev/AdverseCleaner` 的授權（未查證）。

### 6.2 IMPRESS

5. **授權**。repo 無 `LICENSE` 檔，GitHub API `license` 為 `null`。
   引用其程式碼需另行確認。
6. **論文 Algorithm 1 中 `σ`（初始化雜訊標準差）的正文數值**。論文附錄 B
   只寫了 p、α、步數、學習率，**沒寫 σ**。σ 的值只能從 README / 腳本取得
   （Glaze 情境 0.1、PhotoGuard 情境 0.05）。
7. **`glaze_pur.py` 的 `--pur_noise` 為何 argparse 預設是 `0.` 而 README／論文寫 0.1**。
   來源本身不一致，無從判定何者為實驗真值；本文件採 README／論文。
8. **IMPRESS 在 DiffVax 實驗中所用的具體超參數**。DiffVax 只寫
   `applying the IMPRESS defense (Cao et al., 2023)`，未列參數，其 repo 亦無此程式碼。

### 6.3 DiffPure

9. **APA（arXiv:2506.01511）所用的 DiffPure 設定**。APA §4.3 只列 DiffPure 為
   preprocessing defense 之一，全文無 `t`、無擴散模型、無步數。
10. **512×512 解析度的官方設定**。DiffPure 官方三個檢查點最高 256×256，
    無 512 的設定可依循。
11. CelebA-HQ 情境的 `t*` 在論文正文中的對應數值（僅原始碼有 `--t 500`）。

### 6.4 CNN 去噪（NTIRE 2023 冠軍）

12. **冠軍方法 IPTV2 的官方 repo**。GitHub search 查無；Papers with Code 標示
    「No code available」。**未找到。**
13. **IPTV2 的預訓練權重**。`ofsoundof/NTIRE2023_Dn50` 的 `model_zoo/` 只有
    主辦方 baseline `team00_sgn.ckpt`。**未找到。**
14. **IPTV2 的完整架構超參數**（各 stage 通道數／block 數、window size、
    三個損失項的權重、學習率、訓練步數、ensemble 組成）。挑戰賽報告只有
    factsheet 等級描述，不足以重建。
15. **DiffVax 所用「CNN-based denoiser」的具體身分**。DiffVax 只引挑戰賽報告，
    未指名模型。其公開 repo 無該程式碼。
16. `ofsoundof/NTIRE2023_Dn50` 的授權（無 `LICENSE`，API `license` 為 `null`）。
17. 該去噪器被套用在防禦擾動上時的雜訊等級假設（DiffVax 未說明；
    模型本身的訓練假設是 σ=50 AWGN，與防禦擾動不匹配）。

### 6.5 Crop & Resize

18. **DIA 的「10%」是邊長還是面積**。
19. **DIA 用中心裁切還是隨機裁切**。
20. **DIA 的縮放插值方法**。
21. **DIA 的官方 repo**（`SOURCE_AUDIT_2026-08-05.md` 已記為未找到，本次未推翻）。

---

## 7. 對 `DESIGN_2026-08-05.md` §3 淨化算子表的直接影響

| 原表列 | 需修正之處 |
|---|---|
| Crop & Resize「依 DIA 設定」 | 可執行：**裁切 10%**。但四個細節未指定（§5.3），需自行定案並在論文明記 |
| Adverse Cleaner「預設」 | 可執行：§1.3 的七個參數齊全，且 DIA 未覆寫 |
| CNN 去噪「NTIRE 2023 模型」 | **不可執行**：冠軍模型程式碼與權重皆未公開。此列必須改寫為替代方案並明記 |
| IMPRESS「預設」 | 可執行，但「預設」有三組（§2.4）。需指名採用 PhotoGuard 情境那一組 |
| DiffPure「預設」 | 參數可執行（t=100/150/500），但**檢查點最高 256×256**，512 的用法是本專案的改動，需明記 |
