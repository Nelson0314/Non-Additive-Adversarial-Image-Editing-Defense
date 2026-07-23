# SPEC.md — 非加性 editing 防禦：技術規格（v2）

## §0 用途與修訂說明

本文件為實作依據。實作任一模組前先讀對應章節；文件未載明處先詢問，勿臆測或自行上網找替代方案。

本版為 v1 經逐篇核對原始論文後的修正版，共修正 25 處。標記說明：

- `[原文確認]` 已對照論文原文
- `[待確認]` 無法從公開資料解決，須靠原始碼或指導者
- `[本專案設計]` 非原論文內容，為本專案自行加入

---

## §1 共用定義

### 1.1 任務

保護方擁有影像 `x`，欲使攻擊方無法以現成 Stable Diffusion 配合文字指令對其進行有意義的編輯。攻擊發生於**推論階段**（training-free），攻擊方不訓練任何模型。

### 1.2 加性與非加性

| | 加性 | 非加性 |
|---|---|---|
| 形式 | `x' = x + δ`，`‖δ‖ ≤ κ` | `x' = G(x)`，由生成過程產出 |
| 相似性約束 | pixel norm | perceptual 度量（LPIPS 上限）`[待確認]` 是否可接受 |
| 與原圖差異 | 幾乎不可見 | 允許可見差異，語意須一致 |

### 1.3 統一介面（`src/protect/base.py`）

```python
class ProtectionMethod(ABC):
    @abstractmethod
    def protect(
        self,
        image: torch.Tensor,   # (1,3,H,W)，值域 [0,1]
        concept: str,          # 欲保護之語意概念，如 "dog"
        config: dict,
    ) -> torch.Tensor:         # (1,3,H,W)，值域 [0,1]
        ...

    @property
    @abstractmethod
    def is_additive(self) -> bool: ...
```

### 1.4 裝置與資源

GPU 廠牌不確定，可能非 NVIDIA。全專案禁用 `.cuda()`，一律經 `src/utils/device.py`。不得依賴 xformers。記憶體上限以 **V100 32GB** 為準，所有方法 `batch_size = 1`。

---

## §2 評測協定 `[原文確認：DAYN §4]`

### 2.1 模型

- 生成保護時攻擊之模型：**Stable Diffusion V1.4**
- 評測模型：**V1.4 與 V2.0**

註：PhotoGuard 原文自身使用 SD **v1.5**（`runwayml/stable-diffusion-v1-5`）。本專案以 DAYN 為主 baseline，統一用 v1.4，不採 PhotoGuard 的版本。

### 2.2 編輯階段生成超參數 `[原文確認：PhotoGuard Table 8]`

DAYN 未給出，沿用 PhotoGuard：

```
height 512, width 512, guidance_scale 7.5, num_inference_steps 100, eta 1
```

### 2.3 Seed 協定 `[原文確認：PhotoGuard Appendix A.1]`

**先在原圖上搜尋一個能產生合理編輯結果的 random seed，再用同一 seed 編輯受保護影像。** 目的為確保兩者編輯條件完全相同，使差異可歸因於保護本身。此為實驗公平性的必要條件，不可省略。

DAYN 另在此基礎上以 **20 個 random seed** 平均。

### 2.4 測試資料 `[原文確認 + 待確認]`

DAYN 未公開資料集。已知構造：**150 張影像、3 類物件、每類 2 個編輯 prompt**（對應兩種惡意情境：改變特定內容、保留特定內容而改動其他區域），每結果以 20 seed 平均。

`[待確認]` 三類物件為何、prompt 內容、生成參數，均在 supplementary。**本專案規劃向 DAYN 作者索取原始資料集**；取得前先以合理設定實作並在 config 中明確標記。

另實作載入真實資料集（如 COCO 子集）之介面作為補充。

### 2.5 評測流程

```
對每張測試影像 x：
  1. protected = method.protect(x, concept)
  2. edited_orig      = edit(x,         prompt, seed)   # 參照
  3. edited_protected = edit(protected, prompt, seed)   # 同一 seed
  4. metrics(edited_protected, edited_orig)
  5. 20 seed 平均
```

指標衡量的是**兩個編輯結果之間**的差異，差異越大代表保護越成功。

### 2.6 指標與方向 `[原文確認：DAYN Table 1]`

| 指標 | 方向 | 說明 |
|---|---|---|
| PSNR | ↓ 越低越好 | |
| SSIM | ↓ 越低越好 | |
| VIFp | ↓ 越低越好 | |
| FSIM | ↓ 越低越好 | |
| LPIPS | ↑ 越高越好 | 距離型 |
| FID | ↑ 越高越好 | 補充 |
| CLIP score | ↓ 越低越好 | 補充，與惡意 prompt 的對齊程度 |

**箭頭慣例的陷阱**：DAYN 的箭頭表示「對免疫有利的方向」；PhotoGuard Table 6 的箭頭表示「影像相似度增加的方向」，兩者相反。**本專案一律採 DAYN 慣例**，並在程式碼與報表中明確註記。

**指標實作一律使用 piq**（`github.com/photosynthesis-team/piq`）`[原文確認：PhotoGuard §4.2 註 8]`。DAYN 指標組合與 PhotoGuard 相同，用同一套件方可比。APA 用的是 IQA-PyTorch，不同套件，勿混用。

### 2.7 DAYN Table 1 引用數值與校準錨點 `[原文確認]`

**DAYN 不在本專案中實作或執行，其數值直接引用作為 baseline。**

下表為 DAYN 論文報告值。其中 **Encoder 與 Diffusion 兩欄為本專案的校準錨點**：本專案實作並執行 PhotoGuard，所得數字須與此二欄比對，以驗證本專案的設定與實作是否正確。DAYN 欄僅供引用比較，不重現。

| | SD V1.4 ||| SD V2.0 |||
|---|---|---|---|---|---|---|
| 指標 | Encoder | Diffusion | DAYN | Encoder | Diffusion | DAYN |
| PSNR ↓ | 18.8437 | 18.2617 | **15.1487** | 18.5955 | 19.3797 | **18.0589** |
| SSIM ↓ | 0.6318 | 0.6504 | **0.4470** | 0.6045 | 0.6440 | **0.4719** |
| VIFp ↓ | 0.2118 | 0.2656 | **0.1462** | 0.1618 | 0.1832 | **0.1008** |
| FSIM ↓ | 0.7757 | 0.7693 | **0.6584** | 0.7453 | 0.7794 | **0.7313** |
| LPIPS ↑ | 0.4131 | 0.4056 | **0.5901** | 0.5799 | 0.4869 | **0.6019** |

**校準判準**：取得原始資料集後，本專案的 PhotoGuard 數值應與表中 Encoder、Diffusion 兩欄接近。若差距顯著，代表編輯設定（尤其 SDEdit strength）或實作有誤，須先解決再進行後續比較。

### 2.8 編輯方法

**SDEdit（img2img）** `[原文確認：Meng et al., ICLR 2022, arXiv 2108.01073]`：對輸入加噪至 t₀ 後依 prompt 去噪。Stable Diffusion 的 img2img 即 SDEdit 之實作，官方 repo 明確列出此對應。核心參數為加噪程度 t₀，在 diffusers 中即 `strength`。

`[待確認]` **DAYN 未指明 strength 值**。此參數直接決定編輯強度並影響 Table 1 全部數字，須向作者索取。

**Inpainting**：給定 mask，僅重繪 mask 區域。

---

## §3 加性方法規格

### 3.1 參數設定的兩套來源（勿混用）

| 項目 | DAYN 重現用 `[原文確認：DAYN §4.1]` | PhotoGuard 原設定 `[原文確認：PG Table 9]` |
|---|---|---|
| Norm | ℓ∞ | ℓ∞ |
| κ / ε | 0.06 | 16/255 ≈ 0.0627 |
| step size | 未給（建議 κ/10） | 2/255 |
| 步數 N | 100（所有方法一致） | 200 |
| T（diffusion timesteps） | 10 | 未給 |

**本專案以對齊 DAYN Table 1 為目標，一律採左欄設定。**

**v3 補充**：上表的 Norm 與 ε 兩列均有未定之處，見 §3.2.1（值域尺度）與 §3.2.2（範數類型）。兩者皆列為 config 參數，於 T2 校準階段以實驗決定。

### 3.2 PhotoGuard — encoder attack `[原文確認：PG 式 4、Algorithm 1；官方程式碼核對 v3]`

論文式 (4) 為通式：

```
δ_encoder = arg min_{‖δ‖≤ε} ‖ ℰ(x + δ) − z_targ ‖²₂
```

**官方實作核對結果（v3 修正）**：官方 demo 實際最小化 `‖ℰ(x')‖`，即 **z_targ = 0（零向量）**，為論文通式的特例。v2 偽碼寫的 `z_targ = ℰ(x_gray)` 與官方不符——灰階影像的 latent 並非零向量，兩者是不同目標。

**PhotoGuard 為本專案校準錨點，凡官方與論文偽碼衝突處一律以官方為準。**

```python
def photoguard_encoder(x, vae, eps, n_iter=100, step=None,
                       target_latent="zeros", norm="linf"):
    """encoder attack。

    參數依 configs/additive.yaml：
      target_latent: "zeros"（官方預設）| "gray"（論文文字描述）
      norm:          "linf" | "l2"（官方 demo 預設 L2，ℓ∞ 僅於註解提供）
      eps:           其值域尺度由 epsilon_scale 決定，見 §3.1.1
    """
    step = step or eps / 10

    if target_latent == "zeros":
        z_targ = torch.zeros_like(vae.encode(x).latent_dist.mean).detach()
    else:  # "gray"
        z_targ = vae.encode(torch.full_like(x, 0.5)).latent_dist.mean.detach()

    delta = init_random(x, eps, norm)      # 官方有隨機初始化，v2 偽碼無
    x_im = x.clone()

    for i in range(n_iter):
        step_i = step * (1 - i / n_iter)   # 官方有 step size 線性衰減，v2 無

        x_im_ = x_im.detach().requires_grad_(True)
        z = vae.encode(x_im_).latent_dist.mean
        loss = F.mse_loss(z, z_targ)
        grad = torch.autograd.grad(loss, x_im_)[0]

        delta = project(delta + step_i * grad_direction(grad, norm), eps, norm)
        x_im = clamp_to_range(x - delta)   # 從原圖重建，見下方註
    return x_im
```

**註（投影方式）**：PG Algorithm 1 第 9 行字面為 `x_im ← x_im − δ`，若 δ 跨迭代累積則總偏移將超過 ε。標準 PGD 應從**原圖**重建。以官方 notebook 之實際實作為準。

### 3.2.1 值域尺度問題 `[待確認 — 最高優先]`

官方 notebook 在 **[-1,1]** 像素空間運作，其 demo 的 `eps=0.06` 換算至 [0,1] 尺度僅為 **0.03**。DAYN 論文只寫 `κ=0.06`，**未指明尺度**。

此差異使擾動強度相差兩倍。若判斷錯誤，PhotoGuard 數字必然無法對上 DAYN Table 1，校準錨點即失效。

**處理方式**：新增 config 參數 `epsilon_scale: "pm1" | "01"`，兩種尺度均須可執行。**不事先假設何者正確**——於 TWCC 的 T2 校準階段各跑一次，以能對上 DAYN Table 1 者為準。此舉將未知數轉為可由實驗回答的問題。

### 3.2.2 範數類型 `[待確認]`

三方不一致：v2 SPEC 假設 ℓ∞；官方 demo 預設 **L2（eps=16）**，ℓ∞ 版僅以註解提供；DAYN §4.1 稱 ℓ1/ℓ2/ℓ∞ 皆可但未指明實驗所用。

**處理方式**：新增 config 參數 `norm: "linf" | "l2"`，同樣於 T2 校準階段以實驗決定。

### 3.3 PhotoGuard — diffusion attack `[原文確認：PG 式 5、Algorithm 2]`

```
δ_diffusion = arg min_{‖δ‖∞≤ε} ‖ f(x + δ) − x_targ ‖²₂
```

`f` 為 LDM，`x_targ` 為灰階或隨機噪聲影像。

**記憶體處理** `[原文確認：PG §3 註 7]`：原文明言在 A100 40GB 上完整反傳仍爆記憶體，故**僅反傳穿過少數幾步**而非完整過程。本專案取 T = 10（對齊 DAYN），必要時用 gradient checkpointing。

**EOT（v3 新增，官方程式碼核對）**：官方 diffusion attack 的梯度以 **10 次平均**（Expectation over Transformation）取得，用以對抗 SD 採樣過程的隨機性、穩定梯度方向。v2 SPEC 完全未提此項，須補上。

此設計亦為 diffusion attack 成本約為 encoder attack 十倍的主因，須納入成本估算。

結構同 3.2，兩處差異：
1. `loss` 換為 `F.mse_loss(sd.img2img_differentiable(x_im_, prompt, num_steps=T), x_targ)`
2. 每次迭代的梯度為 10 次前傳的平均：

```python
grad = torch.zeros_like(x_im_)
for _ in range(10):                    # EOT，官方設定
    out = sd.img2img_differentiable(x_im_, prompt, num_steps=T)
    loss = F.mse_loss(out, x_targ)
    grad += torch.autograd.grad(loss, x_im_)[0]
grad /= 10
```

### 3.4 DAYN（引用 baseline，不實作）

DAYN（Lo et al., CVPR 2024）為 editing 情境的主要對照方法，**本專案不實作、不執行**，其 Table 1 數值直接引用（見 §2.7）。

其機制簡述供理解對照關係：以 cross-attention 的注意力抑制損失為目標（式 2–5），配合 TUGU（Algorithm 1）將完整擴散過程拆解為獨立 timestep 以降低記憶體。屬 additive，κ=0.06、N=100、T=10。

其式 (3) 的跨層聚合為「先 bicubic 上採樣、再逐像素相加」，式 (5) 才對聚合結果取一次 ℓ1-norm；式 (4) 的 mask 由**原始影像**的聚合注意力圖以 threshold τ 二值化取得。

註：非加性方法的 reward 方案一（§4.2）借用其注意力抑制的思路，但為本專案自行實作於 editing pipeline 上，與 DAYN 的完整方法不同，不應視為 DAYN 的重現。

---

## §4 非加性方法規格

### 4.1 前置：DDIM inversion `[原文確認：APA 式 4]`

```
z_t = √ᾱ_t · (z_{t-1} − √(1−ᾱ_{t-1})·ε_θ) / √ᾱ_{t-1} + √(1−ᾱ_t)·ε_θ
```

由參考影像的 `z_0` 迭代 T 次得 `z_T`，保留原圖資訊、去噪可近似重建。

**與淨化的區別**：DiffPure 加**隨機**高斯噪聲、目的為**遺忘**原圖、不可逆；DDIM inversion 為**確定性**反推、目的為**保存**原圖、可逆。方向相反。

### 4.2 關鍵轉換：victim 改為 editing pipeline `[本專案設計]`

AdvDiff 與 APA 原始 victim 為分類器，reward 為 `∇log p_f(y_a|·)`（targeted，需目標標籤）。**editing 情境不存在「目標標籤」**，故須改用 untargeted 形式的類比。

AdvDiff 附錄 H 的 untargeted 版本為 `−∇_{x_{t-1}} log p_f(y|x_{t-1})`（y 為真實標籤），可作為形式參考。

**兩個 reward 方案，先實作方案一**：

**方案一（優先）**：沿用注意力抑制損失作為 reward。梯度僅穿過單步 UNet，計算圖淺、記憶體可控。

**方案二**：`R = ‖edit(x', prompt) − edit(x, prompt)‖`。更直接但計算圖深，須配 gradient checkpointing 並縮減步數。

**已知風險**：editing pipeline 為 training-free 推論、參數凍結、計算圖較分類器深（AntiPure ICCV'25 §4 已分析此類困難）。實作順序**必須**為：先以單張影像、極少步數驗證梯度可算且方向正確，再逐步放大。

### 4.3 AdvDiff-based `[原文確認：AdvDiff 式 9、11、Algorithm 1/2、附錄 E]`

**兩個注入點，DDPM 與 DDIM 形式不同**：

| | DDPM `[Alg.1 line 8, 12]` | DDIM `[Alg.2 line 6, 11]` |
|---|---|---|
| 每步注入 | `x*_{t-1} = x_{t-1} + σ²_t·s·∇ log p_f(y_a\|x_{t-1})`（後置加法） | `ε̂_t = ε̃_t − √(1−ᾱ_t)·∇ log p_f(y_a\|x_t)`（改 epsilon） |
| 起始噪聲注入 | `x_T = x_T + σ̄²_T·a·∇_{x_0} log p_f(y_a\|x_0)` | `x_T = x_T + a·∇_{x_0} log p_f(y_a\|x_0)`（**無係數**） |

**本專案在 LDM 上實作，採 DDIM 形式。** 註：Alg.2 第 6 行原文未帶 `s`，附錄式 25 說明可將常數 C 替換為 guidance scale，故實作時 `s` 為可調參數。

**guidance 施加區間** `[原文確認：AdvDiff 附錄 E]`：原文 §4.5 消融指出反向過程早期（高 t）影像仍為噪聲、目標模型無法判斷，guidance 無效。ImageNet 設定為僅在 **(0, 0.2]** 區間施加（MNIST 為 (0, 0.5]）。

**原文參數**（ImageNet / LDM+DDIM）：`N=5, s=0.7, a=0.5`，採樣 200 步。原文自承對 s、a 敏感（s=10 產生噪聲紋理、a=10 產生噪聲影像）。

```python
def advdiff_based_protect(x, sd, concept, config):
    prompt_emb = sd.encode_text("")
    z_0 = sd.vae.encode(x).latent_dist.mean
    z_T = ddim_inversion(z_0, sd, prompt_emb, config.T)   # [本專案設計] 原文由隨機噪聲出發
    z_T_orig = z_T.clone()

    for _ in range(config.N):
        z = z_T.clone()
        for t in descending_timesteps(config.T):
            eps = sd.unet(z, t, prompt_emb)

            if in_guidance_range(t, config.t_range):       # 附錄 E：(0, 0.2]
                z_g = z.detach().requires_grad_(True)
                R = protect_reward(z_g, t, sd, concept)    # §4.2 方案一
                g = torch.autograd.grad(R, z_g)[0]
                eps = eps - config.s * sqrt(1 - alpha_bar(t)) * g   # Alg.2 line 6

            z = ddim_step(z, eps, t)

        x_prot = sd.vae.decode(z)
        R_final = protect_reward_on_image(x_prot, sd, concept)
        g_T = torch.autograd.grad(R_final, z_T)[0]
        z_T = z_T + config.a * g_T                          # Alg.2 line 11，無係數

        # [本專案設計]：原文對 x_T 無投影約束；保護任務須貼近原圖故加入
        z_T = project_to_ball(z_T, z_T_orig, config.eps_latent)

    return clamp01(x_prot)
```

### 4.4 APA-based `[原文確認：APA 式 5–12、§4.1]`

**Stage 1（式 6）**：`R_s(∆θ) = E_{t,ε}[ −‖ε − ε_{θ+∆θ}(z_t,t,c)‖² ]`，更新 `∆θ = ∆θ + α∇_{∆θ}R_s`。

**Stage 2**：

- 式 (7) trajectory-level：`g_tr = ∇_{z_T}R_a`；`m^i = m^{i-1} + g_tr/‖g_tr‖₁`；`z_T = Π_{z⁰_T+ε_a}(z_T + µ·sgn(m^i))`
- 式 (8) step-level：`ε = ε − √(1−ᾱ_t)·∇_{z_t}R_a`，**原文無 s 係數**；且式 (11) 之後明確以 **`sgn(m^t_st)`** 取代該梯度項
- 式 (9)：`z^t_0 = (z_t − √(1−ᾱ_t)ε)/√ᾱ_t`
- 式 (10)：`z^t_in = √(1−ᾱ_t)·z_0 + (1−√(1−ᾱ_t))·z^t_0`
- 式 (11)：`g_st = ∇_{z_t}R_a(f(x^t_in))`；`m^t_st = m^{t+1}_st + g_st/‖g_st‖₁`
- 式 (12)：`x^t_0 = ϱ((D(z^t_0) + D(z̄_0))/2)`；`g_tr = ∇_{z_T}(1/T)Σ_t R_a(f(x^t_0))`

**參數** `[原文確認：APA §4.1]`：`T_a=10, N=10, ε_a=0.4, µ=0.04`；**APA-SG 用 T=50（完整 inversion），APA-GC 用 T=10**。基於 SD V1.5。attack guidance 僅在最後 T_a 步施加。

`[待確認]` LoRA rank，主文未給（在附錄）。v1 誤植為 4，該值實為 AntiPure 之設定。

### 4.5 Hybrid `[本專案設計]`

以 APA 為骨架（inversion + 兩階段解耦），Stage 2 的注入改用 AdvDiff 的兩注入點形式。假設兩者優勢互補，實驗須驗證是否確實優於單獨任一者；若未優於，如實報告。

---

## §5 淨化方法規格

**DiffPure 已移出獨立實驗條件，列為保留項目。** 但其機制仍存在於 GrIDPure 內部（每個 grid 以小步 SDEdit 淨化即 DiffPure），故不刪除說明。

### 5.1 AdverseCleaner（= BF+GF）`[原文確認]`

**v1 錯誤修正**：AdverseCleaner 與 Fragile by Design 的「BF+GF」是**同一個方法**，v1 誤列為兩項。

由 **cv2 bilateral filter + guided filter** 兩步組成，作者 lllyasviel（Zhang, 2023），原 repo `lllyasviel/AdverseCleaner`（現僅見 fork）。全長 16 行、不需 GPU、1024px 約 3 秒 CPU。需 `opencv-contrib-python`（`cv2.ximgproc.guidedFilter`）。

```python
def adverse_cleaner(x, n_bf_iter=3):
    y = x
    for _ in range(n_bf_iter):
        y = cv2.bilateralFilter(y, d, sigma_color, sigma_space)
    return cv2.ximgproc.guidedFilter(guide=x, src=y, radius=r, eps=eps)
```

**作者自述之疑慮**：以受保護影像本身作 guidance 並不安全，guidance 中已含對抗噪聲，guided filter 可能把噪聲帶回。故**實作兩個變體**：僅 bilateral filter、以及完整 BF+GF，以觀察 guided filter 是否削弱淨化效果。

### 5.2 輕量淨化 `[原文確認：Pixel is a Barrier 附錄 D；DiffusionGuard 附錄 F]`

```python
def jpeg_compress(x, quality=65):     # 標準設定 quality 65
def gaussian_blur(x, sigma):          # sigma ∈ {0.5, 1.0, 1.5}
def crop_resize(x, ratio=0.2):        # 中心裁切 20% 後放大回原尺寸
```

### 5.3 GrIDPure `[原文確認：Zhao et al., CVPR 2024, §5.2, Fig. 9]`

正確引用：Zhao, Duan, Xu, Wang, Zhang, Du, Guo, Hu. "Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable Diffusion?" CVPR 2024, pp. 24398–24407. arXiv 2312.00084. Code: `github.com/ZhengyueZhao/GrIDPure`。全名 Grid Iterative Diffusion-based Purification。

**四步驟**：

1. 高解析影像切成多個 grid，確保每一部分**至少與兩個 grid 重疊**
2. 每個 grid 以小步 SDEdit（無條件 diffusion model）淨化
3. 合併回原解析度，重疊區**取平均**
4. 與前一輪影像混合：`x_{i+1} = (1−γ)·x̃_i + γ·x_i`

**grid 切分細節（512×512）**：stride = 128，切出九個 256×256 grid（相鄰共享 256×128），**四個 128×128 角落另組成第十個 grid**，共 **10 個**。256×256 之限制來自淨化用無條件 diffusion model 的原生解析度。

**參數（v3 依官方 repo 核對補充）**：參數尺度為**1000 步 DDPM 的原始 timestep**。官方提供兩組性質不同的設定：

| 設定 | pure_steps | iterations | 性質 |
|---|---|---|---|
| README 建議 | 10 | 20 | **多次淺淨化**，GrIDPure 的核心設計 |
| 腳本預設 | 100 | 1 | **單次深淨化**，行為上退化為 DiffPure |

v2 SPEC 將「0.1T」與「t=10、20 迭代」並列，實為上述兩種不同設定。configs 以原始 timestep 為準即可。

**階段二應兩組都測**：既符合淨化強度掃描的需求，亦可檢驗 GrIDPure 的迭代設計是否確實優於單次深淨化。

AntiPure 使用 t^p=10、γ=0.1、2 輪×20 迭代，屬多次淺淨化一類。

**成本警告** `[原文確認：§5.3]`：**單張 512×512 影像在單張 V100 上約需 2 分鐘**。以 150 張 × 5 方法估算即約 25 小時。原文指出流程可平行加速，實作時須納入考量並先估算總時數。

**負面結果**：以 LDM 跑 SDEdit 作為淨化器無效（對抗保護在不同 LDM 間遷移性佳），淨化必須用 pixel-space 無條件 diffusion model。

**checkpoint**：所需之 pixel-space 無條件 guided diffusion checkpoint 約 **2GB**，TWCC 階段須另行下載（見 §9）。

---

## §6 實驗階段

### 6.1 階段零：相似性約束校準（`stage0_calibrate.py`）`[本專案設計]`

非加性無 pixel norm 約束，若不校準則「效果較好」可能僅因允許更大改動。

```
1. 對加性方法（κ=0.06）計算保護影像與原圖之 LPIPS，得基準 L_ref
2. 對非加性方法掃描相似性上限，找出使 LPIPS ≈ L_ref 之設定
3. 寫入 configs/nonadditive.yaml，輸出校準曲線
```

### 6.2 階段一：乾淨情況比較（`stage1_clean.py`）

方法矩陣：加性（PhotoGuard-encoder、PhotoGuard-diffusion）× 非加性（AdvDiff-based、APA-based、Hybrid）× 模型（V1.4、V2.0）× 編輯（SDEdit、Inpainting）。

輸出 csv：方法、模型、編輯方式、七項指標、記憶體峰值、耗時。

**校準檢查點**：PhotoGuard 數值須與 §2.7 表中 Encoder、Diffusion 兩欄接近，方可確認設定正確。

**通過條件**：非加性至少一變體在多數指標上不劣於 §2.7 表中的 DAYN 欄。

### 6.3 階段二：淨化後比較（`stage2_purify.py`）

淨化清單（五項）：JPEG、Gaussian blur、crop-and-resize、AdverseCleaner（含僅 BF 變體）、GrIDPure。

```
核心輸出：drop = (clean_effect − purified_effect) / clean_effect
```

**待驗證假設**（非預設結論）：非加性之下降比例顯著小於加性。

**注意**：流程為「淨化 → 編輯 → 測編輯劣化」，與 customization 論文的「淨化 → fine-tune → 測身份」不同，僅借用其淨化方法。GrIDPure 之 t 須掃描多值並繪製「淨化強度 vs 防禦效果」曲線，以捕捉可能之交叉現象。

---

## §7 公式—出處對照表

| 公式/設定 | 出處 |
|---|---|
| encoder attack 目標 | PhotoGuard 式 (4)、Algorithm 1 |
| diffusion attack 目標 | PhotoGuard 式 (5)、Algorithm 2 |
| PG 超參數 ε=16/255, step=2/255, N=200 | PhotoGuard Table 9 |
| SD 生成超參數 | PhotoGuard Table 8 |
| Seed 協定 | PhotoGuard Appendix A.1 |
| 指標實作 piq | PhotoGuard §4.2 註 8 |
| attention map | DAYN 式 (2) |
| 跨層聚合（bicubic + 逐像素相加） | DAYN 式 (3) |
| mask（由原圖、threshold τ） | DAYN 式 (4) |
| 注意力抑制損失 | DAYN 式 (5) |
| TUGU | DAYN Algorithm 1 |
| DAYN 超參數 κ=0.06, N=100, T=10 | DAYN §4.1 |
| DAYN 引用數值 | DAYN Table 1 |
| AdvDiff 每步注入（DDPM） | AdvDiff 式 (9)、Alg.1 line 8 |
| AdvDiff 每步注入（DDIM） | AdvDiff Alg.2 line 6、附錄 C 式 24 |
| AdvDiff 起始噪聲注入 | AdvDiff 式 (11)、Alg.1 line 12 / Alg.2 line 11 |
| AdvDiff guidance 區間 | AdvDiff 附錄 E |
| AdvDiff 參數 N=5, s=0.7, a=0.5 | AdvDiff §4 實作細節 |
| **非加性抗淨化之證據** | **AdvDiff Table 2**（DiffPure 下 AutoAttack 22.2% vs AdvDiff-Untargeted 75.2%） |
| APA Stage 1 reward | APA 式 (6) |
| APA trajectory-level | APA 式 (7) |
| APA step-level | APA 式 (8)、(11) |
| APA 中間步淨化 | APA 式 (9)、(10) |
| APA diffusion augmentation | APA 式 (12) |
| APA 參數 | APA §4.1 |
| DDIM inversion | APA 式 (4) |
| GrIDPure 四步驟與混合式 | Zhao et al. CVPR'24 §5.2 |
| GrIDPure grid 切分 | Zhao et al. CVPR'24 Fig. 9 |
| GrIDPure V100 成本 | Zhao et al. CVPR'24 §5.3 |
| 淨化標準參數（crop 20%、JPEG 65、0.1T） | Pixel is a Barrier 附錄 D |
| SDEdit | Meng et al., ICLR 2022, arXiv 2108.01073 |

**v1 引用錯誤更正**：v1 之 survey 與 HTML 中「APA 對 DiffPure 的 ASR 遠高於 additive 攻擊（APA Table 2）」為誤引。APA Table 2 僅比較各 unrestricted 攻擊之間，無 additive 對照。正確證據為 **AdvDiff Table 2**。

---

## §8 待確認清單（依優先序）

### 須向 DAYN 作者索取

1. **測試資料集**（150 張、3 類物件、prompt、生成參數）
2. **編輯所用的 SDEdit strength**——直接影響 Table 1 全部數字
3. **κ=0.06 的值域尺度**——[-1,1] 或 [0,1]，兩者擾動強度差兩倍（見 §3.2.1）
4. **擾動的範數類型**——論文稱 ℓ1/ℓ2/ℓ∞ 皆可，未指明實驗所用（見 §3.2.2）
5. **PhotoGuard baseline 的 encoder attack 使用何種 target latent**——零向量（官方預設）或灰階 latent（見 §3.2）

第 3、4、5 項若無法取得，可於 T2 校準階段以實驗決定：各參數組合各跑一次 PhotoGuard，以能對上 Table 1 者為準。

### 其他

6. APA 的 LoRA rank（在其附錄）
7. 非加性之相似性約束改用 LPIPS 是否為可接受之假設

不需要 DAYN 的官方程式碼（不實作），但測試集與編輯設定為對齊的前提。

---

## §9 Claude Code 第一優先任務

在環境偵查與骨架建立之後、實作演算法之前，先完成：

1. `git clone https://github.com/MadryLab/photoguard` 並讀 `notebooks/` 內之 PGD 實作，確認投影方式（解決待確認第 3 項）。**PhotoGuard 為本專案的校準錨點，其實作正確性決定整個比較的有效性，此項優先於其他 clone 任務。**
2. `git clone https://github.com/ZhengyueZhao/GrIDPure` 讀其預設參數與 grid 切分實作
3. `git clone https://github.com/ermongroup/SDEdit` 作為 SDEdit 參考（實際使用 diffusers 之 img2img）
4. 安裝 `piq` 與 `opencv-contrib-python`，確認 VIFp、FSIM、`cv2.ximgproc.guidedFilter` 可用
5. **TWCC 階段須另行下載**：GrIDPure 所需之 pixel-space 無條件 guided diffusion checkpoint，約 **2GB**。記入 TWCC_CHECKLIST.md
