# 原始碼查證：論文正文 vs 官方實作

> 2026-08-05。目的是把 `SURVEY_2026-08-05.md` 中「依論文正文擷取」的規格，
> 逐項對照官方原始碼確認。**落差本身就是要寫進論文的東西**——
> 若照正文實作而原始碼另有其事，baseline 的數字會系統性偏低，
> 整篇論文的比較基礎就不成立，且事後補不回來。
>
> 適用範圍：**除 N1／N2（本專案自己的方法）之外全部**。
> N3（APA 移植）的階段一必須忠實，階段二的替換已在 `DESIGN` §4 記錄理由。

## 狀態

| # | 對象 | repo | 正文查證 | 原始碼查證 | 落差 |
|---|---|---|---|---|---|
| 1 | PhotoGuard-c | `MadryLab/photoguard` | ✅ | ✅ 見 §3 | **重大** |
| 2 | Mist | `mist-project/mist` | ✅ | ✅ 見 §4（已複驗） | **重大** |
| 3 | DIA | 未找到公開 repo | ✅ PDF 全文 | 進行中 | — |
| 4 | AdvPaint | `JoonsungJeon/AdvPaint` | ✅ | ✅ 見 §1 | **重大** |
| 5 | PromptFlare | `NAHOHYUN-SKKU/PromptFlare` | ✅ | ✅ 見 §2 | **重大** |
| 6 | DiffVax | `ozdentarikcan/DiffVax` | ✅ | ✅ 見 §5 | **不可用** |
| 7 | APA | 進行中 | ✅ | 進行中 | — |
| 8–11 | 四個淨化算子 | 進行中 | 進行中 | 進行中 | — |
| 12 | Crop & Resize | DIA 論文內 | ✅ | 進行中 | — |

## 查證的總結論（截至目前）

**已查的五篇，五篇都有落差，其中兩篇的落差會直接改變比較基礎。**
這證實了「依論文正文實作」不足以支撐嚴格比較。逐項見下。

一個反覆出現的型態：**值域**。三篇 baseline 在 `[-1, 1]` 上做最佳化，
而論文報的 eps 是像素域的數字。兩者差兩倍，且不會有任何症狀——
輸出仍是一張合理的防禦圖，只是強度錯了一倍。本專案的張量介面是 `[0, 1]`，
每一篇的換算都必須逐篇確認，不可共用一套。

---

## 0. 先修正一項出處錯誤

**DIA 是 ICCV 2025**，不是只有 arXiv 預印本。
正式出處：[ICCV 2025 open access](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_DIA_The_Adversarial_Exposure_of_Deterministic_Inversion_in_Diffusion_Models_ICCV_2025_paper.pdf)，
作者 Seunghoo Hong, Geonho Son, Juhun Lee, Simon S. Woo。
`SURVEY_2026-08-05.md` 需更正。

---

## 1. AdvPaint — 原始碼查證完成，**發現三處落差**

來源：[`JoonsungJeon/AdvPaint/AdvPaint.py`](https://github.com/JoonsungJeon/AdvPaint)

### 1.1 落差一：迭代次數

| 來源 | 值 |
|---|---|
| 論文正文（我先前擷取） | **250** |
| 原始碼預設 | **100** |

`SURVEY` §2.1 記的 250 若直接拿來用，等於跑了一個原論文預設之外的設定。
**處置：以原始碼的 100 為準**，並在報表註記正文與程式的差異。

### 1.2 落差二：eps 與 step_size 有兩組值

| 參數 | 函式預設 | CLI 預設 |
|---|---|---|
| `eps` | 0.06 | **0.1** |
| `step_size` | 0.03 | **0.05** |

論文正文報的是函式預設那一組（0.06 / 0.03）。CLI 是實際跑實驗的入口。
**處置：採論文正文那組（0.06 / 0.03），因為那是論文報告數字所依據的設定**，
並在 `BaselineSpec` 註記 CLI 另有一組值。

### 1.3 落差三：損失只用四個成分，不是「全部 self- 與 cross-attention」

原始碼：

```python
loss = (loss_query + loss_key + loss_value + loss_cross_q) / length

loss_query   -= (GT_self_query[timestep][path]  - self_query[timestep][path]).norm(p=2)
loss_key     -= (GT_self_key[timestep][path]    - self_key[timestep][path]).norm(p=2)
loss_value   -= (GT_self_value[timestep][path]  - self_value[timestep][path]).norm(p=2)
loss_cross_q -= (GT_cross_query[timestep][path] - cross_query[timestep][path]).norm(p=2)
```

即：**self-attention 的 Q／K／V 三者 + cross-attention 的 Q 一者**。
cross-attention 的 K 與 V **不在損失內**。論文摘要寫的
「targets the self- and cross-attention blocks」在字面上為真但不精確，
照字面實作會多攻擊兩個成分，是另一個方法。

### 1.4 其餘已確認的實作細節

| 項目 | 原始碼 |
|---|---|
| PGD 更新 | `X_adv = X_adv - grad.sign() * actual_step_size`（**減號**，因損失已取負） |
| 投影 | `torch.minimum(torch.maximum(X_adv, X_ori - eps), X_ori + eps)` |
| 值域 | 最佳化在 **[-1, 1]**，推論時才回到 [0, 1] |
| 步長衰減 | `step_size - (step_size - step_size/100)/iters * iter`，線性降到原值的 1% |
| timestep | **只取 `timesteps[0]`**，不是掃全部。與論文正文「optimization timestep T only」一致 |
| 優化器 | **無**（不是 Adam/SGD），手動梯度符號更新 |
| mask | `X_ori_masked = X_ori * (mask_512 < 0.5)`，逐 mask 檔迴圈 |

**對本專案的意義**：`ARCH` §4 規劃的「共用 PGD 骨幹 + 各自的損失」成立——
AdvPaint 用的是標準 sign-PGD，與 PhotoGuard 系同一形式。
但**值域 [-1, 1] 必須注意**：本專案的張量介面是 [0, 1]，
eps 0.06 在兩個值域下代表的實際擾動量差兩倍。

---

## 2. PromptFlare — 部分查證，**發現一項與本專案設計衝突**

來源：[`NAHOHYUN-SKKU/PromptFlare/promptflare.py`](https://github.com/NAHOHYUN-SKKU/PromptFlare)

### 2.1 衝突：它**換掉 attention processor**

```python
for n, m in pipe.unet.named_modules():
    if n.endswith('attn2'):
        m.set_processor(MyAttnProcessor2_0(attn_controller, n))
```

本專案的 `src/models/attention.py` 刻意**不換 processor**，改用 forward pre-hook，
理由記在該檔 docstring：換掉 processor 會連帶改變 UNet 自己的注意力計算路徑
（SDPA 融合核心改為手寫實作），「有沒有開這個目標」就不再是單一變因。

**這不是矛盾，是兩個不同用途**：本專案的 pre-hook 是為了**擷取**注意力分佈供
我們自己的 N1 損失使用；PromptFlare 是**它自己的方法**，換 processor 是其實作的一部分。
**處置：重現 PromptFlare 時照它的原樣換 processor**，不可改成 pre-hook——
那會變成一個我們自己設計的變體。兩者在程式上必須隔離，不共用注意力擷取層。

### 2.2 BOS 遮罩的實際建構

```python
encoder_attention_mask = torch.ones(2, 77).to(device=pipe.device)
encoder_attention_mask[1][1:] = 0
```

即第二批次**只保留位置 0**（BOS），其餘 76 個位置遮蔽。
批次 0 是未遮蔽的完整 prompt。這與論文的 `L_CA` 式子
（`CA(φ(z̃_t), e, M^BOS)` 對 `CA(φ(z̃_t), e)` 取差）一致。

### 2.3 已確認的其他值

| 項目 | 原始碼 |
|---|---|
| `num_inference_steps` | **4** |
| `k`（擾動步驟數） | 1 |
| `grad_reps`（梯度累積） | 1 |
| PGD 更新 | `adv = adv - avg_grad.sign() * step_size`，同 AdvPaint 的形式 |
| attention 層 | 只有 `attn2` |

⚠️ `num_inference_steps = 4` 與論文正文擷取到的「400 步最佳化、步長 2/255」
是兩件不同的事（前者是去噪步數、後者是 PGD 步數），需要在
`attention_control.py` 與 `protect.py` 中確認 PGD 側的預設值。

### 2.4 未完成

- `AttnController.cal_loss()` 的實際計算式（在 `attention_control.py`）
- `protect.py` 的 CLI 預設值（eps / step_size / iters）
- `loss_mask` 與 `loss_depth` 兩個參數的意義

---

## 2.5 PromptFlare 的其餘落差（agent 查證）

詳見 `_audit_promptflare_photoguard.md`。四項：

| # | 落差 | 後果 |
|---|---|---|
| 1 | **eps 實際只有論文宣稱的一半。** `protect.py:23` 的 `eps /= 255` 施加在 `utils.py:10` 產生的 `[-1,1]` 張量上，故論文的 12/255 在**像素域只有 6/255**。以 repo 內 sample 實測原圖與 protected 圖的最大絕對差為 7（多數 ≤6），與推論一致 | 照論文數字實作會給它兩倍的預算，baseline 被不當增強 |
| 2 | `cal_loss` 是**跨層加總**，不是論文 Eq. 12 的 `E_l`（期望值） | 層數改變時損失尺度跟著變，與論文式子不等價 |
| 3 | 記錄的是 `to_out` 之後的 `A·V·W_out`，**不是** Eq. 10 的 `A·V` | 攻擊的對象與論文所述差一個線性投影 |
| 4 | processor 用 `F.scaled_dot_product_attention`，**attention map 從未實體化**；不記錄 Q/K/V 也不記錄 softmax 機率 | 「cross-attention decoy」的實作方式與直覺理解不同 |
| 5 | 全 repo **無任何 seed**，`torch.randn` 未固定 | 保護結果不可逐位元重現 |

`loss_depth = [1024, 256, 64]` 是 **token 數白名單**（即依 latent 解析度篩層），
不是層編號。此值在 SDXL 上必須重新決定——1024² 影像的 latent token 數與 SD v1.5 不同。

---

## 3. PhotoGuard-c — **三項重大落差，已取得逐字佐證**

來源：`MadryLab/photoguard`，HEAD `686bea75c786cb46c88fc396a0cd0ee3d7d28c2e`，
`notebooks/demo_complex_attack_inpainting.ipynb`。

> WebFetch 取不到 attack 函式的原因已查明：該檔 2,858,833 字元，
> attack 函式在 21.4% 位置（char offset 610,869），前面全是 cell 4/6 內嵌的 base64 PNG。
> 需以 `curl` 存檔後 `json.load` 解析。**這是後續要查大型 notebook 時的可重用作法。**

### 3.1 論文寫 ℓ∞，實跑的是 L2

cell 8 的函式定義（原文）：

```python
d_x = X_adv - X.detach()
d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)
```

更新式是**歸一化梯度**而非 sign：

```python
grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
grad_normalized = grad.detach() / (grad_norm + 1e-10)
actual_step_size = step_size
X_adv = X_adv - grad_normalized * actual_step_size
```

cell 10 的實際呼叫（原文）：`eps=16, step_size=1, iters=200, grad_reps=10,
clamp_min=-1, clamp_max=1`。

cell 11 的 ℓ∞ 呼叫**整段 17 行全部以 `#` 開頭**，其參數為 `eps=0.1, step_size=0.006`。
`super_linf` 函式本身在 cell 8 有定義（sign 更新 + min/max 投影），但從未被呼叫。

| 來源 | 內容 |
|---|---|
| 論文 Table 9 | ℓ∞、16/255、step 2/255、200 步 |
| **實跑** | **L2 renorm、maxnorm=16、step 1、200 步、grad_reps=10** |

**repo 中不存在任何一處同時滿足 Table 9 四欄的程式碼。**

### 3.2 `maxnorm=16` 的實際幾何

已實測 `torch.renorm` 的行為（torch 2.13.0）：

| 測試 | 結果 |
|---|---|
| `torch.renorm(randn(1,3,512,512)*10, p=2, dim=0, maxnorm=16).norm()` | 16.000 |
| 同上逐通道 norm | 9.237 / 9.252 / 9.224 |

即約束作用在**整張圖展平後的單一向量**（786,432 元素），不是逐通道。
`X` 形狀為 `[1,3,512,512]`，batch=1 故 `dim=0` 只有一個切片。
值域為 `[-1,1]`（`clamp_min=-1`、`prepare_mask_and_masked_image` 的 `/127.5 - 1.0`）。

換算（**此段為算術推導，非原始碼直述**）：L2 半徑 16 在 `[-1,1]` 相當於 `[0,1]` 的 8；
若均攤到全部元素 RMS = 16/√786432 ≈ 2.30/255 灰階；但 `compute_grad` 有
`grad = ... * (1 - cur_mask)`，擾動只存在於非重繪區，以其 sample 實測該區佔 24.9%，
RMS ≈ 4.61/255。

**關鍵含意**：L2 球允許少數像素遠超過該值（極端可達滿量程），
與 ℓ∞ 16/255 是**兩種不同的失真幾何，不可互相換算**。
引用 PhotoGuard-c 的失真預算時，不得用論文 Table 9 的數字描述實跑結果。

### 3.3 沒有 img2img 版的 complex attack

以 GitHub tree API 確認全檔案清單，只有 `demo_complex_attack_inpainting.ipynb`。
本專案的威脅模型是 img2img／SDEdit，故 **PhotoGuard-c 必須由我方移植**，
並在論文標註為改寫，與 AdvPaint／PromptFlare 的 mask 改寫同一位階。

### 3.4 target 實際被歸零

```python
target_image_tensor = prepare_image(target_image)
target_image_tensor = 0*target_image_tensor.cuda()  # ...or simply the zero tensor
```

`prepare_image` 下載的圖被乘 0 覆蓋後才傳入。故 complex attack 的 target 是
**`[-1,1]` 的零張量，即中性灰**，不需外部影像。
Mist 需要特製的高對比 `MIST.png`，兩者**不可混用同一份 target**。

### 3.5 一項對本專案設計的佐證

cell 10 的 `prompt = ""`——**PhotoGuard-c 本身就是 prompt-free 的**。
這獨立佐證了 `DESIGN` §2.1 把全部條件改為 prompt-free 的決定。
另記：`strength=0.7`、`guidance_scale=7.5`、`num_inference_steps=4`、`SEED=786349`。

### 3.6 本專案的處置建議

**以原始碼版本（L2）為準實作**，理由：

1. 我們在**匹配 τ_LPIPS** 上比較（射線縮放），故 eps 的絕對值不是操作變因；
   真正影響結果的是**更新規則的幾何**（歸一化梯度 vs sign），那必須取自原始碼。
2. 我們不引用其論文報告的數字，全部自行重跑，故無須遷就 Table 9。

`BaselineSpec` 需記錄此落差，報表在該列加註
「原始碼實跑 L2 renorm maxnorm=16；論文 Table 9 記為 ℓ∞ 16/255」。

---

## 4. Mist — **三項落差，已由我方獨立複驗**

我以 WebFetch 直接取 `mist-project/mist/mist_v3.py` 複驗，agent 的三項指控全部屬實。

### 4.1 版本確認

`mist-project/mist`（主程式 `mist_v3.py`）即 arXiv:2305.12683。
`mist-v2` 對應 arXiv:2310.04687，是**另一篇**（針對 LoRA 微調的水印工具），不可誤用。

### 4.2 落差一：論文寫 `‖·‖₂`，程式碼是平方和

```python
self.fn = nn.MSELoss(reduction="sum")      # 即 ‖·‖₂²，不是 ‖·‖₂
```

**直接影響 `w = 1e4` 的有效尺度**。照論文式子用 `‖·‖₂` 再配 w=1e4，
兩項的相對權重會與原實作差一個平方，融合模式等於跑在完全不同的操作點上。

### 4.3 落差二：mode 編號與 README 相反

```python
if self.mode == 0:
    return - loss_semantic                              # semantic
elif self.mode == 1:
    return self.fn(zx, zy)                              # textural
else:
    return self.fn(zx, zy) - loss_semantic * self.rate  # fused
```

`Readme.md` 的對照表寫 0 = textural，**與程式碼相反**。以程式碼為準。

### 4.4 落差三：eps 名目為 16/255 而非 17/255

```python
'epsilon': epsilon/255.0 * (1-(-1)),
'alpha':   alpha/255.0 * (1-(-1)),
```

值域是 `[-1, 1]`（`img/127.5 - 1.0`），故乘 `(1-(-1))=2` 是**正確的換算**，
換回像素域即 16/255。論文正文的 17/255 是對取整後上界的宣稱，無對應程式碼。

> 注意 Mist 這裡**做對了**值域換算（乘 2），而 PromptFlare **沒做**（§2.5 第 1 項）。
> 兩篇看起來都寫 `eps/255`，實際差兩倍。這正是必須逐篇查原始碼的理由。

### 4.5 其餘與論文一致

steps = 100、step_size = 1/255、w（`rate`）= 1e4、預設 fused、SD v1.4 原始 ckpt、
值域 `[-1,1]`、target 為 repo 根目錄的 `MIST.png`（黑底白字密集平鋪，高對比硬邊）。

---

## 5. DiffVax — **在本專案的威脅模型下不可用**

詳見 `_audit_mist_diffvax.md`。

| # | 發現 | 後果 |
|---|---|---|
| 1 | 論文稱 immunizer「takes an input image I」，實際餵入的是**編輯區域已歸零的 masked image**；完整原圖載入後從未使用 | **免疫階段必須先知道遮罩**，這與「防禦方不知道攻擊方要編輯哪裡」的前提衝突 |
| 2 | `attack()` 硬編碼 9 通道 inpainting 輸入；全 repo 對 `img2img`／`SDEdit`／`InstructPix2Pix` 命中 **0 筆** | 只支援 inpainting |
| 3 | 無任何 L∞ 硬預算（`self.eps = 32/255` 是死參數，未被使用） | 無法納入以 τ_LPIPS 為共同貨幣的比較 |
| 4 | **counter-attack 評測完全未實作**：CNN 去噪、JPEG 0.75、IMPRESS 三者在 repo 中不存在；`evaluate.py` 只算 PSNR/SSIM/FSIM/CLIP | 論文 §2.3 記的抗淨化評測無原始碼可依 |
| 5 | `loss = L_edit + α·L_noise`，**α = 4 出自 `configs/train.yml`**；論文正文未給任何 α、lr、batch、epoch | 正文不足以重現，需依 repo 設定 |

**建議：把 DiffVax 移出 baseline 清單。** 理由不是它不好，而是在
「無 mask 的全圖 SDEdit」這個威脅模型下，忠實重現它在結構上不可能——
它的免疫器要吃 masked image。強行改寫等於我們自己設計一個新方法再冠上它的名字。

處置建議是在論文的相關工作中引用它並說明為何未納入比較，
這比放一個改到面目全非的版本誠實。**此項需你裁決。**

---

## 6. DIA — repo 找到，值域落差

官方：[`sohn1029/DIA`](https://github.com/sohn1029/DIA)（共同第一作者帳號）。
**只有攻擊端**，無淨化／評測／baseline 程式碼。

| 項目 | 內容 |
|---|---|
| step_size | `attack_setting.json` 的 `lr = 1/255`（論文未給） |
| 值域 | `[-1,1]`（前處理 `2·ToTensor()−1`、`clamp ∓1`） |
| **eps 換算** | `eps=0.05` 在 `[-1,1]` → **`[0,1]` 尺度只有 0.025（≈6.4/255）** |

**第三篇值域陷阱。** 做失真匹配時不可把 0.05 當成 `[0,1]` 的 0.05。

淨化的完整數字在 **arXiv 版 Supplementary Table 6**，CVF 版無補充資料。
`SURVEY` §2.3 原記「Fig 6」需更正為此。

---

## 7. APA — 論文與程式碼互相矛盾

官方：[`deep-kaixun/APA`](https://github.com/deep-kaixun/APA)，兩階段皆完整。

### 7.1 階段一的超參數在論文中不存在

APA 的 arXiv v1 是唯一版本且**沒有 Appendix**，論文四處引用它卻查無該節。
故階段一全部超參數只能取自程式碼：

| 項目 | 值 |
|---|---|
| LoRA rank / alpha | 8 / 8 |
| `target_modules` | `["to_k","to_q","to_v","to_out.0"]`，掛在整個 UNet |
| **涵蓋範圍** | **attn1 + attn2**（self- 與 cross-attention 皆有） |
| optimizer | AdamW，`lr=1e-4` constant |
| 步數 | 200 |
| `noise_offset` | 0.1（**論文未提**） |
| 粒度 | per-image 一組 LoRA |
| prompt | ImageNet 類別名 |

> **已據此修正程式。** `src/residual/site_weight.py` 原本寫死只掃 `.attn2.`，
> 照 APA 實作會少掉一半目標層（實測 4 → 8 層）。已新增 `blocks` 參數，
> 預設維持 `("attn2",)` 使既有行為不變，`site_apa.py` 改用 `APA_BLOCKS`。
> 這是一個不會有症狀的容量差異：訓練跑得完、曲線正常，只是階段一的對齊能力被削弱。

### 7.2 論文自我矛盾（不影響本專案，但須在論文中說明）

APA §4.5 明確批評 `R_a − λ‖z_0−z̄_0‖²` 這種 one-stage 形式會 reward hacking，
但**釋出的 APA-GC 主方法程式碼裡就有 `−10·MSE(ori_latents, la_t)`**（APA-SG 沒有）。
另 APA-GC 的「T=10」實際是沿用 50 步排程只跑 11 步，`ϱ(·)` 的 brightness 在程式碼中被註解掉。
照論文重現得不到 Table 3 的 LPIPS 0.23 / SSIM 0.69。

**對本專案的影響有限**：我們只移植階段一，階段二本來就已替換（`DESIGN` §4）。
矛盾落在被替換掉的那一半。但論文中須寫明「本專案移植的是其階段一，
依官方程式碼而非論文正文，因後者無該節」。

---

## 8. 四個淨化算子

| 算子 | 可否忠實實作 | 依據 |
|---|---|---|
| **Adverse Cleaner** | ✅ | 原 repo `lllyasviel/AdverseCleaner` 已 404，由兩個獨立鏡像交叉驗證出 16 行原碼：`bilateralFilter(d=5, σc=8, σs=8) × 64` → `guidedFilter(guide=原圖, radius=4, eps=16) × 4`。值域 [0,255] BGR，OpenCV **不可微**，需直通代理 |
| **IMPRESS** | ✅ | 官方 `AAAAAAsuka/Impress`。損失 `MSE(VAE(x), x) + α·max(LPIPS(x, x_adv) − L, 0)`，**原生可微**。⚠️ 有三組「預設」（函式簽名／Glaze／PhotoGuard），取 PhotoGuard 組：eps 0.1、iters 1000、lr 0.005、α 0.01、σ 0.05、值域 `[-1,1]`。**無授權檔** |
| **DiffPure** | ⚠️ 部分 | `NVlabs/DiffPure`，t=100(CIFAR)/150(ImageNet)/500(CelebA-HQ)。SDE 版以 `sdeint_adjoint` 可微，授權允許研究用途。**缺口：檢查點最高 256×256**，本專案在 1024² 下使用屬自訂改動 |
| **CNN 去噪（NTIRE 2023）** | ❌ **致命缺口** | 冠軍為 Team Apply AI 的 IPTV2（29.96 dB），**程式碼與權重皆未公開**。報告聲稱在 `ofsoundof/NTIRE2023_Dn50`，該 repo 實際只有主辦方 baseline SGN。且 DiffVax 只引挑戰賽報告、未指名模型、repo 亦無該程式碼 |
| **Crop & Resize** | ⚠️ 部分 | DIA 補充材料明寫「cropped 10%」（**不是**二手來源說的 20%），但未指定邊長／面積、中心／隨機、插值方法 |

---

## 9. 裁決（2026-08-05）

以下五項為我方裁決。每一項都標明「依原始碼」還是「我方指定」，
**凡我方指定者一律在報表與論文中標註，不得混入原論文設定**。

| # | 事項 | 裁決 | 可回退 |
|---|---|---|---|
| 1 | **DiffVax** | **移出 baseline 清單**，改在相關工作中引用並說明未納入的理由。它的免疫器吃 masked image、只支援 inpainting、無 L∞ 預算、counter-attack 評測未實作——在無 mask 的 SDEdit 下忠實重現結構上不可能 | ✅ 需你確認 |
| 2 | **PhotoGuard-c 的 norm** | **依原始碼取 L2**（`torch.renorm` maxnorm，歸一化梯度更新）。理由：我們在匹配 τ_LPIPS 上比較，eps 絕對值非操作變因，真正影響結果的是更新規則的幾何 | ✅ |
| 3 | **CNN 去噪** | **取 Restormer**（使用者 2026-08-05 裁決）。見下方說明 | ✅ 已定案 |

### 為何 CNN 去噪需要裁決（使用者問）

**因為「還原原始碼」這個選項不存在。** 這是它與其他三項的根本差別——
那三項都有原始碼可依，只是要決定依到什麼程度。

| 事實 | 出處 |
|---|---|
| DiffVax 論文稱其 counter-attack 用「NTIRE 2023 挑戰賽的冠軍去噪模型」 | 論文 §A.3 |
| 冠軍為 Team Apply AI 的 IPTV2（29.96 dB），**程式碼與權重皆未公開** | 挑戰賽報告 |
| 報告聲稱程式在 `ofsoundof/NTIRE2023_Dn50`，該 repo 實際只有主辦方 baseline SGN | 實查該 repo |
| **DiffVax 自己的 repo 沒有任何 counter-attack 實作**（JPEG 與 IMPRESS 兩項也沒有） | 實查該 repo |

故任何 CNN 去噪器都是我方指定的替代品，差別只在指定哪一個——
這是一個無法由查證消除的選擇，因此需要裁決。

**取 Restormer 的理由**：DiffVax 用該算子的目的是「以一個強去噪器攻擊防禦擾動」。
Restormer 是通用影像修復的標準強 baseline、權重公開、有官方 repo，滿足該目的。
NAFNet 與 SCUNet 同樣可行，選擇不影響結論的方向，只影響絕對數值。

**標註要求**（已由測試釘住）：算子命名為 `cnn_denoise_substitute`、
docstring 與例外訊息均寫明「非 NTIRE 2023 冠軍，為我方替代」、
報表在該列加註。**不得聲稱重現 DiffVax 的該項評測。**
| 4 | **DiffPure 解析度** | 檢查點只到 256²。作法：**降取樣到 256 → 淨化 → 升回原尺寸**，並**額外加一條「只做同樣降升取樣、不淨化」的對照**，把 resize 本身的破壞力與擴散淨化分開。不加這條對照就無法歸因 | ✅ |
| 5 | **Crop & Resize** | 依 DIA 的「10%」，其餘**我方指定**：中心裁切、每邊各裁 10%、bicubic 升回原尺寸。標註為我方指定 | ✅ |

## 11. 實作階段新發現：DIA-PT 的 L1 起點會超出預算 —— **待裁決**

> 2026-08-05，補測試時發現。由 `tests/test_baselines.py::test_DIA的L1起點在某些輸入下超出預算`
> 以 `xfail(strict=True)` 釘住——缺陷還在時它如預期失敗，一旦有人修掉會變成
> 非預期通過而立刻顯現，不會被靜默吸收。

### 現象

DIA-PT 的 `init_rule="l1_ball"` 在某些輸入下產生遠超 eps 的起點。
實測（16² 影像、eps = 0.05、起點 generator 固定為 0，只改 `x` 的 seed）：

| x 的 seed | ‖d‖₁ | ‖d‖∞ | 相對 eps |
|---|---|---|---|
| 1234 | 0.05004 | 0.050029 | 正常 |
| 99 | 0.05004 | 0.050029 | 正常 |
| **7** | **14.01168** | **1.499097** | **30 倍** |

### 根因

`_l1_projection` 逐字取自 DIA 官方 `attack/DIA_PT.py:22-86`，其出處為
Croce & Hein 的 AutoAttack。**該演算法的箱型約束假設 `x ∈ [0,1]`**，
而 DIA 把它套用在 `[-1,1]` 的影像張量上（`DIA_PT.py:241-248`）。

程式中的分支：

```python
c  = eps1 - y.abs().sum(dim=1)
s1 = -u.sum(dim=1)              # u = min(0, min(1-x-y, x+y)) ≤ 0，故 s1 ≥ 0
c5 = s1 + c < 0
c2 = c5.nonzero().squeeze(1)    # 需要投影的批次索引
```

`s1 + c ≥ 0` 時 `c2` 為空，投影分支整個不執行，`d` 停在初值 `u.clone()`。
`u` 的值域在 `x ∈ [-1,1]` 下可達 −1，故回傳的 delta 可達 ±1。
`x ∈ [0,1]` 時 `u ∈ [-1, 0]` 且該情形不會發生——箱型約束的前提正是那個值域。

**這是 DIA 原始碼自身的缺陷，本專案忠實轉寫了它。** 轉寫沒有錯誤。

### 後果

`run_pgd` 在起點只做值域夾限、不做投影（與 DIA 自身的迴圈一致），
故**第一次梯度是在一個可能遠超預算的點上計算的**。
這使 DIA-PT 的比較基礎與其他四篇不對等。

### 三個選項

| # | 處置 | 代價 |
|---|---|---|
| a | **照抄不改**，在論文中如實陳述此缺陷 | 最忠實。但 DIA-PT 的失真預算實際上不受控，「同 τ 比較」對它不成立 |
| b | 在 `run_pgd` 的起點加一次 `project()`，標為 `modified_from_paper` | 保住比較基礎。是標準 PGD 作法，不改更新規則，但確實是我方改動 |
| c | **只用 DIA-R**（`init_rule="none"`，不受影響），DIA-PT 列為未納入 | 最乾淨。代價是少一個變體；查證顯示 DIA-R 在多數格點上本來就較強 |

### 裁決（使用者 2026-08-05）

**取 (a) 的變體：保留 DIA-PT 的規格與程式碼，但本輪不納入實驗，並註明原因。**

理由是使用者定下的總原則——**除非不得已，一律完全還原論文原始碼**。
選項 (b) 的加投影雖然能保住比較基礎，但那是**我方改動 DIA 的攻擊程序**，
在「不得已」尚未成立時不該做：DIA-R 是同一篇論文的另一個變體，
不受此缺陷影響，且已在實驗中，故 DIA 這一篇仍有忠實的代表。

具體處置：

| 項目 | 作法 |
|---|---|
| `src/baselines/dia.py` 的 `dia_pt` spec | **保留**，逐字忠於原始碼，不加投影 |
| `grid.py::CONDITIONS` | **移除 `dia_pt`**，本輪不跑 |
| `tests/test_baselines.py` 的 xfail | **保留**，繼續釘住該缺陷 |
| 論文 | 在 baseline 章節註明 DIA 取 DIA-R 變體，並說明 DIA-PT 未納入的原因 |

這樣既沒有改動別人的方法，也沒有讓一個失真預算不受控的條件混進匹配比較。
若日後要納入 DIA-PT，程式已在，改的是 `CONDITIONS` 一行。

---

## 10. 值域對照表（實作時必須逐篇查表，不可共用）

**這是本次查證最重要的單一產出。** 三篇 baseline 在 `[-1,1]` 上最佳化，
兩篇的 eps 換算方式還不一樣。

| 方法 | 最佳化值域 | 程式碼的 eps 寫法 | **像素域 `[0,1]` 的實際 eps** |
|---|---|---|---|
| PhotoGuard-c | `[-1,1]` | `maxnorm=16`（**L2 半徑**，非 L∞） | L2 8（幾何不同，不可換算成 L∞） |
| Mist | `[-1,1]` | `epsilon/255.0 * (1-(-1))` — **有乘 2** | 16/255 ✅ |
| DIA | `[-1,1]` | `eps=0.05` 直接用 | **0.025（≈6.4/255）** |
| AdvPaint | `[-1,1]` | `eps=0.06` 直接用 | **0.03** |
| PromptFlare | `[-1,1]` | `eps/=255` — **沒乘 2** | **6/255**（論文宣稱 12/255） |

本專案的張量介面是 `[0,1]`。**每一篇的換算必須逐篇實作，不可共用一套。**
