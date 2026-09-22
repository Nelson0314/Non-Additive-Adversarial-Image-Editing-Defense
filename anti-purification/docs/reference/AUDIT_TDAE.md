# TDAE 論文全文查證

用途：供 baseline 嚴格重現使用。本文件所有數值與公式均標註來源；查不到者一律列於
「未找到的項目」一節，**不作推斷、不補值**。反推出來的數標星號並註明是反推
（`BASELINE_PROVENANCE.md` 規則 4）。

## 查證所用來源

| 代號 | 內容 | 位置 |
|---|---|---|
| `[TDAE-HTML]` | TDAE，arXiv:2512.14341v2，`[cs.CV] 02 Mar 2026`，**已被 IEEE TPAMI 接受** | https://arxiv.org/html/2512.14341v2 |
| `[TDAE-ABS]` | 同篇 abstract 頁 | https://arxiv.org/abs/2512.14341 |

`[TDAE-HTML]` 的章節結構為 I–V ＋ References，**沒有 Appendix、沒有補充資料**。
引用格式為「節次／公式編號／表號」。

## 0. 官方程式碼：沒有

- 全文檢索 `github.com`、`code is available`、`Our code`、`released`、`supplementary`、
  `appendix`：**皆為零筆**。
- 摘要、Introduction 的貢獻列表、Conclusion 都沒有釋出程式碼或權重的敘述。

**後果：本 repo 的 `src/baselines/tdae.py` 是依論文重建，不是官方實作的重現。**
其他六篇 baseline 的 `AUDIT_*` 可以逐行引「檔案:行號」，這一篇不行——凡是論文
正文沒寫的，只有兩種下場：列在 §5「未找到的項目」，或列在 §4「本專案的決定」。

---

## 1. 論文寫什麼

### 1.1 問題設定（§III-A Preliminaries）

- 來源影像 `x₀`、文字編輯嵌入 `c`、期望的編輯輸出 `y₀`，編輯系統
  `f_θ: X × C → Y` 滿足 `f_θ(x₀, c) ≈ y₀`。
- 免疫目標 Eq. (1)：`δ_v* = argmax_{‖δ_v‖_p ≤ ε} L(f_θ(x₀+δ_v, c), y₀)`。
- `L` 的定義只有一句：「a differentiable loss metric (e.g. ℓ₂-distance)」。
- Eq. (2) 是標準 PGD：`δ ← Π_{‖·‖_p ≤ ε}(δ + α·sign(∇_δ L))`。
- `L_∞` 被點名為 imperceptibility 的常用選擇（§III-A 第一段）。

### 1.2 FDM（§III-C）

出發點是 TPA（`[37]` Fan et al., NeurIPS）的 Eq. (3)：對鄰域取樣求**期望**梯度範數。
TDAE 指出取樣成本太高，改成鄰域內的**最大**梯度範數，Eq. (4)：

```
min_{‖δ_v‖_p ≤ ε_v}  −L(f_θ(x₀+δ_v, e), y₀)
                     + λ · max_{‖δ_v′−δ_v‖_q ≤ ρ} ‖∇_{δ_v′} L(f_θ(x₀+δ_v′, e), y₀)‖₂
```

**但論文自己放棄了 Eq. (4) 的內層 max**（§III-C：「Directly optimizing the inner
maximization problem in Eq. (4) is computationally intractable」），實跑的是 Eq. (5)：

```
min_{‖δ_v‖_p ≤ ε_v}  −L_{δ_v} + λ · ‖∇_{δ_v} L_{δ_v}‖₂ ,
    其中 L_{δ_v} = L(f_θ(x₀+δ_v, e), y₀)
```

接著三步化簡：

1. Eq. (6)：`s = ∇L_{δ_v} / ‖∇L_{δ_v}‖₂`，並明文寫「assuming ∇L ≠ 0; otherwise s = 0」。
   論文說這個方向的選法是「Drawing inspiration from ... SAM `[57]`」。
2. Eq. (7)：把 `‖∇L‖₂` 寫成沿 `s` 的方向導數，再用有限差分近似
   `≈ (L_{δ_v+h·s} − L_{δ_v}) / h`。
3. Eq. (8)：取絕對值以保證非負，得到實際最佳化的目標

```
min_{‖δ_v‖_p ≤ ε_v}  −L_{δ_v} + (λ/h) · | L_{δ_v+h·s} − L_{δ_v} |
```

梯度 Eq. (9)（用 `∇|u| = sign(u)∇u`，令 `z = L_{δ_v+h·s} − L_{δ_v}`）：

```
g_FDM = −∇L_{δ_v} + (λ/h) · sign(z) · ( ∇L_{δ_v+h·s} − ∇L_{δ_v} )
```

Eq. (10)：`∇L_{δ_v+h·s}` 取「在 `δ′ = δ_v + h·s` 這一點對 `δ̃_v` 的一階梯度」，
論文明寫 `Following [57], this can be estimated efficiently without computing
second-order derivatives`，即 **Hessian-free**。

> **它不是 SAM 式的 ascent-then-descent。** SAM 是用鄰域最壞點的梯度**取代**原梯度；
> FDM 是在原目標上**加一個梯度範數懲罰**，鄰域點只用來做有限差分。兩者只有「先
> 往梯度方向走一步再取梯度」這個動作長得像，組合方式不同：SAM 的更新是 `g₂`，
> FDM 的更新是 `−g₁ + (λ/h)·sign(z)·(g₂ − g₁)`。

### 1.3 DPD（§III-D）

- 在原文字嵌入 `c` 上加可學的 `δ_p`，約束 `‖δ_p‖_∞ ≤ ε_p`。
- Eq. (11)（文字側，**最小化**）：`min_{‖δ_p‖_∞ ≤ ε_p} L(f_θ(x₀+δ_v, c+δ_p), y₀)`
  ——找一個能讓惡意編輯重新奏效的 prompt。
- Eq. (12)（影像側）：用 `e = c+δ_p` 跑 Eq. (5)。
- 兩者交替：每 `S` 步 FDM 之後，固定 `δ_v`，跑 `M` 步 PGD 更新 `δ_p`。

### 1.4 Algorithm 1（逐行）

Input：`x₀`、`c`、`L`、`ε_v`、`α`、`N`、`λ`、`S`、`ε_p`、`η`、`M`、`y₀`、`h`。

```
 1  δ_v ← 0
 2  δ_p ← 0
 3  x_imu ← x₀
 4  for n = 1..N:
 5      e ← c + δ_p
 6      if n ≡ 0 (mod S):
 7          δ_p ← 0
 8          for m = 1..M:
 9              e_DPD ← c + δ_p
10              L_DPD ← L(f_θ(x_imu, e_DPD), y₀)
11              g_p ← ∇_{δ_p} L_DPD
12              δ_p ← δ_p − η·sign(g_p)
13              δ_p ← Π_{‖·‖_∞ ≤ ε_p}(δ_p)
15          e ← c + δ_p
17      x_imu_current ← x₀ + δ_v
18      g₁ ← ∇_{δ_v} L(f_θ(x_imu_current, e), y₀)
19      s  ← g₁ / ‖g₁‖₂
20      δ_v′ ← δ_v + h·s
21      x_imu′ ← x₀ + δ_v′
22      g₂ ← ∇_{δ_v} L(f_θ(x_imu′, e), y₀) |_{δ_v = δ_v′}
23      z  ← L(f_θ(x_imu′, e), y₀) − L(f_θ(x_imu_current, e), y₀)
24      g_FDM ← −g₁ + (λ/h)·sign(z)·(g₂ − g₁)
25      δ_v ← δ_v − α·sign(g_FDM)
26      δ_v ← Π_{‖·‖_∞ ≤ ε_v}(δ_v)
27      x_imu ← x₀ + δ_v
29  x_adv ← x₀ + δ_v
```

由此可直接讀出四欄：起點 `δ_v = 0`（無隨機初始化）、更新為 `sign` 且是**減號**、
投影為 `L_∞`、每步只算 `g₁`／`g₂` 各一次（`grad_reps = 1`）。

### 1.5 實驗設定（§IV-A）

- 受害／代理模型：InstructPix2Pix（INS）、Stable Diffusion v1-4（SD14）、
  Stable Diffusion 3（SD3），皆取自 Hugging Face。
- 資料集：自建，從 `timbrooks/instructpix2pix-clip-filtered`（`[61]`）挑 100 張，
  35 人像 ／ 35 風景 ／ 30 畫作；**編輯 prompt 用資料集附帶的原始編輯指令**。
- 指標：PSNR↓、SSIM↓、LPIPS↑、VIFP↓、FSIM↓，量的是**編輯結果**對
  「乾淨圖的 benign 編輯結果」的差距（↓代表免疫較強）。
- 對照：ACE `[36]`、MIST `[35]`、PGD `[32]`、PGE `[32]`、SA `[34]`。
  `[32]` 是 Salman et al., ICML 2023，即本 repo 的 `photoguard_*`。

### 1.6 唯一給了數值的超參數（§IV-E）

> 「Based on these results, we adopt **λ/h = 0.3** for other experiments involving
> TDAE with PGD and SA.」

Fig. 5 是 `λ/h` 的敏感度圖，橫軸掃了哪些值**只有圖、正文未列**。

### 1.7 效率（§IV-C、Table IV）

FDM 對 TPA 的每次迭代時間：`SA+FDM` 6.49 s vs `SA+TPA` 71.37 s；
`PGD+FDM` 8.83 s vs `PGD+TPA` 89.65 s。硬體未載。

---

## 2. 本 repo 實作什麼

檔案：`src/baselines/tdae.py`，`SPEC_PAPER`（`name="tdae"`，
`modified_from_paper=True`）。

| 論文 | 本 repo | 說明 |
|---|---|---|
| Eq. (8) 的目標 | `flatgrad_objective` 回傳值 | 逐項相同 |
| Eq. (9) ＋ Eq. (10) 的 `g_FDM` | 同一個回傳值的 autograd 梯度 | `tests/test_tdae.py` 以逐行照抄的 `_algorithm1` 對照，三個 `h` 下 `allclose(rtol=1e-11)` |
| Eq. (6) 的 `s`，`∇L = 0` 時 `s = 0` | `n.clamp_min(tiny)` ＋ `(n > 0)` 遮罩 | 直接寫 `g₁/‖g₁‖` 會得到 `nan`，而 `nan` 經 `sign()` 之後會把 δ 整片打爆，骨幹不報錯 |
| Algorithm 1 第 18、22 行各一次前向 | 一步兩次 `sd.edit` | 測試 `test_一步只做兩次編輯鏈前向` 釘住 |
| Algorithm 1 第 25 行 `δ ← δ − α·sign(g_FDM)` | `objective="minimize"`、`update_rule="sign"` | `run_pgd` 不必改 |
| Algorithm 1 第 1 行 `δ_v ← 0` | `init_rule="none"` | |
| Algorithm 1 第 26 行 `Π_{‖·‖_∞ ≤ ε_v}` | `norm="linf"` | |
| §IV-E `λ/h = 0.3` | `LAMBDA_OVER_H = 0.3` | |
| §III-A `L` 為 ℓ₂-distance | `‖f_θ(x,e) − y₀‖₂`，不平方 | `y₀ = f_θ(x₀, e)` |
| §III-D DPD | **未實作**，`prepare(dpd=...)` 拋 `NotImplementedError` | 理由見 §3 |
| SD3 | **未做** | 理由見 §6 |

實作上的一個細節：`flatgrad_objective` 回傳的張量是
`value + (surrogate − surrogate.detach())`，其值為 Eq. (8)、梯度為 Eq. (9)。
這樣寫的原因是 `s` 必須先對 `L_{δ_v}` 反向一次才算得出來，若再讓 autograd 對同
一張圖反向第二次，就會撞到 `sd.edit` 內的 `torch.utils.checkpoint`
（`use_reentrant=False`）區塊；改為前向三次則把每步成本從論文的兩次 `f_θ`
變成三次。代數對照寫在該函式的 docstring 裡，並由測試釘住。

---

## 3. 文字側（DPD）與本專案威脅模型的衝突

**論文寫什麼**：DPD 的 `c` 是「該張圖在 InstructPix2Pix-clip-filtered 裡附帶的
原始編輯指令」的嵌入（§IV-A Target Models and Dataset）。Eq. (11) 在 `c` 的
`ε_p`-鄰域裡搜一個能繞過當前防禦的 `δ_p`。

**本專案的威脅模型**：防禦方**看不到攻擊指令**。沒有 `c` 就沒有「在 `c` 的鄰域
裡搜」這件事——在別的 prompt 的鄰域裡搜，搜到的不是論文那個量。

**怎麼處理**（已寫入 `SPEC_PAPER.modification_note`）：

1. **只實作影像側（FDM）**。`c` 固定為中性的空 prompt（`TDAE_PROMPT = ""`），
   `δ_p ≡ 0`。這不是我們發明的組態：論文 Table VI 的 `SA + FDM`、
   Table I–III 的 `PGE + FDM` 就是只開 FDM 的那一列。
2. `prepare(dpd=...)` 只要不是 `None` 就拋 `NotImplementedError`，訊息同時寫出
   威脅模型的衝突**與**論文沒給 `S`／`M`／`ε_p`／`η` 任何一個數。不預設一組
   看起來合理的值。
3. **附帶後果要一起記**：`encode_text("")` 與 `uncond_prompt()` 在 SD v1.x 上相同，
   CFG 兩支前向數值相等，退化為無條件預測（`photoguard.py` 記過同一件事）。
   於是 `y₀` 是一次**無指令的 SDEdit 重建**，`L` 量的是「免疫圖的無指令重建」
   對「乾淨圖的無指令重建」的 ℓ₂ 距離。這是威脅模型的直接後果，不是論文的設定，
   引用本條件的數字時必須連這句一起引。

---

## 4. 本專案的決定（論文沒有，但不決定就跑不起來）

| 項目 | 取值 | 理由 |
|---|---|---|
| 值域 | `[-1,1]` | 論文無程式碼、未載值域。取 diffusion VAE 輸入慣例，與 `photoguard`／`mist`／`dia` 同（`src/baselines/pgd.py` 模組 docstring 的對照表） |
| `ε_v` | `[0,1]` 的 16/255 | TDAE 是 plug-and-play，預算屬於被套的那一篇。論文的 `PGD [32]` 即 Salman et al.，本 repo 依其 Table 9 重建的臂是 `photoguard_linf`（16/255）。取同一組可與既有 baseline 在同一個 `eps01` 上頭對頭 |
| `α` | `[0,1]` 的 2/255 | 同上 |
| `N` | 200 | 同上 |
| `h` | **呼叫端必填**，無預設 | 論文只固定比值 `λ/h`，`h` 與 `λ` 各自的值查不到 |
| `strength` | **呼叫端必填**，無預設 | 論文未說明 SD14／SD3 怎麼當編輯模型用，見 §6 |
| 代理編輯鏈步數 | **呼叫端必填**，無預設 | 同上。一步 PGD 跑兩次完整編輯鏈，步數直接乘上去 |
| `guidance_scale` | 預設 7.5 | §IV-A Threat Model 寫攻擊方用「standard inference settings (e.g. default schedulers, CFG scale, step counts from the original editing models)」，SD v1.x pipeline 的預設 CFG 即 7.5。但那句講的是**攻擊方**，不是代理模型，故仍當本專案的設定記 |
| 編輯噪聲 | 整次求解固定同一個 | Eq. (7) 的有限差分要求 `L_{δ}` 與 `L_{δ+h·s}` 是同一個 `f_θ` 在兩點上的值；換噪聲則 `z` 是兩個不同函式的差，除以 `h` 之後不近似任何方向導數。論文全文無 `seed` |
| batch | 只接受 1 | Eq. (6) 的 `s` 與 Eq. (8) 的 `z` 都是逐張定義；batch>1 時 `z` 變成整批的和 |

### 4.1 一個反推的數（＊）

論文 Table V（免疫圖對**原圖**的保真度，INS 上）：

| | ACE | ACE+TDAE | MIST | MIST+TDAE | PGD | PGD+TDAE | PGE | PGE+TDAE | SA | SA+TDAE |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR↑ | 34.68 | 34.24 | 34.27 | 34.09 | 36.04 | 35.09 | 34.37 | 34.00 | 34.71 | 34.41 |
| SSIM↑ | 0.9027 | 0.8976 | 0.8914 | 0.8870 | 0.9136 | 0.8972 | 0.8812 | 0.8818 | 0.8916 | 0.8877 |
| LPIPS↓ | 0.2317 | 0.2369 | 0.2430 | 0.2473 | 0.2262 | 0.2474 | 0.2245 | 0.2309 | 0.2364 | 0.2384 |

`PGD+TDAE` 的 35.09 dB 若把 sign-PGD 視為飽和（絕大多數像素落在 `±ε_v`），
則 `rms ≈ ε_v`，反推 **`ε_v*` ≈ 255·10^(−35.09/20) ≈ 4.5/255**；用 `PGD` 的
36.04 dB 反推則是 4.0/255。這與本 repo 採用的 16/255 差約四倍。

**這是反推的數，不是量到的數**（`BASELINE_PROVENANCE.md` 規則 4），且反推用的
飽和假設論文沒有背書。只列作對照，不作為設定值；引用 Table V 的 PSNR 時要連
「INS 上、100 張自建資料集、免疫圖對原圖」這個協定一起引。

---

## 5. 未找到的項目

以下每一項都檢索過 `[TDAE-HTML]` 全文（含 Algorithm 1 的 Input 列、§III-C、
§III-D、§IV-A 的三個小節、§IV-B–§IV-E、所有表格標題與 Fig. 1–5 的說明文字），
並確認全篇無 Appendix、無補充資料、無官方程式碼可查（§0）。

| 項目 | 狀態 | 查過哪裡 |
|---|---|---|
| `ε_v`（影像預算） | **未找到** | 只出現在 Eq. (1)(3)(4)(5)(8)(12) 與 Algorithm 1 Input 的符號位置，無數值。全文檢索 `255` 零筆 |
| `α`（影像步長） | **未找到** | 同上 |
| `N`（總步數） | **未找到** | 同上 |
| `λ` | **未找到** | 只有 §IV-E 的比值 `λ/h = 0.3` |
| `h`（有限差分步長） | **未找到** | 同上 |
| `ρ`（Eq. 4 內層半徑） | **未找到**，且**用不到** | §III-C 只說「`ρ > 0` defines the radius of a small local region under an `L_q` norm, often `L_∞`」；Eq. (4) 的內層 max 已被論文自己放棄 |
| `S`、`M`、`ε_p`、`η`（DPD 四個數） | **未找到** | 只出現在 Algorithm 1 Input 與 §III-D 的敘述 |
| `L` 的具體形式 | **部分**：§III-A 給「e.g. ℓ₂-distance」，§III-D 說 `L` 是被套的那一篇的損失。TDAE 單獨跑時用哪一個，未明寫 | §III-A、§III-C、§III-D |
| `y₀` 怎麼取得 | **未明寫**。Algorithm 1 Input 列把它當給定常量；§III-A 稱之為「the desired edited output」、`f_θ(x₀,c) ≈ y₀` | §III-A、Algorithm 1 |
| 影像解析度 | **未找到** | §IV-A；全文 `512`／`1024` 只出現在參考文獻字串裡 |
| 推論步數、scheduler、CFG 尺度的**數值** | **未找到** | §IV-A Threat Model 只有「standard inference settings ... from the original editing models」一句定性敘述 |
| 隨機種子、重複次數 | **未找到** | 全文檢索 `seed` 零筆 |
| 硬體、單張耗時 | **部分**：Table IV 有每次迭代的秒數，硬體型號未載 | §IV-C、Table IV |
| Fig. 5 橫軸掃了哪些 `λ/h` | **未找到**（只有圖，正文未列） | §IV-E |
| ACE／MIST／PGD／PGE／SA 各自重跑時的預算 | **未找到** | §IV-B 只給結果表，不給對照組的設定 |

---

## 6. SD3 的缺口（核對本 repo 先前的紀錄）

### 6.1 論文有沒有說明 SD3 是怎麼當編輯模型用的？**沒有。**

本 repo 先前查證時記錄「未說明」，核對結果為**正確**。證據：

- §IV-A Target Models and Dataset 的原句是：「we test against multiple versions of
  StableDiffusion (SD) **configured for editing tasks**, including SD14 `[9]` and the
  recent SD3 `[58]`, as well as the dedicated InstructPix2Pix (INS) model `[59]`.」
  ——`configured for editing tasks` 之後沒有任何說明。
- 全文檢索：`img2img` **0 筆**、`image-to-image` 1 筆（僅出現在參考文獻標題）、
  `SDEdit` 3 筆（**全部**在 §II-A 的相關工作引用與參考文獻，不在實驗設定裡）、
  `strength` 1 筆（參考文獻字串）、`inference step`／`denoising step`／`noise level`
  **各 0 筆**、`inversion` 2 筆（相關工作）。
- 因此「SD14 與 SD3 以什麼流程把一張既有影像變成編輯結果」（SDEdit？inversion 後
  重建？某個 pipeline 的哪些參數？）在論文中**完全沒有交代**。這同時使
  `strength` 與推論步數無從引用，見 §4。

### 6.2 SD3 這一側本輪不做

- `src/models/sd.py` 只有 `SDWrapper`（SD v1.x）、`SDInpaintWrapper`、`SDXLWrapper`
  三種封裝；全 repo 檢索 `SD3`／`SD3Transformer`／`stable-diffusion-3` 為零筆。
- SD3 是 MMDiT（非 UNet）、三個 text encoder（CLIP-L ＋ CLIP-G ＋ T5）、
  flow-matching 排程，`SDWrapper` 的 `_eps`／`alphas_cumprod`／`_eps_cfg`
  在它上面全部不成立，不是換一個 `model_name` 就能跑的。
- **後果**：論文 Table II（以 SD3 為代理）與所有 `→SD3` 的轉移列在本 repo
  不可重建。要補這一塊必須先寫 SD3 封裝，屬於另一件事。

### 6.3 論文自己記錄的 SD3 排除項

§IV-B（Table II 的討論段）逐字：

> 「Note that ACE and MIST baselines are excluded not due to TDAE limitations, but
> because their architectures are inherently incompatible with SD3's Diffusion
> Transformer (DiT) design.」

即：ACE 與 MIST 在 SD3 上被排除，論文歸因於架構不相容，**不是** TDAE 的限制。
注意這只適用於 SD3 **當代理模型**的那張表（Table II）；Table I 的
`INS → SD3` 轉移列裡 ACE 與 MIST 都有數字，因為那裡 SD3 只當攻擊方的目標模型用。

---

## 7. 要加到 `scripts/baseline_run.py` 的幾行

本模組不改該檔（同時有別的 baseline 在改）。要跑時加這四處：

1. `from src.baselines import dia, mist, photoguard, tdae`
2. `run_additive` 的 spec 表加 `"tdae": tdae.SPEC_PAPER,`
3. `run_additive` 的分派加一支：
   ```python
   elif name == "tdae":
       kw = {"strength": strength, "num_inference_steps": 4, "h": 1.0}
   ```
4. `CONDITIONS` 加 `"tdae"`。

第 3 點的三個數論文都沒有，必須顯式寫出來並在報表標為本專案設定。`h = 1.0` 是
`[-1,1]` 值域上的 ℓ₂ 半徑（`s` 已單位化）：512² RGB 在 `ℓ∞ 32/255` 下，球內最遠
點的 ℓ₂ 距離是 `(32/255)·√(3·512·512) ≈ 111`，故 `h = 1.0` 約是可行域尺度的 1%。
換 `h` 要同時記在報表上，因為 `λ = 0.3·h` 會跟著變。

**`tdae` 沒有登記進 `src/baselines/REGISTRY`**：`tests/test_baselines.py` 要求
`AUDIT` 表與 `REGISTRY` 逐鍵相等，新 baseline 進 REGISTRY 之前要先進那張表。
`diffusionguard`、`advdrop`、`dct_shield` 也是同樣的獨立模組形態。

---

## 8. 為什麼不進本次的外部比較

§7 那四處接線**沒有被套用**。`scripts/defence_run.py` 的 `CONDITIONS` 不含
`tdae`，CSV 也沒有 `tdae_*` 欄。被移除的是「把 TDAE 放進這一批外部比較」
這件事，**不是**本 repo 依論文重建的模組：`src/baselines/tdae.py` 與
`tests/test_tdae.py` 原樣保留。

### 8.1 機制：論文的起點在本專案的損失上是零梯度點

本 repo 的實作裡 `L = ‖edit(x_adv) − y₀‖₂`，而 `y₀ = edit(x₀)`
用的是**固定的編輯噪聲**（`tdae.prepare` 固定噪聲以求可重現，理由見 §4
「編輯噪聲」那一列：Eq. (7) 的有限差分要求兩點上是同一個 `f_θ`）。
論文 Algorithm 1 第 1 行是 `δ_v ← 0`，`tdae.SPEC_PAPER` 忠實地寫成
`init_rule="none"`。兩者相遇時：

1. δ = 0 ⟹ `x_adv = x₀` ⟹ `edit(x_adv)` 與 `y₀` **逐位元相同**；
2. 於是 `L = ‖0‖₂ = 0`，而 `‖v‖₂` 在 `v = 0` 的梯度是 0；
3. `s = ∇L/‖∇L‖₂` 被 `nonzero` 守衛設成 0（避免除以零）；
4. `z = L_{δ+h·s} − L_δ = 0`，`k = 0.3·sign(0) = 0`；
5. 整個 `g_FDM` **精確為零**，PGD 一步都不動。

這個退化不會拋錯，交付的會是原圖。它與 `runs/advcf_objective/` 的
`advcf_expect_noid` 是同一種失效：起點落在目標函數的零梯度點上。

### 8.2 實測證據

| 量測 | 條件 | 讀數 |
|---|---|---|
| `flatgrad_objective` 的 grad absmax | CPU、16×16 替身、恆等起點（`δ_v ← 0`） | **0.0** |
| 同上 | 離開恆等之後 | **0.132** |
| 遠端實跑（照論文起點） | 15 分鐘後 | 仍停在 `step 0  L=0.000000e+00  Linf01=0.0000` |

### 8.3 先前的處置與它被否決的理由

先前為了讓這一列跑得動，起點改成標準 PGD 的 `uniform_linf` 隨機起點
（Madry et al.），並逐列寫進 CSV 的 `init_rule` 與 `modification_note`。
跑完的四張讀數是 PSNR 26.12–27.07、LPIPS 0.58–0.64。

**使用者裁定不接受這個偏離**：換掉 Algorithm 1 第 1 行之後，那一列已經不是
論文的方法，掛著 TDAE 的名字報出去會讓讀者以為比較的是該篇。因此整個條件
連同已跑出的結果一起移除，而不是保留結果再加註。

### 8.4 成本

實測 **13,350 秒／圖**，是這一批裡最貴的條件——`photoguard_c` 是 6,406 秒／圖，
TDAE 是它的兩倍有餘。成因是 Algorithm 1 一步 PGD 要跑兩次完整的代理編輯鏈
（第 18、22 行），步數直接乘上去（§1.7 的 Table IV 也記錄了 FDM 的每次迭代
時間，只是硬體未載，無法與這裡的秒數對齊）。

### 8.5 這不是對該篇的評價

上面的零梯度是**本專案的威脅模型與論文起點互相作用**的結果，不是
「TDAE 無效」。鏈條是：

> 防禦方看不到攻擊指令（§3）→ 求解端只能代入空 prompt → `y₀` 變成
> **無指令重建**＋固定噪聲 → `x₀` 在這個 `y₀` 上恰好是損失的零點 → δ=0 的
> 起點就落在零梯度點上。

論文自己的設定裡 `c` 是資料集附帶的原始編輯指令（§IV-A），`y₀` 是一次**帶
指令**的編輯，`x₀` 不會是該損失的零點，同一個起點不會退化。本檔對該篇方法
本身不下判定；要在本專案的威脅模型下跑它，需要的是換一個不依賴
「`y₀` 與 `edit(x₀)` 逐位元相同」的目標定義，而不是換起點。
