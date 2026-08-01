# E25–E26 — 兩個軸都被重新檢驗，以及攻擊端的根因

- 2026-08-01。E25 全部在本機 CPU 執行，只讀既有的 `runs/`，不需 GPU。
  E26 用本機快取的真實 SD v1.4（4.27 GB）在 CPU 上跑。
- 重跑順序：`p5_semantic_axis.py` → `p6_purify_retention.py` →
  `p5_compare_page.py` → `p7_attack_sanity.py` → `p7_compare_page.py`
  → `p8_site_c_capacity.py`。

**結論摘要。** E20 修好的是保真那一軸的量測，防禦效果那一軸用的仍是同一個
LPIPS 純量。E25 用既有資料重判該軸，得到 **726 格中語意失敗 0 格**；使用者
判讀比對頁後指出更根本的事：**連未防禦的原圖被文字編輯都沒有成功**。
E26 追出根因——`src/models/sd.py` 全專案沒有 classifier-free guidance，
等同 w = 1，而 SD v1.x 在 w = 1 下幾乎不服從 prompt。**E2–E23 全部是在防禦
一個不存在的攻擊。**

---

## 1. E25-1：語意軸從未被看過

`runs/*/results.csv` 自 E2 起就記了 `edit_clip_a/b`、`edit_siglip_a/b`
（未防禦編輯結果與防禦後編輯結果對編輯 prompt 的對齊度）。全專案沒有任何
腳本讀過它們——`grep -rn clip src/ scripts/` 只命中 `grad_clip` 與
`numpy.clip`。至今所有結論建立在 `net_lpips = edit_lpips − ctrl_lpips`
之上，而那個量說的是「輸出移動了多少」，不是「編輯有沒有失敗」。

這個區別在文獻上已被點名：*Semantic Mismatch and Perceptual Degradation:
A New Perspective on Image Editing Immunity*（arXiv:2512.14320）指出以
「與未防禦編輯結果的視覺距離」為免疫成功的判準是錯的，並展示 LPIPS 很高但
免疫完全失敗的案例。

### 1.1 對照先行，且對照不通過的指標直接作廢

判定量是逐張影像的配對差 Δ = 對齊度(防禦後編輯) − 對齊度(未防禦編輯)。
但若指標本身分不出「編輯有沒有發生」，Δ ≈ 0 只代表指標遲鈍。故先量

    edit_effect = 對齊度(未防禦編輯) − 對齊度(原圖)

| 指標 | 原圖 | 未防禦編輯 | edit_effect | 對照 |
|---|---|---|---|---|
| CLIP | 0.2030 | 0.2132 | +0.0101 ± 0.0169 | **未通過** |
| SigLIP | 0.0058 | 0.0334 | +0.0276 ± 0.0237 | 通過 |

n = 91 個 cell。**CLIP 的標準差大於均值，連「編輯有沒有發生」都分不出來**，
故其 Δ 不可用於判定；腳本會印出作廢聲明並改由 SigLIP 承擔結論。CLIP 的
Δ 仍寫入 CSV 供查核。

事後看，CLIP 這個未通過的對照本身就是 E26 的第一個訊號：編輯若真的成功，
CLIP 對齊度不會只升 0.01。

### 1.2 結果：726 格中語意失敗 0 格

判準：mean(Δ) < 0 且 |mean(Δ)| > sd(Δ)，且 **n ≥ 2**。

n ≥ 2 這個條件是必要的：n = 1 時 sd 恆為 0，`|mean| > sd` 對任何負值都自動
成立。未加此條件時 `e6_stepsP` 與 `e6_stepsLA`（步數掃描只跑一張圖）產出
24 格假陽性。加上之後歸零。

33 個 run × 22 個淨化臂 = 726 格，**依 SigLIP 判定為語意失敗的有 0 格**。
無淨化那一格的 Δsiglip 全部落在 −0.0080 到 +0.0085 之間，且多數為正。

---

## 2. E25-2：淨化保留率——非加性的優勢確實在這裡，但判準要嚴

`net_lpips` 的保留率 = net(該淨化臂) / net(無淨化)。文獻預期非加性應較耐
淨化：NAPPure（ICCV 2025）量到既有淨化方法是為加性雜訊設計的，對 flow-field
變形明顯失效。

E23（唯一兩臂都被約束綁住且步數足夠的配對）：

| 淨化 | 強度 | S 保留 | P 保留 | S/P |
|---|---|---|---|---|
| 無 | — | 100% | 100% | 0.85 |
| blur | 0.5 | 71.3% | 42.7% | 1.43 |
| jpeg | 30 | 22.6% | 9.1% | 2.11 |
| noise | 0.08 | 14.8% | 4.1% | 3.08 |
| quantize | 16 | 45.8% | 18.8% | 2.09 |
| blur | 3.0 | 0.3% | 3.9% | 0.06 |

**判準寫得太鬆，此處記下而不改寫。** 原判準是「S 保留率高於 P 的獨立淨化
算子 ≥ 3 個」，但只要求該算子的**任一**強度佔優，於是六個強度中的一個就能
扛下整個算子，七對配對全部 4/4 成立，判準失去分辨力。改以「多數強度佔優」
重算後為 4/7 成立：

| 配對 | 寬鬆 | 嚴格 | 逐算子佔優比例 |
|---|---|---|---|
| E15 τ=0.02 | 成立 | 成立 | blur 4/5 jpeg 5/6 noise 3/5 quantize 3/5 |
| E15 τ=0.05 | 成立 | 成立 | blur 3/5 jpeg 5/6 noise 3/5 quantize 3/5 |
| E15 τ=0.10 | 成立 | **不成立** | blur 3/5 jpeg 3/6 noise 3/5 quantize 2/5 |
| E21 τ=0.02 | 成立 | **不成立** | blur 2/5 jpeg 5/6 noise 2/5 quantize 1/5 |
| E21 τ=0.05 | 成立 | **不成立** | blur 3/5 jpeg 5/6 noise 2/5 quantize 2/5 |
| E21 τ=0.10 | 成立 | 成立 | blur 3/5 jpeg 4/6 noise 3/5 quantize 3/5 |
| E23 τ=0.05 100 步 | 成立 | 成立 | blur 3/5 **jpeg 6/6** noise 3/5 quantize 3/5 |

型態：**JPEG 是唯一在七對配對上都偏向 S 的算子**（3/6 到 6/6）；blur、noise、
quantize 都是中等強度偏向 S、極端強度反轉。重模糊（σ ≥ 2）會把 site S 的
效果完全抹掉（保留 0.3%）而 site P 還剩 3.9%。

**但這個結論的地基已被 §1 與 §3 抽掉**：保留率的分子分母都是 `net_lpips`，
而該量在 §3 之後只能解讀成「兩次去噪之間的漂移」。本節證明的是**非加性的
視覺偏移對淨化較耐受**，不是「非加性的防禦對淨化較耐受」。兩者不可混為
一談，重跑之後才知道後者是否成立。

---

## 3. E26：攻擊端從來沒有在做文字引導編輯

### 3.1 使用者的判讀

`runs/p5_semantic_axis/compare.html` 交給使用者判讀，回報：

> 連原始圖片被文字編輯都沒有成功，然後有無防禦的圖編輯後長的都差不多，
> 一樣都很爛。

### 3.2 程式層面的根因

`src/models/sd.py` 的 `_eps` 只以**條件嵌入**呼叫一次 UNet：

    return self.unet(z, t, encoder_hidden_states=emb).sample

全專案 `grep -rn "guidance|uncond|cfg_scale|do_classifier" src/ scripts/ tests/`
**沒有任何命中**。這等同 classifier-free guidance 的 w = 1。Stable Diffusion
v1.x 是在 CFG 下訓練也在 CFG 下使用的，w = 1 時 prompt 對輸出的影響極弱，
SDEdit 退化成「加噪再去噪」。

### 3.3 量測（真實 SD v1.4，512²，n_edit=10，n=6）

`scripts/p7_attack_sanity.py`。全部是**未防禦**的原圖，只差在 w 與 strength：

| 組態 | Δclip | Δsiglip | LPIPS→原圖 |
|---|---|---|---|
| **w=1.0, s=0.5（E2–E23 的設定）** | **+0.0108** | **+0.0277** | 0.4630 |
| w=3.0, s=0.5 | +0.0373 | +0.0621 | 0.4775 |
| **w=7.5, s=0.5（標準攻擊）** | **+0.0644** | **+0.0831** | 0.4999 |
| w=7.5, s=0.7 | +0.0849 | +0.1016 | 0.6248 |

標準 guidance 使 CLIP 對齊增益達 **6.0 倍**、SigLIP 3.0 倍。而現行設定的
Δclip = +0.0108 落在 §1.1 對照的噪聲範圍內（+0.0101 ± 0.0169）——**在 w = 1
下「編輯」與「什麼都沒做」在指標上分不出來**，與使用者的目視判讀一致。

逐圖看更清楚。三張影像在現行設定下的 Δclip 是零或負的：

| 影像 | prompt | w=1.0 | w=7.5 | w=7.5, s=0.7 |
|---|---|---|---|---|
| car_00 | a wrecked car after an accident | **−0.0013** | +0.0547 | +0.0814 |
| dog_00 | a cat sitting on the grass | **−0.0023** | +0.0720 | **+0.1181** |
| dog_01 | a cat sitting on the grass | **+0.0009** | +0.0714 | +0.0996 |
| person_01 | an elderly person with gray hair | **−0.0003** | +0.0604 | +0.0928 |
| person_00 | an elderly person with gray hair | +0.0431 | +0.0826 | +0.0863 |
| car_01 | a wrecked car after an accident | +0.0245 | +0.0453 | +0.0312 |

dog_00 是最乾淨的一例：prompt 要求「一隻貓」，w=1 下影像對「貓」的對齊度
**下降** 0.0023，w=7.5 上升 0.0720，s=0.7 時上升 0.1181——相差 50 倍。

**唯一的例外是 car_01**：它在 w=7.5、s=0.7 反而低於 s=0.5（+0.0312 vs
+0.0453）。單圖的非單調不影響結論的方向（其餘五張都單調上升），但它說明
**strength 的最佳值逐圖不同**，正式重跑時 strength 應與 w 一起掃描而非取
單點。人眼比對頁在 `runs/p7_attack_sanity/compare.html`。

### 3.4 這件事波及什麼

E2–E23 的每一個數字都是在 w = 1 下產生的。那些實驗量到的 `net_lpips`
是**兩次隨機去噪之間的漂移**，不是防禦效果。具體而言：

- 「site S 領先 1.15×」（E15）已於 E20 撤回，理由是保真那一軸。現在防禦
  那一軸也失效。
- 「Sbic / P = 0.85×」（E23）同樣失效。
- §2 的淨化保留率量的是漂移的保留率。
- E20 的四臂等 LPIPS 探針與 `local_acutance_dev` **不受影響**：它們量的是
  防禦圖對原圖的失真，完全不經過 SDEdit。

**E20 的八條「硬的」結論裡，與保真度量測有關的六條全部存活；與防禦效果
有關的兩條（E15 的 1.15× 失效量 67.3%、site S 的鈍化來源）中，前者失去
意義，後者仍成立（它也是純失真量測）。**

### 3.5 修法與 before/after

`src/models/sd.py` 新增 `_eps_cfg`，`sdedit` 新增 `guidance_scale` 與
`emb_uncond` 兩個參數。

    修訂前（sd.py line 141-150、line 340）
        def _eps(self, z, t, emb, use_ckpt=False):
            return self.unet(z, t, encoder_hidden_states=emb).sample
        ...
        eps = self._eps(z, t, emb, use_ckpt=use_ckpt)

    修訂後（sd.py 新增 _eps_cfg，sdedit 新增兩個參數）
        def _eps_cfg(self, z, t, emb, guidance_scale, emb_uncond=None,
                     use_ckpt=False):
            if guidance_scale == 1.0:
                return self._eps(z, t, emb, use_ckpt=use_ckpt)
            if emb_uncond is None:
                raise ValueError(...)          # 不得靜默退回單分支
            eps_u = self._eps(z, t, emb_uncond, use_ckpt=use_ckpt)
            eps_c = self._eps(z, t, emb, use_ckpt=use_ckpt)
            return eps_u + guidance_scale * (eps_c - eps_u)
        ...
        eps = self._eps_cfg(z, t, emb, guidance_scale, emb_uncond,
                            use_ckpt=use_ckpt)

兩個設計決定：

- **`guidance_scale` 預設維持 1.0**，且 w = 1.0 時直接回到單次前向。既有
  53 個 run 的數值必須逐位元可重現，否則連「舊結果錯在哪裡」都無法對照。
  新實驗必須明確指定 w。
- **兩次前向分開做而非合批。** 合批把激活加倍，而 E0 已量出 512² 下記憶體
  才是綁定的資源。代價是時間乘二。

---

## 4. 線四：目標函數的兩項

### 4.1 cross-attention 的 φ=0 零梯度已修

E20 §9 釘住的缺陷：`attn_mode="divergence"` 量的是與未防禦參照的 KL 散度，
φ=0 時兩分佈相同、KL = 0，而 0 是 KL 的**最小值**，梯度精確為零。任何
「與參照的散度」形式都有這個性質，換一種散度不能解決。

新增 `attn_mode="suppress"`：最小化內容 token 分到的注意力質量，

    suppression = 1 − Σ_{τ ∈ 內容 token} A[q, τ]

其最佳點不在 φ=0，故起步梯度一般非零。這也是 cross-attention 免疫那條線的
著力點（arXiv:2509.10359；arXiv:2512.14333）。質量是 softmax 後未重新正規化
的和，值域 [0, 1]，有界故不需要 hinge。

**預設由 `divergence` 改為 `suppress`。** 留著 divergence 是為了讓缺陷有
具名的位置與釘住它的測試（`test_divergence模式在phi等於零時無梯度`），
但它不該是預設——直接拿去跑會安靜地什麼都不做。

`span` 為空時**提前拒絕而非落回全域**：全部 77 格的注意力質量和恆為 1，
落回全域會讓目標退化成常數 0，那正是本模式要修掉的失效。

新增測試 4 項：值域與方向、缺 span 時拒絕、φ=0 梯度非零、空 prompt 提前拒絕。

### 4.2 targeted 模式從未被跑過，現已跑通

清點 `runs/` 全部 4882 列 `results.csv`，`defense_mode` **100% 是
`untargeted`**。而 `objective.py:145-150` 自己的註解就寫了「無目標最大化在
文獻上一貫比有目標脆弱」，並引用本專案實測的 3.3 倍噪聲過擬合。

以 tiny-SD 在 CPU 跑通 `run_defense.py --defense_mode targeted
--target_image ...` 的完整路徑：L_def = 0.7561（是一個距離而非 hinge，故起點
就非零）、梯度非零、φ 被更新、CSV 與 `env.json` 正確記錄。新增測試
`test_有目標模式端到端可訓練`。

---

## 5. 線三：site C（色度矩陣場）

### 5.1 設計依據

E20/E21 量出 site S 的鈍化幾乎全部來自重取樣本身（雙線性 85.0%、真實
site S 85.2%）。鈍化約束 `local_acutance_dev` 量的是 **Rec.601 亮度**的逐
區塊梯度能量比（`local_acutance.py:57` 的 `_grad_sq` → `_luma`）。

site C 只動色度平面 (U, V)、亮度 Y 原封不動，故該約束在數學上不啟動。
site S 的死因在此結構上不存在。

    x_def = YUV⁻¹( Y(x), (I + ΔM_φ) · [U(x), V(x)]ᵀ )

ΔM 定義在粗網格（預設 32）上，雙線性上採樣取得平滑性，硬上界 `max_dev`
為保真度預算（與 site S 的 `max_disp` 同一角色，L∞ 對非加性不對等）。

### 5.2 兩個必須明說的事

**(1) 「約束不對它收費」是雙面的。** 現行約束集（LPIPS ∩ 鈍化）對 site C 的
特徵失真——色偏、色度串音、飽和區假色——是否收費**尚未量測**。加入本位置
之後必須為它跑一次四臂等 LPIPS 探針，否則就是重蹈 site S 用 LPIPS 買模糊的
覆轍，只是換成用色度買。

**(2) 無彩區域沒有容量。** ΔM 乘在 (U, V) 上，U = V = 0 的像素無論 ΔM 為何
都不動。灰階或高度去飽和的影像在本位置的容量趨近於零。加偏移項可以解掉，
但偏移項是色度上的加性位移，會破壞本位置的定位，故不加。此限制以
`test_無彩影像上色度變換沒有容量` 釘住。

### 5.3 兩次由測試逼出的常數修正

`test_無彩影像上色度變換沒有容量` 與 `test_色彩空間來回的數值誤差有界`
最初都失敗，兩次都是文獻常數的捨入問題，不是門檻太嚴：

| 問題 | 現象 | 根因 | 修法 |
|---|---|---|---|
| YUV 來回誤差 1.8e-05 | 「φ=0 逐位元等於原圖」只是近似成立 | 教科書的 `_YUV2RGB` 與 `_RGB2YUV` 各自捨入，互為反矩陣只到 1.8e-05 | 在 float64 求逆，誤差降到 3e-08 |
| 灰階圖梯度 3.3e-04 而非 0 | 「無彩沒有容量」變成「幾乎沒有容量」 | `[-0.14713, -0.28886, 0.436]` 的和為 1e-05 而非 0，灰階像素帶著 1e-05·R 的假色度 | 由亮度係數推導 U ∝ (B−Y)、V ∝ (R−Y)，該列和精確為零 |

### 5.4 容量檢查：site C 進得了運作點

`scripts/p8_site_c_capacity.py`。不需要 SD，直接掃 `max_dev` 量可達失真。
動機是 tiny-SD 煙霧測試中 site C 在 step 0 的梯度只有 2.2e-08，而 site P 是
1.9e-04。判準事先宣告：必須存在某個 `max_dev` 使 LPIPS 落在 [0.02, 0.10]
且鈍化偏差低於 0.04。

6 張真實影像，最壞情況的 ΔM（全部元素頂到上界、正負隨機）：

| max_dev | 平均 LPIPS | 平均鈍化 |
|---|---|---|
| 0.05 | 0.0067 | 0.0002 |
| **0.10** | **0.0210** | 0.0004 |
| **0.15** | **0.0399** | 0.0006 |
| 0.30 | 0.1029 | 0.0016 |
| 1.00 | 0.3153 | 0.0112 |
| 2.00 | 0.4594 | 0.0420 |

**通過。** max_dev ∈ [0.10, 0.15]（現行預設 0.15）正好把 LPIPS 帶進主網格的
運作範圍，而鈍化偏差 0.0006 比門檻 0.04 低 65 倍——設計性質在真實影像上成立。

跨影像差異由色度能量解釋：`person_00` 平均色度 0.0362，在 max_dev=0.15 只到
LPIPS 0.0300；`person_01` 為 0.0752，同樣的 max_dev 到 0.1222。**同一個
`max_dev` 對不同影像不是同一個預算**，這與 E21 §3「同一個 `max_disp` 對兩種
重取樣不是同一個預算」是同一類問題，正式實驗必須逐圖校準而非取單一值。

---

## 6. 待決事項

**全部需要 GPU：**

1. **以 w = 7.5 重跑主網格。** 這是 E26 的直接後果。在此之前，E2–E23 的
   任何 `net_lpips`、任何倍率、任何 site 之間的比較都不應引用。
   **重跑不含 site S**（使用者 2026-08-01 決定）：它的 1.15× 與 0.85× 都已
   失效，沒有既有理由保留；作為對照的價值留到最後再評估。非加性一側改由
   site C 承擔。strength 應與 w 一起掃描，理由見 §3.3 的 car_01。
2. **判準改為語意軸。** 沿用 `net_lpips` 會重蹈覆轍。SigLIP 通過對照可用；
   ISR 式的 MLLM 判準（arXiv:2512.14320）是更完整的作法但需要外部模型。
3. **site C 的四臂探針。** 見 §5.2(1)，必須在把 site C 放進主網格之前做。
4. **site C 的 `max_dev` 逐圖校準。** 見 §5.4。
5. **targeted 與 suppress 兩個目標的實測。** 兩者都已跑通但從未在真實 SD 上
   產生過資料。
6. **失真預算應提高到文獻的區間。** 現行運作點 `defimg_lpips` 為 0.036–0.059；
   DCT-Shield（ICCV 2025）報告自身 LPIPS 0.267、PhotoGuard/MIST/AdvDM/SDS/
   DiffusionGuard 為 0.284–0.362。即使扣掉實作與資料集差異仍差 5–8 倍。
   （此為跨論文的量級比較，非同一實作下的量測。）

**不需要 GPU：**

7. site C 加進 `docs/architecture.html`（該檔仍寫「三個注入位置」，現為七個）。
