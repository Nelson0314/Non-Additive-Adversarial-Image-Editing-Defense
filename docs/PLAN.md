# 現行計劃

2026-08-13 定案。判準與結論以 `FINDINGS.md`／`DECISIONS.md` 為準；本檔說明
**要做什麼、為什麼、怎麼判成功**。做完的部分移入那兩個檔，本檔隨之縮短。

前置閱讀：`MAINLINE.md`（弱 baseline 的定義）、`reference/ROBUSTNESS_TESTS.md`
（三份外部檢定協定）。

---

## 1. 主張

使用者 2026-08-13 定案，取代 `DESIGN.md` §1 的三層階層：

| 層級 | 主張 | 讀數 |
|---|---|---|
| **主** | 非加性**更**抗淨化：淨化後的效果**衰減率**低於加性 baseline | `retention` |
| **並列** | 防禦效果本身不輸：未淨化時的位移量不低於加性 baseline | `effect(·, identity)` |
| 三 | 保真受控，報全部指標不挑選 | LPIPS／DISTS／PSNR／SSIM／NIQE／銳利度 |

**不再追求語意抵抗。** CLIP-T 對齊掉幅仍照報，但不作為成敗判準——
FND-024／029／030 四個軸全部否證，且 arXiv:2506.04394（ICIP 2025）獨立測到
同一現象（加擾動反而可能提高輸出與 prompt 的關聯）。主讀數改為**位移量**。

## 2. 判定式

```
effect(cond, P)     = edit_lpips( edit(P(x_def)),  edit(P(x)) )      # 位移讀數
retention(cond, P)  = effect(cond, P) / effect(cond, identity)

主張成立 ⇔ 在多數淨化算子 P 上，retention(非加性) > retention(每個加性 baseline)
前提     ⇔ effect(cond, identity) ≥ 3σ（σ 為跨 seed 標準差），否則該列標為不可用
```

### 2.1 分母的可用性：改用位移讀數後這張表可以出

`METRICS.md` §6 記載 `retention` 在本專案「五次拒絕出表」，原因是分母落在
雜訊裡。那是用 `effect_siglip` 讀數時的情形——65 組只過 1 組。
改用 `edit_lpips` 讀數時，FND-018 的 1425 列有 **1350 列**通過 3σ 閘門。
故本輪的主讀數必須是位移，`METRICS.md` §6 需據此改寫。

### 2.2 對照 R 的定義必須修正

FND-018／020 記載：`Ra`（**同一個非加性 site 上的隨機方向**）的 retention
普遍高於我方方法（diffpure150：Ra 2.846 對 apa 1.580）。在舊主張下這是
壞消息；在「非加性更抗淨化」之下要重新歸位：

| 對照 | 定義 | 在新主張下的角色 |
|---|---|---|
| **R** | **同失真的加性隨機**（像素空間加同 LPIPS 量的隨機噪聲） | **主張的對照組**。非加性要贏它 |
| `Ra` | 同 site 的隨機方向 | **非加性族的成員**，是該族的**下界**。它 retention 高 → 支持主張 |

**若我方方法贏不過 `Ra`**，論文的貢獻點就是**參數化（site）本身**而不是
損失函數。這要寫進論文，不得省略。FND-018／020 的既有數據已指向這個結論。

## 3. 路線 A：對淨化的 min-max

### 3.1 目標

```
max_φ   min_{P ∈ 𝒫}   effect( P( G(x; φ) ) )
```

外層是防禦參數 φ，內層在訓練用淨化算子集合 𝒫 上取**最壞情況**。

### 3.2 只換一個位置

維持弱 baseline 的四個原生位置（階段一官方 LoRA、階段二 dual-path、
latent L∞ 球 ε_a=0.4、L1 正規化動量 + sign），**只在 reward 外面包一層
`min over 𝒫`**。與 DEC-023「唯一替換是 reward」的紀律一致，可歸因。

**非加性參數化這一輪不動。** 換參數化（flow field 等）是第二個變因，
要在確認 min-max 本身買到 retention 之後才做。

### 3.3 內層用確定性最壞情況，不用隨機輪替

每步對 𝒫 中每個算子各求一次 reward，取最小者反傳。

**理由**：`DEFECTS.md` 記載「N2 的五個候選末端損失全距 0.9%，小於當步輪到
哪個淨化算子造成的差異」。隨機輪替會讓算子噪聲蓋過方法訊號。

**成本**：每個 iteration 由 1 次 trajectory 變 |𝒫| 次。取
`src/purify/ops.py::default_train_set()` 的三個算子（identity、blur σ=1.0、
jpeg q=75，該函式註解已寫明「訓練期的 𝒫，必須包含恆等算子」），
故階段二成本乘 3。N=10 迭代、每次 trajectory 11 步 UNet，可接受。

### 3.4 直通代理與 proxy_gap

`jpeg` 走 `straight_through`（前向真實、反向恆等），`blur` 與 `identity`
原生可微。訓練用 `Purifier.forward`、評測用 `Purifier.evaluate`，
每批必須輸出 `proxy_gap` 到 `calib/`，否則「代理夠好」是無憑據的宣稱。

### 3.5 失真幅度不受影響，這對比較有利

FND-028：latent 球從未生效（`ε_a = µ × N = 0.4`），失真幅度完全由
「步長 × 迭代數」決定、與 reward 無關。故 A 與弱 baseline 的失真量級相同，
`retention` 的比較是在**同失真**下做的，不需要額外的預算對齊。

## 4. 路線 B：cross-attention target injection reward

### 4.1 與已否證的注意力抑制不是同一件事

| | 舊（FND-024 已否證） | 新 |
|---|---|---|
| 對象 | attention **map**（probs）在遮罩內的 L1 總和 | attn2 模組的**輸出張量** |
| 動作 | 壓低——把質量拿走 | 對齊——把質量搬給另一個語意 |

**機制論證**：FND-024 第 4 條實測比例式損失時「質量確實被移出遮罩
（horse_00 −56%）而編輯照樣成功」。移出之後**沒有東西接管那個位置**，
去噪過程照 prompt 重新填回。injection 是讓目標語意接管。

外部依據：arXiv:2602.14679（2026-02）用 target injection ＋ source
suppression 雙損失，在單一 image-agnostic 擾動的嚴格約束下仍宣稱匹配
逐圖方法。該文是**加性**的，本專案取其機制、換非加性載體。

### 4.2 reward 形式

```
R_B = −‖A(z̄₀, c) − A(y_target, c)‖²  +  β · ‖A(z̄₀, c) − A(x, c)‖²
          ↑ injection：對齊到目標          ↑ suppression：推離原圖
```

- `A(·, c)`：attn2 模組輸出，**逐層算完再平均**，不跨解析度合併
  （沿用 `CrossAttentionRecorder` 的既有紀律——合併需重採樣，會抹掉
  「哪一個解析度的綁定被破壞」）
- `c`：階段二既有的 `emb_cond = sd.encode_text(class_name)`。這是物件類別名，
  **不是攻擊方的編輯 prompt**，符合 DESIGN §2 的 prompt-free 條件
- `β` 預設 1.0，需掃描
- 正規化沿用 DEC-021：除以第 0 次迭代的絕對值，`fidelity_lambda` 維持 10.0
- **成本**：`A(z̄₀, c)` 要在 trajectory 末端的 `z_0` 上多跑一次 UNet 前向
  （固定小 t），約為 11 步中的 1 步，overhead ≈ 9%。`A(y_tgt, c)` 因 y_tgt、
  t、c 全部固定，整輪只算一次並快取，成本可忽略。
  沿途在既有前向裡記到的是中間態 `z_t` 的注意力而不是 `z̄₀` 的，不能
  拿來省掉這一次前向

### 4.2a 正規化常數的一個實務風險（跑之前要看的第一個數字）

DEC-021 的正規化是「除以第 0 次迭代的絕對值」。`_targeted_reward` 恆為負，
其絕對值有意義；但 injection 的 `−inject + β·suppress` **可正可負**，若第 0 次
迭代恰好落在零附近，正規化常數會極小而使 reward 爆掉。`clamp_min(1e-12)` 只
防除以零，不防「很小」。

**處置**：每格的第一行 log 必須印出 `reward_raw` 與正規化常數。若 |R₀| 落在
1e-3 以下就停下來看，不要讓它跑完——症狀會表現成失真量級異常，而那與
FND-028 記的「失真由 µ×N 釘死」矛盾，容易被誤讀成別的問題。

### 4.3 目標影像：由灰圖換成具體概念

**現況**：`scripts/apa_baseline.py:53` 的 `TARGET_IMAGE = "data/targets/gray.png"`，
弱 baseline 的 targeted reward 推向一張 512² 純灰圖。

**問題**：灰圖在 SD 的條件分布裡不是一個點，而是接近退化的原點。推向它等於
推向「無內容」，去噪過程沒有競爭的吸引子，prompt 直接接管。這與 FND-024 §4.3
的機制記載一致（`x_def` 被當噪聲照 prompt 重畫），也與該筆第 4 條一致
（注意力質量被移出遮罩但沒有東西接管）。

**決定**（使用者 2026-08-13）：改用**名人人像**當注入目標。

| 項目 | 值 |
|---|---|
| 檔案 | `data/targets/obama.png`，512²、RGB |
| 來源 | Commons `File:President Barack Obama.jpg`，2687×3356 |
| 授權 | **Public domain**（Official White House Photo by Pete Souza） |
| 處理 | 800px 標準縮圖 → 中心裁切正方（side 800）→ Lanczos 縮到 512² |
| 來源紀錄 | `data/targets/provenance.json`（含 sha256 與授權欄位） |

**為什麼不是使用者原本說的 LeBron James**：其 Commons 照片為 **CC BY 2.0**
（Erik Drost），而專案規則只收 CC0／公有領域——姓名標示會在論文附圖與衍生
資料集上產生持續義務。取同性質的替代（名人人像 ＋ 公有領域）。

**為什麼不是 Einstein 那張公有領域人像**：黑白。`src/metrics/chroma.py` 是
本專案的保真指標之一，單色目標會系統性偏移色度讀數，使消融格 2 的比較出現
混淆。`File:Donald Trump official portrait.jpg` 同樣是公有領域彩色肖像，
要換只需改一個常數。

**殘留的代價，不得掩蓋**：公有領域只放棄著作權，不處理被攝者的肖像權。
本圖是方法內部的注入目標，不是評測資料集的成員，此區別要在論文寫明。

### 4.4 man_00 的注入是身分層而非類別層，必須分開報

`data/lo_aligned` 的三個來源是 horse／man／bird，而目標是**一個人**。故：

| 來源 | 位移的層級 |
|---|---|
| horse_00、bird_03 | **類別層**（馬／鳥 → 人） |
| man_00 | **身分層**（某人 → Obama） |

兩者不可平均成一個數字。arXiv:2602.14679 的 Ronaldo 注入正是身分層，兩種
都是有效的設定，但它們回答的是不同的問題，報表須分欄。

### 4.5 最小消融（否則無法歸因）

| 格 | reward | 隔離的是什麼 |
|---|---|---|
| 1 | 像素 targeted → `gray.png` | 現況弱 baseline（DEC-023） |
| 2 | 像素 targeted → `obama.png` | **只換目標圖**的效果 |
| 3 | attention injection，β = 0 | 換到 cross-attention 層的效果 |
| 4 | attention injection ＋ suppression，β > 0 | 完整雙重 loss |

**格 2 是關鍵**：若只換目標圖就有效，則不需要 attention loss，方法大幅簡化。
四格共用同一組其餘設定（階段一官方 LoRA、dual-path、latent 球、sign），
符合 DEC-023「唯一替換是 reward」的紀律。

### 4.6 實作狀態（2026-08-13 完成，892 passed / 1 xfailed）

| 位置 | 內容 |
|---|---|
| `src/models/attention.py` | `CrossAttentionOutputRecorder`：記 attn2 **輸出**（`register_forward_hook`）。既有記分佈的 `CrossAttentionRecorder` 未改動 |
| `src/defense/apa_native_stage2.py` | `REWARD_MODES`、`NativeStage2Config.{reward_mode, injection_beta, injection_t}`（`__post_init__` 驗證）、`_injection_reward`、`_attn_outputs`、`_reward` 分派 |
| `scripts/apa_baseline.py` | `build_parser()`（由 `main` 抽出以便測試）、`--target`／`--reward-mode`／`--injection-beta` |
| `tests/test_attention_injection.py` | 10 個測試 |

**`attn_ref` 的快取**：`y_target`、`injection_t`、`emb_cond` 全程固定，故在
`attack_native` 開頭算一次 `{"tgt", "src"}` 並 `detach`，逐迭代沿用。

**dual-path 的兩條線共用同一個 reward**（`_reward` 同時被 `_step_guidance` 與
trajectory 末端呼叫）。只換其中一條會讓「唯一替換是 reward」的歸因失效。

**跑消融四格**：

```
# 格 1（現況弱 baseline，預設值即是）
python scripts/apa_baseline.py --out runs/B1 --data data/lo_aligned     --images horse_00 man_00 bird_03 --conditions apa_weak
# 格 2：只換目標圖
... --target data/targets/obama.png
# 格 3：換到 attention，β=0
... --target data/targets/obama.png --reward-mode injection --injection-beta 0
# 格 4：完整雙重 loss
... --target data/targets/obama.png --reward-mode injection --injection-beta 1.0
```

### 4.7 原始的改動清單（已完成，保留以對照）

1. `src/models/attention.py`：新增記錄 attn2 **輸出**的 recorder。現有
   `CrossAttentionRecorder` 用 `register_forward_pre_hook` 記 probs，輸出要用
   `register_forward_hook`，是**新的一個類別**而非改既有的（既有的有 16／70
   層數驗證與 `enabled` 開關，那些邏輯要沿用）
2. `src/defense/apa_native_stage2.py`：`_targeted_reward` 旁新增
   `_injection_reward`，由 `NativeStage2Config` 的欄位選擇；`A(y_tgt, c)`
   在 `attack_native` 開頭算一次並快取
3. `scripts/apa_baseline.py`：`TARGET_IMAGE` 由常數改為 CLI 參數，預設維持
   `gray.png` 使既有結果可重現，消融格 2–4 由旗標指定

其餘四個位置維持原生。

## 5. 淨化算子集合

`reference/ROBUSTNESS_TESTS.md` §4 有完整對應表。本輪要新增三個：

| 算子 | 用途 | 依據 |
|---|---|---|
| `jpeg75_then_resize` | **評測**：C&R 串接 | ROBUSTNESS_TESTS §1。既有七個算子全部單獨施加，該文證明這會嚴重高估 robustness |
| `nappure_flow` | **評測**：針對非加性的白盒淨化 | ROBUSTNESS_TESTS §3。不納入的話「非加性更抗淨化」可被一句「沒測針對性淨化」推翻 |
| `pdm_pure` | **評測上界對手**，需 DeepFloyd-IF 權重 | ROBUSTNESS_TESTS §2 |

`pdm_pure` 需要 stock SD 以外的模型權重，嚴格說已離開「攻擊方使用 stock
Stable Diffusion」的前提。**處置**：跑得動就報並誠實承認結果；跑不動或
權重不可得則在論文中明寫排除理由。兩種情形都不得靜默略過。

訓練內層 𝒫 維持 `default_train_set()` 的三個，不加新算子——內層要便宜。

## 6. 文件與程式重整

使用者 2026-08-13 裁決：**舊主線程式移到 `legacy/` 並保留**，不刪除。

### 6.1 程式分層

| 層 | 內容 | 位置 |
|---|---|---|
| 主線 | `scripts/apa_baseline.py`、`src/defense/apa_native_stage2.py`、`src/residual/site_apa.py`、`site_weight.py`、`src/defense/optimize.py::align_apa_native` | 原地 |
| 支援 | `src/metrics/`、`src/models/`、`src/purify/`、`src/baselines/`（三個加性對照）、`src/utils/` | 原地 |
| 舊主線 | 見下方 §6.1a 的實測清單 | **`legacy/`** |

### 6.1a 依賴實測與兩次抽出（2026-08-13 執行）

用 AST 追 `scripts/apa_baseline.py` 的遞移依賴。**`from pkg import mod` 形式
必須把 `pkg.mod` 也當模組解析**，漏了這條會低估依賴、把主線需要的檔案搬走
（本次先前的盤點正是漏了它，得出「主線只碰 17 支」的錯誤結論）。

| 階段 | 主線遞移依賴 |
|---|---|
| 起點 | **45** |
| 抽出 `src/utils/io.py` 後 | 34 |
| 抽出 `src/defense/apa_stage1.py` 後 | **23** |

**抽出一：`src/utils/io.py`。** `apa_baseline` 只為 `load_image_tensor` 與
`write_csv` 匯入 `src.experiment.executors`，而 `executors` 拉進整個
`src/experiment/` 與 `objective.py`、`purify/`、`site_warp.py`。

**抽出二：`src/defense/apa_stage1.py`。** `align_apa_native` 只用 `sd` 與
`lora`，與 `optimize.py`（1570 行）其餘部分零耦合，但該檔匯入
`generator`／`objective`／`purify.ops`／`calibration`，一支帶進 16 個模組。

兩處都在原位置**重新匯出**而非複製實作——`executors.write_csv` 與
`optimize.align_apa_native` 這兩個名字仍被舊主線的批次使用，兩份實作會慢慢
分岔而沒有症狀（FND-027 就是拿 `align_apa_native` 量出來的）。

脫離主線的有：`src/experiment/`（6 支）、`src/purify/`（5 支）、
`src/defense/objective.py`、`generator.py`、`optimize.py`、`recon.py`、
`src/residual/site_warp.py`、`site_embedding.py`、`src/data/`（3 支）、
`src/metrics/` 的 4 支。

### 6.1b `src/` 不搬到 `legacy/`，理由是套件會撞名

`legacy/src/` 與 `src/` 同名，Python 只會載入 `sys.path` 上先出現的那一個。
這不是風格問題，是會壞掉。三個可行的形態：

| 形態 | 代價 |
|---|---|
| **A. `src/` 原地不動，用清單與模組標註分層**（建議） | 目錄看起來沒變，靠 `MAINLINE.md` 的 23 支清單與各檔頂端的狀態標註 |
| B. 非主線移到 `src/legacy/` 子套件 | 不撞名，但要改 36 個模組與 31 支 script 的 import；漏改的症狀是 ImportError（會報錯，不會靜默） |
| C. 整個 repo 拆成兩個套件 | 過大 |

`scripts/` 沒有這個問題（平坦、無套件語意），**31 支可直接搬到
`legacy/scripts/`**，只需把 `sys.path` 的 `parents[1]` 改成 `parents[2]`。
`tests/` 不搬：混在一起跑一次 `pytest` 就能同時保護主線與 legacy，拆開要
維護兩套設定，而測試正是 legacy 程式不腐爛的唯一保護。

### 6.1c 原始的預估（已被上表取代）

用 AST 追 `scripts/apa_baseline.py` 與主線測試的遞移依賴（**`from pkg import mod`
形式必須把 `pkg.mod` 也當模組解析**，漏了這條會低估依賴、把主線需要的檔案
搬走）：

| | 檔數 |
|---|---|
| 主線遞移依賴 | **45** |
| 可搬到 `legacy/` | **12** |

可搬的 12 支：`src/data/`（3）、`src/defense/linf_attack.py`、
`src/metrics/battery.py`、`src/metrics/spectrum.py`、`src/residual/site_embedding.py`，
以及五個 `__init__.py`（不可搬，套件結構）。**實際可搬的只有 7 支。**

**為什麼遠少於預估**：`apa_baseline.py` 匯入 `src.experiment.executors`
（載入影像、寫 CSV），而 `executors` 拉進整個 `src/experiment/` 與
`src/defense/objective.py`、`src/purify/`、`src/residual/site_warp.py`。
`src/baselines/` 的三個加性對照又拉進 `advpaint`／`promptflare`。

**兩個選項**：
1. **只搬那 7 支**——誠實但幾乎沒有效果，`src/` 看起來不會變乾淨
2. **先解依賴再搬**：把 `executors` 裡主線用到的三個函式
   （`load_image_tensor`、`write_csv`、`save_image` 附近）抽成
   `src/utils/io.py`，切斷 `apa_baseline → experiment` 這條邊，之後
   `src/experiment/` 整包可搬。這是一次真實的重構，需要測試保護

`scripts/` 那 32 支的搬動不受此限：其中 22 支與主線零交集，可直接搬。

`tests/` 目前是 32 支平坦檔案。搬動程式時對應的測試同步移到
`legacy/tests/`（`test_objective.py`、`test_executors.py`、`test_grid.py`、
`test_runner.py`、`test_warp_mask_gate.py` 等），主線 `tests/` 只留支援層與
弱 baseline 的測試。**搬動前後測試數必須相同且全綠**，不得因 collect 不到
而靜默減少——搬完先比對 `pytest --collect-only -q | tail -1` 的總數。

**例外**：`site_warp.py` 與 `src/purify/ops.py` 在路線 A 的後續階段
（換非加性參數化）可能取回，搬動前逐支確認。

### 6.2 `CLAUDE.md` 的「注入位置」章節整段重寫

現行那張 A 類／B 類、site L／E／W／P／PF／S 的表是舊主線遺物，與
「只認弱 baseline」直接衝突。改寫成以弱 baseline 為唯一起點的形式。
before／after 依 `CLAUDE.md` 的工作要求記錄具體行號與原貌。

### 6.3 docs 骨架

主線五份 ＋ 兩個目錄：

```
docs/
  START_HERE.txt     新 session 讀檔須知（併入原 INDEX.md 的編碼制度說明）
  MAINLINE.md        弱 baseline 的定義與最小程式集合
  PLAN.md            本檔：現在要做什麼
  FINDINGS.md        測得的事實（FND-）
  DECISIONS.md       裁決（DEC-）
  METRICS.md         指標定義與其陷阱（MET-）
  RUNBOOK.md         操作流程
  DEFECTS.md         犯過的錯（DEF-）
  reference/         外部論文的查證紀錄
  archive/           降級的逐次紀錄
```

`INDEX.md` **併入 `START_HERE.txt` 後移除**：它的功能是「哪一種編碼放在哪個
檔」，主線收斂到八份之後那張對照表只剩八列，獨立成檔反而多一次跳轉。
`CLAUDE.md` 現行寫著「先讀 `docs/INDEX.md`」，須同步改。
`EXP-`／`MTH-`／`DEF-` 三種編碼中，`MTH-` 隨 `METHODS.md` 一併降級，
`EXP-` 隨 `EXPERIMENTS.md` 降級；新批次的紀錄直接寫進 `FINDINGS.md`。

`DESIGN.md`、`ARCHITECTURE.md`、`METHODS.md`、`SYNTHESIS.md`、`EXPERIMENTS.md`、
兩份 `HANDOFF_*.md` 移入 `archive/`，其中仍生效的內容（DESIGN §2 威脅模型、
ARCHITECTURE §2.4 淨化介面）摘要進 `MAINLINE.md`。

## 7. 已知陷阱（動手前必讀）

1. **算子噪聲蓋過方法差異**：`DEFECTS.md` 記的 0.9% 全距。故 §3.3 用確定性
   最壞情況
2. **DISTS 在 512² 上先降採樣到 256²**（FND-026），加性與非加性的失真比較
   會翻轉。比絕對值時先確認量在哪個解析度
3. **latent 半徑不是保真約束**（FND-028），同一 ε_a 在不同影像上是不同的
   可見失真，且 LPIPS／DISTS／SSIM 三者排序都可能與人眼相反
4. **未防禦的編輯必須真的成功**（DEC-022），否則抗編輯那一欄的分母不成立。
   換資料集或 prompt 後先看 `edit_orig` 的圖
5. **階段二的 CFG 必須是 1.0、反演必須走淺噪聲帶**（50 格排程只執行前 11 格）。
   弄錯不會報錯，只會讓失真量級整個不同
6. **`cnn_denoise_substitute` 缺權重**（METRICS.md §7），七個算子中有一個
   從頭到尾沒有資料

## 8. 順序與驗收

| 步 | 內容 | 驗收 |
|---|---|---|
| 1 | 重整（§6） | `pytest` 維持全綠；`git status --porcelain --ignored` 確認無結果檔被排除 |
| 2 | **路線 B**（低成本探針） | `edit_lpips` 是否超過弱 baseline 的 targeted（FND-029 記兩圖平均 0.435）；`compare.html` 每格有圖，人眼為主判準 |
| 3 | 新增三個淨化算子（§5） | `proxy_gap` 有數字；`nappure_flow` 在 512² 上的參數重新定過而非照抄 32² |
| 4 | **路線 A** | `retention` 是否高於三個加性 baseline；同時檢查是否高於 `Ra`（§2.2） |

步 2 與步 3 互不相依，可並行。步 4 依賴步 3。
