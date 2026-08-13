# 主線

2026-08-13 改版。判準與結論以 `FINDINGS.md`／`DECISIONS.md` 為準，本頁只回答
三個問題：**主線是什麼、程式在哪、已知什麼**。

---

## 1. 研究目標

白盒條件（攻擊方使用 stock Stable Diffusion）、外掛模組形態下，找出非加性方法，
在匹配人眼可辨失真下勝過加性基準。**不再追求語意抵抗**（FND-024／029／030
四個軸全部否證），主讀數是位移量與抗淨化的衰減率。

判準以人眼為主、數值指標為輔。`compare.html` 是主要產出物，每一格都必須有影像
可看；指標與人眼矛盾時以人眼為準並記錄。

## 2. 三個條件，沒有第四個

2026-08-13 使用者裁定方向收斂。**現行只保留三類條件**，其餘歷史變體的程式與
文字紀錄已刪除（`runs/` 全部保留，是唯一的證據來源）：

| 條件 | 是什麼 | 程式 |
|---|---|---|
| **弱 baseline** | 完全原生 APA，只把 reward 換成 targeted output（DEC-023） | `scripts/apa_baseline.py --conditions apa_weak` |
| **強 baseline** | 三個已發表的加性方法：`photoguard_c`／`mist`／`dia_r` | 同上，`src/baselines/` |
| **site F** | **紋理重相位**，本專案的正式方向 | `src/residual/site_phase.py` |

`add`（加性 δ 走同一個 encoder-targeted 損失）與 `phase_rand`（同失真隨機相位，
即 RPN）是 site F 消融的兩個**內部對照**，不是獨立條件。

## 3. 程式：六支腳本、35 支 src

`scripts/`：

| 腳本 | 用途 |
|---|---|
| `apa_baseline.py` | 弱 baseline ＋ 三個強 baseline 的訓練與評測；`--conditions apa_phase` 走 B 臂 |
| `phase_ablation.py` | A 臂參數化消融：`add`／`phase`／`phase_rand`，對齊預算 |
| `phase_distortion_sweep.py` | 失真掃描，供人眼定門檻。十六項指標，不跑編輯評測 |
| `phase_retention.py` | 抗淨化的 retention，跑在已存的防禦圖上，不重跑攻擊 |
| `phase_gate_probe.py` | 紋理閘有效面積與 θ_max 校準（CPU，不載入 SD） |
| `phase_compare_page.py` | `compare.html` 產生器 |

`src/` 的 35 支：

| 目錄 | 內容 |
|---|---|
| `residual/` | `site_phase.py`（site F）、`site_weight.py`／`site_apa.py`（APA 階段一 LoRA）、`site_latent.py`、`base.py`／`composite.py`／`lowrank.py` |
| `defense/` | `apa_stage1.py`、`apa_native_stage2.py`（含 B 臂的相位分支）、`param_pgd.py` |
| `baselines/` | `pgd.py` 骨幹 ＋ 五篇（`REGISTRY` 需要五篇齊全）＋ `encoder_target.py` |
| `models/` | `sd.py`（SD 包裝、DDIM／BDIA 反演） |
| `metrics/` | `suite.py`、`aesthetic.py`、`acutance.py` |
| `purify/` | `ops.py`（含 C&R 串接 `jpeg_then_resize`）、`diffpure.py`、`impress.py`、`adverse_cleaner.py` |
| `utils/` | `io.py`、`artifacts.py`、`device.py` |

**沒有 `legacy/`、沒有 `docs/archive/`、沒有 `src/experiment/`。** 舊主線的
33 支腳本、24 支 src 模組、21 支測試與逐次紀錄已於 2026-08-13 刪除。
取回：`git checkout 6bb656280 -- <path>`。

## 4. site F 的構造

```
x_def = OLA( irfft2( rfft2(w·P_b) · exp(i·g_b·m_ω·θ_b) )·w ) / OLA(w²)
```

32×32 區塊、hop 16、Hann 窗，512² 上共 1089 個區塊 × 32×17 個頻格，
約 59 萬個參數（加性 δ 為 78.6 萬）。三個由構造保證的性質：

1. `θ = 0` 時輸出**逐位元等於原圖**，由 `OLA(w²)` 正規化保證，不依賴 COLA。
2. **幅度譜逐位保留**，係數乘上模長為 1 的複數而非拆 `abs`／`angle`。
3. 輸出為實數；`fx = 0` 與 `fx = N/2` 兩行的閘取 0，避免破壞共軛對稱。

`g_b` 是結構張量 coherence 導出的紋理度閘（邊緣與平坦區皆為 0），`m_ω` 是徑向
頻率閘。兩者由原圖算一次即固定，**不參與最佳化**。

定案的三個參數：`block = 32`、`r_min = 0.12`、紋理閘分位數 `0.5`。理由見
`docs/superpowers/specs/2026-08-13-texture-rephasing-design.md` 與 FND-032。

## 5. 資料

| 目錄 | 內容 |
|---|---|
| `data/apa_native/` | APA 官方三張圖 |
| `data/lo_aligned/` | 本專案的馬／人／鳥／貓／狗／女六類真實照片 |
| `data/targets/` | `gray.png`（targeted 目標）、`obama.png`、`MIST.png` |

## 6. 已知什麼

- **site F 的主讀數成立**：retention 在 10 個淨化算子上勝過加性 9/10（DISTS 0.04）
  與 8/10（0.075），含最接近真實的 C&R 串接。FND-033
- **site F 勝過同失真隨機 12/12**，倍率 2.6–2.7×。本專案第一次（FND-004 打平、
  FND-018 落後）。FND-032
- **位移量在人眼門檻上達標**：θ=1.30 對 ε=1.2/255 是 **1.55×**、逐圖 22/24。
  FND-032 的 0.90–0.93× 是 DISTS 對齊軸下的悲觀下界。FND-035
- **B 臂無效**：相位參數化搬進 APA 的 latent 上沒有收益，優勢是像素空間特有的。
  FND-032
- **沒有共用的預算軸**：兩個家族的人眼門檻在十六項指標上沒有一項落在同一個值
  （DISTS 差 4.5 倍、最接近的 LPIPS 也差 1.28 倍）。每個條件跑在自己的可接受
  上限上，才是「匹配人眼可辨失真」的字面意思。FND-035
- **反演與重建**：BDIA 精確反演使生成路徑幾乎不再額外收費（FND-002）；原生
  階段一 LoRA 在 BDIA 管線上有害（FND-027）
- **量測陷阱**：DISTS 在 512² 上先降採樣到 256²（FND-026）；latent 半徑不是保真
  約束（FND-028）；`retention` 的分母塌陷時不可解讀（METRICS.md §6）

## 7. 現在在做什麼

`runs/hb5` 批次已備妥，待使用者下令開跑。五張圖、五個類別
（`man_02`／`woman_02`／`dog_03`／`horse_03`／`cat_01`），七個條件：

| 組 | 條件 | 預算 |
|---|---|---|
| A 臂 | `phase`／`add`／`phase_rand` | 人眼門檻 θ=1.30、ε=1.2/255（FND-035） |
| 弱 baseline | `apa_weak` | 原生 ε_a = 0.4 |
| 強 baseline | `photoguard_c`／`mist`／`dia_r` | 各自論文的原生預算 |

使用者 2026-08-14 裁定 baseline **不做預算對齊**。加性跑在原生 16/255，是
人眼門檻的 13 倍，故該欄只能寫成「原生預算下的參考點」——匹配人眼失真那一
軸由 A 臂三條件承擔。

階段一（攻擊＋抗編輯）與階段二（retention）的分片與相依見
`scripts/run_hb5.sh`；平行分片的併批用 `scripts/merge_runs.py`。

之後：

1. 對 θ 加空間平滑（粗網格上採樣，由構造保證），壓掉塊狀斑駁
2. 補一次雙盲 2AFC 使用者研究，把 FND-035 的門檻從單一評分者升上去
3. 把 **DCT-Shield（ICCV 2025，arXiv:2504.17894）** 加進 baseline 清單——同場景、
   同主張（更少視覺瑕疵、抗 JPEG），必須正面比較
