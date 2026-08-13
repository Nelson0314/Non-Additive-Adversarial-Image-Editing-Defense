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
- **位移量尚未達標**：對加性是 0.90–0.93×，而該比較用的 DISTS 軸對 site F 有利。
  FND-032
- **B 臂無效**：相位參數化搬進 APA 的 latent 上沒有收益，優勢是像素空間特有的。
  FND-032
- **指標與人眼**：DISTS 與 LPIPS 都低估相位擾動的可見失真，PSNR、GMSD、HaarPSI
  與人眼一致。GMSD 在同一個 DISTS 上判 site F 差 5.7 倍。FND-034
- **反演與重建**：BDIA 精確反演使生成路徑幾乎不再額外收費（FND-002）；原生
  階段一 LoRA 在 BDIA 管線上有害（FND-027）
- **量測陷阱**：DISTS 在 512² 上先降採樣到 256²（FND-026）；latent 半徑不是保真
  約束（FND-028）；`retention` 的分母塌陷時不可解讀（METRICS.md §6）

## 7. 現在在做什麼

等使用者在失真掃描頁上劃定人眼門檻，之後：

1. 把 `fit_to_budget` 的 `distortion_fn` 換成該門檻對應的指標
2. 對 θ 加空間平滑（粗網格上採樣，由構造保證），壓掉塊狀斑駁
3. **對 site F 做多種淨化方式與串接的測試**（協定見 `reference/ROBUSTNESS_TESTS.md`）
4. 把 **DCT-Shield（ICCV 2025，arXiv:2504.17894）** 加進 baseline 清單——同場景、
   同主張（更少視覺瑕疵、抗 JPEG），必須正面比較
