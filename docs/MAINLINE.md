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
| **紋理重相位** | **把影像切塊、只轉傅立葉相位**，本專案的正式方向 | `src/residual/texture_rephase.py` |

`add`（加性 δ 走同一個 encoder-targeted 損失）與 `phase_rand`（同失真隨機相位，
即 RPN）是紋理重相位消融的兩個**內部對照**，不是獨立條件。

## 3. 程式：十支腳本

`scripts/`：

| 腳本 | 用途 |
|---|---|
| `apa_baseline.py` | 弱 baseline ＋ 三個強 baseline 的訓練與評測 |
| `phase_ablation.py` | 像素臂參數化消融：`add`／`phase`／`phase_rand`。`--human-threshold` 走人眼門檻半徑，否則對齊 DISTS |
| `phase_distortion_sweep.py` | 失真掃描，供人眼定門檻。十六項指標，不跑編輯評測 |
| `phase_retention.py` | 抗淨化的 retention，跑在已存的防禦圖上，不重跑攻擊 |
| `phase_gate_probe.py` | 紋理閘有效面積與 θ_max 校準（CPU，不載入 SD） |
| `phase_compare_page.py` | `compare.html` 產生器 |
| `merge_runs.py` | 併平行分片。**平行必須分片**：`write_csv` 每次整份覆寫，兩個行程寫同一個目錄會互相蓋掉 |
| `run_hb5.sh` | hb5 批次的具名工作表與分派 |
| `hb5_arch_assets.py` | 紋理重相位架構圖的每一張中間張量，由真實前向路徑算出 |
| `hb5_purify_gallery.py`／`hb5_report.py` | 淨化圖庫與報告產生器 |

`src/`：

| 目錄 | 內容 |
|---|---|
| `residual/` | `texture_rephase.py`（紋理重相位）、`lora_weights.py`／`apa_port.py`（APA 階段一 LoRA）、`latent_inject.py`、`base.py`／`composite.py`／`lowrank.py` |
| `defense/` | `apa_stage1.py`、`apa_native_stage2.py`、`param_pgd.py` |
| `baselines/` | `pgd.py` 骨幹 ＋ 五篇（`REGISTRY` 需要五篇齊全）＋ `encoder_target.py` |
| `models/` | `sd.py`（SD 包裝、DDIM／BDIA 反演） |
| `metrics/` | `suite.py`、`aesthetic.py`、`acutance.py` |
| `purify/` | `ops.py`（含 C&R 串接 `jpeg_then_resize`）、`diffpure.py`、`impress.py`、`adverse_cleaner.py` |
| `utils/` | `io.py`、`artifacts.py`、`device.py` |

**沒有 `legacy/`、沒有 `docs/archive/`、沒有 `src/experiment/`。** 舊主線的
33 支腳本、24 支 src 模組、21 支測試與逐次紀錄已於 2026-08-13 刪除。
取回：`git checkout 6bb656280 -- <path>`。

## 4. 紋理重相位的構造

> **完整紀錄見 `PHASE_METHOD.md`**——自足、可獨立讀完，含被推翻過的說法。


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
| `data/lo_aligned/` | 本專案的馬／人／鳥／貓／狗／女六類真實照片，各 4 張 |
| `data/targets/` | `gray.png`（targeted 目標）、`obama.png`、`MIST.png` |

`runs/` 的相位批次：

| 批次 | 內容 |
|---|---|
| `phase_sweep` | 失真掃描 108 格，人眼門檻由此劃定 |
| `phaseA_full` | 像素臂 DISTS 對齊，6 張 × 3 條件 × 2 預算 ＋ retention 360 列 |
| `phaseA_human` | 像素臂人眼門檻，**24 張** × 3 條件（FND-035 的來源） |
| `phaseB` | latent 臂（相位搬進 latent），已否決 |
| `hb5`／`hb5_pgc` | 人眼門檻 vs 原生預算的七條件對照 ＋ 40 格 retention（FND-036／037） |

## 6. 已知什麼

- **紋理重相位的主讀數成立**：retention 在 10 個淨化算子上勝過加性 9/10（DISTS 0.04）
  與 8/10（0.075），含最接近真實的 C&R 串接。FND-033
- **紋理重相位勝過同失真隨機 12/12**，倍率 2.6–2.7×。本專案第一次（FND-004 打平、
  FND-018 落後）。FND-032
- **位移量在人眼門檻上達標**：θ=1.30 對 ε=1.2/255 是 **1.55×**、逐圖 22/24。
  FND-032 的 0.90–0.93× 是 DISTS 對齊軸下的悲觀下界。FND-035
- **latent 臂無效**：相位參數化搬進 APA 的 latent 上沒有收益，優勢是像素空間特有的。
  FND-032
- **沒有共用的預算軸**：兩個家族的人眼門檻在十六項指標上沒有一項落在同一個值
  （DISTS 差 4.5 倍、最接近的 LPIPS 也差 1.28 倍）。每個條件跑在自己的可接受
  上限上，才是「匹配人眼可辨失真」的字面意思。FND-035
- **反演與重建**：BDIA 精確反演使生成路徑幾乎不再額外收費（FND-002）；原生
  階段一 LoRA 在 BDIA 管線上有害（FND-027）
- **量測陷阱**：DISTS 在 512² 上先降採樣到 256²（FND-026）；latent 半徑不是保真
  約束（FND-028）；`retention` 的分母塌陷時不可解讀（METRICS.md §6）

## 7. 現在在做什麼

`runs/hb5` 是最後一個完整批次（2026-08-14）：五個類別各一張真實照片、七個
條件、十個淨化算子。像素臂三條件走使用者裁定的人眼門檻（θ=1.30、ε=1.2/255），
四個 baseline 走各自論文的原生預算。

**inpainting 方向已於 2026-08-15 完全放棄**，相關程式、資料、遮罩、批次與
紀錄全部刪除。威脅模型只有 img2img（SDEdit strength 0.55）一種。

接下來，依優先序：

1. **跑空白地板控制組** `effect(purify(原圖))`——把未防禦的原圖淨化後再編輯，
   量算子自己的位移地板。沒有它，主張一沒有可判定的讀數（FND-037 的
   r = −0.83）。成本約 55 分鐘一張卡，**最高優先**
2. 人眼裁定 `photoguard_c` 的失真是否可接受（FND-036 的 0.99× 打平）
3. 對 θ 加空間平滑，順帶解決固定 θ 不等於固定失真（PSNR 逐圖 23.2–30.6）
4. 補雙盲 2AFC，把 FND-035 的門檻從單一評分者升上去
5. 把 **DCT-Shield（ICCV 2025，arXiv:2504.17894）** 加進 baseline 清單
