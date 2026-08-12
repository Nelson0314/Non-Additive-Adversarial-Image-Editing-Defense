# 主線

2026-08-13 更新。判準與結論以 `FINDINGS.md`／`DECISIONS.md` 為準，現行工作
以 `PLAN.md` 為準。本頁只回答三個問題：**主線是什麼、程式在哪、已知什麼**。

---

## 1. 研究目標

白盒條件、外掛模組形態下，找出非加性方法，在匹配人眼可辨失真下勝過加性
基準。主張階層見 `PLAN.md` §1（2026-08-13 改為「非加性**更**抗淨化」為主、
防禦效果不輸為並列，主讀數是位移量，不再追求語意抵抗）。

判準以人眼為主、數值指標為輔。`compare.html` 是主要產出物，每一格都必須有
影像可看；指標與人眼矛盾時以人眼為準並記錄。

## 2. 內部弱 baseline（DEC-023）

「完全原生 APA，只把 reward 換成 targeted output」。五個位置裡四個維持原生：

| 位置 | 設定 | 出處 |
|---|---|---|
| 階段一 | APA 官方 LoRA：denoising MSE、AdamW 1e-4、200 步固定、rank=8、noise_offset=0.1 | 官方 `visual_alignment.py` |
| 階段二 | dual-path attack guidance（trajectory ＋ step-level） | 官方 `pipe_ours.py` |
| 約束 | latent L∞ 球 ε_a = 0.4 | 官方 Eq.7 |
| 更新 | L1 正規化動量 ＋ sign，µ=0.04、N=10 | 官方 Eq.7 |
| 反演 | DDIM，50 格排程只執行前 11 格、T_a=10、CFG=1 | 官方 APA-GC |
| **reward** | **`−‖D(z̄₀) − y_target‖²`** | PhotoGuard-c／Mist 形式 |

叫「弱」是因為它的語意抵抗接近零——**它是位置基準，不是有效的防禦**。

跑它：

```
python scripts/apa_baseline.py --out runs/<批次名> \
    --data data/lo_aligned --images horse_00 man_00 bird_03
```

不給 `--data` 時讀 `data/apa_native`（APA 官方那三張圖）。
`--conditions` 可只挑其中幾個（`apa_weak`／`photoguard_c`／`mist`／`dia_r`）。
路線 B 的四格消融見 `PLAN.md` §4.6。

## 3. 程式：主線的 23 支

`scripts/` 只有 `apa_baseline.py` 一支；其餘 33 支已移到 `legacy/scripts/`。
`src/` 的檔案**原地不動**（`legacy/src/` 會與 `src/` 撞名，Python 只會載入
`sys.path` 上先出現的那一個，見 `PLAN.md` §6.1b）。以下是
`scripts/apa_baseline.py` 的完整遞移依賴，用 AST 實測而非估計：

### 3.1 弱 baseline 自身

| 檔案 | 用途 |
|---|---|
| `scripts/apa_baseline.py` | **主驅動**：弱 baseline ＋ 三個加性 baseline，訓練與評測 |
| `src/defense/apa_stage1.py` | 階段一：官方 Eq.6 的 denoising MSE |
| `src/defense/apa_native_stage2.py` | 階段二：dual-path、latent 球、sign 更新、reward |
| `src/residual/site_apa.py` | 官方階段一的超參數常數 |
| `src/residual/site_weight.py` | LoRA 掛載（`WeightResidual`） |
| `src/residual/base.py`、`composite.py`、`lowrank.py`、`site_latent.py` | 上面兩支的相依 |

### 3.2 加性對照

`src/baselines/`：`pgd.py`（共用骨幹）、`photoguard.py`、`mist.py`、`dia.py`。
`__init__.py` 的 `REGISTRY` 需要五篇齊全，故 `advpaint.py`、`promptflare.py`
也在依賴內——那是完整清單的一部分，不是冗餘。

### 3.3 模型與指標

| 檔案 | 用途 |
|---|---|
| `src/models/sd.py` | SD 包裝、DDIM／BDIA 反演 |
| `src/models/attention.py` | cross-attention 擷取（分佈 ＋ 輸出兩個 recorder） |
| `src/metrics/suite.py`、`aesthetic.py`、`acutance.py` | 指標 |
| `src/utils/io.py` | 影像張量與 CSV 讀寫 |
| `src/utils/artifacts.py`、`device.py` | 存圖、裝置與精度 |

### 3.4 不在主線上的（原地保留，仍可用）

`src/experiment/`（五段流程 6 支）、`src/purify/`（5 支）、
`src/defense/objective.py`、`generator.py`、`optimize.py`、`recon.py`、
`linf_attack.py`、`src/residual/site_warp.py`、`site_embedding.py`、
`src/data/`（3 支）、`src/metrics/` 的 `battery`／`spectrum`／`chroma`／
`local_acutance`／`ray_scale`。

其中兩支在 `PLAN.md` 的路線 A 會被取回：**`src/purify/ops.py`**（`forward`／
`evaluate`／`proxy_gap` 三件式與 `default_train_set()` 已具備）與
**`src/residual/site_warp.py`**（換非加性參數化時）。

### 3.5 兩個重新匯出，不要複製實作

| 舊名字 | 現在指向 |
|---|---|
| `src/experiment/executors.py` 的 `write_csv`、`load_image_tensor` | `src/utils/io.py` |
| `src/defense/optimize.py` 的 `align_apa_native` | `src/defense/apa_stage1.py` |

舊主線的批次仍用舊名字。**兩份實作會慢慢分岔而沒有症狀**——CSV 的欄位規則
分岔會讓既有 `runs/` 不可比，而 FND-027 正是拿 `align_apa_native` 量出來的。

## 4. 資料

| 目錄 | 內容 |
|---|---|
| `data/apa_native/` | APA 官方三張圖 |
| `data/lo_aligned/` | 本專案的馬／人／鳥等六類真實照片 |
| `data/targets/` | `gray.png`（弱 baseline 的目標）、`obama.png`（路線 B 的注入目標）、`MIST.png`（Mist 用） |

`data/targets/provenance.json` 記錄來源與授權。注入目標只收 CC0／公有領域。

## 5. 已知什麼

- **反演與重建**：生成路徑有逐影像的重建下限（FND-001）；BDIA 精確反演讓該
  下限幾乎不再額外收費（FND-002）；LoRA 在 UNet 上碰不到 VAE 的誤差（FND-016）
- **階段一**：原生 LoRA 在 BDIA 管線上不只無效而是有害（FND-027）；換掉它買到
  的是保真度與時間，不是防禦（FND-023）
- **階段二**：latent 球從未生效（`ε_a` 恰等於 `µ×N`），且同一半徑在不同影像上
  是不同的可見失真，三個指標的排序都與人眼相反（FND-028）
- **四個軸的分工**：reward 決定位移量、更新規則決定保真度，**沒有任何一個軸讓
  語意抵抗成立**（FND-029）。原文自己的分類器 reward 與抗文字編輯無交集
  （FND-030）；注意力抑制與「編輯失敗」之間沒有因果關係（FND-024）
- **抗淨化**：非加性勝過三個加性 baseline，7/7 算子嚴格成立，且在 img2img 與
  inpainting 兩個威脅模型上獨立重現（FND-018／020）。**但優勢來自非加性參數化
  而非 Lo 式 (5)**——同 site 的隨機方向 `Ra` 處處等於或高於我方
- **量測陷阱**：DISTS 在 512² 上先降採樣到 256²，使加性與非加性的失真比較翻轉
  （FND-026）；`retention` 的分母在 `effect_siglip` 讀數下不可用，但在位移讀數
  下 1425 列有 1350 列可用（`PLAN.md` §2.1）

## 6. 現在在做什麼

見 `PLAN.md`。兩條路線：**A** 對淨化的 min-max、**B** cross-attention target
injection reward（程式已完成，待跑四格消融）。
