# 主線

2026-08-12 起的主線。判準與結論以 `FINDINGS.md`／`DECISIONS.md` 為準，
本頁只說明「主線是什麼、程式在哪、已知什麼」。

舊主線（注意力抑制損失、相對 DISTS 預算、投影約束、淨化與 inpainting 批次）
已降級為次要紀錄，見 `archive/LEGACY_*.md`。降級不代表作廢——那些結論仍然
成立，只是不再是現行工作的判準來源。

## 研究目標（未變）

白盒條件、外掛模組形態下，找出非加性方法，在匹配人眼可辨失真下，於抵抗
文字引導編輯上勝過加性基準。判準以人眼為主、數值指標為輔。

## 內部弱 baseline（DEC-023）

「完全原生 APA，只把 reward 換成 targeted output」。五個位置裡四個維持原生：

| 位置 | 設定 |
|---|---|
| 階段一 | APA 官方 LoRA（denoising MSE、AdamW 1e-4、200 步、rank=8） |
| 階段二 | dual-path attack guidance（trajectory + step-level） |
| 約束 | latent L∞ 球 ε_a = 0.4 |
| 更新 | L1 正規化動量 + sign，µ=0.04、N=10 |
| **reward** | **`−‖D(z̄_0) − y_target‖²`（唯一替換）** |

叫「弱」是因為它的語意抵抗接近零——它是**位置基準，不是有效的防禦**。

## 程式（最小集合）

| 檔案 | 用途 |
|---|---|
| `src/defense/apa_native_stage2.py` | 階段二：dual-path、latent 球、sign 更新、targeted reward |
| `src/defense/optimize.py::align_apa_native` | 階段一：官方 Eq.6 目標 |
| `src/residual/site_weight.py` | LoRA 掛載（`WeightResidual`） |
| `src/residual/site_apa.py` | 官方階段一的超參數常數 |
| `scripts/apa_baseline.py` | **主驅動**：弱 baseline + 三個加性 baseline，訓練與評測 |
| `src/metrics/suite.py`、`aesthetic.py` | 指標 |

資料：`data/apa_native/`（APA 官方三張圖）、`data/lo_aligned/`（本專案的
馬／人／鳥等六類）、`data/targets/`（targeted reward 與 Mist 的目標圖）。

專案既有的 `scripts/run_stage.py`／`src/experiment/` 五段流程與
`src/baselines/` 五篇實作**保留**——前者是主架構、後者是比較基準，不是冗餘。

## 已知什麼（主線相關）

- **反演與重建**：生成路徑有逐影像的重建下限（FND-001）；BDIA 精確反演讓
  該下限幾乎不再額外收費（FND-002），而 LoRA 在 UNet 上碰不到 VAE 的誤差
  （FND-016）
- **階段一**：原生 LoRA 在 BDIA 管線上不只無效而是有害（FND-027）；換掉它
  買到的是保真度與時間，不是防禦（FND-023）。但在真實照片那批上兩者實質
  打平（見 EXP-apa2 的 2×3×2 格點）
- **階段二**：latent 球從未生效（`ε_a` 恰等於 `µ×N`），且同一半徑在不同影像
  上是不同的可見失真，三個指標的排序都與人眼相反（FND-028）
- **四個軸的分工**：reward 決定位移量、更新規則決定保真度，**沒有任何一個軸
  讓語意抵抗成立**（FND-029）。原文自己的分類器 reward 與抗文字編輯無交集
  （FND-030）；注意力抑制與「編輯失敗」之間沒有因果關係（FND-024）
- **量測陷阱**：DISTS 在 512² 上會先降採樣到 256²，使加性與非加性的失真比較
  翻轉（FND-026）；分片目錄與 `_merged` 曾逐位元重複（FND-015）

## 下一步的空間

`FND-029` 把四個軸都試過了：**「把編輯推遠」做得到，「讓編輯不服從 prompt」
做不到**。已否證的包括提高失真預算（FND-024 第 1 條：τ=0.50 下抑制 89–94%
而編輯成功率 100%）、放大既有方向（FND-025）、換 reward（含與評測同族的
CLIP loss）、換約束、換更新規則、換階段一。

尚未試過的組合中，最有依據的是**把最會推離的 reward 與最保真的訓練方法
合起來**——targeted 只跟原生的球+sign 配過，從未配過 Adam 或影像空間投影。
