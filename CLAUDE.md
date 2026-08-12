# WACV — 白盒非加性抗文字編輯防禦

## 研究範圍

唯一必要目標（使用者於 2026-07-30 明確界定，先前的發散方向已作廢）：

> 在白盒條件（已知攻擊方使用 stock Stable Diffusion）、外掛模組形式下，
> 找出非加性方法，在**匹配人眼可辨失真**下，於**抵抗文字引導編輯**上
> 勝過加性基準。

明確不是重點：

- **low rank 完全不深究。**
- 架構不設限，只需維持外掛模組的形態。

要多做 paper survey；不要在會失敗的方向持續深究。

**2026-08-05 主張階層改版**（使用者定案，見 `docs/DESIGN.md` §1）：
原本「抗淨化只是附帶結果」的定位已改。現行階層為

| 層級 | 主張 |
|---|---|
| **主** | 非加性**抗淨化**勝過加性 |
| 次 | 抗編輯持平或小輸：同時滿足 (a) ≥ 0.85 × 最佳 baseline、(b) > 同失真高斯隨機對照 R |
| 三 | 保真受控，報全部指標不挑選 |

**判準以人眼為主、數值指標為輔**（同日定案）。`compare.html` 因此由附屬產物
升格為主要產出物，每一格都必須有影像可看；指標與人眼矛盾時以人眼為準並記錄。

## 注入位置

分兩類，差別在於是否經過 VAE 來回：

- **A 類／生成路徑**：site L（latent ε）、site E（文字嵌入）、site W（權重 LoRA）。
  全部經 `decode(... encode(x) ...)`，共用同一個重建誤差下限。BDIA 精確反演已實作
  並驗證（latent 來回誤差降 5 個數量級），消掉反演那一半後仍剩純 VAE 來回
  LPIPS 0.1434 / 27.51 dB。E17 顯示 latent 最佳化可把下限砍到 0.0760。
- **B 類／像素路徑**：site P（加性低秩）、site PF（加性全秩）、site S（空間變形）。
  無下限。

關鍵框定：那個下限是**我們選的 G** 造成的，不是威脅模型強加的。攻擊方用
stock SD；防禦方換掉自己 G 裡的 decode 完全合法。

## 主線與弱 baseline（2026-08-12 起）

**先讀 `docs/MAINLINE.md`。** 現行內部弱 baseline 是「完全原生 APA，只把
reward 換成 targeted output」（DEC-023）——四個位置（階段一 LoRA、dual-path
階段二、latent L∞ 球、sign 更新）維持原生，只換 reward。它的語意抵抗接近零，
是**位置基準不是有效防禦**。使用者正在尋找其他方法。

舊主線（注意力抑制損失、相對 DISTS 預算、投影約束、淨化與 inpainting 批次）
已降級到 `docs/archive/LEGACY_*.md`：仍然成立，但不再是現行判準來源。

## 程式位置

| 用途 | 路徑 |
|---|---|
| **主驅動（主線）** | **`scripts/apa_baseline.py`**（弱 baseline + 三個加性 baseline） |
| 階段二（主線） | `src/defense/apa_native_stage2.py` |
| 階段一（主線） | `src/defense/optimize.py::align_apa_native` |
| LoRA 掛載 | `src/residual/site_weight.py`、常數在 `site_apa.py` |
| 指標 | `src/metrics/suite.py`、`aesthetic.py` |
| baseline 攻擊 | `src/baselines/`（`pgd.py` 為共用骨幹，五篇各一檔） |
| 專案五段流程（保留） | `scripts/run_stage.py`、`src/experiment/` |
| 目標函數／優化（舊線仍在用） | `src/defense/objective.py`、`optimize.py` |
| 重建下限 | `src/defense/recon.py` |
| BDIA 精確反演 | `src/models/sd.py` 的 `bdia_inversion` |
| cross-attention 擷取 | `src/models/attention.py` |

已測過並否決的變體（注意力抑制／分類器 CE／latent／CLIP 四種 reward、
DISTS 進 loss 的軟約束、Adam 更新規則）與本輪四支一次性腳本已移除，結論在
FND-027…030。取回：`git checkout a4f93451f -- <path>`。

`src/defense/generator.py` **依模塊提供的能力分派，不比對 site 名稱**。
新增 site 時提供 `pixel_residual` 或 `eps_hook` 即可，不要在此加 `if site == ...`。

## 文件

先讀 `docs/INDEX.md`：它列出全部編碼（`EXP-`／`FND-`／`DEC-`／`DEF-`／`MTH-`／`MET-`）與各自在哪個檔案。每一筆都自足、可單獨讀完，編碼只用來互相指認，**不代表先後或依賴**。
結論與判準一律以 `docs/FINDINGS.md`、`docs/DECISIONS.md` 為準；`docs/archive/` 是逐次紀錄，不是判準來源。

## 環境

- 本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**，base 沒有 pytest）。
- 測試：`python -m pytest -q`，基準為 **629 passed / 1 skipped / 1 xfailed**。
  xfailed 是刻意釘住的 DIA-PT L1 起點缺陷（原始碼自身的問題，`strict=True`）。
- 遠端 TWCC 容器：host/port 每次重開都不同，由使用者提供；密碼與 GitHub token 同樣由使用者提供，**不得寫入任何入庫檔案**。
- 遠端持久儲存 `/work/nelson0314` 跨容器保留：conda env `wacv`、repo 在
  `/work/nelson0314/WACV`、`hf_cache` 5.9 G（SD v1.4 已下載）。
  每次執行前先 `source /work/nelson0314/WACV/env.sh`。
- 容器預裝的 NGC torch **不支援 V100 的 sm_70**，必須用 conda env 的 torch cu118。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔衝突而 abort。作法是先把它們
  `mv` 到 `/work/nelson0314/pull_backup/` 再 pull。

## 工作要求

- 一律用繁體中文回答，客觀學術語氣；程式碼關鍵字、函式名、套件名維持英文。
  **commit message 用英文。**
- 動手前先驗證假設（讀檔、跑指令），不要憑記憶猜 API。
- 修改論文方法要記 before/after：具體行號、原貌、原因。
- 架構或實驗設計需先提計劃討論再寫程式。環境問題直接修掉，不用寫進報告。
- 宣告完成前必須實際跑過並看到成功輸出；失敗就直說失敗。
- **未經明確授權不得把分支併入 main。**
- 禁止用 try/except 或條件跳過來掩蓋症狀，要找根本原因。

## 資料保全

`runs/` 是唯一的證據來源，TWCC 容器會被刪除，實驗無法重跑。所有 CSV / JSON /
log / PNG 一律入版控。`.gitignore` 的 `runs/` 區塊曾有一條 `runs/*/**` 讓 git
停止遞迴而靜默漏掉 273 個檔案（見 commit `1942e38`）；改動該區塊時必須用
`git status --porcelain --ignored` 確認沒有結果檔被排除。
