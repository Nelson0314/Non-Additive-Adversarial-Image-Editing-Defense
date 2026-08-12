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

**2026-08-13 主張階層改版**（使用者定案，判定式見 `docs/PLAN.md` §1–2）：

| 層級 | 主張 | 讀數 |
|---|---|---|
| **主** | 非加性**更**抗淨化：淨化後的效果**衰減率**低於加性 baseline | `retention` |
| **並列** | 防禦效果本身不輸：未淨化時的位移量不低於加性 baseline | `effect(·, identity)` |
| 三 | 保真受控，報全部指標不挑選 | LPIPS／DISTS／PSNR／SSIM／NIQE／銳利度 |

**不再追求語意抵抗。** CLIP-T 對齊掉幅仍照報，但不作為成敗判準——四個軸全部
否證（FND-024／029／030），且 arXiv:2506.04394（ICIP 2025）獨立測到同一現象。
**主讀數是位移量。**

`docs/archive/DESIGN.md` §1 的 2026-08-05 版階層（次要判準為「≥ 0.85 × 最佳 baseline」）
已被上表取代。對照組 R 的定義同時修正為**同失真的加性隨機**，見 `PLAN.md` §2.2。

**判準以人眼為主、數值指標為輔。** `compare.html` 是主要產出物，每一格都必須
有影像可看；指標與人眼矛盾時以人眼為準並記錄。

## 注入位置

**現行只有一個：APA 的 latent 擾動（site apa）。** 弱 baseline 在階段一把
LoRA 掛上 UNet、階段二在 latent 上做 dual-path guidance，其餘位置都不使用。

早期探索過六個位置（A 類經 VAE 來回：latent ε／文字嵌入／權重 LoRA；
B 類走像素路徑：加性低秩／加性全秩／空間變形），結論留在
`docs/archive/LEGACY_findings.md`。其中兩件事仍然有效：

1. **生成路徑有逐影像的重建下限**（FND-001），BDIA 精確反演讓它幾乎不再
   額外收費（FND-002）。那個下限是**我們選的 G** 造成的，不是威脅模型強加
   的——攻擊方用 stock SD，防禦方換掉自己 G 裡的 decode 完全合法。
2. **空間變形（`src/residual/site_warp.py`）在路線 A 的第二階段可能取回**
   （`docs/PLAN.md` §3.2），屆時它是第二個變因，不與 min-max 同時改。

`src/defense/generator.py` 依模塊提供的能力分派，**不比對 site 名稱**。
新增位置時提供 `pixel_residual` 或 `eps_hook` 即可，不要在此加
`if site == ...`。該檔目前不在主線依賴內。

## 主線與弱 baseline（2026-08-12 起）

**先讀 `docs/MAINLINE.md`。** 現行內部弱 baseline 是「完全原生 APA，只把
reward 換成 targeted output」（DEC-023）——四個位置（階段一 LoRA、dual-path
階段二、latent L∞ 球、sign 更新）維持原生，只換 reward。它的語意抵抗接近零，
是**位置基準不是有效防禦**。使用者正在尋找其他方法。

舊主線（注意力抑制損失、相對 DISTS 預算、投影約束、淨化與 inpainting 批次）
已降級到 `docs/archive/LEGACY_*.md`：仍然成立，但不再是現行判準來源。

## 程式位置

**完整清單見 `docs/MAINLINE.md` §3**（用 AST 實測的 23 支遞移依賴，不是估計）。
此處只列最常用的六個：

| 用途 | 路徑 |
|---|---|
| **主驅動** | **`scripts/apa_baseline.py`**（`scripts/` 只有這一支） |
| 階段一 | `src/defense/apa_stage1.py::align_apa_native` |
| 階段二 | `src/defense/apa_native_stage2.py` |
| LoRA 掛載 | `src/residual/site_weight.py`，常數在 `site_apa.py` |
| 指標 | `src/metrics/suite.py`、`aesthetic.py` |
| cross-attention 擷取 | `src/models/attention.py`（分佈與輸出各一個 recorder） |

**三層結構**：

- **主線**：上表 ＋ `src/baselines/`（加性對照）、`src/models/sd.py`、
  `src/utils/`。共 23 支
- **原地保留、不在主線依賴內**：`src/experiment/`、`src/purify/`、
  `src/defense/objective.py`／`generator.py`／`optimize.py`／`recon.py`、
  `src/residual/site_warp.py`／`site_embedding.py`、`src/data/`。
  **不搬到 `legacy/`**：`legacy/src/` 會與 `src/` 撞名，Python 只會載入
  `sys.path` 上先出現的那一個
- **`legacy/scripts/`**：舊主線的 33 支腳本（含五段流程的 `run_stage.py`）。
  平坦目錄無套件語意，故可以搬

**兩個重新匯出，不要複製實作**：`executors.write_csv`／`load_image_tensor`
指向 `src/utils/io.py`；`optimize.align_apa_native` 指向
`src/defense/apa_stage1.py`。兩份實作會慢慢分岔而沒有症狀。

已測過並否決的變體（注意力抑制／分類器 CE／latent／CLIP 四種 reward、
DISTS 進 loss 的軟約束、Adam 更新規則）已移除，結論在 FND-027…030。
取回：`git checkout a4f93451f -- <path>`。

## 文件

新 session 先讀 `docs/START_HERE.txt`，接著三份：

| 檔 | 回答什麼 |
|---|---|
| `docs/MAINLINE.md` | 主線是什麼、程式在哪、已知什麼 |
| `docs/PLAN.md` | **現在要做什麼、為什麼、怎麼判成功** |
| `docs/FINDINGS.md`／`DECISIONS.md` | 測得的事實與裁決，**判準一律以這兩份為準** |

外部論文的查證紀錄在 `docs/reference/`（含 `ROBUSTNESS_TESTS.md`：三份抗淨化
檢定協定的精確設定）。`docs/archive/` 是降級的逐次紀錄，不是判準來源。
編碼（`FND-`／`DEC-`／`MET-`／`DEF-`）每一筆自足、可單獨讀完，只用來互相
指認，**不代表先後或依賴**。

## 環境

- 本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**，base 沒有 pytest）。
- 測試：`python -m pytest -q`，基準為 **900 passed / 1 xfailed**（2026-08-13）。
  xfailed 是刻意釘住的 DIA-PT L1 起點缺陷（原始碼自身的問題，`strict=True`）。
- **GPU 工作一律在 NYCU BASIC lab 跑**（兩台各 8 張 RTX 3090，home 目錄跨機同步）：

      ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-1
      ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-2
      source ~/env.sh        # PATH／venv／HF_HOME，並 cd 到 repo

  repo 在 `/nfs/home/nelson0314/WACV-s3`。金鑰認證已設好，**密碼與 token
  不得寫入任何入庫檔案**。卡是多人共用，跑之前先看 `nvidia-smi`。
- 本機 RTX 2050 4GB 跑不動本專案的 GPU 工作，只用於寫程式、跑 pytest、看報表。
- 遠端也用 sparse-checkout。`git pull` 之後若某個頂層目錄沒出現，
  先 `git sparse-checkout add <目錄>`。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔衝突而 abort，先把它們 `mv` 到
  暫存目錄再 pull。

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

`runs/` 是唯一的證據來源，遠端機器不保證保留，實驗無法重跑。所有 CSV / JSON /
log / PNG 一律入版控。`.gitignore` 的 `runs/` 區塊曾有一條 `runs/*/**` 讓 git
停止遞迴而靜默漏掉 273 個檔案（見 commit `1942e38`）；改動該區塊時必須用
`git status --porcelain --ignored` 確認沒有結果檔被排除。
