# WACV — 白盒非加性抗文字編輯防禦

在白盒條件（攻擊方使用 stock Stable Diffusion）、外掛模組形式下，尋找非加性
方法，在**匹配人眼可辨失真**下，於**抵抗文字引導編輯**上勝過加性基準。

主張分三層，順序即強度：

| 層 | 主張 | 現況（2026-08-05） |
|---|---|---|
| **主** | 非加性在**抗淨化**上勝過加性 | 待跑。先驗資料無法回答：以 `net_lpips` 計曾看似成立，換成 Δsiglip 後七組配對全部不成立（`HANDOFF.md` §3.2） |
| **次** | 抗編輯能力達最佳 baseline 的 0.85 倍以上，**且**高於同失真高斯隨機對照 R | 待跑。兩個條件須同時滿足（`LOGIC_CHECK` §C2） |
| **三** | 失真受控 | 射線縮放到同一 τ_LPIPS，並回報全部保真度指標（既知缺陷 A3：「匹配失真」曾被四度證偽） |

**截至 2026-08-04 的所有實驗一律視為先驗實驗**，程式與評測流程已整批重造。
`runs/` 下的 20 個目錄全部屬於先驗階段，新一輪批次尚未建立。

基準論文是 Lo, Yeo, Shuai, Cheng, *Distraction is All You Need*, CVPR 2024；
其第一作者是本專案的指導者，故其約束、判準與 baseline 為必要對齊項
（見 `ADVISOR.md`）。

---

## 從哪裡開始讀

| 你想知道 | 讀這個 |
|---|---|
| **哪份文件現行、哪個 run 屬於哪個實驗** | **`docs/INDEX.md`**（檔案索引，先讀這份） |
| **接手執行**：環境、實作進度、五段指令、失敗處理 | **`docs/RUNBOOK_2026-08-05.md`** |
| 要證什麼、怎麼算成立 | `docs/DESIGN_2026-08-05.md` |
| 為什麼選這些 baseline／這個威脅模型 | `docs/SURVEY_2026-08-05.md` |
| 程式怎麼組起來、續跑語意 | `docs/ARCH_2026-08-05.md`、`docs/CODE_2026-08-05.md` |
| 哪裡曾經錯過、為什麼現在的寫法是那樣 | `docs/LOGIC_CHECK_2026-08-05.md`（既知缺陷 A1–A13） |
| 某個 baseline 的參數為什麼是這個值 | `docs/SOURCE_AUDIT_2026-08-05.md` 與 `docs/_audit_*.md`（逐字原始碼佐證） |
| 先驗階段的結果、量測陷阱、參考文獻 | `HANDOFF.md` |
| 工作規範 | `CLAUDE.md` |

`8e0ffbc`（Reset the project to the minimal reproducible set）刪除了 3775 個檔案，
包含 `EXPLAINER.md`、`LEDGER.md`、`CONVERGENCE.md`、`NEXT_SESSION.md`、
逐實驗的 `RESULTS_*.md` 等。內容已由 `HANDOFF.md` 承接；要查原文用
`git show 8e0ffbc^:docs/<檔名>`。

---

## 程式在哪裡

| 用途 | 路徑 |
|---|---|
| 注入位置 | `src/residual/site_{latent,embedding,weight,warp,apa}.py` |
| 目標函數 | `src/defense/objective.py`（LPIPS 為綁定約束，`beta_linf` 可關） |
| 優化 | `src/defense/optimize.py`（`optimize` / `optimize_encoder` / `optimize_crossattn` / `align`） |
| cross-attention 擷取 | `src/models/attention.py` |
| baseline 攻擊 | `src/baselines/`（`pgd.py` 為共用骨幹，五篇各一檔） |
| 保真度指標 | `src/metrics/`（`local_acutance.py`、`chroma.py`、`suite.py`、`ray_scale.py`） |
| 淨化算子 | `src/purify/ops.py` |
| **主驅動** | **`scripts/run_stage.py`**（五段：calib／train／rayscale／eval／report） |
| 格點與續跑 | `src/experiment/`（`grid.py` 決定跑哪些格、`runner.py` 決定要不要跑、`executors.py` 實際計算） |
| 進度監察 | `scripts/dashboard.py` |

`scripts/run_defense.py`（先驗階段的驅動）與 site P／PF／color 三個注入位置
已於 2026-08-05 依 `ARCH` §2.3 刪除。`runs/` 下 20 個先驗批次由它產生，
要查當時的作法用 `git checkout 4d2332c -- scripts/run_defense.py`。

`src/defense/generator.py` **依模塊提供的能力分派，不比對 site 名稱**。
新增 site 時提供 `pixel_residual` 或 `eps_hook` 即可。

---

## 執行

```bash
# 測試（基準見下方；全部在 CPU 上跑，不需要 SD 權重）
python -m pytest -q

# 五段流程。stage 是**位置引數**；--gpu-tag 與 --precision 必填。
# --dry-run 在耗掉機時之前回答「會跑幾格」。
COMMON="--batch b1 --gpu-tag RTX-5090 --precision bf16"
python scripts/run_stage.py calib    $COMMON --mist-target data/targets/MIST.png
python scripts/run_stage.py train    $COMMON --mist-target data/targets/MIST.png
python scripts/run_stage.py rayscale $COMMON
python scripts/run_stage.py eval     $COMMON --mist-target data/targets/MIST.png
python scripts/run_stage.py report   $COMMON

# 進度監察（唯讀，不動 GPU）
python scripts/dashboard.py runs/b1 --json
```

**上機前先讀 `docs/RUNBOOK_2026-08-05.md`**——那份是自足的執行手冊，
含研究背景、五段內容、失敗處理、機時估計與判讀指南。
**GPU 工作不可並行**——CPU 密集工作（例如 pytest）與 GPU 工作並行時，
實測把單張 SDEdit 由 222 s 拉長到 30 分鐘以上。

本機直譯器是 `C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**）。
指令前加 `PYTHONIOENCODING=utf-8`——Windows 預設的 cp950 編不了 `²` 這類字元，
會在印出結果時才炸掉一支已經跑完的腳本。

**本機不要並行跑兩個 GPU 工作**（RTX 2050 只有 4 GB），也不要讓 CPU 密集工作
與 GPU 工作並行：實測後者會把 GPU 工作的 Python 執行緒餓住，單張耗時由 222 s
拉長到 30 分鐘以上。

---

## 資料保全

`runs/` 是唯一的證據來源，雲端容器會被刪除，實驗無法重跑。所有 CSV / JSON /
log / PNG / HTML 一律入版控。`.gitignore` 的 `runs/` 區塊曾有一條 `runs/*/**`
讓 git 停止遞迴而靜默漏掉 273 個檔案（見 commit `1942e38`）；改動該區塊時必須用
`git status --porcelain --ignored` 確認沒有結果檔被排除。

---

## 分支

目前在 `claude/e20-fidelity-constraint`，**未併入 main**。
未經明確授權不得併入。
