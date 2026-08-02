# 交接 prompt（2026-08-03，E31 進行中）

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。取代 2026-08-02 版（該版的用途是「重新決定方向」，方向已定） |
| **用途** | 以下橫線之後的內容可直接貼進新 session |
| **前一版** | `docs/archive/2026-08-01-HANDOFF_PROMPT.md`（更早的一版） |

---

WACV 專案接手 — 白盒非加性抗文字編輯防禦

## 這一輪要你做的事

執行 E31 的剩餘部分。方向已定，設計與計畫都寫好了，**不要重新發散**。
動雲端 GPU 之前先確認 gate 通過。

## 先讀這些（順序固定）

| # | 檔案 | 為什麼 |
|---|---|---|
| 1 | `docs/NEXT_SESSION.md` | 現況、進度、待辦、本機／雲端分工 |
| 2 | `docs/specs/2026-08-02-e31-positive-control.md` | E31 的設計依據。**§12 是最新的變更**（兩道次要門檻改為逐預算） |
| 3 | `docs/plans/2026-08-02-e31.md` | 逐步的作法，含每步的驗證方式 |
| 4 | `CLAUDE.md` | 研究範圍與工作規範 |

要查東西時用這三份索引，不要從頭讀所有文件：

- `docs/LEDGER.md` —— **主張**的索引。某個結論還算不算數、出處在哪、被什麼推翻。
- `docs/INDEX.md` —— **檔案**的索引。哪份文件現行、哪個 run 屬於哪個實驗。
- `docs/gallery.html` —— **人眼比對頁**的索引。某個門檻當初看哪張圖定出來的。

要文獻脈絡讀 `docs/SURVEY.md`（按問句組織，不是逐篇摘要）。

## 現況

E29 是修好攻擊端（E26 補上 classifier-free guidance）、判準（E25 改語意軸）、
校準（E27）與保真約束（E28）之後的第一次實測。結論是否定的：**在試過的每一個
運作點上，防禦都沒有阻止文字編輯達成 prompt**，加性與非加性皆然，且是在對防禦
最有利的雜訊條件下量的。

E31 因此不再比較兩臂——兩個零之間的比較沒有內容——改為在加性基準（site P）上
尋找**任何一個擋得下編輯的運作點**，為本專案從未驗證過的量測裝置建立正對照。
沿三個軸掃描：目標函數（`untargeted`／`targeted`／`crossattn:suppress`）×
τ_lpips（0.10／0.28）× strength（0.5／0.3），12 格 × 2 圖。

判準改為 ISR 式的聯集：**語意不符 prompt 或明顯的感知劣化**。E25 之後只取了
前半，而在高 strength 的全域編輯下前半幾乎不可達成。

## 已完成（全部在本機，零雲端成本）

| 項 | 產出 | 結果 |
|---|---|---|
| ISR 重判既有 run | `runs/p12_isr_rejudge/` | 828 格語意失敗 0 格。`edit_niqe_*` 自 E2 起就在 CSV 裡而從未被讀過 |
| 預算探針 | `runs/p13_budget_probe/` | **τ=0.28 在原約束集下不可達**，且與防禦方法無關——連 i.i.d. 白高斯雜訊都超標 |
| 逐預算門檻 | `runs/p14_budget_thresholds/` | 兩道次要門檻改為隨預算而定（規格 §12） |
| 劣化階梯 | `runs/p11_degrade_ladder/` | 待使用者判讀 `compare.html` |
| 強度掃描 | `runs/e31_strength_sweep/` | 既有防禦圖在 strength 0.2–0.5 下的響應（遷移設定，為下界） |
| 本機能力 | `runs/logs/e31_*_probe.log` | 無梯度 512² SDEdit 可跑（4873 MB、222.5 s）；**含梯度訓練不可跑**（256² 即 178 s/step） |
| 程式改動 | `optimize.py`、`suite.py`、`run_defense.py` | 三個 `defense_mode` 都可跑；RMS 與尖峰比例進 CSV；`--tau_acut` 可指定 |

測試基準：**276 passed / 1 skipped / 0 failed**。

## 待辦（照順序）

1. **使用者判讀** `runs/p11_degrade_ladder/compare.html`，定出感知劣化的門檻。
2. **R1**（雲端，約 20 分）：`TA_028=... TC_028=... bash scripts/drivers/e31_calibration.sh`。
   門檻的值取自 `runs/p14_budget_thresholds/thresholds.csv` 的 budget=0.28 那一列。
3. **Gate**：R1 的綁定者判定必須全部是 LPIPS hinge。不通過的處置見規格 §8，
   **不得以放寬約束草率繞過**——本專案已踩過四個假的綁定者，每一個都會讓整批
   網格變成無效資料。
4. **R2**（雲端，約 1.5–2 小時）：`LR_028=... TA_*=... TC_*=... bash scripts/drivers/e31_grid.sh`。
5. **判定與報告**（本機）：`scripts/e31_report.py --degrade_tau <值>`，寫 `docs/RESULTS_E31.md`。
   報告必須逐項對照規格 §9 事先寫下的四種預期否定結果，說明實際落在哪一種。

## 這個專案最重要的幾條經驗

- **「匹配失真」已經三次被證明是假的**（site S 買模糊、site C 買色調偏移）。
  每引入一個新參數化，先用等 LPIPS 多臂探針量出現行約束對它收不收費。
- **不得憑文獻聲譽選指標。** ΔE00、NLPD、VIF 都是被實測推翻的前例。
- **指標之間矛盾時，把影像做成比對頁給人眼判斷。** 兩個門檻與 E29 的否定結果
  都是這樣定或確認的。
- **綁定者診斷是常設步驟**（`scripts/e27_binding_check.py`）。
- **推翻自己先前的判斷時，把錯誤的假設與推翻它的資料一起留在文件裡**，
  不要改寫成正確版本。`docs/LEDGER.md` §7 就是這麼記的。
- **先量再說。** 本輪的兩個關鍵決定（τ=0.28 不可達、本機不能跑網格）都是
  零成本的探針量出來的，不是推論出來的。

## 環境

- 本機：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**），
  i5-12500H + RTX 2050 4 GB。指令前加 `PYTHONIOENCODING=utf-8`——cp950 編不了
  `²` 這類字元，會在印出結果時才炸。
- **本機不要並行跑兩個 GPU 工作**，也不要讓 CPU 密集工作與 GPU 工作並行
  （實測單張耗時由 222 s 拉長到 30 分鐘以上）。用 `scripts/drivers/local_night.sh`。
- 雲端：Lightning AI H100 80GB，環境準備用 `scripts/drivers/colab_setup.sh`
  （**不是** `remote_setup.sh`，後者會 `pip install torch` 而有換版風險）。
  換機器先跑 `scripts/colab_probe.py`。
- 連線資訊與 token 由使用者提供，**不得寫入任何入庫檔案**。

## 工作要求

- 一律用繁體中文、客觀學術語氣；程式碼關鍵字、函式名、套件名維持英文。
  **commit message 用英文。**
- 動手前先驗證假設（讀檔、跑指令），不要憑記憶猜 API。
- 修改論文方法要記 before/after：具體行號、原貌、原因。
- 架構或實驗設計先提計劃討論再寫程式。
- 宣告完成前必須實際跑過並看到成功輸出；失敗就說失敗。
- **未經明確授權不得併入 main**（目前在 `claude/e20-fidelity-constraint`）。
- 禁止用 try/except 或條件跳過掩蓋症狀，要找根本原因。
- `runs/` 是唯一的證據來源，所有 CSV / JSON / log / PNG / HTML 一律入版控。
  改動 `.gitignore` 的 `runs/` 區塊時必須用 `git status --porcelain --ignored` 確認。
