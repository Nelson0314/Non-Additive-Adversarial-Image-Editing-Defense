# WACV — 白盒非加性抗文字編輯防禦

在白盒條件（攻擊方使用 stock Stable Diffusion）、外掛模組形式下，尋找非加性
方法，在**匹配人眼可辨失真**下，於**抵抗文字引導編輯**上勝過加性基準。

論證分兩層，順序不可顛倒：

| 層 | 要回答的 | 現況（2026-08-04） |
|---|---|---|
| **第一層：重現** | 在基準論文的協定（L∞ ≤ 0.06、N = 100、Table 1 判準）上，我們的加性實作是否達到他報的水準？ | **兩個 PhotoGuard 變體 已達到或超過**（`runs/lo_baseline/`，24 張 × 20 種子）。`semantic` 未重現，原因是協定只跑了一半，已決定不追 |
| **第二層：貢獻** | 非加性在匹配人眼可辨失真下能否勝過該基準？ | **尚未成立。** 加性位置只有兩格可用，非加性位置零格可用——該條件跑在其校準學習率的 1/12.5，三道約束一次都沒啟動 |

基準論文是 Lo, Yeo, Shuai, Cheng, *Distraction is All You Need*, CVPR 2024；
其第一作者是本專案的指導者，故其約束、判準與 baseline 為必要對齊項。

---

## 從哪裡開始讀

| 你想知道 | 讀這個 |
|---|---|
| **從頭到尾是怎麼回事**（含架構圖與真實影像） | **`docs/EXPLAINER.md`** |
| **對外進度報告**（可直接寄出的單一 HTML） | **`docs/REPORT_2026-08-04.html`** |
| 現在在做什麼、下一步是什麼、接手要貼的 prompt | `docs/NEXT_SESSION.md` |
| 某個結論還算不算數、出處在哪 | `docs/LEDGER.md`（**主張**的索引） |
| 哪份文件現行、哪個 run 屬於哪個實驗 | `docs/INDEX.md`（**檔案**的索引） |
| 某個門檻當初看哪張圖定出來的 | `docs/gallery.html`；最新的是 `runs/figs/compare.html` |
| 文獻脈絡 | `docs/SURVEY.md`（按問句組織，不是逐篇摘要） |
| 現階段的實驗設計與作法 | `docs/specs/2026-08-03-lo-aligned-protocol.md`（§6 已由 §6.1 取代） |
| 為什麼主題會發散、怎麼收斂 | `docs/CONVERGENCE.md` |
| 逐個實驗的承接紀錄 | `docs/RESULTS_E13-E23.md`、`docs/RESULTS_E25-E31.md` |
| 工作規範 | `CLAUDE.md` |

`docs/archive/2026-08-02-e31-positive-control.md` 與 `docs/archive/2026-08-02-e31-plan.md`
的**設計部分已作廢**（12 格網格由 L1 取代），其量測結果仍有效，2026-08-04 移入封存。

每一份承接紀錄的每一節下都有同一塊狀態表（狀態／日期／硬體／承接／後續／資料）。
`grep -l STATUS-BLOCK docs/*.md` 可列出全部。九份逐實驗的承接文件於 2026-08-04
合併為上表的兩份，原文一字未改，引用格式（例如「E20 §5.2」）不變。

---

## 程式在哪裡

| 用途 | 路徑 |
|---|---|
| 注入位置 | `src/residual/site_{pixel,pixel_full,latent,embedding,weight,warp,color}.py` |
| 目標函數 | `src/defense/objective.py`（LPIPS 為綁定約束，`beta_linf` 可關） |
| 優化 | `src/defense/optimize.py`（`optimize` / `optimize_encoder` / `optimize_crossattn` / `align`） |
| cross-attention 擷取 | `src/models/attention.py` |
| 保真度指標 | `src/metrics/`（`local_acutance.py`、`chroma.py`、`suite.py`） |
| 淨化算子 | `src/purify/ops.py` |
| 主驅動 | `scripts/run_defense.py` |
| 實驗驅動 | `scripts/drivers/`（見該目錄的 `README.md`） |

`src/defense/generator.py` **依模塊提供的能力分派，不比對 site 名稱**。
新增 site 時提供 `pixel_residual` 或 `eps_hook` 即可。

---

## 執行

```bash
# 測試（基準 346 passed / 1 skipped / 0 failed）
python -m pytest -q

# 報表與人眼比對頁（本機，只讀 runs/ 的 CSV 與 PNG，秒級）
python scripts/report_table1.py --out docs/RESULTS_TABLE1.md
python scripts/lo_compare_page.py

# L1：基準的三個基準方法（雲端，約 2.9 小時）
bash scripts/drivers/lo_l1.sh 0

# L4′：本專案的條件走同一條評測路徑（雲端）
# --lr / --tau_acut / --tau_chroma 一律留空由腳本自己決定，
# 手動傳單一 --lr 正是 runs/ours_lo/ 失效的原因（LEDGER 6.16）
bash scripts/drivers/ours_l2.sh
```

`scripts/drivers/e29_calibration.sh`、`e30_grid.sh`、`e31_calibration.sh`、
`e31_grid.sh` **對應的實驗已中止**，檔案保留為紀錄，不要拿來跑。

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
