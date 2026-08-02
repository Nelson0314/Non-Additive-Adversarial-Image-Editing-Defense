# WACV — 白盒非加性抗文字編輯防禦

在白盒條件（攻擊方使用 stock Stable Diffusion）、外掛模組形式下，尋找非加性
方法，在**匹配人眼可辨失真**下，於**抵抗文字引導編輯**上勝過加性基準。

現階段的實際問題比這一句更前面：**至今沒有任何一個運作點觀察到防禦成功**，
包含加性基準。E31 因此改為先建立正對照。詳見 `docs/NEXT_SESSION.md`。

---

## 從哪裡開始讀

| 你想知道 | 讀這個 |
|---|---|
| 現在在做什麼、下一步是什麼 | `docs/NEXT_SESSION.md` |
| 某個結論還算不算數、出處在哪 | `docs/LEDGER.md`（**主張**的索引） |
| 哪份文件現行、哪個 run 屬於哪個實驗 | `docs/INDEX.md`（**檔案**的索引） |
| 某個門檻當初看哪張圖定出來的 | `docs/gallery.html`（**人眼比對頁**的索引） |
| 文獻脈絡 | `docs/SURVEY.md`（按問句組織，不是逐篇摘要） |
| 現階段的實驗設計與作法 | `docs/specs/2026-08-02-e31-positive-control.md`、`docs/plans/2026-08-02-e31.md` |
| 接手時要貼給新 session 的東西 | `docs/HANDOFF_PROMPT.md` |
| 工作規範 | `CLAUDE.md` |

每一份 `docs/RESULTS_*.md` 的標題下都有同一塊狀態表（狀態／日期／硬體／承接／
後續／資料）。`grep -l STATUS-BLOCK docs/*.md` 可列出全部。

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
# 測試（基準 284 passed / 1 skipped / 0 failed）
python -m pytest -q

# 本機的分析與評測工作，串起來跑（不要並行，見下）
bash scripts/drivers/local_night.sh

# 雲端的兩支，須傳入逐預算的次要門檻
TA_028=... TC_028=... bash scripts/drivers/e31_calibration.sh
LR_028=... TA_010=... TC_010=... TA_028=... TC_028=... bash scripts/drivers/e31_grid.sh
```

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
