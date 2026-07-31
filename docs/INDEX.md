# 文件與實驗資料索引

最後整理：2026-07-31。

本檔的用途是讓接手者在不讀 git log 的情況下，知道**哪一份文件是現行的、
哪一份已被取代、每一個 run 目錄屬於哪個實驗**。

---

## 1. 文件

### 現行

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `docs/REPORT.html` | 主報告，涵蓋 E0–E12。含推翻與撤回的結論、規格層級更正 | 現行。表頭的「25 個 run、289 個資料檔」為 07-30 的數字，現為 48 個 run |
| `docs/RESULTS_E13-E17.md` | 承接 E13–E17：warp vs additive 主網格、VAE 地板拆解 | 現行 |
| `docs/specs/2026-07-28-lowrank-residual-defense.md` | 設計規格 v1 | 現行的**設計依據**，但其後的推翻不回頭改寫本文，須與上兩份對讀 |
| `docs/NIGHT_RUN_2026-07-29.md` | 2026-07-29 夜間自主執行的完整紀錄 | 歷史紀錄，非報告。記錄了 site L 防禦不存在這個否定結果的推導過程 |
| `docs/architecture.html` | 方法解說：威脅模型、低秩外積參數化、loss 與梯度路徑 | **內容過時**：寫的是「三個注入位置」，現已有六個（P/PF/L/E/W/S）。概念部分仍可讀 |

### 已封存（`docs/archive/`）

| 檔案 | 為何封存 |
|---|---|
| `2026-07-26-RESULTS_v2.html` | 前一代研究框架的結果（pg_enc / pg_diff / advdiff / apa / hybrid，以 LPIPS=0.464 為錨）。該路線已整批作廢，改為單一外掛殘差模組。程式碼可由 commit `02cf175` 取回 |
| `2026-07-26-v2-evidence/` | 上一列那份報告的**原始數值來源**（stage0/1/2 的 calibration、fairness、results.csv、manifest 與各階段 config/env 快照，18 檔 0.33 MB）。原本只存在於 TWCC 持久儲存、從未入庫。另有約 152 MB 的逐圖 PNG 留在 `/work`，理由見該目錄的 README |
| `2026-07-29-RESULTS_lowrank.html` | 由 `scripts/make_report.py` 生成，只涵蓋 E0–E3，被 `docs/REPORT.html` 取代。保留是因為 E0 成本表與 E0c 地板表只有這份有 |

---

## 2. 實驗資料（`runs/`）

48 個 run 目錄。所有 CSV / JSON / log / PNG 自 commit `1942e38` 起全部入版控
（在那之前 `.gitignore` 有一條規則讓 273 個檔案靜默漏掉，詳見該 commit）。

| run 目錄 | 實驗 | 說明 |
|---|---|---|
| `pilot` | — | 最早的可行性試跑 |
| `e0`, `e0_vae`, `e0b` | E0 / E0b | 成本實測與記憶體歸因。結論：瓶頸是 VAE 激活，非步數 |
| `e0c`, `e0c_tmax` | E0c | site L 的保真地板。促成 `t_max` 由 999 改為 500 |
| `e0d_L` | E0d | 學習率校準 |
| `e2`, `e2_phi0` | E2 | 主網格 site × rank。`e2_phi0` 是 φ=0 對照，**它推翻了 site L 有防禦這個結論** |
| `e2_la` | E2 | site LA（latent anchored）。該 site 只存在於未合併分支 `claude/anchored-site-l` |
| `e4_scale` | E4 | 注入強度掃描 |
| `e5_la10` | E5 | 三臂比較 |
| `e6_stepsP`, `e6_stepsLA` | E6 | 步數掃描 |
| `e7_stepsP_20/100`, `e7_stepsLA_20/100` | E7 | 步數掃描（擴充） |
| `e8_rank_tau0.02/0.05/0.10` | E8 | 全秩 vs 低秩，三個 LPIPS 預算。24 MB，本目錄群最大 |
| `e9_align_probe` | E9 | 對齊容量探測（2 圖、200 步） |
| `e10_eot_all` | E10 | EOT 梯度平均 |
| `e11_wlr_0.001/0.008` | E11 | 權重空間 LoRA 的學習率 |
| `e12_align_L_fixed`, `e12_align_W_r4/r16` | E12 | 對齊階段重跑（修正 PSNR 加權損失後） |
| `e13_slr_0.008/0.03/0.1/0.3` | E13 | site S 學習率。**發現 L∞ hinge 把 site S 完全節流** |
| `e14_P_*`, `e14_S_*` | E14 | 兩個 site 在同一判準下重新校準學習率 |
| `e15_P_tau0.02/0.05/0.10`, `e15_S_*` | E15 | **主網格**。6 圖、未見種子、無淨化、`beta_linf=0` |
| `e16_S_disp3.0/6.0` | E16 | 位移上界。`disp6.0` 只跑完 12 格中的 8 格 |
| `e17_vae_floor` | E17 | VAE 重建地板拆解。四分支：roundtrip / latent_opt / asym_free / asym_leak |
| `logs` | — | 各 run 的 driver 與 stdout 紀錄 |

### 涉及已作廢方向的資料

`e8_rank_*` 系列（24 MB）與低秩本身的探討有關，而低秩已於 2026-07-30 明確
列為不深究。資料保留不刪：它是 E8 結論的唯一證據，且 TWCC 容器已刪除，
無法重跑。

`e2_la`、`e5_la10`、`e6_stepsLA`、`e7_stepsLA_*` 使用 site LA，該 site 的
程式碼只存在於分支 `claude/anchored-site-l`，未併入 main。若要重跑須先取回該分支。

---

## 3. 未合併的分支

**未經明確授權不得併入 main。**

| 分支 | 未合併 commit 數 | 內容 |
|---|---|---|
| `claude/anchored-site-l` | 11 | site LA（latent anchored）的實作與 E4–E6 資料 |
| `claude/net-defense-objective` | 1 | 提議在防禦 hinge 內扣除未防禦對照 |

---

## 4. 重跑報告的指令

```bash
python scripts/e13_report.py      # E13 site S 學習率
python scripts/e14_report.py      # E14 兩 site 重新校準
python scripts/e15_report.py      # E15 主網格 + 淨化殘存 + 泛化
python scripts/e16_report.py      # E16 位移上界
python scripts/make_report_figures.py   # docs/figures/*.png（E2/E8/E9/E10/E12）
python scripts/make_report.py     # docs/archive/2026-07-29-RESULTS_lowrank.html（E0–E3）
```

`scripts/e17_vae_floor.py` 需要 GPU 與 SD 權重，不能在本機重跑。
