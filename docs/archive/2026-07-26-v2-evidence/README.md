# v2 框架的原始數值證據（2026-07-26）

這是 `docs/archive/2026-07-26-RESULTS_v2.html` 的資料來源。該報告封存時，
它引用的每一個數字在 repo 內都沒有可追溯的出處——原始檔只存在於 TWCC
持久儲存 `/work/nelson0314/WACV/experiments/`，從未入庫。本目錄補上這一段。

## 這是哪一條研究線

v2 框架比較的是 **pg_enc / pg_diff（加性）對 advdiff / apa / hybrid（非加性）
在淨化下的耐受性**，以 PhotoGuard-encoder 的感知擾動量為公平性錨點。
該路線已於 2026-07-28 整批作廢，改為現行的「單一 SD 外掛殘差模組」。
v2 的程式碼可由 git 歷史 commit `02cf175` 取回。

**這些數字不可與現行 E0–E18 的結果並列解讀**：兩者的威脅模型、公平性
校準方式與失真預算定義都不同。

## 內容

| 路徑 | 內容 |
|---|---|
| `experiments/stage0/20260726_032310/` | 校準階段。`fairness.csv` 是把各非加性方法的強度旋鈕對齊到同一 LPIPS footprint 的實測值（3 張影像）；`calibration.csv` 為掃描過程 |
| `experiments/stage1/20260726_070400/` | 未淨化下的防禦強度。`results.csv`、`manifest.json`、`protect_info.json` |
| `experiments/stage2/20260726_122219/` | 淨化後的殘存。`results.csv`（204 KB，主表）、`report_v2_tables.csv`、`summary.md` |
| `figures_untracked_backup/comp_run.log` | 比較圖產生過程的執行紀錄 |

每個 stage 都附 `config_snapshot.yaml` 與 `env.json`，可還原當時的參數與環境。

## 未取回的部分

同目錄下另有約 152 MB 的逐圖 PNG（保護圖、編輯圖、淨化圖），留在
`/work/nelson0314/WACV/experiments/` 與 `figures_untracked_backup/`。
未入庫的理由：屬於已作廢路線的逐格影像傾印，且 `/work` 為持久儲存、
跨容器保留，沒有流失風險。若日後需要，路徑仍在。
