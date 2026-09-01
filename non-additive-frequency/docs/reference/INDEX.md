# 文獻查證索引

外部論文的查證紀錄。**這裡是索引，不是判準來源**——判準見
[../DECISIONS.md](../DECISIONS.md)，測得的事實見 [../RESULTS.md](../RESULTS.md)。

| 檔 | 內容 |
|---|---|
| [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) | **全部文獻與網址的分類索引**，含每一篇的取得狀況 |
| [SURVEY_FREQUENCY.md](SURVEY_FREQUENCY.md) | 頻域／相位方法與抗淨化，逐篇的具體運算 |
| [SURVEY_PHASE_PRIORART.md](SURVEY_PHASE_PRIORART.md) | 相位擾動的前例，新穎性主張的邊界 |
| [SURVEY_NOISE_RESISTANCE.md](SURVEY_NOISE_RESISTANCE.md) | 穩健浮水印的工具箱，以及它為什麼有一半不能移植到防護擾動 |
| [SURVEY_FRONTIER.md](SURVEY_FRONTIER.md) | 新穎性侵蝕的盤點、2026 的攻防前沿、以及不動點框架這個沒被佔走的定位 |
| [SURVEY_GENERAL.md](SURVEY_GENERAL.md) | 早期的通論式 survey |
| [ROBUSTNESS_TESTS.md](ROBUSTNESS_TESTS.md) | 三份抗淨化檢定協定的精確設定 |
| [BASELINE_ALIGNMENT.md](BASELINE_ALIGNMENT.md) | 威脅模型不對齊時，文獻上的四種處理方式 |
| [SOURCE_AUDIT.md](SOURCE_AUDIT.md) | 移植他人方法時「哪些值有出處、哪些是我方指定」的逐項清單 |
| [CODE_CONTRACTS.md](CODE_CONTRACTS.md) | 本專案的介面約定 |

## 逐方法的原始碼查證

| 檔 | 涵蓋 |
|---|---|
| [AUDIT_DIA_APA.md](AUDIT_DIA_APA.md) | DIA、APA |
| [AUDIT_MIST_DIFFVAX.md](AUDIT_MIST_DIFFVAX.md) | Mist、DiffVax |
| [AUDIT_PROMPTFLARE_PHOTOGUARD.md](AUDIT_PROMPTFLARE_PHOTOGUARD.md) | PromptFlare、PhotoGuard |
| [AUDIT_INPAINTING_METHODS.md](AUDIT_INPAINTING_METHODS.md) | AdvPaint、DIA、PromptFlare 的 inpainting 變體 |
| [AUDIT_PURIFIERS.md](AUDIT_PURIFIERS.md) | 淨化算子（DiffPure、IMPRESS、GrIDPure、FD-Pure 等） |

`apa_paper/` 是 APA 論文的表格截圖，弱 baseline 的定義依賴它。

## 狀態欄的意思

- **已實作** — 程式在 `src/baselines/` 或 `src/purify/`，跑得出數字
- **已查證** — 讀過論文或原始碼並留下紀錄，未實作
- **僅引用** — 只在論述中引用，未查證細節
