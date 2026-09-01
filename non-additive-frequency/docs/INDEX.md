# 索引

本專案的入口。每一份文件回答一個問題，彼此不重複；需要交叉查閱的地方一律在
此處或該文件內給出明確路徑。

## 文件

| 文件 | 回答什麼 | 什麼時候讀 |
|---|---|---|
| [GOAL.md](GOAL.md) | 研究目標、成敗判準、明確不做的事 | 開始任何工作之前 |
| [METHOD.md](METHOD.md) | 本方法（紋理重相位）的構造、參數、可調旋鈕 | 要改動方法時 |
| [BASELINES.md](BASELINES.md) | 每個對照組是什麼、來源、重現狀態、程式位置 | 要加或比較對照組時 |
| [EVALUATION.md](EVALUATION.md) | 指標、資料集、攻擊模型、淨化算子、對齊協定 | 要量任何東西之前 |
| [RESULTS.md](RESULTS.md) | 測得的事實。每一筆自足，含證據路徑 | 想知道某件事有沒有被測過 |
| [DECISIONS.md](DECISIONS.md) | 已裁決的事項與理由 | 想改變既定作法時 |
| [DEFECTS.md](DEFECTS.md) | 已知缺陷與修正紀錄 | 遇到可疑行為時 |
| [PENDING.md](PENDING.md) | **量過但尚未裁定**的事項與缺口 | 接手工作時；引用任何新數字之前 |
| [OPERATIONS.md](OPERATIONS.md) | 環境、遠端機器、執行與資料保全 | 要跑實驗時 |
| [reference/](reference/) | 外部文獻的查證紀錄 | 要引用或移植別人的方法時 |
| [superpowers/specs/perceptual-budget-design.md](superpowers/specs/perceptual-budget-design.md) | 頻率閘由二值改為知覺價目表的設計說明 | 要動頻率閘時 |

## 查閱路徑

**「這個結論的證據在哪？」**
→ [RESULTS.md](RESULTS.md) 每一筆都有「證據」欄，指向 `runs/` 底下的目錄。

**「這個對照組怎麼跑？」**
→ [BASELINES.md](BASELINES.md) 的程式欄 → [OPERATIONS.md](OPERATIONS.md) 的執行段。

**「這個指標為什麼這樣選？」**
→ [EVALUATION.md](EVALUATION.md)；若涉及裁決則另見 [DECISIONS.md](DECISIONS.md)。

**「這個方向試過了嗎？」**
→ [RESULTS.md](RESULTS.md) 的「已否決的方向」一節；若是最近量的但還沒裁定，
在 [PENDING.md](PENDING.md)。

**「別的領域怎麼讓訊號活過失真？」**
→ [reference/SURVEY_NOISE_RESISTANCE.md](reference/SURVEY_NOISE_RESISTANCE.md)：
穩健浮水印的工具箱，以及它為什麼有一半（同步、模板、不變域、處理增益）
依賴一個防護擾動沒有的解碼端。

**「這篇論文我們讀過嗎？」**
→ [reference/BIBLIOGRAPHY.md](reference/BIBLIOGRAPHY.md) 是總索引，逐篇細節在
同目錄的其他檔。

## 命名規則

**目錄名、檔名、實驗組名一律不含日期、流水號或順序詞**，要一眼看出在做什麼。

| 不可以 | 應該寫成 |
|---|---|
| `runs/s0817`、`runs/t0820` | `runs/sdedit_mainline`、`runs/ip2p_pilot` |
| `docs/SURVEY_2026-08-18.md` | 併入 `reference/` 的主題檔 |
| 「第二輪的結果」 | 「等失真錨點上的頭對頭結果」 |
| 「2026-08-19 裁定」 | 直接寫現行作法與理由 |

編碼（`FND-`／`DEC-`／`DEF-`／`MET-`）**只用來互相指認，不代表先後或依賴**。
每一筆都必須能單獨讀完。

## 現況一句話

**這個方向已經告一段落**，本目錄是封存的完整記錄。新階段的入口在
[`../../anti-purify/start.md`](../../anti-purify/start.md)。

成立的結果：在**等失真**的頭對頭上（`runs/ip2p_matched_headtohead/`，
六算子、三種子、兩張影像、與 DCT-Shield 同一批），本方法**未淨化輸、
JPEG 淨化後贏**——協定完全對稱的那一組（兩邊都不量化交付）JPEG 75 是
3.4 倍、JPEG 30 是 4.3 倍。

不成立的：**模糊**。高斯模糊在頻域乘實正數，構造上不可能改相位，它抽掉的是
載體（σ=1 之後擾動的能量存活率只剩 0.169，σ=2 剩 0.061）。十二條路只有加性項
的期望存活加權動得了它，而且只有兩成。**裁切不是失敗點**——換過參照之後它是
六個算子裡保留最好的一欄（77%），輸給 DCT-Shield 但沒有失效。

主讀數維持位移、錨點維持等失真；位移的三個已知污染隨表註明，見
[EVALUATION.md](EVALUATION.md#防禦成功的讀數)。

## 交接

常設判準以 `DECISIONS.md` 與 `RESULTS.md` 為準。跨階段的背景、量測協定、
專案規定與機器使用，統一寫在
[`../../anti-purify/start.md`](../../anti-purify/start.md)。
