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
| [OPERATIONS.md](OPERATIONS.md) | 環境、遠端機器、執行與資料保全 | 要跑實驗時 |
| [reference/](reference/) | 外部文獻的查證紀錄 | 要引用或移植別人的方法時 |

## 查閱路徑

**「這個結論的證據在哪？」**
→ [RESULTS.md](RESULTS.md) 每一筆都有「證據」欄，指向 `runs/` 底下的目錄。

**「這個對照組怎麼跑？」**
→ [BASELINES.md](BASELINES.md) 的程式欄 → [OPERATIONS.md](OPERATIONS.md) 的執行段。

**「這個指標為什麼這樣選？」**
→ [EVALUATION.md](EVALUATION.md)；若涉及裁決則另見 [DECISIONS.md](DECISIONS.md)。

**「這個方向試過了嗎？」**
→ [RESULTS.md](RESULTS.md) 的「已否決的方向」一節。

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

主線影像定案為六張（`runs/ip2p_fair_comparison/images6.txt`）——25 張的人眼
稽核顯示只有 13 張的未防禦編輯真的執行了指令，而**指令的型態決定服從率**，
故選圖一律取「針對畫面主體」的指令。

在**未淨化的防禦**上，紋理重相位要付 **4.36 倍**的失真才追得平 DCT-Shield，
而歸因指向參數化本身（純加性都比它強）。在**抗淨化**上有一組對它有利的讀數，
但那依賴一個未被採納的錨點。詳見 [RESULTS.md](RESULTS.md#目前的整體處境)。

主讀數維持位移、錨點維持等失真；位移的三個已知污染隨表註明，見
[EVALUATION.md](EVALUATION.md#防禦成功的讀數)。
