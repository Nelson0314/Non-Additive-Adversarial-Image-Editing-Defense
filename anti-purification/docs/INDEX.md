# 索引

| 文件 | 回答什麼 |
|---|---|
| [DIRECTION.md](DIRECTION.md) | 研究方向：走過哪兩段、現行的三個載體 |
| [EVALUATION.md](EVALUATION.md) | 指標與淨化的比較方式 |
| [CONDITIONING_PROBE.md](CONDITIONING_PROBE.md) | 文字條件反應那個量為什麼不能當目標 |
| [OPERATIONS.md](OPERATIONS.md) | 遠端環境、卡的規則、repo 的定位 |
| [reference/BIBLIOGRAPHY.md](reference/BIBLIOGRAPHY.md) | 外部文獻 |
| [reference/BASELINE_PROVENANCE.md](reference/BASELINE_PROVENANCE.md) | **每個 baseline：哪一篇、那篇寫什麼、官方程式做什麼、我們跑什麼**，以及可引用的參考數字的確切出處 |
| [reference/BASELINE_ALIGNMENT.md](reference/BASELINE_ALIGNMENT.md) | 預算不對齊時怎麼比（掃強度畫曲線＋兩個錨點） |
| [reference/BASELINE_CANDIDATES.md](reference/BASELINE_CANDIDATES.md) | 比現有條件更新的防護方法，過硬體／程式／可引用三道篩選 |
| [../scripts/edit_preflight.py](../scripts/edit_preflight.py) | **編輯管線的定義與未防禦預檢**（ip2p ＋ inpainting，指令與設定都在它的 docstring） |
| [../scripts/defence_run.py](../scripts/defence_run.py) | 防禦圖求解（只求解、不評測） |
| [../scripts/make_masks.py](../scripts/make_masks.py) | inpainting 遮罩，**dilate 4**（不是模組預設的 16，理由見該檔） |
| [reference/AUDIT_*.md](reference/) | 各 baseline 的逐行原始碼查證（PhotoGuard／PromptFlare、Mist／DiffVax、AdvPaint／DIA／PromptFlare、DIA、淨化算子） |

現況與正在跑的東西在 [`../../HANDOFF.md`](../../HANDOFF.md)。
每一批的設定與量測記在該批自己的 `runs/<批次>/README.md`。

## 命名規則

目錄名、檔名、實驗組名一律不含日期、流水號或順序詞。

| 不可以 | 應該寫成 |
|---|---|
| `runs/s0817`、`runs/t0820` | `runs/objective_pilot` |
| `SURVEY_2026-08-18.md` | 併入主題檔 |
| 「第二輪的結果」「本輪」 | 直接寫現行作法與理由 |

文件內容不得有時間相依性：不寫「本輪」「先前」「稍後會補」，只寫現況、理由、
以及查閱路徑。
