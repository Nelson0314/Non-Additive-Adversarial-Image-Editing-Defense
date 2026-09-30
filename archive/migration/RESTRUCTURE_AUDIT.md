# 1. 現況盤點

建議將整理工作分成三件事：**修正會誤報完成或遺失數值記錄的流程、建立可獨立交付的子專案邊界、整理命名與文件**。三者應分開提交與驗證，避免結構調整同時改變研究協定。

最重要的診斷是：`lab/` 尚未達到文件宣稱的自足程度；`main_table/` 的相依也不限於上層程式與資料集，還包括既有實驗產物。僅搬動目錄或複製入口腳本，無法解決這些相依。

## 1.1 查核範圍與限制

本報告以本機工作目錄及 Git `333cf7f` 為依據。全程只進行本機讀取、文字檢索、AST 解析、CSV 結構檢查與 Git 查詢；沒有修改檔案、執行 Git 寫入、載入研究模型、執行 GPU 工作或連線遠端。

為縮短表格，後文使用下列路徑縮寫：

| 縮寫 | 路徑 |
|---|---|
| `A/` | `anti-purification/` |
| `M/` | `anti-purification/main_table/` |
| `L/` | `lab/` |
| `F/` | `frequency-phase/` |
| `MEM/` | `C:/Users/nelso/.claude/projects/C--image-immunization/memory/` |

| 對象 | 查核方式與範圍 |
|---|---|
| 全部 1,707 個追蹤檔案 | 全量檔案盤點、分類、路徑與命名檢索 |
| 214 個 Python 檔 | 逐檔語法、AST、import、函式／類別、docstring、例外處理、CLI 與路徑檢查；對報告列出的具體問題核讀實作 |
| 11 個 shell 檔 | 入口、工作目錄、相依指令、完成旗標、退出碼、租約與換行檢查 |
| 57 份 Markdown | 入口與分工、內部引用、狀態敘述、歷史紀錄、書面用語檢查 |
| 67 個 JSON、11 個 YAML／YML | 結構、設定用途、資料路徑、條件名稱與來源欄檢查 |
| 697 份 CSV | 全部解析表頭與資料列，合計 65,697 列；欄位值按資料組抽樣，不逐項重新驗算科學數值 |
| `runs/` 的數值輸出 | 掌握逐實驗、逐條件、分片及讀數表的命名與版面；日誌檢索錯誤標記，數值內容抽樣 |
| 562 張追蹤影像、4 份 PDF、1 個 ZIP | 檔案與相依盤點；不對未逐張檢視的影像品質、PDF 全文論證或 ZIP 內資料正確性下結論 |
| 全域規則與持久記憶 | 讀取全域 `CLAUDE.md`、`MEMORY.md` 及相關記憶；記憶目錄共 59 份 Markdown |
| 遠端 | 僅讀 `.tmp/codex_audit/remote_tree.txt`，共 2,138 行；未查遠端檔案內容、符號連結目標或行程 |

**靜態檢查不等於執行驗證。** 214 個 Python 檔中，213 個可由 AST 解析，1 個有確定語法錯誤；未執行 `pytest`，因此不沿用文件中的「657 passed」作為本次測試結果。

遠端摘要只列出深度上限內的目錄及部分非影像檔，且清單結尾停在 `runs/advcf_objective/`。因此，本報告能列出已知遠端相依與遷移前必查項目，**不能聲稱已窮盡所有遠端呼叫端**。

## 1.2 區塊角色與規模

| 區塊 | 追蹤檔案數 | 主要內容 | 診斷 |
|---|---:|---|---|
| 根目錄 | 4 | 3 份 Markdown、`.gitignore` | 總入口已失效，交接與歷史紀錄混放 |
| `.claude/` | 2 | 報告技能、HTML 模板 | 技能依賴已移除的 builder，模板帶有特定實驗內容 |
| `A/`，不含 `main_table/` | 1,543 | 共用實作、舊實驗、資料、文獻查證 | 同時承擔套件、歷史專案、資料庫及產物紀錄四種角色 |
| `M/` | 111 | 16 個 Python 入口／模組、1 個測試檔、2 個 shell、84 份 CSV | 已完成的比較研究，但程式與產物仍跨目錄取用 |
| `L/` | 34 | 9 個 Python、6 個 shell、13 份 CSV | 入口有複製，核心模組仍依賴 `A/src/` |
| `F/` | 13 | 12 個 Python、README | 已結束的部分程式封存，不是可獨立執行的專案 |

`A/` 的主要規模：

| 路徑 | 規模 |
|---|---|
| `src/` | 77 個 Python，20,140 行 |
| `scripts/` | 43 個 Python、3 個 shell |
| `tests/` | 56 個 Python，含空的 `__init__.py` |
| `configs/` | 34 份 JSON |
| `data/` | 604 個檔案，約 220 MB |
| `runs/` | 696 個檔案，其中 599 份 CSV、31,132 列 |

遠端摘要記錄：`runs/` 約 12 GB、`main_table/` 約 2.1 GB、`data/` 約 593 MB、`lab/` 約 552 MB。這些是摘要中的容量，不能視為本次即時量測。證據：`remote_tree.txt:1723–1737`。

## 1.3 實際相依圖

```text
根 README／HANDOFF／Claude 記憶
          │ 指向入口、腳本、實驗路徑、已發布報告
          ▼
main_table/code ── paths.py／sys.path ──► anti-purification/src
       │                                      ▲
       ├──► anti-purification/data             │
       ├──► 主線 runs 中的防禦、淨化、編輯圖    │
       └──► lab/runs/defence/color             │
                （color 匯入來源）             │
                                              │
lab/code ─────── paths.py／sys.path ────────────┘
   │
   ├──► 複製的 portraits 輸入
   ├──► 共用未防禦編輯產物／文件記載的連結
   └── color_defence ◄────► style_prompt_defence
                  helper 與 objective 相互引用

anti-purification/src/defense/param_pgd
       └── 延遲 import ──► frequency-phase/src/residual

anti-purification/src/purify/ops
       └── 延遲 import ──► frequency-phase/src/purify/freq_grid

frequency-phase/scripts
       ├──► 主線 baseline_run、models、metrics、utils
       └──► 已不存在的 src.defense.codefense
```

具體證據：

| 相依 | 證據 |
|---|---|
| `M` 與 `L` 尋找主線 | 兩份 `code/paths.py:35–57` 搜尋 `anti-purification`、`non-additive-frequency` 或 `IMMUNISATION_SOURCE_HOME` |
| 修改 Python 搜尋路徑 | 兩份 `paths.py:66–71`；`M` 14 個入口、`L` 8 個入口呼叫 `add_source_to_syspath()` |
| 資料集跨目錄 | `paths.py:60–63` 將 portraits、targets 指向主線 |
| 主表取用主線淨化產物 | `M/code/edit_ultraedit_full.py:73–83` |
| 主表匯入 lab 方法產物 | `M/code/color_row_chain.sh:29–36`、`M/STATUS.md:25–28` |
| lab 的程式耦合 | `L/code/style_prompt_defence.py:35`、`L/code/color_defence.py:349` |
| 頻域反向相依 | `A/src/defense/param_pgd.py:217`、`A/src/purify/ops.py:589–592` |
| 已不存在的模組 | `F/scripts/phase_retention.py:372` |

可解析的靜態 import 閉包中，`M` 涉及 30 個主線模組，`L` 涉及 24 個，聯集 37 個、交集 17 個。這尚不包含所有動態載入及外部套件，已足以說明「只複製幾支入口」不足以自足。

## 1.4 資料版面與已確認的完整性

| CSV 區塊 | 檔案數 | 資料列 |
|---|---:|---:|
| `A/data/` | 1 | 189 |
| `A/runs/` | 599 | 31,132 |
| `M/results/` | 84 | 33,312 |
| `L/data/`、`L/results/` | 13 | 1,064 |

對主表五份及 lab 兩份主要讀數表，以各表的科學鍵值檢查，未發現重複列：

| 表 | 列數 | 鍵值 |
|---|---:|---|
| `M/results/displacement.csv` | 768 | condition、scenario、image、prompt_index |
| `M/results/retention.csv` | 5,376 | 上述鍵＋purifier |
| `M/results/displacement_flux.csv` | 384 | condition、image、prompt_index |
| `M/results/displacement_ultraedit.csv` | 384 | condition、scenario、image、prompt_index |
| `M/results/retention_ultraedit.csv` | 2,688 | 上述鍵＋purifier |
| `L/results/displacement.csv` | 32 | condition、scenario、image、prompt_index |
| `L/results/retention.csv` | 224 | 上述鍵＋purifier |

主表是 **12 個防禦條件**；跨編輯器原始編輯表的 **13 arms 包含 `undefended`**，兩者不矛盾。

另須更正背景中的概括：主表實跑的七道是 `blur1`、`blur2`、`jpeg80`、`jpeg50`、`jpeg30`、`crop_resize0.1`、`rotate15`，**不含 DiffPure**。`identity` 是未淨化參照。DiffPure 存在於共用算子實作及文獻查證，不能據此稱它已納入主表七道。證據：`M/code/purify_run.py` 的 `PURIFIERS`、`A/docs/EVALUATION.md:38–51`。

# 2. 問題清單

嚴重度定義：

- **P0**：再次派工或更新記錄前應處理，可能誤報成功或漏存證據。
- **P1**：會破壞獨立執行、遷移正確性或資料解讀。
- **P2**：維護與可讀性問題，宜在行為凍結後整理。

| 優先度／類別 | 問題與證據 | 建議 |
|---|---|---|
| **P0／程式風格** | [L/scripts/readout.sh](C:/image-immunization/lab/scripts/readout.sh:6) 使用 `set -uo pipefail`；18、24 行的 Python 失敗後仍執行 `echo`，28 行仍印完成，最後退出碼可為 0。`queue_worker.sh:120、134–138` 依退出碼記完成。 | 保存並傳回失敗碼；兩份輸出通過列數、鍵值及必要欄位檢查後才能寫完成狀態。 |
| **P0／結構** | `L/.gitignore:1` 排除整個 `runs/`，但 `color_defence.py:710`、`style_prompt_defence.py:583、751` 在其中寫 `results.csv`、`trace.csv`。與「數值 CSV 一律入版控」不相容。 | 數值證據移到可追蹤位置，或調整 ignore 例外；先盤點遠端所有未入庫數值，不可假定文件摘要足以替代。 |
| **P0／程式風格** | `M/code/color_row_chain.sh:40、51` 僅憑目錄存在就略過編輯階段；目錄可能在中斷前已建立。 | 以預期格數、產物清單及協定摘要驗收，不能以目錄存在代表完成。 |
| **P1／程式風格** | `M/code/edit_flux_preview.py:105–109` 的續跑鍵僅含 image、prompt_index；`edit_ultraedit_full.py:143–146` 的鍵未包含完整參數，並移除找不到圖的舊列。更換參數或搬路徑可能被誤判為已完成／需要覆寫。 | 續跑鍵加入 protocol／config digest；路徑失效先報錯，不刪除原始列。 |
| **P1／自足性** | [L/HANDOFF.md:3](C:/image-immunization/lab/HANDOFF.md:3) 宣稱所有程式、資料、產出都在 lab；實際 `paths.py:35–71` 仍載入主線，`DESIGN.md:13` 還記載產物連結。 | 修正文案；交付完整固定版本相依及輸入快照。 |
| **P1／自足性** | `M/code/paths.py:60–63`、`edit_ultraedit_full.py:73–83`、`color_row_chain.sh:29–36` 分別依賴主線資料、主線產物、lab 產物。 | 將「資料集」「方法產物匯入」「評估輸出」設成三個明確介面。 |
| **P1／結構** | 三組編輯／淨化／讀數入口已分歧。`M`、`L` 的 `edit_retention.py:70–91` 有 `purified_mask()`，`A/scripts/edit_retention.py` 沒有。 | 先建立行為差異表，再抽出共用實作；不可直接以任一副本覆蓋其餘副本。 |
| **P1／文件、程式風格** | [M/STATUS.md:58–60](C:/image-immunization/anti-purification/main_table/STATUS.md:58) 記載 11 個舊條件的幾何分區讀數使用未變換遮罩。`A/docs/EVALUATION.md:76` 的受影響列數仍沿用舊範圍。 | 在資料目錄加機器可讀限制。原生主表受影響範圍為 `11×2×8×4×2=1,408` 列的兩個分區欄；全圖欄不受此錯誤影響。更正讀數另立工作，不混入改名。 |
| **P1／程式風格** | [L/code/style_prompt_defence.py:727–736](C:/image-immunization/lab/code/style_prompt_defence.py:727) 在 `--select last` 時先令 `best=last`，再以 `best is not None` 寫 `feasible`。此時旗標表示有最後一步，不再表示通過限制。 | 分開記錄 `selection_policy`、`selected_feasible`；是否已有受影響資料需核對 trace，不能直接推定所有歷史列錯誤。 |
| **P1／程式風格** | `A/src/utils/io.py:35` 及多個 driver 以 `"w"` 重寫 CSV；分片與重試可能寫相同位置。`L/HANDOFF.md:63–67` 已記錄互相覆寫風險。 | 單一 writer、原子替換、不可變實驗目錄；彙整另存新表，不把重試當覆寫授權。 |
| **P1／結構** | `L/scripts/run_on_card.sh:11–21` 的檢查與寫租約分離；`style_prompt_round.sh:21` 的鎖只保護該排程。`MEM/parallel-run-on-card-races.md:11–20` 有既往碰撞紀錄。 | 所有派工入口共用一套跨行程取卡與租約協定；不能只修某一支排程。 |
| **P1／程式風格** | `L/scripts/style_prompt_round.sh:6` 預設 `CAP=6`，18 行只計算 style 工作；與 `L/HANDOFF.md:47–49` 的跨 session 合計 5 張不一致。 | 容量政策集中設定；既往單次例外不可成為預設值。 |
| **P1／程式風格** | `M/code/metrics_union.py:227`、`A/scripts/metrics_union.py:186`、`readout_panel.py:69` 捕捉所有 `Exception` 後略過指標；後者直接省略欄。 | 必要指標失敗使階段失敗；選配指標使用固定 schema，記 `status`、`reason`，保留 traceback。 |
| **P1／自足性** | `environment.yml:1–3` 明示不鎖版本，還引用不存在的環境文件。程式直接使用 `facenet_pytorch`、`insightface`、`skimage` 等，環境檔未完整列示。 | 各專案有可安裝描述、鎖定環境、選配 extras、模型 revision／checkpoint manifest。 |
| **P1／自足性** | `F` 與 `A` 都使用泛稱 `src`，但 `A/src/__init__.py` 是一般套件；僅把兩個根加入 `sys.path` 不能可靠合併它們。另缺 `src.defense.codefense`。 | 使用明確 namespace；封存文件停止宣稱加兩條路徑即可執行。 |
| **P1／文件** | [根 README.md:11](C:/image-immunization/README.md:11) 的 `anti-purify/`、`start.md`、`COLOR_AND_DECOY.md` 等均不存在；14、41 行重複指向失效入口。 | 根 README 只列四個區塊的角色、入口與共用規則。 |
| **P1／文件** | 根 `HANDOFF.md:307` 稱另一個 artifact 為「主表（現行）」；`M/STATUS.md:75–76` 指向不同 artifact，且明說發布內容是舊顏色列。 | 建立唯一發布清單，記錄每個 artifact 對應的資料與來源 commit。 |
| **P2／程式風格** | [A/scripts/summarise_diagnosis.py:137](C:/image-immunization/anti-purification/scripts/summarise_diagnosis.py:137) 連續兩個 `add_argument` 缺右括號；AST 報 `SyntaxError`。 | 修復語法或明確封存停用；加入全檔案語法檢查。它不是主表入口，不應誇大為整個專案無法執行。 |
| **P2／程式風格** | `A/src/utils/io.py:59` 只判斷寬是否等於 `size`，高不同而寬相等時會略過 resize。 | 明確定義尺寸契約，檢查兩個空間維度；與搬檔分開修正。 |
| **P2／程式風格** | `A/scripts/fetch_imagenet_subset.py:81–84` 以 `except Exception: continue` 靜默丟掉樣本；`test_direction_similarity.py:17–20` 將各種初始化錯誤都轉成 skip。 | 縮小例外型別，區分不可解碼輸入、缺權重及程式缺陷，記錄排除原因。 |
| **P2／結構** | `L/code/color_defence.py` 717 行、`style_prompt_defence.py` 760 行，相互引用；後者也被 fidelity／readout 間接牽連。 | 共用 carrier、objective、CSV writer 移到不依賴 CLI 的模組。 |
| **P2／程式風格** | lab 注意力 objective 包裝模型方法，見 `style_prompt_defence.py:212–216`、`color_defence.py:391–407`；檔內未見對應還原生命週期。 | 增加 context manager／`close()` 契約及重複建立測試；目前只能確認生命週期風險，不能宣稱已污染數值。 |
| **P2／命名** | `edit_flux_preview.py` 被用於全表；`flux_full_queue_a.sh` 接收任意 arms；`prompt_sets_ultraedit_round2.json` 實際是名詞及放置句型變體。 | 改為任務或科學變因名稱，詳見第 4 章。 |
| **P2／文件** | `L/HANDOFF.md:31` 說參數只定義在 shell，但 `defence_cmd.sh:2、10` 依賴 Python 預設；shell 宣稱只有 color，11–24 行卻有七個可執行分支。 | 明確列出四個保留條件及其他三個分支的狀態，參數正本移到設定。 |
| **P2／文件** | `.claude/skills/defence-report/SKILL.md:43–50` 引用不存在的 `build_field_report.py`；模板仍用特定實驗標題、固定資料量與判定語句。 | 技能與通用模板重整；不要因技能存在而恢復已刻意刪除的報告 builder。 |
| **P2／結構** | `git ls-files --eol`：695/697 份 CSV 工作目錄為 CRLF，Git 全為 LF；`patch_pipeline_run.sh` 是 `i/lf w/crlf attr/text eol=lf`。 | 分別保存 Git blob 與工作目錄雜湊；換行整理不得混入科學資料遷移。 |

需要保留的既有設計也很明確：baseline 的來源、值域與預算落差已有詳細紀錄；多數數學測試使用替身模型；不少缺檔、缺遮罩、無梯度情況會直接失敗。清理時應保存這些契約，不能把所有例外處理或長註解一概刪除。

# 3. 目標結構

## 3.1 自足方式的取捨

| 方式 | 能否不依賴兄弟目錄 | 優點 | 代價／限制 | 建議 |
|---|---|---|---|---|
| 只複製入口腳本 | 否 | 搬動量小 | 現有 lab 即屬此情況；核心與資料仍外借 | 不採用 |
| 完整複製相依來源 | 可以 | 容易離線交付、各線可固定行為 | 修正容易分歧，來源與版本難追蹤 | 只用於封存快照 |
| 固定版本的共用套件 | 可以 | 統一維護、可測試、可逐專案升版 | 需套件發佈與相容性管理 | 採用 |
| 固定版本套件＋專案內附相依快照 | 可以，且不需另取內部套件 | 符合可單獨搬走的要求，來源可驗證 | 有受控重複，需記錄版本與雜湊 | **建議交付形式** |
| repo 內單一 `shared/`，直接改 `sys.path` | 否 | 最少副本 | 共用修改會立即影響所有研究線，仍依賴目錄位置 | 不符合本次目標 |
| 符號連結共用資料與產物 | 否 | 節省空間 | 連結失效、來源更動與遠端映射增加隱性相依 | 僅可作儲存實作，不可作交付契約 |

建議是：**共用程式有一份維護正本；每個可執行子專案鎖定版本，交付時附上該版本的套件來源快照或 wheel。** 子專案安裝自己的相依後，不能再從兄弟目錄 import。

資料亦採相同原則：

- portraits、masks、prompts、來源 metadata：各專案附所需的固定輸入快照。
- lab 的 `color` 防禦图交給主表：使用帶雜湊與方法設定的匯出包。
- 共用未防禦編輯：以明確 protocol ID 的輸入產物匯入，不讀另一條線正在寫的目錄。
- 模型權重不進 Git；記錄模型 ID、revision、精度、量化方式與必要檔案雜湊。

## 3.2 建議目錄樹

以下樹列出完整責任層級；歷史數值檔的既有葉節點保留，不展開數百個實驗目錄。

```text
image-immunization/
├── README.md                         # 專案地圖與入口
├── CLAUDE.md                         # repo 共通工作規則
├── .gitignore
├── .gitattributes
├── .claude/
│   └── skills/
│       └── defense-report/
│           ├── SKILL.md
│           └── assets/template.html  # 模板來源，可入版控
│
├── docs/
│   ├── ARCHITECTURE.md               # 邊界、自足與資料流
│   ├── OWNERSHIP.md                  # 維護範圍及共用套件責任
│   ├── NAMING.md
│   ├── ARTIFACTS.md                  # 遠端產物與發布報告索引
│   └── migration/
│       ├── paths.json               # 舊路徑到新位置
│       └── identifiers.json         # 識別值定義，不任意合併條件
│
├── packages/
│   └── immunization_core/
│       ├── pyproject.toml
│       ├── src/immunization_core/
│       │   ├── baselines/
│       │   ├── defenses/
│       │   ├── editors/
│       │   ├── metrics/
│       │   ├── purifiers/
│       │   ├── pipelines/            # 編輯、淨化、讀數共用流程
│       │   ├── artifacts/            # schema、路徑解析、CSV 寫入
│       │   └── runtime/              # 裝置、精度、明確設定
│       ├── tests/
│       └── docs/
│           └── reference/            # 方法來源、移植差異、固定引用
│
├── main_table/
│   ├── README.md                     # 使用與內容
│   ├── STATUS.md                     # 唯一交接入口
│   ├── CLAUDE.md                     # 本區範圍與特殊規則
│   ├── pyproject.toml
│   ├── requirements.lock
│   ├── vendor/immunization_core/      # 固定版本交付快照
│   ├── vendor.lock.json
│   ├── src/immunization_benchmark/
│   │   ├── cli/
│   │   ├── conditions.py
│   │   ├── protocol.py
│   │   └── third_party/ultraedit/
│   ├── scripts/                     # shell 派工入口
│   ├── configs/
│   │   ├── protocols/
│   │   ├── editors/
│   │   └── prompts/
│   ├── data/
│   │   ├── portraits/
│   │   ├── targets/
│   │   └── manifest.json
│   ├── results/                     # CSV、trace、manifest，全部入版控
│   │   ├── 原生主表檔案
│   │   ├── aligned/
│   │   ├── flux/
│   │   ├── ultraedit/
│   │   └── sweeps/
│   ├── artifacts/                   # 大型影像、模型衍生產物，不入版控
│   ├── reports/                     # 生成 HTML，不入版控
│   ├── runtime/                     # queue、lease、sentinel，不作研究證據
│   ├── docs/
│   │   ├── PROTOCOL.md
│   │   ├── DATA.md
│   │   └── LIMITATIONS.md
│   └── tests/
│
├── lab/
│   ├── README.md
│   ├── HANDOFF.md                    # 保留作為既有唯一交接入口
│   ├── CLAUDE.md
│   ├── pyproject.toml
│   ├── requirements.lock
│   ├── vendor/immunization_core/
│   ├── vendor.lock.json
│   ├── src/immunization_lab/
│   │   ├── methods/color/
│   │   ├── methods/style_prompt/
│   │   ├── objectives/
│   │   └── cli/
│   ├── scripts/
│   ├── configs/
│   │   ├── color/
│   │   ├── style_prompt/
│   │   └── evaluation/
│   ├── data/
│   │   ├── portraits/
│   │   └── color_lpips_ref.csv
│   ├── results/
│   │   ├── color/
│   │   └── style_prompt/
│   ├── artifacts/
│   ├── reports/
│   ├── runtime/
│   ├── docs/
│   │   ├── color/DESIGN.md
│   │   ├── style_prompt/DESIGN.md
│   │   └── references/
│   └── tests/
│
└── archive/
    ├── anti_purification/
    │   ├── README.md                 # 封存狀態、來源 commit、可執行範圍
    │   ├── src/                      # 歷史來源保留
    │   ├── scripts/
    │   ├── configs/
    │   ├── data/
    │   ├── tests/
    │   ├── runs/                     # 原始數值樹保留
    │   └── docs/
    │       ├── HANDOFF.md
    │       └── COLOR_LINE.md
    └── frequency_phase/
        ├── README.md
        ├── src/
        └── scripts/
```

`vendor/` 是有版本與雜湊的交付快照，不允許各 session 就地修改；修正應回到套件正本，再由子專案明確升版。

封存區是否也必須可執行，需要裁定。現在 `F/` 的內容不足以承諾可執行；若要求四塊都能單獨執行，就必須另行恢復其完整歷史依賴與設定。不能只補 README 宣稱自足。

## 3.3 放置與命名規則

| 層級 | 放什麼 | 命名及禁止事項 |
|---|---|---|
| 根目錄 | 專案地圖、共通規則 | 不再放單一研究線的長篇交接或實驗結果 |
| `packages/` | 跨專案且契約一致的程式 | 不能依賴 `main_table/`、`lab/`、`runs/`；不得以使用者家目錄尋找來源 |
| 子專案 `src/` | 該專案特有方法、協定、入口 | 使用明確 namespace；禁止 `sys.path` 拼接兄弟專案 |
| `configs/` | 實驗設定、協定、方法變體 | 以方法或變因命名；不用 `round2`、`queue_a`、`latest`、日期 |
| `data/` | 固定輸入與來源 | 原图、遮罩属于輸入，可依既有規則入版控；不得混入生成結果 |
| `results/` | 不可重現的數值證據 | CSV、trace、排除原因、設定快照都入版控；資料列識別值不可為改名方便而改寫 |
| `artifacts/` | 生成影像等大型產物 | 不入版控；以 manifest 描述，不把絕對位置當資料身份 |
| `runtime/` | 派工狀態 | 不與研究 trace 混放；完成狀態須綁定設定與輸入雜湊 |
| `reports/` | 生成 HTML | 不入版控；模板來源另放 |
| `archive/` | 已結束研究的證據 | 保留原文與數值；補封存 metadata，避免重寫歷史為新方法 |

參數數字、模型版本及既有影像主鍵不是流程流水號。例如 `jpeg50`、`sd3`、`man_00` 有資料意義，不應套用「全部去掉數字」的機械規則。

# 4. 改名對照表與相依性

## 4.1 遷移原則

1. **路徑、Python 識別字、CSV 識別值分開遷移。**
2. 原始 CSV 內容不因改名而重寫；可以搬檔，但須保存對照及雜湊。
3. 舊條件名稱只有在方法、設定、輸入及產物身份完全相同時，才可建立等價別名。
4. 歷史資料無法確認語意時保留原名，另加描述欄；不根據縮寫猜新名。
5. 封存區只先搬外層，不同時全面改寫內部檔名。

下表相依代碼供後續對照表引用；每一列都必須檢查該列列出的集合。

| 代碼 | 必查相依範圍 |
|---|---|
| **P** | Python import、`sys.path`、動態載入、module／function 字串 |
| **R** | Python 與 shell 的相對路徑、絕對路徑、CWD、環境變數 |
| **C** | JSON／YAML、CLI 預設值、條件 registry |
| **T** | 測試 import、fixture、CSV 路徑、斷言 |
| **G** | `.gitignore`、`.gitattributes`，含搬出原作用範圍後失效的規則 |
| **D** | README、HANDOFF、STATUS、docstring、技能及模板 |
| **V** | CSV 的檔案位置、路徑欄、arm／condition／variant、來源說明 |
| **N** | 遠端程式、影像目錄、工作清單、sentinel、家目錄腳本及連結 |
| **M** | repo 外的 Claude 記憶及其相互連結 |
| **U** | 已發布 artifact 與報告來源清單 |

## 4.2 目錄與入口文件

| 舊路徑 | 建議新路徑 | 相依及處置 |
|---|---|---|
| `A/main_table/` | `main_table/` | **P R C T G D V N M U**。移出前先解除 `paths.py`；CSV 中 NFS、Windows 與相對路徑走 resolver。遠端本來就是 `main_table/`，不能再套一層 `anti-purification/`。 |
| `A/`，排除主表 | `archive/anti_purification/` | **P R C T G D V N M U**。在共用套件完成移植、兩條活動線不再讀它後才搬；原 `runs/` 內部樹保留。 |
| `F/` | `archive/frequency_phase/` | **P R T D N M**。先處理兩條延遲 import；文件標示封存可執行性。 |
| `M/code/` | `main_table/src/immunization_benchmark/`，shell 另入 `scripts/` | **P R C T D N M**。同目錄 import 改為套件 import；不用 `code/` 同時放 Python、shell、JSON。 |
| `L/code/` | `lab/src/immunization_lab/` | **P R C T D N M**。拆除 color／style 的互相 import，入口與方法分開。 |
| 共用的 `A/src/` 實作 | `packages/immunization_core/src/immunization_core/` | **P R C T D M**。是抽取固定版本，不是直接搬空歷史來源；移植閉包與動態依賴需驗證。 |
| 根 `HANDOFF.md` | `archive/anti_purification/docs/HANDOFF.md` | **D M U**。移除其全域入口權限；根 README 指向各區交接入口。 |
| 根 `COLOUR_LINE.md` | `archive/anti_purification/docs/COLOR_LINE.md` | **D M U**。保留已刪除實驗的紀錄、四個 artifact 及撤回標示；不把舊方法改成 `color`。 |
| `L/docs/DESIGN.md` | `lab/docs/color/DESIGN.md` | **D M**，加上 `HANDOFF.md` 與程式 docstring 引用 |
| `L/docs/STYLE_PROMPT.md` | `lab/docs/style_prompt/DESIGN.md` | **D M**，加上 paper metadata、排程引用 |
| `.claude/skills/defence-report/` | `.claude/skills/defense-report/` | **D R M U**。同步技能內 asset 相對路徑及指令名稱；歷史發布頁不改 URL。 |

## 4.3 活動程式逐檔對照

下表的新檔名置於所屬專案 `cli/`，共用邏輯由 `immunization_core.pipelines` 提供；不能再維護三份完整 driver。

| 舊檔案 | 新名稱／拆分 | 直接相依證據與必要處置 |
|---|---|---|
| `M/code/defence_run.py` | `generate_defenses.py` | **P R C T D N**；`test_conditions.py:27`、主表條件表；`color_row_chain.sh`、原生／等失真 CSV schema |
| `M、L/code/edit_preflight.py` | `run_edits.py` | **P R C T D V N M**；`color_row_chain.sh:42、53`、`L/arm_chain.sh:51、61`、`style_prompt_round.sh:47`；FLUX／SD 系列 import 它的 helper |
| `M、L/code/purify_run.py` | `apply_purifiers.py` | **P R C D V N**；`edit_retention.py:51` import `PURIFIERS`、`label`；兩條 chain 的七道清單 |
| `M、L/code/edit_displacement.py` | `measure_edit_displacement.py` | **P R D V N**；`edit_retention.py:50`、`passthrough_readout.py:51`、`style_prompt_readout.py:26` import `subject_mask` |
| `M、L/code/edit_retention.py` | `measure_purified_displacement.py` | **P R C D V N**；`L/scripts/readout.sh:24`；幾何遮罩版本不得被舊副本覆蓋 |
| `M/code/edit_displacement_flux.py` | `measure_flux_displacement.py` | **P R D V N**；FLUX CSV 及未防禦 arm 配對 |
| `M/code/edit_flux_preview.py` | `run_flux_edits.py` | **P R C D V N**；`color_row_chain.sh:66`、`flux_full_queue_a.sh:11`；續跑鍵與固定輸出 CSV 必須一起處理 |
| `M/code/edit_ultraedit_full.py` | `run_ultraedit_edits.py` | **P R C D V N**；`color_row_chain.sh:60`；讀取主線淨化輸入及輸出供共用讀數使用的版面 |
| `M/code/edit_sdedit_preview.py` | `sweep_sdedit_parameters.py` | **P R C D V N**；模型、strength、guidance 掃描是其實際職責 |
| `M/code/edit_sd_family_preview.py` | `sweep_editor_parameters.py`＋`editors.py` | **P R C D V N**；`edit_ultraedit_full.py:54` 重用載入函式，不能只改 CLI 檔名 |
| `M/code/sd_family_offtarget_readout.py` | `measure_off_target_changes.py` | **P R D V N**；47 行直接讀 `sys.argv[1]`，48、72 行由 batch 組 CSV 名稱 |
| `M/code/immunise_as_condition.py` | `import_defense_artifacts.py` | **P R C D V N M**；`color_row_chain.sh:34`、`STATUS.md:26`；它是匯入與重算保真，不是求解器 |
| `M/code/metrics_union.py` | `measure_additional_metrics.py` | **P R D V N M**；`STATUS.md:72–74` 的 CPU／影像鏡像說明、四類讀數與 VMAF 路徑 |
| `M/code/passthrough_readout.py` | `measure_additive_transfer.py` | **P R D V N M**；`results/PASSTHROUGH.md`、native／aligned 防禦圖；移除 import 時強制找目錄 |
| `M、L/code/paths.py` | `artifacts/layout.py`＋明確設定 | **P R C T D N M**；取消尋找兄弟目錄，舊環境變數只作明確過渡別名 |
| `M/code/ultraedit_sd3_pipeline.py` | `third_party/ultraedit/pipeline.py` | **P R D**；載入端、原始 Apache header；補來源 commit／內容雜湊，不能只記 `main` |
| `L/code/color_defence.py` | `methods/color/`＋`generate_color_defenses.py` | **P R C D V N M**；style、fidelity、style readout import helper；`defence_cmd.sh:10` |
| `L/code/style_prompt_defence.py` | `methods/style_prompt/`＋`generate_style_prompt_defenses.py` | **P R C D V N M**；color 的 `CrossAttnObjective`、style 排程、可行性旗標與 hook 生命周期 |
| `L/code/defence_fidelity.py` | `measure_defense_fidelity.py` | **P R D V N M**；`queue_worker.sh:121`；歷史錨點 0.3344 與來源字串保留 |
| `L/code/style_prompt_readout.py` | `measure_style_prompt_edits.py` | **P R D V N M**；`style_prompt_round.sh:69`；未防禦分母路徑、空資料與配對完整性 |

對 `A/scripts/` 的同名副本，建議封存原名與行為，不同步改寫成活動入口；若要重新啟用，再依上表移植。

## 4.4 Shell、設定與內部識別字

| 舊名稱 | 新名稱 | 相依範圍／理由 |
|---|---|---|
| `M/code/flux_full_queue_a.sh` | `M/scripts/run_flux_conditions.sh` | **R C D N**；內容是任意條件迴圈，不是固定 queue A |
| `M/code/color_row_chain.sh` | `M/scripts/evaluate_color_condition.sh` | **R C D V N M**；匯入、編輯、淨化、跨編輯器及狀態檔一起遷移 |
| `L/scripts/arm_chain.sh` | `evaluate_condition.sh` | **R C D N M**；實作只跑 ip2p，檔頭卻寫兩場景 |
| `L/scripts/defence_cmd.sh` | `generate_condition.sh` | **R C D N M**；條件 registry 改由設定維護 |
| `L/scripts/readout.sh` | `measure_condition_results.sh` | **R D V N M**；先修退出碼再改名 |
| `L/scripts/run_on_card.sh` | `run_with_gpu_lease.sh` | **R D N M**；所有 queue、style、記憶中的呼叫必須同步 |
| `L/scripts/style_prompt_round.sh` | `run_style_prompt_jobs.sh` | **R C D N M**；工作清單、鎖、租約名称及容量政策 |
| `M/code/prompt_sets_ultraedit.json` | `M/configs/prompts/ultraedit_templates.json` | **R C D N**；內容為六種句型，variant 值保留 |
| `M/code/prompt_sets_ultraedit_round2.json` | `M/configs/prompts/ultraedit_noun_placement_variants.json` | **R C D V N**；內容為 `add`、`add_noun`、`placed`，不是一般「第二輪」 |
| `src/defense/immunise.py` 的活動移植版 | `defenses/optimization.py` | **P T D M**；`immunise.py`、`immunise_patch.py`、`paper_baseline.py`、lab 方法與相關測試 |
| `optimise_carrier`、`randomise_carrier`、`quantise` | `optimize_carrier`、`randomize_carrier`、`quantize` | **P T D M**；保留一段明確 API 相容期；不改 CSV 的 `free_*` 欄 |
| `baselines/dct_watermark.py` | `baselines/djsma.py`，僅在保留並啟用時 | **P T D**；模組 docstring 與 `run_djsma()` 表明實作為 DJSMA，並非一般 watermark |
| `WACV_ALLOW_TF32` | `IMMUNIZATION_ALLOW_TF32` | **R C T D N M**；見 `A/src/utils/device.py:10`；預設值與精度行為保持不變 |
| `IMMUNISATION_SOURCE_HOME` | 取消來源定位用途 | **P R D N M**；改用套件安裝；產物位置另設 `--artifacts-root`，不能混為同一變數 |

`NCF`、`PGD`、`SDS`、`DCT`、`LPIPS`、`IP2P` 等有明確文獻或模型意義，可保留並在 glossary 解釋。`geomctl`、`geommulti`、`maj`、`pub_id` 等本地縮寫則應在新介面展開。

## 4.5 舊設定檔的名稱

這些設定屬歷史實驗。**封存時不必逐一實體改名**；下表提供重新啟用或建立描述性索引時的名稱。原 `variants[].name`、seed、科學參數與既有結果對應維持不變。

所有列共同相依為 **R C D V N M**：設定內 `manifest`／target 路徑、呼叫腳本、`runs/*/README.md`、CSV variant、遠端啟動腳本及記憶。

| `A/configs/` 舊檔名 | 描述性名稱 |
|---|---|
| `advcf_anchor.json` | `advcf_distortion_caps.json` |
| `advcf_eot.json` | 保留 |
| `advcf_objective.json` | `advcf_objective_comparison.json` |
| `advcf_pilot.json` | `advcf_loss_transfer.json` |
| `advcf_portraits.json` | `advcf_portrait_curves.json` |
| `advcf_portraits_cap.json` | `advcf_portrait_deltae_cap.json` |
| `advcf_steps900.json` | `advcf_iteration_budget.json` |
| `advcf_term_scale.json` | `advcf_loss_scaling.json` |
| `advcf_texture_sds.json` | 保留 |
| `evaluate_colour_purify.json` | `evaluate_color_normalization.json` |
| `evaluate_geom_multi.json` | `evaluate_patch_geometry.json` |
| `evaluate_heldout.json` | 保留 |
| `evaluate_instruction.json` | 保留 |
| `evaluate_instruction_seeds.json` | `evaluate_instruction_seed_sweep.json` |
| `evaluate_patch.json` | 保留 |
| `evaluate_pilot.json` | `evaluate_accessory_edits.json` |
| `evaluate_pilot_preflight.json` | `evaluate_undefended_accessory_edits.json` |
| `evaluate_plateau.json` | `evaluate_plateau_curves.json` |
| `evaluate_screen.json` | `evaluate_candidate_defenses.json` |
| `immunise.json` | `generate_color_defenses.json` |
| `immunise_patch.json` | `generate_material_patches.json` |
| `immunise_patch_canvas.json` | `generate_canvas_patches.json` |
| `immunise_patch_geomctl.json` | `generate_geometric_patch_control.json` |
| `immunise_patch_geommulti.json` | `generate_geometric_patches.json` |
| `immunise_patch_lowlevel.json` | `generate_figure_patches.json` |
| `immunise_patch_print.json` | `generate_geometric_prints.json` |
| `immunise_patch_print_face.json` | `generate_face_objective_prints.json` |
| `immunise_patch_print_full.json` | `generate_garment_prints.json` |
| `immunise_patch_print_ladder.json` | `generate_print_distortion_sweep.json` |
| `immunise_patch_print_ring.json` | `generate_ring_prints.json` |
| `immunise_patch_smiley.json` | `generate_smiley_patches.json` |
| `immunise_patch_universal.json` | `generate_image_target_patches.json` |
| `paper_baseline_classifier.json` | `color_classification_baselines.json` |
| `paper_baseline_trio.json` | `color_ip2p_baselines.json` |

`immunise_patch_print_full.json` 的 `full` 指整件衣物支撐，不是流程完成程度；因此应按內容改為 `garment`，不能統一刪除 `_full`。

## 4.6 結果檔與資料目錄

以下是**純搬檔／改檔名**方案，CSV bytes 與列值保持不變。相依均含 **R T G D V N M U**，其中 `V` 必須區分「CSV 自己的位置」與「CSV 裡記錄的影像位置」。

| 舊路徑／檔名 | 新路徑／檔名 |
|---|---|
| `M/results/defence_<condition>.csv` | `results/defense_<condition>.csv` |
| `M/results/aligned/defence_<condition>_aligned.csv` | `results/aligned/defense_<condition>.csv` |
| `M/results/flux_full_<arm>.csv` | `results/flux/edits_<arm>.csv` |
| `M/results/displacement_flux.csv` | `results/flux/displacement.csv` |
| `M/results/ultraedit_full/<arm>.csv` | `results/ultraedit/edits/<arm>.csv` |
| `M/results/displacement_ultraedit.csv` | `results/ultraedit/displacement.csv` |
| `M/results/retention_ultraedit.csv` | `results/ultraedit/retention.csv` |
| `flux_preview.csv` | `sweeps/flux/guidance_3p5.csv` |
| `flux_preview_g2.csv` | `sweeps/flux/guidance_2p0.csv` |
| `flux_preview_truecfg.csv` | `sweeps/flux/true_cfg_3p5.csv` |
| `sdedit_preview.csv` | `sweeps/sdedit/sd15_strength.csv` |
| `sdedit_preview_stable-diffusion-2-1-base.csv` | `sweeps/sdedit/sd21_base_strength.csv` |
| `sdedit_preview_stable-diffusion-2-1.csv` | `sweeps/sdedit/sd21_v_prediction.csv`，加相容性限制 |
| `sdedit_preview_stable-diffusion-v1-5_sdedit_guidance_sweep_sd15.csv` | `sweeps/sdedit/sd15_guidance.csv` |
| `sd_family_sd3_ultraedit_quick.csv` | `sweeps/ultraedit/image_guidance.csv` |
| `sd_family_sd3_ultraedit_low_guidance.csv` | `sweeps/ultraedit/text_image_guidance.csv` |
| `sd_family_ultraedit_prompt_round2.csv` | `sweeps/ultraedit/noun_placement_variants.csv` |
| `sd_family_ultraedit_prompt_sweep_a.csv` | `sweeps/ultraedit/prompt_templates_man_00_woman_00.csv` |
| `sd_family_ultraedit_prompt_sweep_b.csv` | `sweeps/ultraedit/prompt_templates_man_01_woman_01.csv` |
| 上述三份對應的 `sd_family_offtarget_*.csv` | 同目錄、同語意名稱加 `_off_target.csv` |
| `sd_family_sdxl_ip2p_grid.csv` | `sweeps/sdxl_ip2p/guidance_portrait_pair.csv` |
| `sd_family_sdxl_ip2p_all8.csv` | `sweeps/sdxl_ip2p/guidance_portraits.csv` |
| `sd_family_sdxl_ip2p_high_image_guidance.csv` | `sweeps/sdxl_ip2p/high_image_guidance.csv` |
| 對應的 SDXL off-target 表 | 同目錄加 `_off_target.csv` |
| `sd_family_ip2p_si18_reference.csv` 及其 off-target 表 | `sweeps/ip2p/reference_edits.csv`、`reference_off_target.csv` |
| `L/results/exp/` | `L/results/color/variants/` |
| `A/data/set0817/` | 封存保留原名；索引顯示為 `curated_editing_inputs` |
| `data/carrier_catalogue.yaml`、`decoy_catalogue.yaml` | 活動移植版用 `carrier_catalog.yaml`、`decoy_catalog.yaml` |
| `portrait_manifest.json`、`portraits_manifest.json` | 活動移植版分別用 `omniedit_portraits.json`、`curated_portraits.json`；不可互換 |

上述 preview／sweep 名稱已按 CSV 中實際模型、參數、影像集合核對；例如 true-CFG 表只有 2 列，不能沿用 README「各 4 格」的概括。

## 4.7 不應改寫的識別值

| 識別值 | 處置與理由 |
|---|---|
| `colour_curve_ours` → `color` | **禁止建立等價替換。** `M/STATUS.md:29–37` 明示兩者設定及防禦圖 LPIPS 不同；等失真錨點仍屬舊方法。 |
| `L/results/*fidelity*.csv` 中 `colour_curve_ours` | 保留。全量檢索找到 4 份表、32 列，出現在 `anchor_source`，是歷史錨點來源，不是待更新的 active condition。 |
| `ab_warp_ch_comm_max` | 記憶將它描述為 `color` 的前名，但任何跨檔等價仍須以設定及產物 hash 驗證；不能擴及其他 `ab_warp_*`。 |
| `dia_pt`、`dia_r`、`photoguard_c`、`photoguard_linf`、`dct_shield_y` | 有方法與預算差異，保留為穩定主鍵，另加顯示名稱。 |
| `r11`、`r13`、`cls` | 歷史字串保留；可在 catalog 加完整設定描述。缺完整遠端工作規格前，不替它們發明精確新方法名。 |
| `rand_a`／`rand_b`／`rand_c`、`*_repeat`、`shard1` | 歷史抽樣、重複或分片身份；不得為命名一致而合併。新資料使用明確 seed／replicate／shard 欄。 |
| `free_*`、`D_*`、`DT_*`、`disp_*` | 原始 schema 保留；對外表頭用描述性文字。更換 schema 必須版本化。 |
| 科學 seed、`jpeg50`、`rotate15`、影像 ID | 保留。它們影響重現或資料配對。 |
| UltraEdit CSV 的 `scenario=ip2p` | 已存在的相容版面不得原地改成另一鍵值；新 schema 增加獨立 `editor`／`protocol_id`。 |

## 4.8 路徑欄、遠端、記憶與發布連結

CSV 已存在下列四種路徑，不能用一次字串替換處理：

| 型態 | 實際例子 | 遷移方式 |
|---|---|---|
| NFS 絕對路徑 | `results/aligned/displacement_aligned.csv` 的 `/nfs/.../runs/edit_defended_aligned/...` | 依來源根及產物角色解析 |
| Windows 絕對路徑 | `metrics_fidelity_union.csv` 的 `C:/image-immunization/anti-purification/main_table/images/...` | 使用 `PureWindowsPath`／明確字串映射；不可依執行平台的 `Path.is_absolute()` 猜測 |
| 遠端 repo 相對路徑 | `displacement.csv` 的 `runs/edit_defended/...` | manifest 指定原始根與產物組 |
| lab 相對路徑 | `L/results/displacement.csv` 的 `lab/runs/...` | 解析為 lab 自有 artifact，不能退回主表同名檔 |

必查路徑欄包括 `data_root`、`source_png`、`input_png`、`png`、`original_png`、`defended_png`、`undefended_png`、`defence_png`、`output_png`、`file`、`path`、`screen_dir`。`dir_norm`、`direction` 是數值，不是路徑，不能用欄名包含 `dir` 的規則替換。

**遠端相依**

| 已知位置 | 依據及要求 |
|---|---|
| `~/image-immunization/{src,scripts,configs,data,runs}` | 本機 `A/` 對應遠端根；不能同步本機整棵樹後假定位置相同 |
| `~/image-immunization/main_table/`、`lab/` | 分別對應本機 `M/`、`L/` |
| `~/env.sh` | `remote_tree.txt:17`；shell 明確 source，內容本次不可驗 |
| `~/edit_aligned.sh`、`eps_aligned.sh`、`eps_scan_coarse.sh`、`leg_{a,b,c,d,e}.sh`、`pg_and_sifm.sh`、`readout_aligned.sh`、`pg_interp.py` | 摘要列出，但 repo 沒有完整鏡像；改名前須逐檔取得內容並查引用 |
| `runs/*/_launch/` | 目錄摘要可見，深層腳本未完整列出；不能宣告沒有舊名 |
| `lab_leases/`、style 鎖、queue 規格與 sentinel | 會引用工作名稱及路徑；搬目錄不得複製舊完成狀態後直接繼續 |
| 遠端 `main_table` 舊顏色資料及 `.bak` | 摘要仍有本機未保留項目；同步不可使用未審核的刪除規則 |

**Claude 記憶**

| 記憶檔 | 需要更新的內容 |
|---|---|
| `MEMORY.md`、`current-state-in-handoff.md` | 全域入口改為根 README；後者 `:11` 的根 HANDOFF 路徑與遠端舊根需要分開處理 |
| `main-table-entry-is-status.md` | 新主表根、管轄與 vendor 版本邊界 |
| `lab-line-current-state.md` | 自足程度、入口、已移除臂、共用套件使用方式 |
| `color-is-the-lab-method.md` | color 程式及設定位置；歷史方法身份保留 |
| `style-prompt-line.md`、`dont-touch-others-work.md` | color／style 模組拆分後的責任路徑 |
| `main-table-next-three-jobs.md` | `metrics_union.py`、UltraEdit 結果路徑 |
| `additive-passthrough-inflates-dt.md`、`no-distortion-metric-locks-the-others.md` | PASSTHROUGH、fidelity、aligned 路徑與適用資料版本 |
| `orchestrator-adoption-relaunches-stopped-jobs.md`、`parallel-run-on-card-races.md` | 排程與租約入口 |
| `all-gpu-work-goes-remote.md`、`remote-gpu-gotchas.md` | 分別仍有 `WACV-s4`、`WACV-s3`；不能一律當成現行根 |
| `python-write-text-breaks-shell-scripts.md` | `:25` 指向不存在的 `test_shell_line_endings.py` |
| `ab-warp-is-the-preferred-look.md` | `:15` 指向已刪除 `L/docs/AB_WARP_NEXT.md` |
| `sdedit-strength-swaps-the-face.md`、`no-standing-multiseed-requirement.md`、`readout-is-not-face-identity-only.md` | 已退役產物或刪除腳本只能作歷史引用，不再當執行指示 |
| `main-table-per-image-gpu-cost.md` | 舊顏色方法的成本不能沿用到 color；應連結指定方法版本的量測 |

不要為統一英式／美式拼法而批次改名整個記憶庫；它的內部連結也是相依。优先修正文中的路徑與適用範圍。

**已發布報告**

共找到 **14 個不同 artifact URL**：

| 來源 | 數量 | 處置 |
|---|---:|---|
| `COLOUR_LINE.md:14–17` | 4 | 全數登錄，保留撤回標記 |
| 根 `HANDOFF.md:307–312` | 6 | 移入歷史發布清單，取消未核實的「現行」標籤 |
| `M/STATUS.md:75–76` | 2 | 主表明記舊顏色列版本；aligned 保留其錨點 |
| `A/runs/print_patch/README.md:143–144` | 2 | 保留歷史實驗與資料範圍 |

本機改名不會自動更新 artifact 內容。本次未開啟外部連結，因此不判定它們是否仍可存取。新 `ARTIFACTS.md` 應記 URL、來源 commit、CSV 雜湊、方法版本、是否撤回／被取代。

# 5. 程式風格與文件清理規範

## 5.1 可直接執行的規範

| 項目 | 規範 |
|---|---|
| 語言 | 自有程式註解與說明採客觀繁體中文；識別字採英文。第三方來源保留原文及授權 header。 |
| 拼法 | 新的自有 API 使用 `color`、`defense`、`immunize`、`optimize`、`normalize`、`gray`、`center`。文獻標題、上游 API、歷史 CSV 不改。 |
| 模組 docstring | 首段寫責任；接著寫輸入／輸出、張量值域、遮罩極性、單位、副作用、失敗條件及來源。實驗歷程移入文件。 |
| 函式 docstring | 公開函式與非直觀數學操作必寫契約；簡單私有 helper、getter 不為提高覆蓋率而補贅述。 |
| 註解 | 解釋公式來源、非直觀限制及選擇理由；不逐行翻譯程式，不寫「這次修好」「先前踩坑」作為永久說明。 |
| 密度 | 不訂固定註解比例。長度問題以重複、過時及阻礙查找判斷；數學與來源說明應保存。 |
| 型別與命名 | 公開邊界加型別；避免讓 `src`、`BASELINES`、`R`、`C` 承擔不明角色。數學局部變數可保留 `x`、`z`、`eps`。 |
| CLI | 保留現有一致的 kebab-case；統一 `--data-root`、`--artifacts-root`、`--output-dir`／`--output-csv`、`--config`。舊參數提供明確別名，避免立即破壞遠端腳本。 |
| CLI 載入 | `--help`、設定驗證及產物清單檢查不得載入權重、聯網或建立 CUDA context。 |
| 設定 | 協定參數只有一份正本；記錄解析後完整設定，不能依賴 shell 與 Python 預設的隱性組合。 |
| 錯誤 | 必要輸入、必要指標、協定不符立即失敗；選配功能缺席須有固定欄位與原因。禁止以空欄、零值或成功退出碼混淆失敗。 |
| 產物 | 原始數值不可覆寫；逐列記錄可採單一 writer 的 journal，彙整 CSV 使用原子替換。 |
| 完成判定 | 退出碼、預期鍵集合、必要檔案、協定 digest 都通過才完成。這是工程驗收，不是新增研究成功門檻。 |
| 裝置 | GPU 工作裝置不符應明確失敗；純 CPU 分析明確標示。不能讓錯誤 GPU 編號靜默轉 CPU。 |
| 測試 | 區分無模型單元測試、固定資料契約測試、需權重測試；缺依賴與程式錯誤不得使用同一 skip。 |
| 科學協定 | 結構遷移不改 LPIPS 後端、resize、插值、遮罩、seed、量化、precision 或樣本排除規則。需修正時另立資料版本。 |
| 文件敘述 | 寫「在指定資料組觀察到……」，列樣本、設定、數值與限制；避免由代理量直接推定免疫成效。 |
| 歷史文件 | 保留原始證據與原有決策，外加「適用版本／已取代段落」說明；不得將舊結論改寫成新方法的結果。 |
| 時間 | 正文避免「最新」「這輪」「稍後補」；使用穩定狀態與來源 commit。機器 metadata 可保存必要的時間戳。 |
| 提交 | 英文 commit message；機械搬檔、行為修正、文件更新分開提交。 |

註解密度的具體例子：`dayn.py` 850 行中 docstring 約 321 行，`sd.py` 1,256 行中約 499 行；另一方面，lab 的 color／style 檔將大量方法、求解與 I/O 集中在單檔。問題是責任與資料來源難查，不能簡化成「註解太多」或「註解太少」。

## 5.2 活動程式逐檔清理

第 4.3 節已逐檔列出 25 個活動 Python 檔的改名與相依；以下補充清理驗收。

| 檔案 | 清理內容 |
|---|---|
| `M、L/edit_preflight.py` | 抽出資料載入、編輯協定、模型呼叫與身份讀數；保留已儲存分母的協定檢查 |
| `M、L/purify_run.py` | 七道設定由協定檔讀取；避免 shell、Python、文件各抄一份 |
| `M、L/edit_displacement.py` | 兩份內容相同，可作優先抽取對象；LPIPS 分區契約與配對鍵獨立測試 |
| `M、L/edit_retention.py` | 保存 `purified_mask()` 行為；lab 的條件過濾選項納入顯式 API |
| `M/defence_run.py` | 條件 registry、求解與記錄分開；不把原生與 aligned 設定混成預設 |
| `M/edit_displacement_flux.py` | 與通用配對讀數共用核心，保留 FLUX 的輸入尺寸協定 |
| `M/edit_flux_preview.py` | 加完整 resume identity、輸出位置參數；既有 CSV 不因新設定被跳過 |
| `M/edit_ultraedit_full.py` | 明確讀淨化輸入 manifest；缺圖不刪舊列；編輯器身份不再借用 scenario 表示 |
| `M/edit_sd_family_preview.py` | 模型 loader 脫離 sweep driver，UltraEdit 不再 import 掃描入口 |
| `M/edit_sdedit_preview.py` | 保存 prediction-type 限制；記錄每格完整掃描參數 |
| `M/sd_family_offtarget_readout.py` | 改 `argparse`；處理空表、缺檔與跨平台 CSV 路徑；保留遮罩外配件的解讀限制 |
| `M/immunise_as_condition.py` | 匯入來源須具備方法設定、影像 hash、輸入 hash；匯入與保真量測分開記錄 |
| `M/metrics_union.py` | 明確 required／optional 指標；統一 CSV resolver；CPU 模式列入環境契約 |
| `M/passthrough_readout.py` | 延後檢查實際使用的 native／aligned 輸入；保留 additive transfer 的定義 |
| `M、L/paths.py` | 刪除跨專案探索及 import 時檔案搜尋；以專案設定定位資產 |
| `M/ultraedit_sd3_pipeline.py` | 保存第三方原碼、license；獨立記錄補丁。127 行超過 100 字元，宜作 formatter 排除，避免製造無關差異 |
| `L/color_defence.py` | carrier、限制、objective、solver、CLI、CSV writer 拆分；移除對 style CLI 的 import |
| `L/style_prompt_defence.py` | 修 `feasible`；明確管理 hook；保存選點政策；trace 與最終 CSV 納入證據保存 |
| `L/defence_fidelity.py` | 更新仍描述十二臂的 docstring；明確區分實際限制與文字 `budget` 標籤，歷史錨點保持原來源 |
| `L/style_prompt_readout.py` | `:94–95` 在空結果時會索引空列表，應先報明缺哪組輸入；對照集合與缺格也應明列 |

Shell 逐檔處置：

| 檔案 | 清理內容 |
|---|---|
| `A/scripts/free_cards.sh` | 作為容量查詢工具，政策由統一派工層執行；不要將其「空卡」輸出當完整租約保證 |
| `A/scripts/conditioning_probe_run.sh` | `:4` 的 `WACV-s4` 移到設定；封存原協定 |
| `A/scripts/patch_pipeline_run.sh` | 同上；工作目錄 CRLF 必須獨立修正與驗收 |
| `M/code/color_row_chain.sh` | 以 manifest 取代目錄存在判定；輸入匯入與產物引用解除 lab 路徑依賴 |
| `M/code/flux_full_queue_a.sh` | `source env.sh` 後再設定 CWD；使用共用租約與明確條件清單 |
| `L/scripts/arm_chain.sh` | 檔頭兩場景說明改為實際 ip2p；完成旗標綁定輸入與設定 |
| `L/scripts/defence_cmd.sh` | 七個 case 與文件保留四條件的狀態對齊；設定移出 shell |
| `L/scripts/queue_worker.sh` | 全域取卡原子性、單一結果 writer、取消與失敗分開；啟動前驗證 `FID_ARMS` |
| `L/scripts/readout.sh` | 必先修失敗傳遞，再整理名稱 |
| `L/scripts/run_on_card.sh` | 租約獲取、擁有者驗證及釋放成為同一協定，避免刪掉別人的租約 |
| `L/scripts/style_prompt_round.sh` | 移除預設六卡及 style-only 計數；統一 CWD、取消、續跑與租約管理 |

## 5.3 共用來源與歷史 driver 清單

同列列出的每個檔案適用該處置；未列出確定缺陷的檔案，不代表已完成執行正確性驗證。

| `A/src/` 檔案 | 清理要求／依據 |
|---|---|
| `baselines/{advpaint,danp,dayn,dia,diffvax,mist,photoguard,promptflare,sifm,tdae}.py` | 保存來源與移植差異；將大段研究歷程移到各方法文件，程式保留公式、shape、值域與 hook 契約。TDAE 未進主表是已記錄決策，不當死碼刪除 |
| `baselines/{advdrop,blurguard,dct_shield,dct_watermark,diffusionguard}.py` | 區分原生實作、重建與變體；`dct_watermark` 的命名另處理；保留缺 SAM 等依賴時明確失敗的設計 |
| `baselines/{encoder_target,jpeg_codec,pgd,__init__}.py` | 核心算式、值域與 registry 契約保留；registry 的六個 spec 不等於主表十二條件，不應硬湊成同一清單 |
| `defense/{assets,ncf_library,ncf_runner}.py` | 輸入來源、雜湊與資料載入契約保留；消除依賴 CWD 的路徑 |
| `defense/{color_amplitude,delta_e_torch}.py` | 保存求解用可微版本與量測用 skimage 的區分，不為去重而混用 |
| `defense/{color_field,color_param,lowfreq_color,ncf_param,recoloradv_param,lab_offset_field}.py` | 公開 carrier 介面統一；明記 amplitude 是否真正改變 render。`lab_offset_field` 無查得的靜態呼叫端，先標封存候選 |
| `defense/{carrier_search,color_search,composite}.py` | 搜尋、參數化、組合各自維持邊界；移除「旋鈕」等比喻式說明 |
| `defense/{carrier_mask,subject_mask}.py` | 主體／重繪／支撐三種遮罩極性明確命名；資料集預處理不與評估遮罩變換共用含糊函式 |
| `defense/{material_patch,patch_canvas,print_patch,target_patch,plateau_curve}.py` | 保留支撐外逐位元不變等數學契約；外觀判斷移出程式敘述，不宣稱參數化自動保證自然 |
| `defense/{conditioning_response,instruction_free,mainstream_terms,outside_terms,readout_terms,sds_terms}.py` | 公式、符號、固定驗證抽樣與訓練抽樣分離；研究觀察放文件，不由 loss 值宣稱編輯效果 |
| `defense/{eot,purify_aware,param_pgd}.py` | 不同 EOT 與兩階段協定明確命名；相位載體改選配相依，不能隱性讀封存區 |
| `defense/immunise.py` | 拆出 optimization API；`:17–18` 稱可行時懲罰為零，但 `:142` 含 `λ·g`，應精確說明線性項與平方違反項，不能只改文字掩蓋公式差異 |
| `defense/{criterion,uniformity,victim_classifier}.py` | 區分歷史複合判準、診斷量與正式讀數；保留原 schema，不將其自動升格為研究成功門檻 |
| `metrics/{acutance,aesthetic,arcface,identity,layout,naturalness,regional,standard,suite}.py` | 定義輸入、輸出、缺臉／空區域與後端；`naturalness` 無查得的靜態呼叫端，先記封存候選；`standard.py:1` 的日期式敘述移到 provenance |
| `models/ip2p.py` | 保存影像條件 scaling 與 batch generator 契約 |
| `models/sd.py` | 1,256 行涵蓋 SD、inpainting、SDXL，按模型 adapter 拆分；拆分不能改取樣、precision、prediction type |
| `purify/{adverse_cleaner,diffpure,impress,ops}.py` | registry、依賴可用性、真算子與 proxy 分開；保留明确拒絕近似替代的契約 |
| `utils/{artifacts,device,io}.py` | 產物寫入、裝置、純 I/O 分工；修正尺寸判定、原子寫入及環境變數名稱 |
| 各空 `__init__.py` | 不補形式化註解；新 package 只在需要時明確匯出 API |

43 個歷史 Python driver 的處置：

| `A/scripts/` 檔案 | 清理要求 |
|---|---|
| `baseline_run.py`、`dct_shield_run.py`、`defence_run.py`、`paper_baseline.py` | 標明對應任務與協定，保留歷史行為；共用 dataset／solver helper 移植時另建 API |
| `build_dataset.py`、`crop_portraits.py`、`make_masks.py` | 保留來源與裁切紀錄；覆寫資料集前須顯式目的地，不能當一般無副作用工具 |
| `fetch_commons_pool.py`、`fetch_imagenet_subset.py` | 分開下載、篩選、metadata；保留前者有限重試，修後者廣泛吞例外 |
| `screen_candidates.py`、`select_subjects.py` | 人工選擇與自動量測分開；刪除「身分是唯一主軸」等不再適用的概括 |
| `colour_shift_control.py` | 活動版改 `measure_color_transfer.py`；保存 LUT 重建假設，不與 additive transfer 合併 |
| `conditioning_probe.py` | 量測輸出契約保留，移除由局部反應直接判定研究方向的強敘述 |
| `dct_shield_eps_calibration.py` | 保留係數域預算與像素域不同的說明；清理已刪除掃描腳本引用 |
| `edit_preflight.py`、`edit_displacement.py`、`edit_retention.py`、`purify_run.py`、`immunise_as_condition.py` | 封存副本，不以活動版直接覆蓋；記錄幾何遮罩差異 |
| `edit_distance_panel.py`、`evaluate_defence.py` | 明記 `gain` 正負方向與兩側淨化定義；保留 `map_box` 修正來源 |
| `field_readout.py`、`free_proxy_readout.py`、`readout_panel.py`、`metrics_union.py` | 固定 schema、輸入 hash、失敗狀態；代理量與編輯讀數分開 |
| `immunise.py`、`immunise_patch.py` | 若移植則用 `generate_*` 名稱；維持不讀評估指令的防線 |
| `instruction_preflight.py`、`instruction_probe.py`、`instruction_vqa.py` | 校準用途與研究主讀數分開；不新增常設多種子或 VQA 要求 |
| `ip2p_helmet_contactsheet.py`、`ip2p_helmet_review.py`、`ip2p_helmet_sweep.py` | 保留格數、既有列不變、人工判讀配對的驗證；路徑移到設定 |
| `patch_probe.py`、`plateau_probe.py`、`print_probe.py` | 不適用情況用結構化 status 記錄；不能只有 console 訊息 |
| `reedit_ip2p.py`、`reeval_edits.py` | 新工作不原地覆寫既有 CSV；舊檔明示其覆寫副作用 |
| `sds_sign_check.py`、`term_gradient_scale.py` | 診斷與正式評估分開，保留真模型需求標示 |
| `summarise_audited.py`、`summarise_screen.py`、`summarise_diagnosis.py` | 活動版用 `summarize_*`；先修 diagnosis 語法；無效數字不可無紀錄地變成缺值 |

頻域封存的 12 個 Python 檔：

| 檔案 | 處置 |
|---|---|
| `scripts/freq_baselines_run.py` | 補完整歷史環境後才可稱可執行；`baseline_run` 來源不能靠 CWD 猜測 |
| `scripts/phase_retention.py` | 標示缺 `codefense`；`:1–16` 的非幾何分母定義與主表兩側淨化不同，禁止直接合併 |
| `src/purify/freq_grid.py` | 改明確 namespace 或獨立選配套件，不塞入另一個 `src.purify` |
| `src/residual/{base,composite,latent_inject,lora_weights,lowrank,perceptual_weight,texture_rephase,spectral_split}.py` | 保留來源與能力介面；舊 `spec`／ARCHITECTURE 引用補歷史 commit；`spectral_split` 只標未查得活動引用 |
| `src/residual/__init__.py` | 保留封存，不以空檔補註解作清理成果 |

## 5.4 測試逐檔處理範圍

靜態盤點有 701 個 `test_` 函式定義；這不是測試通過數。

| 測試檔案 | 清理／遷移要求 |
|---|---|
| `M/tests/test_conditions.py` | 保留 11 個資料契約測試；擴及 FLUX／UltraEdit 的鍵集合及條件一致性 |
| `A/tests/test_baselines.py`、`test_blurguard.py`、`test_danp.py`、`test_dayn.py`、`test_diffusionguard.py`、`test_diffvax.py`、`test_sifm.py`、`test_tdae.py` | 隨對應套件移植；保留公式、值域、缺參數及 hook 還原測試，不以 registry 是否啟用刪除方法測試 |
| `test_defence_run_dayn.py`、`test_defence_run_diffvax.py`、`test_defence_run_sifm_danp.py`、`test_edit_preflight_defended.py` | 移到 pipeline／CLI 測試；移除靠檔案位置載入 driver 的方式 |
| `test_acutance.py`、`test_identity_metric.py`、`test_direction_similarity.py`、`test_image_similarity.py` | 將需權重測試與純函式測試分組；修 direction 的廣泛 skip |
| `test_suite_fid.py`、`test_suite_pairwise.py`、`test_suite_semantic.py`、`test_readout_panel.py` | 後端與選配依賴明確；缺 metric 的新 status 契約更新到測試 |
| `test_ip2p.py`、`test_ip2p_batch.py`、`test_inpaint_chain.py` | 保留模型輸入、scaling、generator、九通道與遮罩行為測試 |
| `test_ip2p_helmet_review.py`、`test_ip2p_helmet_sweep.py` | 保存不覆寫舊列與派工驗收測試；入口改名同步更新 |
| `test_carrier_search.py`、`test_color_search.py`、`test_composite.py` | 隨搜尋／組合模組移植；保留重現與參數邊界測試 |
| `test_chroma_isometry.py`、`test_chroma_rotation.py`、`test_color_amplitude.py`、`test_color_field.py`、`test_lowfreq_amplitude.py`、`test_lowfreq_color.py` | 保留同色／支撐外不變、幅度及色域契約；不因拼法統一改資料值 |
| `test_colour_shift_control.py` | 活動版改 `test_color_transfer.py`，保持 LUT 控制的獨立語意 |
| `test_conditioning_response.py`、`test_criterion.py`、`test_delta_e_torch.py`、`test_eot.py`、`test_instruction_free.py`、`test_mainstream_terms.py` | 保留數學與固定驗證抽樣測試；標示 criterion 屬歷史協定 |
| `test_make_masks.py`、`test_material_patch.py`、`test_outside_terms.py`、`test_patch_canvas.py`、`test_plateau_curve.py`、`test_print_patch.py` | 保留遮罩極性、支撐、梯度、可行性與 state round-trip 契約 |
| `test_purify_aware.py`、`test_purify_cr.py`、`test_purify_new_ops.py` | 區分共用算子集合與主表七道；缺權重分支不得導致主表協定被更換 |
| `test_readout_terms.py`、`test_recoloradv.py`、`test_sds_terms.py`、`test_uniformity.py`、`test_victim_classifier.py` | 隨對應模組移植；保留公式符號、正規化與梯度測試 |
| `A/tests/__init__.py` | 不需內容性修改 |

需要新增的測試集中於實際風險：讀數失敗是否傳回失敗碼、半完成目錄能否續跑、跨平台路徑解析、CSV 不覆寫、幾何遮罩、`--select last` 的可行性旗標、hook 還原及獨立目錄執行。无需為純檔名搬動增加模仿實作的測試。

## 5.5 文件分工與逐份清理

文件模板：

| 文件 | 必備內容 | 不應承擔 |
|---|---|---|
| 根 README | 專案目的、區塊地圖、入口、自足與資料保存原則 | 實驗結果、遠端即時狀態 |
| 子專案 README | 安裝、資料、CLI、輸出位置、固定協定入口 | 長篇交接歷程 |
| STATUS／HANDOFF | 責任範圍、已完成項目、已知限制、待裁定事項、下一個可執行工作 | 複製完整方法與結果表 |
| PROTOCOL／DESIGN | 威脅模型、輸入、方法、參數、讀數、配對及排除規則 | 排程時間、卡號、臨時處理 |
| 實驗 README | 問題、固定設定、變因、來源 CSV、樣本、觀察、限制 | 以結果代替未記錄的設定 |
| 方法 audit | 固定來源版本、論文與程式差異、本專案選擇、對應測試 | 未標版本的「目前無程式」 |
| ARTIFACTS | 本機／遠端映射、產物身份、發布 URL、來源版本 | 以一個 URL 宣稱所有內容皆為現行 |

入口及操作文件：

| 文件 | 診斷與處置 |
|---|---|
| 根 `README.md` | `:11、14、41` 失效入口；全面重寫專案地圖 |
| 根 `HANDOFF.md` | `:81–88` 舊顏色方法、`:298` 已移除 builder、`:307` 舊主表發布入口；改為歷史交接 |
| 根 `COLOUR_LINE.md` | `:3–5` 已說明實驗刪除，應封存保留；`:53`「現行操作點」改為該紀錄的基準，不能指向 lab color |
| `A/CLAUDE.md` | 共通規則抽到根；歷史專案範圍留本區；不再讓根 HANDOFF 控制所有子專案 |
| `A/README.md` | 題名仍是 WACV 頻域／相位；`:12–20` 多個不存在文件、`:26` 不存在 residual 位置；重寫封存說明 |
| `A/data/README.md` | 幾乎只描述 DAYN placeholder，指向不存在 `configs/base.yaml`、`src/data/dataset.py`；補實際資料集合與来源 |
| `A/docs/INDEX.md` | `:18` 入口仍是根 HANDOFF；保留命名規則，取消過時全域入口 |
| `A/docs/DIRECTION.md` | 三種載體及研究敘述屬舊線；標示適用範圍，不代替 lab 方向文件 |
| `A/docs/EVALUATION.md` | 保留七道與算子預設區分；更新幾何欄範圍、移動的 phase driver；將重現性觀察與原因推論分開 |
| `A/docs/OPERATIONS.md` | 舊 `WACV` 路徑、卡數及 cache 說明與 STATUS 不一致；改由環境 manifest／部署文件給出 |
| `A/docs/CONDITIONING_PROBE.md` | 保留量測定義与數據範圍；推論需限定該探針設定，不直接否定未測方法 |
| `M/README.md` | 補獨立安裝、輸入與產物契約；preview 列數依實際 CSV 修正；套件清單不等於測試通過 |
| `M/STATUS.md` | 保留唯一入口；「沒有遠端工作」只能作交接紀錄，不能作遷移時行程證據；發布版本與限制維持明示 |
| `M/docs/README.md` | 現為跨目錄參考索引；改連固定版本 audit 文件，而非要求上層存在 |
| `M/results/PASSTHROUGH.md` | 已明示舊顏色版本，應保留；`:75、83、118` 的退役／不存在產物引用需標歷史來源或證據缺口 |
| `M/results/aligned/README.md` | 錨點 0.3344 依使用者指示保留；`:150` 的相對 EVALUATION 路徑解析錯誤 |
| `L/HANDOFF.md` | 修自足、參數正本、保留分支與日期式標題；操作細節移到 runtime 文件 |
| `L/docs/DESIGN.md` | 保存 color 與退役方法的分界；`:13` 的共享產物連結改成匯入 manifest |
| `L/docs/STYLE_PROMPT.md` | 三張人像的結果範圍保留；`r11/r13/cls` 加設定對照；根 PDF 依賴改为明確來源清單 |
| `F/README.md` | `:40–54` 的 `sys.path` 操作不足以恢復執行；補封存完整性與缺項 |
| `.claude/skills/defence-report/SKILL.md` | 修失效 builder 依賴、比喻式標題及固定判定語句 |
| `.claude/skills/defence-report/assets/template.html` | `:242、260、328–335、669–670、731–732` 帶特定標題、數量、修辭與判定；改成資料驅動模板；`:594` 的 `data.js` 生成契約補齊 |
| `L/docs/paper/neucom_134591.json` | 是單行外部 bibliographic metadata，不是論文全文；保存原文，另建人可讀引用，不把出版日期改成工作日期 |

16 份文獻／來源文件：

| 文件，皆位於 `A/docs/reference/` | 清理處置 |
|---|---|
| `AUDIT_DANP.md` | `:19` 的「沒有官方程式」改成查證範圍與來源版本；保留未指定參數及本專案選擇 |
| `AUDIT_DAYN.md` | `:208–240` 是接入提案；補現在由 defence driver 使用的實作入口，不把提案當未完成事項 |
| `AUDIT_DCT_SHIELD.md` | 保留 `:91–101` 對 registry 與測試範圍的精確區分；更新移植後路徑 |
| `AUDIT_DIA.md` | `:28`「目前仍可存取」改成有來源快照的查證紀錄；公式與原碼差異保留 |
| `AUDIT_INPAINTING_METHODS.md` | 將「回答某次審查」整理為問題與證據；`:339` 行號修正應綁定來源 commit |
| `AUDIT_MIST_DIFFVAX.md` | `:626` 增補移植段與前段原始查證區分；保留模型、遮罩與訓練／推論差异 |
| `AUDIT_PROMPTFLARE_PHOTOGUARD.md` | 保留原始碼佐證與預算推導；避免再複製到各入口 docstring；第三方程式區塊不做文風重寫 |
| `AUDIT_PURIFIERS.md` | `:35` 的 404 狀態限於查證紀錄；`:852` 指向舊 DESIGN，改歷史來源；不稱本文所有算子都進主表 |
| `AUDIT_SIFM.md` | `:290–294` 的「本輪沒有改」改為整合狀態與對應 commit |
| `AUDIT_TDAE.md` | 保留 `:334–395` 的排除理由與證據；不可因有接入範例就自動啟用 |
| `BASELINE_ALIGNMENT.md` | `:249–274` 多個刪除腳本；標為歷史協定，不與主表 aligned 錨點混用 |
| `BASELINE_CANDIDATES.md` | `:3` 六條件、`:96` DAYN 未納入均已不合主表；分開歷史篩選與實際 registry |
| `BASELINE_PROVENANCE.md` | 適合作為來源正本；預算單位與参考數值協定繼續保留，路徑隨套件移植更新 |
| `BIBLIOGRAPHY.md` | 文獻與實作清單分開；移除未限定資料範圍的「沒有任何一篇」及「目前是空的」 |
| `SOURCE_AUDIT.md` | `:1–5` 已明示原文恢復，應保存；在外層索引標示 `:335` DiffVax、`:514` DIA-PT 等舊決策已被後續主表取代，不直接改歷史正文 |
| `SURVEY_FRONTIER.md` | `:7` 缺 `SURVEY_POSITIONING.md`；比喻式「佔地／旋鈕」及「唯一還要跑」改為有範圍的文獻比較與待驗證假設 |

20 份 `runs/README.md`：

| 實驗目錄 | 清理處置 |
|---|---|
| `advcf_anchor` | `:82–87` 修正前 out 臂限制保留；`:180` 刪除 builder 引用改歷史來源 |
| `advcf_colour_purify` | 保留四種色彩正規化的實際比較；「天敵／最強」改數值與範圍 |
| `advcf_eot` | 保留固定驗證抽樣修正及 4 格限制；不得將小差距概括為普遍效益 |
| `advcf_objective` | 求解／評估變異的歸因與表中數值分開；不要從少量重跑推出通用可重現性 |
| `advcf_pilot` | 保留設定、指令排除原因與 4 格限制；移除「全場最差／根因確認」等超出對照範圍的敘述 |
| `advcf_steps900` | `:4` 稱只有 steps 變化，`:21–23` 又記錄 probe 與 scheduler 差異；把固定項、衍生改變完整列明 |
| `advcf_term_scale` | 保留梯度尺度與編輯讀數分離；歷程敘述改成設定—結果表 |
| `advcf_texture_sds` | 保留梯度符號、數值精度與取樣限制；圖像判讀標來源與樣本 |
| `baseline_restore` | `:3` 的 `apa_baseline.py` 已不存在；記來源 commit，不能改成另一個腳本假稱等價 |
| `defence_animals` | 資料建置與四條件選擇保留；將「這輪不評測」改為此資料組未含評估 |
| `defence_solver_variation` | `:25` 參照的 portraits README、`:30` 遠端 worker 需歷史索引；不保留隊列位置作操作指示 |
| `immunise_tv000` | `:17–27` 從代理身份量推論防禦／外觀過強；保留原值及不可行 checkpoint 說明，限定為診斷 |
| `instruction_calibration` | 刪除已完成後仍保留的「下一步」；保留非盲測修訂與分布限制 |
| `objective_pilot` | `:4` 腳本與設定已不存在；補歷史來源，保存 `_superseded_no_smoothness` 的原因 |
| `paper_baseline` | 缺 ImageNet manifest 的取得方式明列；WACV-s4 只作歷史產物根 |
| `paper_trio` | 移除「第一／第二層」作主識別，改任務名称；保留同一防禦圖跨任務的比較條件 |
| `patch_canvas` | 保留支撐／失真不匹配限制；外觀敘述標人工判讀，避免代替數值證據 |
| `plateau_probe` | 保留可達性與對照定義；「沒買到東西」改實際讀數範圍 |
| `print_patch` | 标題「目前的結論」改為指定協定結果；兩個 artifact 登錄 |
| `purify_heldout` | `:43–48` 明示 crop 身份讀數來自修正前，必须保留限制並進資料 catalog |

# 6. 遷移計畫

本節是後續可執行方案，**本次沒有執行任何階段的寫入或遠端操作**。

| 階段 | 工作與產出 | 驗證方法 | 回復方式／裁定點 |
|---|---|---|---|
| **A：凍結證據** | 保存追蹤檔清單、Git blob ID、工作目錄 hash、CSV schema／列數／鍵集合；建立已知缺陷及外部相依清單 | 697 份 CSV 全量比對；分開記錄 Git LF 與工作目錄 CRLF；未追蹤檔不納入搬動 | 不改原始資料；這份清單作後續每階段基準 |
| **B：修可靠性缺陷** | 修 readout 退出碼、半完成目錄、style 可行性旗標、必要指標失敗與數值保存政策 | 使用假指令／固定小型資料驗證失敗傳遞、重試、取消、單一 writer；不需 GPU | 每個行為修正獨立英文 commit，可單獨 revert |
| **C：建立資料與協定契約** | 建 `protocol_id`、condition registry、輸入／產物 manifest、跨平台 resolver、固定 schema | 用既有七份主要表驗鍵集合；檢查所有舊路徑可解析或明確標缺；不得默默 fallback | 保留旧 reader；新 resolver 可整批撤回，不改 CSV |
| **D：抽取共用套件** | 由已確認版本抽出核心及 pipelines；保留差異清單；各專案固定版本 | 在移除兄弟目錄可見性的獨立目錄測 import、`--help`、設定驗證；強制離線、不載權重 | 降回舊套件版本或切回舊入口；歷史來源保留 |
| **E：搬活動目錄與改名** | 主表移到根；code 分為套件、shell、configs；依第 4 章做機械改名 | `git grep` 查舊 import／路徑；AST 全檔解析；`bash -n`／shell 檢查；既有資料契約測試 | 使用 rename manifest 反向搬回；不靠人工猜原位置 |
| **F：整理結果與文件** | 結果檔純搬動；建立唯一入口、限制、發布及來源索引；更新記憶 | Git blob ID／工作目錄 hash 對照；內部連結檢查；原始 CSV 列、欄、順序和值不變 | 文件及路徑遷移分開 revert；記憶也保存修改前副本 |
| **G：遠端部署準備** | 另取得家目錄腳本、深層 `_launch`、連結、工作規格、權重版本與運行狀態 | 先做同步 mapping 與 dry-run；比較兩台同一 NFS 的程式 hash；確認無受影響 writer | **需另行授權遠端操作**；本次摘要不足以直接部署 |
| **H：遠端切換** | 新版本先放獨立 release 目錄，驗證後切換；舊根與產物映射保留 | `rsync --dry-run --itemize-changes` 等只檢查方案；正式切換前再查租約／行程；不得就地截斷運行中 shell | 切回舊 release；保留舊輸出根，不搬正在寫入的產物 |
| **I：數值更正工作** | 若使用者決定，重算已知幾何分區讀數及核查 style feasible | 只對受影響欄與列建更正版；原始表保留，記 correction provenance | **獨立研究資料工作**，另定 GPU／CPU 資源與範圍；不能算作改名驗收 |

建議的驗收關卡：

| 關卡 | 必須成立 |
|---|---|
| 原始證據 | 所有原始 CSV 可追溯到搬動前 blob／hash；沒有未解釋刪列、加列、欄位轉型 |
| 自足 | `main_table`、`lab` 各自放到另一個目錄仍可安裝、import、列出 CLI、驗證固定資料；`PYTHONPATH` 不含兄弟專案 |
| 協定 | 模型、seed、precision、resize、遮罩、LPIPS 後端及參數不因抽取改變 |
| 完成狀態 | 子步驟失敗、缺圖、缺必要指標、設定不符時不能得到完成旗標 |
| 文件 | 每區只有一個交接入口；所有現行執行指令的路徑存在 |
| 遠端 | 所有已知呼叫端完成改名核對，未知的摘要外引用已取得並檢查 |

殘留舊名的查核應有白名單，不能要求整個 repo 完全沒有舊字串。例如：

```text
git grep -n -E 'non-additive-frequency|IMMUNISATION_SOURCE_HOME|WACV-s[34]'
git grep -n -E 'edit_flux_preview|flux_full_queue_a|prompt_sets_ultraedit_round2'
git grep -n -E 'sys\.path|from src\.|import src\.'
git ls-files --eol
```

對 `MEM/` 必須另用 `rg`，Git 查不到 repo 外記憶。對 `colour_curve_ours` 的命中應核對用途，不能列為必須清零的殘留。

純結構遷移不要求重新產生防禦圖，也不要求多種子實驗。必要的數值更正與研究擴充須另外裁定。

# 7. 需要使用者裁定的問題

| 問題 | 建議選擇 | 為何無法只從檔案決定 |
|---|---|---|
| 「每一塊自足」是否包含已結束研究也能執行？ | 活動專案必須可獨立執行；封存區明示完整性。若四塊皆要可執行，另列歷史環境恢復工作 | `frequency-phase` 缺設定、環境及部分模組，修復範圍明顯超過改名 |
| 是否接受固定版本共用套件及受控 vendor 快照？ | 接受；禁止兄弟目錄 runtime import | 需決定自主性與重複維護成本的取捨 |
| 共用套件由誰維護？ | 指定一個責任範圍；baseline、color、style 透過版本升級採用 | 既有各 session 的唯讀邊界不等於套件維護制度 |
| 是否採用 `main_table/`、`lab/`、`archive/` 的根層結構？ | 採用，保留兩個活動名稱以降低變動 | 這是專案組織選擇，沒有程式唯一答案 |
| 歷史 CSV 是否只允許搬檔，內容永不因命名整理而改寫？ | 是；新 schema 用衍生表與映射 | 可最大程度維持數值與發布證據；尤其不可把舊色彩方法改成 color |
| lab 的遠端 trace／results 是否全部保全？ | 全部保全數值，再排除真正可重建的暫存狀態 | `runs/` 的 ignore 與規則不一致；本機無法判斷遠端尚存哪些未入庫數值 |
| color driver 額外三個分支如何定位？ | 標為未納入保留結果的實驗選項，裁定後才停用或保留 | `color_xattn`、`color_simple_dayn`、`color_lut3d_dayn` 仍可執行，文件主要列四個保留條件 |
| 是否修正原生主表 1,408 列的幾何分區欄？ | 建立保留原表的更正版；先確認影像仍存在 | 這是數值更正工作，涉及產物與計算資源，不能由改名取代 |
| 是否恢復報告生成器？ | 先建發布索引；只有確定需要新報告時才恢復 builder | HEAD 明確刪除了報告與 builders，不能把刪除當意外 |
| 舊 artifact 是否需要重新發布或標示取代？ | 保留舊 URL，新增版本與適用範圍說明 | 本機改名無法修改已發布內容，發布亦屬另一項操作 |
| 遠端是否一起重整目錄？ | 本機先完成契約與 mapping；遠端另作 staged migration | 摘要不含完整脚本與即時行程，無法安全確定切換條件 |
| 換行以哪份 bytes 作保存基準？ | Git blob 與現有工作目錄各留 hash；換行正規化獨立處理 | 695 份 CSV 的工作目錄與 Git 換行不同，單一 hash 基準不足以描述兩者 |

不需要另行裁定的事項已有檔案依據：`colour_curve_ours` 不等同 `color`；等失真錨點維持 0.3344；沒有要求就不增跑多種子；主表七道不自行加入 DiffPure；數值證據保全優先於命名一致；本次只交付診斷與方案。

Codex session ID: 01a0ee43-524e-70d3-9d7e-427b6fc411be
Resume in Codex: codex resume 01a0ee43-524e-70d3-9d7e-427b6fc411be
