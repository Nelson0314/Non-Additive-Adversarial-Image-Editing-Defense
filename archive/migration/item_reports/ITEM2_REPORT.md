# 第 2 項交接紀錄：移除

基準 HEAD：`89c148c41d527c8d38a379b3744bc311ee1e1c1d`。修改保留於工作目錄，index 為空，未建立 commit、未 push、未連線遠端、未執行 GPU 工作。本輪未進入第 3 項。

## 修改與提交分組

`item2_patches/series.json` 記錄四份 patch、英文提交訊息、各 patch 與工作目錄檔案的 SHA-256。刪除與文件更新分組；本項沒有額外的可靠性行為修正。

| 順序 | 提交訊息 | 檔案與修改 |
|---|---|---|
| 1 | `Remove retired color experiment branches` | `lab/scripts/defence_cmd.sh` 移除 `color_xattn`、`color_simple_dayn`、`color_lut3d_dayn` 派工入口；`lab/code/color_defence.py` 移除專用實作與選項值；新增 `lab/tests/test_color_conditions.py`，共 18 個測試案例。 |
| 2 | `Remove the retired defence report skill and template` | 刪除 `.claude/skills/defence-report/SKILL.md` 與 `assets/template.html`，即該目錄的全部受版控檔案。 |
| 3 | `Remove retired color branch documentation` | `lab/docs/DESIGN.md` 移除 DAYN 與 3D Lab 載體的敘述，保留原有 xattn 數值敘述。 |
| 4 | `Remove artifact links from active main table documentation` | `anti-purification/main_table/STATUS.md` 移除兩個 artifact 網址及專門指向它們的段落。 |

## 共用依賴核對

移除前先核對呼叫位置，專用定義為 `Lut3D`、`RegionAttnScore`、`encode_class`；`torch.nn.functional` 與 `torch.utils.checkpoint` 在 `color_defence.py` 中僅供移除的定義使用，一併刪除。移除 `--objective dayn`、`--carrier lut3d` 及 DAYN 專用驗證、輸出欄位。

四個保留條件為 `color`、`color_simple`、`color_simple_skinbox`、`color_simple_xattn`。共用的 `ColorMap`、`XAttnScore`、`_grad_norm`、`align_weight` 及 `style_prompt_defence.CrossAttnObjective` 保留。`--objective xattn`、`--carrier ab`、`--caps simple`、`--box skin`、`--lr-final-ratio`、`--lam-every` 均保留；後兩者預設仍為 0.2 與 5。既有最佳化、約束、種子、精度、遮罩與輸出協定不變。

`verify_item2_equivalence.py` 對照基準 HEAD 與目前程式：17 個保留函式／類別的 AST 完全一致；依四個保留條件固定分支後，四份 `main()` 運算路徑的 AST 亦完全一致。結果保存在 `item2_patches/retained_equivalence.json`。此核對與 CPU 測試驗證保留路徑，未執行 GPU 數值實驗。

## 驗證結果

| 驗證 | 結果 |
|---|---|
| `python -m pytest lab/tests -q -p no:cacheprovider --basetemp=.tmp/codex_audit/pytest_item2_final`，設定 `CUDA_VISIBLE_DEVICES=''`、`PYTHONIOENCODING=utf-8` | **56 passed**，包含新增的 18 個案例。 |
| 新增案例範圍 | 四個保留條件的派工參數、輸出路徑與額外參數傳遞、預設值、共用選項；三個移除入口及兩個移除選項值均拒絕執行。 |
| `bash -n lab/scripts/defence_cmd.sh`、檔案位元組檢查 | 通過；`.sh` 僅有 LF。 |
| `python lab/code/color_defence.py --help` | 通過；選項為 `--objective {comm,xattn}`、`--carrier {ab}`。 |
| `git diff --check` | 通過。 |
| 活動文件與程式殘留搜尋 | 指定活動文件無 `claude.ai/artifact` 連結；lab 活動程式／文件無三個移除分支及其專用定義。測試內保留名稱以檢查拒絕行為。baseline 的獨立 DAYN 方法未改動。 |
| 基準 SHA-256 比對 | 1,720 個未修改受版控檔案、697 份 CSV、22 份指定保留文件及 92 個既有未追蹤檔案全部相同。 |
| 四份 patch 隔離套用 | `git apply --check` 與實際套用均通過，產生內容與工作目錄一致，刪除檔案均不存在。隔離路徑記錄於 `item2_patches/verification.json`。 |
| `commit_item2.ps1` | PowerShell 語法解析通過；提交腳本未執行。 |

根 `HANDOFF.md`、`COLOUR_LINE.md` 及 `anti-purification/runs/**/README.md` 均未修改。使用者 PDF 與其他既有未追蹤資料未改動。遠端 `lab/runs/` 未處理。

## 協調端執行

在原工作目錄執行：

```powershell
& .\.tmp\codex_audit\item2_patches\commit_item2.ps1
```

腳本先核對 HEAD、空 index、patch 與檔案 SHA-256、未修改基準檔案、既有未追蹤檔案及修改清單，再檢查各 patch 可套用。確認指定技能目錄沒有其他內容後，以非遞迴方式移除空目錄；逐份 patch 套入 index 並建立四個 commit，寫出 `item2_patches/commits.json`。不重設工作目錄，不 stage 其他檔案，不 push。

尚待協調端執行：四個 commit，以及 `.claude/skills/defence-report/assets/` 與其父目錄的空目錄清理。自動審核拒絕本環境的非遞迴 `Remove-Item`，僅回報 `blocked by policy`；檔案刪除已完成，兩個空目錄仍存在。提交腳本包含該清理步驟與非空目錄保護。沒有其他需裁定事項。
