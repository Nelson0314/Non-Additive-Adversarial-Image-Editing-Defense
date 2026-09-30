# 重整工作交接（雲端 session 接手用）

本檔讓一個只能存取 GitHub 的新 session 冷啟動接續 repo 重整。先讀本檔，再讀
`RESTRUCTURE_PLAN.md`（使用者裁定、目標結構、第 0–14 項、執行規則），需要細節時查
`RESTRUCTURE_AUDIT.md`（診斷報告全文，章節編號在計畫中被引用）。

## 分工

| 角色 | 誰 | 做什麼 |
|---|---|---|
| 協調端 | 本機 Claude session（`image-immunization-30`） | 需要 ssh／GPU／本機檔案的項目；合併你推上來的分支；驗收 |
| 執行端 | 你（雲端 session） | 只需 repo 的項目，直接改檔、測試、commit、push |

你**不能**連實驗室遠端（`server.basiclab.lab.nycu.edu.tw`），也沒有 GPU。

| 項 | 由誰做 |
|---|---|
| 0、1、2 | 已完成 |
| 3 | 進行中，第 1、2 段已完成；**你從第 3 段開始** |
| 4、5、6、7、8、9、13 | 你 |
| 10（遠端重整）、11（Claude 記憶）、12（GPU 重算）、14（通知其他 session） | 協調端 |

第 5 項需要的遠端 `lab/runs/` 數值 CSV 已由協調端複製到 `archive/migration/remote_lab_runs/runs/`（132 份，路徑與遠端
`lab/runs/` 相同）。請在第 5 項把它們放進 `/color`、`/style` 的 `results/` 對應位置（依你的命名），再刪除這個暫存目錄。

## 目前狀態

- 分支 `cleanup`。第 0 項基準 commit `8bcaae0`；`archive/migration/pre_migration_manifest.json` 是改動前 1,707 個追蹤檔的
  blob／工作目錄雜湊與 697 份 CSV 的表頭與列數，`snapshot_manifest.py` 可重算。
- 第 1 項：11 個 commit `5e1cd1c`…`89c148c`；第 2 項：`0f04ed3`…`57a28a9`；第 3 項第 1 段：`b7c312e`…`51d58bf`；第 2 段：`ff65396`…`db49113`。
  各項驗證紀錄在 `item_reports/`（第 3 項第 2 段沒有報告，內容見 commit message 與 `core/STATUS.md`）。
- 第 3 項剩餘工作列在 `core/STATUS.md`「第 3 項尚餘」第 2、4、5、6 點（編輯／displacement／retention 主流程、共用最佳化與色彩
  helpers、五支 GPU 租約工具移入 `core/scripts/`、相關測試與獨立副本驗證）。
- `core/tests`：100 passed、21 deselected（需權重或外部後端）。
- `baseline`、`color`、`style` 三個本機 session 處於凍結（只讀）；遠端沒有工作在跑。

## 工作方式

1. 在 `cleanup` 上工作；每完成一項（或一段可獨立驗收的子項）就 push 到 `origin/cleanup`。若你的環境只允許推送到自己的分支，
   就推到該分支，並在 `RESTRUCTURE_LOG.md` 寫明分支名。
2. 每完成一項，在 `archive/migration/RESTRUCTURE_LOG.md` 追加一段：項目、commit 範圍、改了什麼、驗證指令與結果、未完成或需使用者
   裁定的事。使用者要求每完成一項回報一次，這份 log 是回報的正本；同時在你的對話中向使用者簡短回報。
3. 需要使用者裁定、或需要 ssh／GPU 的事，寫進 log 並停下，不要自行假設。
4. 測試只跑 CPU：`python -m pytest -q -p no:cacheprovider core/tests`，以及各專案的 `tests/`；需要權重的案例已用 marker 排除。
   依賴可從 `anti-purification/environment.yml` 與 `core/pyproject.toml` 安裝 CPU 版。

## 規則（本機的全域規則你讀不到，摘錄如下）

- 回覆與文件用繁體中文；程式碼關鍵字、函式名、套件名、指令維持英文。commit message 用英文；機械搬檔、行為修正、文件更新分開 commit。
- **書面用語**（寫進檔案的文件、註解）：客觀、學術、精確，接近教科書。不用誇飾與空洞商業詞、口語語尾、教練式人稱、生活化比喻、
  「並非 X 而是 Y」式對比、「值得注意的是」式鋪陳、無出處的「研究顯示」、冒號堆砌的戲劇性揭示；換到任何主題都成立的句子是填充句，刪除。
  不把對話原話或製作流程用語寫進成品；不寫「最新」「這輪」「稍後」這類時間相依字眼。
- 命名不含日期、流水號或順序詞（`round2`、`queue_a`、`_full`、`_preview` 這類流程用語要改成描述內容的名稱）；參數數字與影像 ID
  （`jpeg50`、`rotate15`、`man_00`）有資料意義，保留。
- **不設判準**：數據擺出來為止，不下「成立／不成立」「值得／不值得」的結論，指標本身也一樣。報數字寫描述性名稱（編輯結果 LPIPS、
  防禦圖 LPIPS），不用 D、D_T 這類代號。
- 禁止用 try/except 或條件跳過掩蓋症狀；必要輸入或指標失敗要讓程式失敗。
- 不改科學協定：LPIPS 後端（`piq.LPIPS`）、resize、插值、遮罩、seed、precision、量化、樣本排除規則。
- `.sh` 必須是 LF；Python 用 `python` 指令。
- 不刪除、不改寫 `/archive` 內部（第 6 項只整體搬入）；刪除只限計畫明文要求的項目。
- GPU 卡數規則（寫進文件或排程工具時用）：全局可用卡數由使用者逐次授權；未說明或說明不清時預設全局合計上限 6 張（所有 session 與排程加總）。

## 容易出錯的地方

- CSV 路徑欄有四種形式（Windows 絕對 `C:/image-immunization/...`、NFS 絕對 `/nfs/home/nelson0314/image-immunization/...`、
  repo 相對 `runs/...`、lab 相對 `lab/runs/...`），第 7 項要依產物角色改寫到新版面，不可全文替換；非路徑欄（`dir_norm`、`direction`）不動。
- `colour_curve_ours` 與 `color` 是不同方法：前者改名為 `color_curve`，不可合併；它也是等失真臂錨點 0.3344 的出處。
- 第 7 項改寫 CSV 後，逐表驗證列數、鍵集合、數值欄與 `pre_migration_manifest.json` 及改寫前內容一致。
- 遠端版面在第 10 項才搬；你改寫 CSV 路徑時以**新版面**為準，並在 log 中列出舊→新目錄對照，供協調端搬遠端時使用。
- UltraEdit 表的 `scenario=ip2p` 是借用欄位，新 schema 加 `editor` 欄。
- 主表幾何淨化的主體／背景分區欄（11 條件 × `crop_resize0.1`／`rotate15`）使用未變換遮罩，第 12 項由協調端重算，你不要改這些數值。
