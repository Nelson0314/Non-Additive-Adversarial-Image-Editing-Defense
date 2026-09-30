# 共通規則

適用於 `core/`、`baseline/`、`color/`、`style/`。各專案的範圍、現況與專屬規則見該專案的 `STATUS.md`；`archive/` 內的文件只描述封存當時的狀態，不適用於現行專案。

## 範圍

- 每個 session 只改自己負責的專案目錄。`core/` 是共用正本，修改後以 `python core/scripts/generate_vendor_snapshot.py <專案>` 重新匯出，不就地修改 `vendor/`。
- `archive/` 只保存，不刪除、不改寫其內部。

## 書面用語

- 回覆與文件用繁體中文；程式碼關鍵字、函式名、套件名、指令維持英文。
- 文件與註解客觀、精確，接近教科書：不用誇飾、口語語尾、比喻、「並非 X 而是 Y」式對比、「值得注意的是」式鋪陳、無出處的主張與填充句；
  不寫對話原話與製作流程用語。
- 不寫時間相依的字眼（「最新」「本輪」「先前」「稍後補」）；寫現行狀態、理由、來源 commit 與查閱路徑。
- 報數字用描述性名稱（編輯結果 LPIPS、防禦圖 LPIPS），不用代號；引用數字時連同協定（資料組、設定、樣本數）。

## 命名

目錄、檔案、條件、實驗組、文件不含日期、流水號或順序詞（`round2`、`queue_a`、`_full`、`_preview`），名稱描述內容。
參數數字與影像 ID（`jpeg50`、`rotate15`、`man_00`）有資料意義，保留。自有 API 使用美式拼法（`color`、`defense`、`optimize`、`gray`、`center`）。

## 實驗與報告

- **不設判準**：把訓練跑到收斂（以與訓練目標無關的固定量測判定），完整回報數據與影像；「成立／不成立」「值得／不值得再跑」由使用者判斷。
- 不改科學協定：LPIPS 後端（`piq.LPIPS`）、resize、插值、遮罩、seed、precision、量化、樣本排除規則。需要修正時另立資料版本並記錄。
- 沒有明確要求不跑多種子；主種子為標準。
- 移植外部方法時使用該方法自己的損失與設定；無法移植的部分回報使用者。

## 程式

- 必要輸入或必要指標失敗時程式必須失敗；不以 try/except、條件跳過或 skip 掩蓋。選配功能缺席時以固定欄位記錄原因。
- 測試不可為了通過而 skip、放寬斷言或刪除；失敗先找根本原因。
- `.sh` 一律 LF（根目錄 `.gitattributes` 固定）；Python 以 `python` 指令執行。
- commit message 用英文；機械搬檔、行為修正、文件更新分開 commit。
- 密碼與 token 不寫入任何入庫檔案。

## GPU

- GPU 工作一律送遠端，不用本機顯卡。
- 全局可用卡數由使用者逐次授權；未說明或說明不清時預設全局合計 6 張，計入所有主機、session 與排程。
- 取卡一律經專案 `vendor/scripts/`（`run_with_gpu_lease.sh`、`run_queue_worker.sh`）；租約目錄 `~/gpu_leases` 由所有入口共用，
  `GPU_CAP=N` 或 `--cap N` 寫入 `~/gpu_leases/.capacity`，`GPU_CAP=default` 回到預設值。
- 卡為多人共用：卡上有他人的 compute app 即不使用。`measure_free_gpus.sh` 只產生候選清單，不保留卡。
- 遠端腳本不可就地覆寫：先寫到暫存位置再 `mv`（執行中的 bash 逐行讀檔）。

## 暫時性嘗試

在專案內以 `bash vendor/scripts/run_trial_lifecycle.sh new <名稱>` 建立 `trials/<名稱>/`（不入版控）。採用的內容搬進 `src/`、`configs/`、`results/`
並提交後 `run_trial_lifecycle.sh promote`；不採用者先在 `docs/TRIALS.md` 寫一列（試了什麼、設定、關鍵數字、結論來源）再 `run_trial_lifecycle.sh drop`，
遠端副本一併刪除。
