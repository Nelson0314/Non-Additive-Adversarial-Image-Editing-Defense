# color：現況與接續指引

設計與結果見 `docs/DESIGN.md`，執行方式見 `README.md`。

## 現況

- 保留的條件：`color`、`color_simple`、`color_simple_skinbox`、`color_simple_xattn`；其餘舊條件、預覽、報告頁與其程式已刪除，數據以文字記錄在 `docs/DESIGN.md` §7。
- 讀數：`results/{displacement,retention,fidelity}.csv` 為 `color`；`results/variants/<條件>/` 為三個 simple 條件。
- 研究方向由使用者決定。遠端是否有工作在跑不記錄於本檔：以遠端 `~/gpu_leases/` 的租約與本專案 `runtime/` 的佇列狀態為準。
- 未防禦編輯（`artifacts/undefended_edits/ip2p_si18`）與淨化後的未防禦分母（`artifacts/purified_edits/undefended/`）為 baseline 同名產物的複本，放在本專案的 `artifacts/`，不連結到 baseline。
- `requirements.lock` 未入庫；由遠端執行環境以 `vendor/scripts/generate_requirements_lock.py` 產生。

## 執行紀錄

- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 兩台共用；本專案位於 `~/image-immunization/color`。
- 實測成本：`color` 防禦圖約 53 分鐘／張（顯存約 18 GB）；只跑 ip2p 的鏈約 50 分鐘；讀數約 15 分鐘；32 格編輯約 8 分鐘。

## 規矩

- **取卡函式的紀錄一律寫 stderr**：`$(...)` 會收走 stdout。
- **殺行程**：`pkill -f <樣式>` 會匹配到下指令的 ssh；佇列 worker 的主迴圈是父程序為 `bash -c` 的那一個。
- **中止佇列工作會觸發重試**（失敗 3 次才 `.GIVEUP`）；重寫共用 CSV 的工作（如 `readout`）被中止後，重試行程會與另開的行程同時寫同一個檔。要改道時先 `touch runtime/queues/<佇列>/<工作>.GIVEUP` 並清除 `.lock`，另以不同 `--output-csv` 執行後再合併。
- **`fid` 工作需要 `FID_ARMS`**；不設時 `run_queue.sh` 拒絕啟動。
- **分母**：新種子或新設定的編輯不寫進 `artifacts/undefended_edits/`。
- **`optimize_carrier` 回傳的字典鍵一律 `free_` 開頭**，與 `--objective` 無關。
- 共通規則見根目錄 `CLAUDE.md`。
