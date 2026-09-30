# color：現況與接續指引

設計與結果見 `docs/DESIGN.md`，執行方式見 `README.md`。

## 現況

- 保留的條件：`color`、`color_simple`、`color_simple_skinbox`、`color_simple_xattn`；其餘舊條件、預覽、報告頁與其程式已刪除，數據以文字記錄在 `docs/DESIGN.md` §7。
- 讀數：`results/{displacement,retention,fidelity}.csv` 為 `color`；`results/variants/<條件>/` 為三個 simple 條件。
- 研究方向待使用者決定；沒有工作在跑。
- 未防禦編輯（`artifacts/undefended_edits/ip2p_si18`）與淨化後的未防禦分母（`artifacts/purified_edits/undefended/`）與 baseline 的同名產物相同；遠端版面切換（重整第 10 項）時需一併放入本專案的 `artifacts/`。

## 執行紀錄

- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 兩台共用。
- 實測成本：`color` 防禦圖約 53 分鐘／張（顯存約 18 GB）；只跑 ip2p 的鏈約 50 分鐘；讀數約 15 分鐘；32 格編輯約 8 分鐘。

## 規矩

- **卡**：取卡一律經 `vendor/scripts/`（`run_queue.sh` 內的 `queue_worker.sh`、`run_with_gpu_lease.sh`）。租約目錄 `~/lab_leases/`（`<主機> <pid> <名稱> <擁有者 token>`，兩台共用）。全局卡數由使用者逐次授權；未說明或說明不清時預設 6 張，所有 session、主機與排程合計。`GPU_CAP=N` 或 `run_with_gpu_lease.sh --cap N` 寫入共用的 `~/lab_leases/.capacity`，所有入口在取卡時重新讀取，維持至下一次設定；`GPU_CAP=default` 回到預設值。`QUEUE_CAP` 只能進一步限制佇列派工。平行的 GPU 工作一律排進同一個佇列。
- **遠端腳本不可就地覆寫**：先寫到暫存位置再 `mv`（執行中的 bash 逐行讀檔，截斷同一個 inode 會中止於 Stale file handle）。`.sh` 必須是 LF。
- **取卡函式的紀錄一律寫 stderr**：`$(...)` 會收走 stdout。
- **殺行程**：`pkill -f <樣式>` 會匹配到下指令的 ssh；佇列 worker 的主迴圈是父程序為 `bash -c` 的那一個。
- **中止佇列工作會觸發重試**（失敗 3 次才 `.GIVEUP`）；重寫共用 CSV 的工作（如 `readout`）被中止後，重試行程會與另開的行程同時寫同一個檔。要改道時先 `touch runtime/queues/<佇列>/<工作>.GIVEUP` 並清除 `.lock`，另以不同 `--out` 執行後再合併。
- **`fid` 工作需要 `FID_ARMS`**；不設時 `run_queue.sh` 拒絕啟動。
- **分母**：新種子或新設定的編輯不寫進 `artifacts/undefended_edits/`。
- **`optimize_carrier` 回傳的字典鍵一律 `free_` 開頭**，與 `--objective` 無關。
- **沒有明確要求就不跑多種子**；主種子的正常編輯為唯一標準。
- **命名**：目錄、檔案、條件名不含日期、流水號或順序詞；commit message 用英文。
- **不設判準**：數據與圖並列為止，不下「成立／不成立」「值不值得再跑」的結論。
