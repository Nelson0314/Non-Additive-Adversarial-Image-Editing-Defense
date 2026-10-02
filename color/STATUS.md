# color：現況與接續指引

設計與結果見 `docs/DESIGN.md`，執行方式見 `README.md`，已刪除的暫時性嘗試見 `docs/TRIALS.md`。

## 現況

- 研究主線：33³ RGB 3D 查找表（`lut3d`）防禦的防禦力與自然度，在 trial `trials/lut3d_defense_and_naturalness/`（不入版控；
  入口為該目錄的 `README.md`：檔案、執行方式、全部結果、執行中工作與下一步候選）。trial `trials/edit_failure_mechanisms/`
  保存 `lut3d_mild`（預算 0.2）的解與讀數，主線的對照由它衍生，不刪除。
- 原 `color`（全域 (a,b) 換色表）與隱含分類器目標（selfrec）暫緩，數據在 `docs/DESIGN.md`。
- 保留的正式條件：`color`、`color_simple`、`color_simple_skinbox`、`color_simple_xattn`；其餘舊條件與其程式已刪除，數據以文字記錄在 `docs/DESIGN.md` §7。
- 讀數：`results/{displacement,retention,fidelity}.csv` 為 `color`；`results/variants/<條件>/` 為三個 simple 條件。
- 研究方向由使用者決定。遠端是否有工作在跑：以遠端 `~/gpu_leases/` 的租約、各 trial 的 `runtime/` 與 `logs/` 為準。
- 未防禦編輯（`artifacts/undefended_edits/ip2p_si18`）與淨化後的未防禦分母（`artifacts/purified_edits/undefended/`）為 baseline 同名產物的複本，放在本專案的 `artifacts/`，不連結到 baseline。
- `requirements.lock` 未入庫；由遠端執行環境以 `vendor/scripts/generate_requirements_lock.py` 產生。

## 執行紀錄

- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 兩台共用；本專案位於 `~/image-immunization/color`。
  兩台的卡常被實驗室其他使用者占用；取卡工具只取本機的卡，需要時在另一台另開 main。
- 實測成本：`color` 防禦圖約 53 分鐘／張（顯存約 18 GB）；`lut3d`（comm）約 50 分鐘／張；`teacher` 目標的續訓約 10 分鐘／張；
  單張的編輯 ＋ 七道淨化 ＋ 淨化後編輯約 15–18 分鐘；只跑 ip2p 的鏈約 50 分鐘；讀數約 15 分鐘；32 格編輯約 8 分鐘。

## 規矩

- **卡數**：依使用者口頭指定；未說明時全局合計 6 張。`run_with_gpu_lease.sh` 的 `--limit` 比的是全局租約總數，不要給；取卡失敗（沒有空卡）時以迴圈每 5 分鐘重試。
- **取卡函式的紀錄一律寫 stderr**：`$(...)` 會收走 stdout。
- **殺行程**：`pkill -f <樣式>` 會匹配到下指令的 ssh；以 PGID 或明確 PID 終止，終止後以 `ps`／租約目錄回頭驗證。`$(...)` 收到多個 PID 時要逐一處理。
- **佇列**：既有正式條件走 `scripts/run_queue.sh`；中止佇列工作會觸發重試（失敗 3 次才 `.GIVEUP`），改道時先 `touch runtime/queues/<佇列>/<工作>.GIVEUP` 並清除 `.lock`。
- **`fid` 工作需要 `FID_ARMS`**；不設時 `run_queue.sh` 拒絕啟動。
- **分母**：新種子或新設定的編輯不寫進 `artifacts/undefended_edits/`。
- **`optimize_carrier` 回傳的字典鍵一律 `free_` 開頭**，與 `--objective` 無關。
- 共通規則見根目錄 `CLAUDE.md`。
