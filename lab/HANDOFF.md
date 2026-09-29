# lab/ 交接

**`lab/` 是獨立的實驗室：所有程式、資料、產出都在這裡。對 `../anti-purification/` 一律唯讀；需要的檔案複製過來。**
lab 分兩條線，各由一個 session 負責：

| 線 | 文件 | 程式 |
|---|---|---|
| 顏色線（現行方法 `color`） | `docs/DESIGN.md` | `code/color_defence.py`、`scripts/defence_cmd.sh`、`arm_chain.sh`、`queue_worker.sh`、`readout.sh` |
| 風格轉換線（SPA 移植） | `docs/STYLE_PROMPT.md` | `code/style_prompt_defence.py`、`code/style_prompt_readout.py`、`scripts/style_prompt_round.sh` |

共用：`code/edit_preflight.py`（編輯）、`purify_run.py`（淨化）、`edit_displacement.py`、`edit_retention.py`、`defence_fidelity.py`（讀數）、`scripts/run_on_card.sh`（取卡）。

**不要讀**：`results/*.csv` 全文（用彙整）、`runs/`（遠端資料）。

---

## 現況（2026-09-30）

- 兩條線的成果只保留最新：顏色線 `color`、`color_simple`、`color_simple_skinbox`、`color_simple_xattn`；
  風格轉換線 `r11`、`r13`、`cls`。其餘舊臂、預覽、報告頁與其程式已刪除，數據以文字記錄在各自的文件。
- 讀數：`results/{displacement,retention,fidelity}.csv` 為 `color`；`results/exp/` 為三個延伸臂。
- 兩條線的研究方向都待使用者決定；目前沒有工作在跑。

---

## 怎麼跑

- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）
  `nelson0314@server.basiclab.lab.nycu.edu.tw`，目錄 `~/image-immunization`（NFS，兩台共用）。
  環境 `source ~/env.sh` 提供 `$PY`。
- 臂的參數只定義在 `scripts/defence_cmd.sh`。
- **工作佇列** `scripts/queue_worker.sh <佇列名> <工作>...`，兩台主機各起一個 worker：
  ```
  cd ~/image-immunization && bash -c 'nohup setsid bash lab/scripts/queue_worker.sh <佇列> <工作...> > lab/runs/logs/queue_<佇列>_$(hostname).log 2>&1 < /dev/null & disown'
  ```
  注意 `cd` 要在 `bash -c` 之外，否則會跟著背景程序走。工作種類（相依寫在檔頭）：
  `def:<臂>:<影像>`（單張防禦圖，分卡平行）、`chain:<臂>`（併分片 → `arm_chain.sh`，只跑 ip2p）、
  `readout`、`fid`、`pilot:<臂>:<影像>:<步數>`。
  狀態在 `lab/runs/queue/<佇列>/`；同一工作失敗 3 次寫 `.GIVEUP`，相依它的工作不再派。
  佇列以外的單次 GPU 指令用 `scripts/run_on_card.sh <名稱> <指令...>`（取卡、租約、CUDA 檢查）。
- `arm_chain.sh <GPU> <臂>`：防禦 → ip2p 編輯 → 淨化 → 七道淨化後編輯，10 個 sentinel 在
  `lab/runs/state/`，只在 rc=0 時寫；防禦圖不滿 8 張時只編輯現有影像。
- 實測成本：`color` 防禦圖約 53 分鐘／張（顯存約 18 GB）；只跑 ip2p 的鏈約 50 分鐘；讀數約 15 分鐘；
  32 格編輯約 8 分鐘。
## 規矩（踩過才寫的）

- **卡**：判定空卡要同時看 `scripts/free_cards.sh`、租約目錄 `~/lab_leases/`（格式 `<主機> <pid> <名稱> <擁有者 token>`，
  兩台共用）與別人 compute app 的顯存合計 < 1 GB。全局卡數由使用者逐次授權；未說明或說明不清時預設 6 張，
  所有 session、主機與排程合計，包含非 style 工作。
  預設值只定義在 `scripts/gpu_policy.sh`；`LAB_CAP=N`、`run_on_card.sh --cap N` 或
  `style_prompt_round.sh <輪名> <清單> N` 可設定明確授權值，CLI 參數優先。
  明確授權寫入共用的 `~/lab_leases/.capacity`，所有入口在取卡時重新讀取。
  該授權維持至下一次設定；以 `LAB_CAP=default` 清除明確授權、回到共用預設值。
  `LAB_MYCAP` 與 style 的第四個參數只能進一步限制派工，不增加全局授權。
  候選卡清單不代表已取卡；queue、color chain 與單次派工共用 `scripts/gpu_lease.sh` 的原子取卡流程。
- **遠端腳本不可就地覆寫**：先解到 `.stage/` 再 `mv`（跑著的 bash 邊讀邊執行，截斷同一個 inode
  會死於 Stale file handle）。Windows 寫出的 `.sh` 要確認是 LF。
- **取卡函式的紀錄一律寫 stderr**：`$(...)` 會收走 stdout，卡號變成一串字，torch 靜默退回 CPU。
- **殺行程**：`pkill -f <樣式>` 會匹配到下指令的那條 ssh 自己；worker 的主迴圈是父程序為
  `bash -c` 的那一個，子程序（正在跑的工作）不要殺。
- **分母**：不要把新種子或新設定的編輯寫進 `runs/edit_preflight/`。
- **命名**：目錄、檔案、臂名不含日期、流水號或順序詞。commit message 用英文。
- **不設判準**：數據與圖擺出來為止，不下「成立／不成立」「值不值得再跑」的結論。
- **沒有明確要求就不跑多種子。** 唯一標準是主種子的正常編輯要正常；五種子平均、VQA
  是加碼的深入檢查，缺了不算資料不完整，不要自己排開多種子的佇列工作。
- **平行 GPU 工作一律排進同一個 `queue_worker.sh` 佇列**，不要自己開兩個獨立的
  `run_on_card.sh` 背景指令——兩個獨立呼叫的取卡檢查會搶到同一張卡（沒有跨行程鎖），
  兩邊都會 OOM 崩潰。詳見記憶 `parallel-run-on-card-races`。
- **殺掉佇列裡的工作會觸發自動重試**（同一工作失敗 3 次才 `.GIVEUP`），而且如果那個
  工作會整份重寫某個共用 CSV（如 `readout` 寫 `results/*.csv`），被殺掉後
  自動重試的行程會跟你另開的替代行程搶著寫同一個檔、互相蓋掉對方的資料。要嘛讓它
  自然跑完，要嘛殺掉後手動 `touch <佇列>/<工作>.GIVEUP` 並清掉 `.lock` 再另起爐灶
  （用不同的 `--out` 目錄，事後再合併，比較安全）。
- **`queue_worker.sh` 的 `fid` 工作型別需要先設 `FID_ARMS` 環境變數**，否則在
  `set -u` 下直接 `unbound variable` 崩潰；不想設就繞過佇列、用 `run_on_card.sh`
  直接跑 `defence_fidelity.py`（不帶 `--arms` 會自動掃 `lab/runs/defence/` 底下全部子目錄）。
- **一次性遠端指令裡的 `"$PY"` 不要直接當 `run_on_card.sh` 的引數**：那一層是被
  「呼叫 `run_on_card.sh` 的那個外層 shell」展開，此時 `env.sh` 還沒被 `run_on_card.sh`
  自己 source，`$PY` 是空字串。要包成 `bash -c "\"\$PY\" ...實際指令..."` 讓展開延後到
  `run_on_card.sh` 內部（它 source 完 `env.sh`、export 完 `CUDA_VISIBLE_DEVICES` 之後才
  執行 `"$@"`），詳見記憶 `command-substitution-ate-the-gpu-id` 的同類問題。
- **`optimise_carrier` 回傳的字典鍵一律 `free_` 開頭**，跟 `--objective` 選了 `comm`
  還是 `free` 無關，是那個函式自己的命名，不要看錯前綴。
- **SSH 有兩種不同的斷線**：`kex_exchange_identification: read: Connection reset`
  通常幾分鐘到十幾分鐘會自己好；`Connection timed out`（兩個 port 都連不上）可能持續
  數小時，原因不明，不是本機這邊能修的。兩種都不影響遠端已經 `nohup`/`setsid` detached
  的工作，只是本機暫時看不到進度；拉大量檔案優先用單一 `tar czf - ... | ssh` 管線，
  不要對同一台主機開幾十個連續的 `scp` 連線。
