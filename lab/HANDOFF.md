# lab/ 交接

**`lab/` 是獨立的實驗室：所有程式、資料、產出都在這裡。對 `../anti-purification/`
一律唯讀（連 `docs/` 也不寫）；需要的檔案複製過來。**

| 要什麼 | 讀哪裡 |
|---|---|
| 現況、怎麼跑、未完成的事 | 本檔 |
| 協定、各臂設計、已量到的結論、已排除的方向 | `docs/DESIGN.md` |
| 工作規矩（不設判準、GPU、命名、commit） | `../anti-purification/CLAUDE.md` 與本檔「規矩」 |

**不要讀**：`report/index.html`、`report/data.js`、`report_budget/*`（報告頁原始檔，數百行）、
`results/*.csv` 全文（用 `code/passthrough_summary.py` 或自寫彙整）、`runs/`（遠端資料）。

---

## 現況

### 已完成
- 十一個防禦臂（`docs/DESIGN.md` §4）的防禦圖、ip2p 編輯、七道淨化、淨化後編輯與讀數。
- **穿透分離讀數**：`results/passthrough/`。各臂 `D`／`P`／`D_T`（主種子與四個評估種子）、
  `D_seed`、淨化底線、VQA 與 `id_edit`。彙整：`python lab/code/passthrough_summary.py`。
  主要數字在 `docs/DESIGN.md` §5。
- 報告頁 v6（只有對比圖與數據表，不放說明文字；每欄最佳值粗體、欄名以 ↑／↓ 標方向）：
  <https://claude.ai/artifact/VdWw6PoWtrQd2xTLc5Qyvt>。額度預覽頁：
  <https://claude.ai/artifact/QPA2zfCfp5TEHxJ7WTjodR>。兩頁都**尚未**放入穿透分離讀數與 B 段新臂。

### 遠端正在跑（自動接續，不需要人在場）
等變殘差目標的比較（`docs/DESIGN.md` §6）：

1. 佇列 `comm`：`ab_warp_ch_comm` 的 `woman_00` 防禦圖（前三次 OOM，已重設）→
   `chain:ab_warp_ch_comm`。`ab_warp_ch_free` 與其餘 15 張防禦圖、`free` 的鏈已完成。
   隨機對照的 gen／chain 已標成 `.GIVEUP`（使用者裁定不做），`ab_warp_ch_free_random_r1`
   是已跑完的殘留，不必使用。
2. 佇列 `commtail`（兩台各一個 worker，以檔案判定相依）：兩臂 × 四種子的編輯 → `ptfit` →
   `ptread:main`、`ptread:<種子>` → `vqa`（輸出 `results/passthrough/vqa_commtail.csv`）→
   `readout`（重寫 `results/displacement.csv`、`retention.csv`）→ `fid`（重寫 `results/fidelity.csv`）。

查進度：`ls ~/image-immunization/lab/runs/queue/{comm,commtail}/`（`.done`／`.lock`／`.fails`／`.GIVEUP`），
log 在 `lab/runs/logs/queue_<佇列>/<工作>.log` 與 `lab/runs/logs/queue_<佇列>_<主機>.log`。
預計約 2026-09-26 15:00 前完成（basic-2 常被他人佔滿，時間會延後）。
卡的上限：使用者為這批工作暫時授權 **6 張**（兩個 worker 以 `LAB_CAP=6 LAB_MYCAP=6` 啟動）；
其他新工作仍用預設（兩個 session 合計 5、lab 自己 4）。

---

## 未完成的事（依序）

1. **B 段的比較（等 `commtail` 跑完）。** 本機拉回 `results/` 後跑 `passthrough_summary.py`
   （`PAIRS` 已含 `ab_warp_ch_comm − ab_warp_ch_free`）。要報的：`D`、`D_T`（全圖／主體，
   主種子與五種子平均，bootstrap 區間）、VQA 出現格數、`id_edit`、兩臂逐張輸入 LPIPS、
   `comm` 驗證值起訖（各臂 `runs/defence/<臂>/results.csv` 的 `comm_val_start`／`comm_val_end`）。
   `comm` 的移動量只作診斷，不作為效果的證據。另報兩臂對既有 `ab_warp_ch` 的逐張輸入 LPIPS 差。
2. **把主表的對抗擾動也做穿透分離，再做等失真比較。** 這是判斷顏色線有沒有真正干擾能力的
   關鍵對照（`docs/DESIGN.md` §5.7：等 LPIPS 下 `dia_r` 等四個 baseline 的 `D` 與
   `colour_curve_ours` 相同，但顏色線的 `D` 約八成是穿透）。做法：`passthrough_readout.py`
   加一個 `additive` 參數族，`T̂(y) = clip(y + (x_def − x))`；對象為
   `../anti-purification/main_table/results/aligned/` 的十個 `_aligned` 臂（防禦圖與 ip2p 編輯圖路徑
   看該目錄 `displacement_aligned.csv` 的 `defended_png`、`undefended_png` 欄與 `README.md`）。
   只重用既有 PNG，不需要擴散呼叫。
3. **報告頁更新**：新增 `D`／`P`／`D_T` 表、`D` 對 `P` 散佈圖、位移對輸入 LPIPS 散佈圖
   （顏色臂、淨化底線、主表條件並列，主表點標「另一批」）、VQA 表、逐格三聯圖
   `edit(x)`／`T̂(edit(x))`／`edit(x_def)`；加入 `ab_warp_ch_comm`、`ab_warp_ch_free`。
   使用者的格式要求：不放說明文字，只放圖與表；每欄最佳值粗體、欄名用箭頭標方向。
4. **研究方向的討論**：見下節，由使用者決定，不代為判定。

---

## 研究方向的現況（供討論，不是結論）

- 全域顏色映射撐得過淨化、外觀可控，但 ip2p 對全域色調近乎等變：`D` 約七到九成是穿透，
  VQA 顯示指令物件照樣畫出（155–157/160，未防禦 157/160）。扣穿透後的 `D_T`（0.13–0.25）
  高於換種子的離散（0.109），放寬額度仍能增加 `D_T`，但與 `colour_curve_ours` 分不出差異。
- 等變殘差目標 `comm` 是直接對準 `D_T` 的最佳化；它與 `free` 的比較（未完成事項 1）回答
  「目標是不是瓶頸」。若兩者的 `D_T` 分不開，瓶頸在載體：全域色調映射本身與編輯器近乎可交換。
- 未完成事項 2 回答「對抗擾動在扣穿透後是否仍高於顏色線」。對抗擾動的保留率 0.42–0.63，
  顏色線約 1.0；兩者的取捨（干擾強度 × 抗淨化）可以用 `D_T × 保留率` 之類的組合讀數並列。
- 可能的改變方向（使用者尚未選擇）：顏色映射 ＋ 抗淨化的對抗擾動的混合；非全域但自然的載體
  （記憶：局部高振幅是還沒量過的角落）；把讀數從 LPIPS 換成以 VQA／身分為主的攻擊成功率；
  擴大評估規模（8 張、單編輯器，區間寬約 ±0.05）。

---

## 怎麼跑

- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）
  `nelson0314@server.basiclab.lab.nycu.edu.tw`，目錄 `~/image-immunization`（NFS，兩台共用）。
  環境 `source ~/env.sh` 提供 `$PY`。
- 臂的參數只定義在 `scripts/defence_cmd.sh`。
- **工作佇列** `scripts/queue_worker.sh <佇列名> <工作>...`，兩台主機各起一個 worker：
  ```
  cd ~/image-immunization && bash -c 'LAB_CAP=5 LAB_MYCAP=4 nohup setsid bash lab/scripts/queue_worker.sh <佇列> <工作...> > lab/runs/logs/queue_<佇列>_$(hostname).log 2>&1 < /dev/null & disown'
  ```
  注意 `cd` 要在 `bash -c` 之外，否則會跟著背景程序走。工作種類（相依寫在檔頭）：
  `def:<臂>:<影像>`（單張防禦圖，分卡平行）、`chain:<臂>`（併分片 → `arm_chain.sh`，只跑 ip2p）、
  `seed:<臂>:<種子>`、`ptfit`、`ptread:<種子|main>`、`ptbase`、`vqa`、`readout`、`fid`、`pilot:`、`gen:`。
  狀態在 `lab/runs/queue/<佇列>/`；同一工作失敗 3 次寫 `.GIVEUP`，相依它的工作不再派。
  佇列以外的單次 GPU 指令用 `scripts/run_on_card.sh <名稱> <指令...>`（取卡、租約、CUDA 檢查）。
- `arm_chain.sh <GPU> <臂>`：防禦 → ip2p 編輯 → 淨化 → 七道淨化後編輯，10 個 sentinel 在
  `lab/runs/state/`，只在 rc=0 時寫；防禦圖不滿 8 張時只編輯現有影像。
- 實測成本：900 步防禦圖約 65 分鐘／張（`comm` 目標約 53 分鐘，顯存約 18 GB）；只跑 ip2p 的鏈
  約 50 分鐘／臂；跨臂讀數約 15 分鐘；一個種子 32 格編輯約 8 分鐘。
- 報告頁重建：遠端 `python lab/scripts/build_report_assets.py --out lab/report_img_new --quality 70`
  → 拉回成 `lab/report/img/` → 本機 `python lab/scripts/build_report_data.py --out lab/report/data.js`
  → 以同一 `file_path` 發佈（網址不變，已移除的圖檔要以 `null` 刪掉）。單一版本上限 64 MB。

## 規矩（踩過才寫的）

- **卡**：判定空卡要同時看 `scripts/free_cards.sh`、租約目錄 `~/lab_leases/`（格式 `<主機> <pid> <名稱>`，
  兩台共用）與別人 compute app 的顯存合計 < 1 GB。上限是兩個 session 合計 5 張、lab 自己 4 張；
  超過需要使用者明確授權。
- **遠端腳本不可就地覆寫**：先解到 `.stage/` 再 `mv`（跑著的 bash 邊讀邊執行，截斷同一個 inode
  會死於 Stale file handle）。Windows 寫出的 `.sh` 要確認是 LF。
- **取卡函式的紀錄一律寫 stderr**：`$(...)` 會收走 stdout，卡號變成一串字，torch 靜默退回 CPU。
- **殺行程**：`pkill -f <樣式>` 會匹配到下指令的那條 ssh 自己；worker 的主迴圈是父程序為
  `bash -c` 的那一個，子程序（正在跑的工作）不要殺。
- **分母**：不要把新種子或新設定的編輯寫進 `runs/edit_preflight/`。
- **命名**：目錄、檔案、臂名不含日期、流水號或順序詞。commit message 用英文。
- **不設判準**：數據與圖擺出來為止，不下「成立／不成立」「值不值得再跑」的結論。

## 檔案地圖

| 路徑 | 內容 |
|---|---|
| `code/ab_warp_defence.py` | `ab_warp` 族防禦（含 `--objective comm`、`--init random`、分通道與 LPIPS 上限） |
| `code/comm_objective.py` | 等變殘差目標 |
| `code/ab_prism_defence.py`、`curve_budget_defence.py`、`style_warp_defence.py`、`inpaint_region_defence.py` | 其餘臂；`curve_budget_defence.py` 另提供共用 helper |
| `code/edit_preflight.py`、`purify_run.py`、`edit_displacement.py`、`edit_retention.py`、`defence_fidelity.py` | 編輯、淨化、讀數 |
| `code/passthrough_readout.py`、`seed_spread_readout.py`、`edit_vqa.py`、`passthrough_summary.py` | 穿透分離、種子離散、VQA、彙整 |
| `code/warp_budget_preview.py` | 無最佳化的額度預覽（報告 `report_budget/`） |
| `scripts/` | `defence_cmd.sh`、`arm_chain.sh`、`queue_worker.sh`、`run_on_card.sh`、`readout.sh`、報告頁建置 |
| `results/` | `displacement.csv`、`retention.csv`、`fidelity.csv`、`passthrough/` |
| `docs/paper/neucom_134591.json` | 使用者指定論文（Wang et al., Neurocomputing 2026）的 metadata；全文未取得 |
