# lab/ — 雲端實驗紀錄（追蹤於版控）

`experiments/` 不進版控（大量影像），故雲端執行的**關鍵紀錄**改存於此，由
`scripts/run_experiment.sh` 於每次執行結束時自動 commit 並 push，供機器自動
關機後回頭查看。

## 命名

```
lab/<YYYYMMDD_HHMMSS>_<label>/
```

- 時間戳為執行起始時間；`<label>` 由 runner 第一個參數指定（如 `quick`、`full`、
  `s0calib`），用以區分不同目的之執行。

## 每個資料夾內容

- `run.log`：該次執行的**完整終端機輸出**（含各 stage 起訖時間、每張保護耗時、
  最後一行 `FINISHED exit=<code>` 標示成功/失敗）。
- `rundirs.txt`：本次對應的 `experiments/<stage>/<時間戳>` 目錄清單。
- `stageN__<時間戳>/`：自對應 run 目錄複製的關鍵產出——`summary.md`、
  `results.csv`、校準/淨化曲線 `*.png`、`config_snapshot.yaml`、`env.json`。
  （逐張影像 png 仍只留在雲端 `experiments/`，不推上來。）

## 查看方式

先看 `run.log` 末行的 `exit=0`（成功）；再看 `stage2__*/summary.md` 的
「加性 vs 非加性 mean drop_lpips」——方向可行性的核心數字。
