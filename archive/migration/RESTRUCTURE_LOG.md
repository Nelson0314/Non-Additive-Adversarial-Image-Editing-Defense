# 重整執行紀錄（雲端 session）

本檔由雲端執行端逐項追加；每段記錄項目、commit 範圍、改動、驗證與待裁定事項。

推送分支：`claude/eloquent-davinci-fvkldc`。本 session 的環境只允許推送到此分支，未推送 `main`；由協調端合併。

## 第 3 項第 3 段（core 剩餘子項 2、4、5、6）

- 基準：`4329881`（與 `origin/main` 相同）。
- commit：`ec44b08`…`e24a2d2`（7 個）。

| commit | 內容 |
|---|---|
| `ec44b08` | `lab/code/edit_{preflight,displacement,retention}.py` 移入 `pipelines.editing`、`pipelines.displacement`、`pipelines.retention`；`--data` 改為必填，其餘參數、CSV 欄位、數值設定、lab 子集順序、`--conditions`、arm 鍵合併與配對檢查不變；core 相依加入 PyYAML；新增 `tests/test_edit_pipelines.py`（替身指標）。 |
| `a2d4096` | 上項文件（README、STATUS、PIPELINE_BEHAVIOR）。 |
| `ff6ebb2` | 色彩與最佳化 helpers 依責任放入 `color.space`、`color.difference`、`color.uniformity`、`optimization.carrier`、`optimization.instruction_free`；`quantise`、`optimise_carrier`、`randomise_carrier` 改為 `quantize`、`optimize_carrier`、`randomize_carrier`；NCF 載體、`lowfreq_color`、幅度求解器不在活動閉包內，未納入；測試隨行，歷史載體以 `tests/carrier_stub.py` 代替。 |
| `8229ccb` | 五支 GPU 租約工具複製至 `core/scripts/`（`run_on_card.sh` → `run_with_gpu_lease.sh`）；遠端根目錄與 `~/env.sh` 改為 `--workdir`、`--env`；queue 的工作執行、驗收、相依改為 `--runner`、`--validator`、`--depends` 注入；`LAB_CAP`→`GPU_CAP`、`LAB_MYCAP`→`QUEUE_CAP`；租約協定、共用租約目錄與預設 6 張不變；測試隨行。 |
| `639e921` | 獨立副本匯入測試加入新模組。 |
| `e24a2d2` | README（含 GPU 租約工具一節）、STATUS、PROVENANCE 更新。 |

來源 SHA-256 與改名對照：`core/docs/pipeline_source_manifest.json`。原 `anti-purification`、`lab` 檔案未修改。

### 驗證

| 指令 | 結果 |
|---|---|
| `python -m pytest -q -p no:cacheprovider core/tests` | 168 passed、21 deselected（本段前 100 passed） |
| 將 `core/` 複製到 scratch 目錄，`PYTHONPATH=<副本>/src python -m pytest -q tests` | 168 passed、21 deselected |
| 副本中 `python -m immunization_core.pipelines.{editing,displacement,retention,purification} --help` | 全部成功 |
| `bash -n core/scripts/*.sh`；`tests/test_gpu_scripts.py` 檢查 LF | 通過 |
| `python -m pytest -q -p no:cacheprovider lab/tests` | 56 passed（原檔未動） |

環境：Python 3.11、CPU 版依賴（torch 2.14、piq 0.8.0、scikit-image 0.26、diffusers 0.40）。需權重的 21 個案例未執行。

### 未完成與待裁定

1. 淨化外部權重算子（DiffPure、IMPRESS、Adverse Cleaner）的真實數值驗證需權重與 GPU，由協調端處理。
2. 租約目錄預設值仍為 `$HOME/lab_leases`：新舊工具必須共用同一租約池，改名需所有取卡入口同時切換，建議於第 10 項由協調端一併處理。
3. `run_with_gpu_lease.sh` 不再自行設定 `PYTHONPATH=<遠端根>`；呼叫端須以 `--env` 檔或環境提供。遠端切換時（第 10 項）的指令需同步。
4. queue 的 lab 專屬部分（pilot／def／chain／readout／fid 語法、分片合併、`validate_job.py`、`FID_ARMS` 預檢、`WAIT_ARMS` sentinel）尚未改寫為注入指令，於第 5 項隨 color 專案處理。
