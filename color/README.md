# color：CIELAB 色度平面全域映射的免疫防禦

防禦圖由全域換色表產生：CIELAB (a,b) 平面的 RBF 位移場加單調亮度曲線，輸出只依像素自身顏色；以等變殘差目標對 InstructPix2Pix 最佳化，並受色差、逐通道 Lab 位移、彩度與 LPIPS 上限約束。設計、協定與結果見 `docs/DESIGN.md`，現況見 `STATUS.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/immunization_color/` | `method.py`（載體、目標、支撐）、`cli/`（命令列入口）、`layout.py`（預設目錄） |
| `configs/conditions.yaml` | 防禦條件與其參數的唯一正本 |
| `scripts/` | `generate_condition.sh`（條件 → 防禦圖指令）、`evaluate_condition.sh`（單卡完整鏈）、`measure_condition_results.sh`（跨條件讀數）、`run_queue.sh` 與 `queue_*.sh`（工作佇列）、`env.sh` |
| `data/` | `portraits/`（原圖、`masks/`、`prompts.yaml`）、`color_lpips_ref.csv`（逐張 LPIPS 上限參考） |
| `results/` | `displacement.csv`、`retention.csv`、`fidelity.csv`（`color`）；`variants/<條件>/`（三個 simple 條件）；`defenses/`、`defense_shards/`、`defended_edits/`、`purified/`、`purified_edits/`（逐條件的求解與編輯紀錄） |
| `docs/DESIGN.md` | 問題、協定、方法、結果與已退役方向的數據紀錄 |
| `tests/` | 條件指令、佇列相依與驗收、讀數退出碼、專案自足性 |
| `vendor/` | `immunization_core` 與 GPU 租約工具的固定版本快照；`vendor.lock.json` 記錄來源 commit 與逐檔雜湊 |
| `docs/TRIALS.md` | 已刪除的暫時性嘗試紀錄；`trial.sh new|promote|drop` 的用法見檔首 |
| `artifacts/`、`runtime/`、`trials/` | 影像產物、排程狀態、暫時性嘗試；不入版控 |

## 條件

| 條件 | 設定 |
|---|---|
| `color` | 預設參數 |
| `color_simple` | 色偏上限改為錨點方框，只保留膚色同色 ΔE00 與暖色兩道逐像素上限 |
| `color_simple_skinbox` | 同上；冷色方向的方框隨離膚色中心的距離放大到 2 倍 |
| `color_simple_xattn` | 同 `color_simple`；目標改為攻擊端對類別詞的交叉注意力 |

條件與其參數的唯一正本為 `configs/conditions.yaml`；`scripts/generate_condition.sh` 經 `python -m immunization_color.conditions` 讀取。七道淨化取自 core 的 `purifiers/protocol.json`。

## 執行

於 color 專案根執行；Python 套件以 `src/` 與 `vendor/` 解析：

```bash
source scripts/env.sh                          # PYTHONPATH、PY；ENV_FILE 可指定機器相關設定
bash scripts/evaluate_condition.sh <GPU> color # 防禦圖 → 編輯 → 淨化 → 淨化後編輯
bash scripts/measure_condition_results.sh <GPU>
python -m pytest tests
```

工作佇列：`nohup setsid bash scripts/run_queue.sh <佇列名> <工作>... &`，工作語法（`pilot`、`def`、`chain`、`readout`、`fid`）見該檔檔頭；排程、租約與驗收由 `vendor/scripts/queue_worker.sh` 執行，全局卡數由使用者逐次授權，未指定時全局合計 6 張。單次 GPU 指令用 `vendor/scripts/run_with_gpu_lease.sh --workdir <color 根> <名稱> <指令...>`。

| 入口（`python -m immunization_color.cli.<名稱>`） | 用途 |
|---|---|
| `generate_color_defenses` | 求解防禦圖 |
| `run_edits`、`apply_purifiers`、`measure_edit_displacement`、`measure_purified_displacement` | 對應 `immunization_core.pipelines` 的 editing、purification、displacement、retention |
| `measure_defense_fidelity` | 防禦圖對原圖的 LPIPS、ΔE00、PSNR、L∞、RMS |
| `validate_queue_job` | 佇列工作的產物驗收 |

`vendor/` 不就地修改；更新方式為在 repo 根執行 `python core/scripts/export_vendor.py color`。
