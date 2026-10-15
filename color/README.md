# color：全域顏色映射的免疫防禦

防禦圖由逐影像最佳化的全域逐像素顏色映射產生（研究主線 color3d：33³ RGB 3D 查找表），輸出只依像素自身顏色，不加性雜訊、不用空間相依色場；目標為使攻擊者以 InstructPix2Pix 做的編輯偏離未防禦的編輯結果，並以 LPIPS 區間與文獻依據的顏色約束控制外觀。主線在 trial `trials/lut3d_defense_and_naturalness/`（不入版控），方法與結果見其 `README.md`，現況見 `STATUS.md`，設計與已退役方向的數據見 `docs/DESIGN.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/immunization_color/` | `cli/`（編輯、淨化、讀數的命令列入口）、`method.py`（全域換色表 `ColorMap`、comm 目標與臉框／膚色支撐函式，trial 引用）、`layout.py`（預設目錄） |
| `scripts/env.sh` | 執行環境：以 `src/` 與 `vendor/` 解析 Python 套件；`ENV_FILE` 可指定機器相關設定 |
| `data/portraits/` | 原圖、`masks/`、`prompts.yaml` |
| `docs/DESIGN.md` | 問題、協定、主線摘要與已退役方向的數據紀錄 |
| `docs/TRIALS.md` | 已刪除的暫時性嘗試紀錄；`run_trial_lifecycle.sh new|promote|drop` 的用法見檔首 |
| `tests/` | 專案自足性（預設目錄、vendor 快照、換行） |
| `vendor/` | `immunization_core` 與 GPU 租約工具的固定版本快照；`vendor.lock.json` 記錄來源 commit 與逐檔雜湊 |
| `artifacts/`、`trials/` | 影像產物（未防禦編輯與淨化後未防禦分母）、暫時性嘗試；不入版控 |

## 執行

於 color 專案根執行；Python 套件以 `src/` 與 `vendor/` 解析：

```bash
source scripts/env.sh   # PYTHONPATH、PY
python -m pytest tests
```

color3d 的求解、編輯、淨化與讀數由 trial 的 `run_lut_queue.sh` 排程（用法見該 trial 的 `HANDOVER.md`）；它呼叫下列入口：

| 入口（`python -m immunization_color.cli.<名稱>`） | 用途 |
|---|---|
| `run_edits`、`apply_purifiers`、`measure_edit_displacement`、`measure_purified_displacement` | 對應 `immunization_core.pipelines` 的 editing、purification、displacement、retention |
| `measure_defense_fidelity` | 防禦圖對原圖的 LPIPS、ΔE00、PSNR、L∞、RMS |

`vendor/` 不就地修改；更新方式為在 repo 根執行 `python core/scripts/generate_vendor_snapshot.py color`。
