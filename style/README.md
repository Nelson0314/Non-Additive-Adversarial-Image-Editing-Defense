# style：以風格編輯為載體的免疫防禦（SPA 移植）

依 Wang et al.（Neurocomputing 702, 134591, 2026）的方法，凍結 InstructPix2Pix、最佳化風格指令的文字 embedding 與暖身後的 latent 位移，使防禦圖在攻擊端 ip2p 的後續編輯中失效。設計與結果見 `docs/DESIGN.md`，現況見 `STATUS.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/immunization_style/` | `method.py`（生成器、載體、目標、選點）、`cli/`（命令列入口）、`layout.py`（預設目錄） |
| `configs/styles.yaml` | 風格指令的唯一正本 |
| `configs/jobs/<輪名>.spec` | 各輪的工作清單（`r11`、`r13`、`cls_p_noedit`、`cls_p_snow`） |
| `scripts/` | `run_style_prompt_jobs.sh`（一輪實驗的排程）、`env.sh` |
| `data/portraits/` | 原圖、`masks/`、`prompts.yaml` |
| `results/` | `defenses/<輪名>/<工作>/`（`results.csv`、`trace.csv`）、`edits/<輪名>/`（逐工作 `preflight.csv`、`readout_<風格>.csv`） |
| `docs/` | `DESIGN.md`；`references/`（論文出處與 metadata） |
| `tests/` | 選點、讀數輸入檢查、工作清單、專案自足性 |
| `vendor/` | `immunization_core` 與 GPU 租約工具的固定版本快照；`vendor.lock.json` 記錄來源 commit 與逐檔雜湊 |
| `docs/TRIALS.md` | 已刪除的暫時性嘗試紀錄；`trial.sh new|promote|drop` 的用法見檔首 |
| `artifacts/`、`runtime/`、`trials/` | 影像產物、排程紀錄、暫時性嘗試；不入版控 |

## 執行

於 style 專案根執行：

```bash
bash scripts/run_style_prompt_jobs.sh <輪名> <清單檔> [全局授權卡數] [最佳化派工上限]
python -m pytest tests
```

清單檔每行為「名稱 影像 風格 其餘參數」；名為 `ref` 的工作（`--lr 0 --updates 1`）產生對照。防禦圖寫到 `artifacts/defenses/<輪名>/`，編輯與讀數寫到 `artifacts/edits/<輪名>/`；取卡經 `vendor/scripts/run_with_gpu_lease.sh`，全局卡數由使用者逐次授權，未指定時全局合計 6 張。

| 入口（`python -m immunization_style.cli.<名稱>`） | 用途 |
|---|---|
| `generate_style_prompt_defenses` | 求解防禦圖（選項見 `--help`） |
| `run_edits` | `immunization_core.pipelines.editing`，預設資料根 `data/portraits` |
| `measure_style_prompt_edits` | 編輯結果 LPIPS（對未防禦編輯、對風格參考圖的編輯）與編輯前後改變量，全圖／主體／背景 |

讀數的未防禦分母預設為 `artifacts/undefended_edits/ip2p_si18`。`vendor/` 不就地修改；更新方式為在 repo 根執行 `python core/scripts/export_vendor.py style`。
