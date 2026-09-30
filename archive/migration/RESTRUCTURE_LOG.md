# 重整執行紀錄（雲端 session）

本檔由雲端執行端逐項追加；每段記錄項目、commit 範圍、改動、驗證與待裁定事項。

推送分支：`claude/eloquent-davinci-fvkldc`。本 session 的環境只允許推送到此分支，未推送 `main`；由協調端合併。

## 第 3 項第 3 段（core 剩餘子項 2、4、5、6）

- 基準：`4329881`（與 `origin/main` 相同）。
- commit：`ec44b08`…`e24a2d2`（6 個），另加本紀錄的 commit。

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

## 第 4 項：建立 `/baseline`

- 基準：`7f65339`。commit：`2cac5b8`…`3bf658f`（7 個）。

| commit | 內容 |
|---|---|
| `2cac5b8` | 純改名：`anti-purification/main_table/` 的 116 個追蹤檔移入 `baseline/`，內容不變（對照見下）。 |
| `48800a0`、`85663a7` | `core/scripts/export_vendor.py`：由 git HEAD 匯出 `immunization_core` 與 `core/scripts` 至 `<專案>/vendor/`，寫 `vendor.lock.json`（commit、樹雜湊、逐檔 SHA-256）；`--check` 驗證。 |
| `61c9f59` | 逐位元複製：`anti-purification/data/{portraits,targets}` → `baseline/data/`；主表閉包內的 13 個 `src/baselines` 模組 → `immunization_baseline/attacks/`；主表引用的 13 份 `docs/reference` 與 `docs/EVALUATION.md` → `baseline/docs/`。原檔留在 anti-purification，第 6 項整體封存。 |
| `95da550` | vendor 快照（core commit `85663a7`）。 |
| `30024a3` | 自足化：`paths.py` 的兄弟目錄搜尋改為 `layout.py`（只由本專案根推定）；`src.*` import 改至 `immunization_core`／`immunization_baseline.attacks`（`cat_cond`→`concatenate_conditioning`、`expand_cond`→`expand_conditioning`）；`run_edits`、`apply_purifiers`、`measure_edit_displacement`、`measure_purified_displacement` 改為呼叫 core pipelines；對 `anti-purification/runs` 的退回搜尋改為明確參數；shell 鏈改由 baseline 根執行並使用 vendor 內的租約工具；新增 `pyproject.toml`、`scripts/env.sh`、`tests/test_project_layout.py`。 |
| `3bf658f` | README、STATUS、`docs/README.md`、`results/*/README` 的路徑與入口更新。 |

### 檔案對照（程式、設定、文件、非逐條件結果）

| 舊（相對 `anti-purification/main_table/`） | 新 |
|---|---|
| `.gitignore` | `baseline/.gitignore` |
| `README.md` | `baseline/README.md` |
| `STATUS.md` | `baseline/STATUS.md` |
| `code/check_edit_complete.py` | `baseline/src/immunization_baseline/cli/check_edit_completion.py` |
| `code/color_row_chain.sh` | `baseline/scripts/evaluate_color_condition.sh` |
| `code/defence_run.py` | `baseline/src/immunization_baseline/cli/generate_defenses.py` |
| `code/edit_displacement.py` | `baseline/src/immunization_baseline/cli/measure_edit_displacement.py` |
| `code/edit_displacement_flux.py` | `baseline/src/immunization_baseline/cli/measure_flux_displacement.py` |
| `code/edit_flux_preview.py` | `baseline/src/immunization_baseline/cli/run_flux_edits.py` |
| `code/edit_preflight.py` | `baseline/src/immunization_baseline/cli/run_edits.py` |
| `code/edit_retention.py` | `baseline/src/immunization_baseline/cli/measure_purified_displacement.py` |
| `code/edit_sd_family_preview.py` | `baseline/src/immunization_baseline/cli/sweep_editor_parameters.py` |
| `code/edit_sdedit_preview.py` | `baseline/src/immunization_baseline/cli/sweep_sdedit_parameters.py` |
| `code/edit_ultraedit_full.py` | `baseline/src/immunization_baseline/cli/run_ultraedit_edits.py` |
| `code/flux_full_queue_a.sh` | `baseline/scripts/run_flux_conditions.sh` |
| `code/immunise_as_condition.py` | `baseline/src/immunization_baseline/cli/import_defense_artifacts.py` |
| `code/metrics_union.py` | `baseline/src/immunization_baseline/cli/measure_additional_metrics.py` |
| `code/passthrough_readout.py` | `baseline/src/immunization_baseline/cli/measure_additive_transfer.py` |
| `code/paths.py` | `baseline/src/immunization_baseline/layout.py` |
| `code/prompt_sets_ultraedit.json` | `baseline/configs/prompts/ultraedit_templates.json` |
| `code/prompt_sets_ultraedit_round2.json` | `baseline/configs/prompts/ultraedit_noun_placement_variants.json` |
| `code/purify_run.py` | `baseline/src/immunization_baseline/cli/apply_purifiers.py` |
| `code/resume_state.py` | `baseline/src/immunization_baseline/resume_state.py` |
| `code/sd_family_offtarget_readout.py` | `baseline/src/immunization_baseline/cli/measure_off_target_changes.py` |
| `code/ultraedit_sd3_pipeline.py` | `baseline/src/immunization_baseline/third_party/ultraedit/pipeline.py` |
| `docs/README.md` | `baseline/docs/README.md` |
| `results/PASSTHROUGH.md` | `baseline/results/ADDITIVE_TRANSFER.md` |
| `results/aligned/README.md` | `baseline/results/aligned/README.md` |
| `results/aligned/deltae.csv` | `baseline/results/aligned/deltae.csv` |
| `results/aligned/displacement_aligned.csv` | `baseline/results/aligned/displacement.csv` |
| `results/aligned/retention_aligned.csv` | `baseline/results/aligned/retention.csv` |
| `results/displacement.csv` | `baseline/results/displacement.csv` |
| `results/displacement_flux.csv` | `baseline/results/flux/displacement.csv` |
| `results/displacement_ultraedit.csv` | `baseline/results/ultraedit/displacement.csv` |
| `results/flux_preview.csv` | `baseline/results/sweeps/flux/guidance_3p5.csv` |
| `results/flux_preview_g2.csv` | `baseline/results/sweeps/flux/guidance_2p0.csv` |
| `results/flux_preview_truecfg.csv` | `baseline/results/sweeps/flux/true_cfg_3p5.csv` |
| `results/passthrough.csv` | `baseline/results/additive_transfer.csv` |
| `results/retention.csv` | `baseline/results/retention.csv` |
| `results/retention_ultraedit.csv` | `baseline/results/ultraedit/retention.csv` |
| `results/sd_family_ip2p_si18_reference.csv` | `baseline/results/sweeps/ip2p/reference_edits.csv` |
| `results/sd_family_offtarget_ip2p_si18_reference.csv` | `baseline/results/sweeps/ip2p/reference_off_target.csv` |
| `results/sd_family_offtarget_sdxl_ip2p_all8.csv` | `baseline/results/sweeps/sdxl_ip2p/guidance_portraits_off_target.csv` |
| `results/sd_family_offtarget_sdxl_ip2p_high_image_guidance.csv` | `baseline/results/sweeps/sdxl_ip2p/high_image_guidance_off_target.csv` |
| `results/sd_family_offtarget_ultraedit_prompt_round2.csv` | `baseline/results/sweeps/ultraedit/noun_placement_variants_off_target.csv` |
| `results/sd_family_offtarget_ultraedit_prompt_sweep_a.csv` | `baseline/results/sweeps/ultraedit/prompt_templates_man_00_woman_00_off_target.csv` |
| `results/sd_family_offtarget_ultraedit_prompt_sweep_b.csv` | `baseline/results/sweeps/ultraedit/prompt_templates_man_01_woman_01_off_target.csv` |
| `results/sd_family_sd3_ultraedit_low_guidance.csv` | `baseline/results/sweeps/ultraedit/text_image_guidance.csv` |
| `results/sd_family_sd3_ultraedit_quick.csv` | `baseline/results/sweeps/ultraedit/image_guidance.csv` |
| `results/sd_family_sdxl_ip2p_all8.csv` | `baseline/results/sweeps/sdxl_ip2p/guidance_portraits.csv` |
| `results/sd_family_sdxl_ip2p_grid.csv` | `baseline/results/sweeps/sdxl_ip2p/guidance_portrait_pair.csv` |
| `results/sd_family_sdxl_ip2p_high_image_guidance.csv` | `baseline/results/sweeps/sdxl_ip2p/high_image_guidance.csv` |
| `results/sd_family_ultraedit_prompt_round2.csv` | `baseline/results/sweeps/ultraedit/noun_placement_variants.csv` |
| `results/sd_family_ultraedit_prompt_sweep_a.csv` | `baseline/results/sweeps/ultraedit/prompt_templates_man_00_woman_00.csv` |
| `results/sd_family_ultraedit_prompt_sweep_b.csv` | `baseline/results/sweeps/ultraedit/prompt_templates_man_01_woman_01.csv` |
| `results/sdedit_preview.csv` | `baseline/results/sweeps/sdedit/sd15_strength.csv` |
| `results/sdedit_preview_stable-diffusion-2-1-base.csv` | `baseline/results/sweeps/sdedit/sd21_base_strength.csv` |
| `results/sdedit_preview_stable-diffusion-2-1.csv` | `baseline/results/sweeps/sdedit/sd21_v_prediction.csv` |
| `results/sdedit_preview_stable-diffusion-v1-5_sdedit_guidance_sweep_sd15.csv` | `baseline/results/sweeps/sdedit/sd15_guidance.csv` |

逐條件結果依規則改名：`results/defence_<c>.csv` → `baseline/results/defense_<c>.csv`；`results/aligned/defence_<c>_aligned.csv` → `baseline/results/aligned/defense_<c>.csv`；`results/flux_full_<arm>.csv` → `baseline/results/flux/edits_<arm>.csv`；`results/ultraedit_full/<arm>.csv` → `baseline/results/ultraedit/edits/<arm>.csv`；`results/metrics_<x>_union.csv` → `baseline/results/additional_metrics/<x>.csv`；`tests/*` 同名移入 `baseline/tests/`。

### 影像產物目錄對照（第 7 項 CSV 路徑改寫與第 10 項遠端搬移使用）

新位置皆相對 `baseline/`。舊位置有兩處：遠端 `~/image-immunization/runs/`（主線根，`evaluate_color_condition.sh` 的前身即寫在這裡）與 `main_table/images/`（部分鏡像）。

| 舊 | 新 |
|---|---|
| `runs/defence_portraits/<c>/`、`main_table/images/defence_portraits/<c>/` | `artifacts/defenses/<c>/` |
| `runs/eps_aligned/<c>/` | `artifacts/defenses_aligned/<c>/` |
| `runs/edit_preflight/`、`main_table/images/edit_preflight/` | `artifacts/undefended_edits/` |
| `runs/edit_defended/<c>/`、`main_table/images/edit_defended/<c>/` | `artifacts/defended_edits/<c>/` |
| `runs/purified/<c>/` | `artifacts/purified/<c>/` |
| `runs/edit_purified/<c>/`、`main_table/images/edit_purified/<c>/` | `artifacts/purified_edits/<c>/` |
| `main_table/images/flux_full/<arm>/` | `artifacts/flux_edits/<arm>/` |
| `main_table/images/ultraedit_full/{edit_preflight,edit_defended,edit_purified}/` | `artifacts/ultraedit_edits/{undefended_edits,defended_edits,purified_edits}/` |
| `main_table/images/sdedit_preview*/`、`main_table/images/sd_family*/` | `artifacts/sweeps/<編輯器>/<與 results/sweeps 同名>/` |
| `main_table/images/masks/` | 不保留；與 `data/portraits/masks/` 逐位元相同 |
| `runs/color_import/` | `artifacts/color_import/` |
| `runs/state_color/` | `runtime/color_condition/` |
| `lab/runs/defence/color/` | `evaluate_color_condition.sh main` 的第三個參數（color 專案的防禦圖目錄，第 5 項決定） |

### 驗證

| 指令 | 結果 |
|---|---|
| 搬移前（`4329881` worktree）`anti-purification/main_table` 的 `pytest tests` | 67 passed |
| `cd baseline && python -m pytest -q -p no:cacheprovider tests` | 71 passed（原 67 項＋專案自足性 4 項） |
| 將 `baseline/` 單獨複製到 scratch 後同一指令 | 71 passed |
| 15 個 CLI 各自 `PYTHONPATH=src:vendor python -m immunization_baseline.cli.<名稱> --help` | 全部成功；全部模組可匯入（UltraEdit pipeline 需 transformers） |
| `python core/scripts/export_vendor.py baseline --check` | 一致 |
| `bash -n baseline/scripts/*.sh` | 通過 |
| `core/tests`、`lab/tests` | 168 passed／21 deselected、56 passed |

### 未完成與待裁定

1. CSV 內的路徑欄仍為舊值；`measure_additive_transfer` 與 `measure_additional_metrics` 讀取 CSV 中的影像路徑時，相對路徑改以 baseline 根為基準（原為主線根，並有 `runs/`→`images/` 的首段替換）。第 7 項改寫路徑欄前，這兩支對既有 CSV 的重算不可用。
2. 續跑 CSV 的 `protocol_id` 雜湊包含資料與輸出的絕對路徑（UltraEdit 另含原 `SOURCE_HOME`，已改為 `--defenses`、`--purified-edits`）；搬移後以既有 CSV 續跑會因協定不符而拒絕，須以新輸出重跑。此雜湊無法由第 7 項改寫。
3. `generate_defenses`、`attacks/*` 與 `results/ADDITIVE_TRANSFER.md` 的 docstring／文字仍引用封存區的 `../scripts/*.py`、`src/models/attention.py` 與 `lab/results/passthrough/`；於第 8 項（文件清理）與第 5 項確定 color 結果位置後處理。
4. （已處理）`third_party/ultraedit/pipeline.py` 的來源版本由使用者提供並寫入該檔標頭與 `baseline/docs/README.md`：commit `70e8ce5bc3bc8a6a02e1b9e0b6a1eb0058d98bc2`（移植時 `main` 為 `1af5f0478d56d62cf6f35781a2ab63e03d656b2d`，其間該檔未變動），逐行比對除 import 與移植註解外一致。
5. `anti-purification/tests/test_metrics_union_failure.py` 引用已移出的 `main_table/code`，在 anti-purification 內已不可執行；第 6 項封存時原狀保存，不修。
6. 鎖定依賴（`requirements.lock`）屬第 9 項。
