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

## 第 5 項：建立 `/color`、`/style`

- 基準：`c0a7647`（含 UltraEdit 來源版本的補記）。commit：`6bb29e7`…`22195d8`（12 個）。

| commit | 內容 |
|---|---|
| `6bb29e7` | 純改名：`lab/` 的 39 個追蹤檔移入 `color/`、`style/`（對照見下），內容不變。 |
| `367ca15` | 刪除 `lab/scripts/{gpu_lease,gpu_policy,run_on_card}.sh` 與其 3 份測試；正本已在 `core/scripts/`、`core/tests/test_gpu_scripts.py`，經 vendor 提供給各專案。 |
| `8cf9ecf` | `archive/migration/remote_lab_runs/runs/` 的 132 份 CSV 逐位元移入 `color/results/` 與 `style/results/`（對照見下），並刪除此暫存目錄（交接文件的指示）。 |
| `2d5d605` | `data/portraits` 快照複製到 `color/data/`、`style/data/`（與 baseline 那份相同）。 |
| `39ca57a` | 解除 color／style 互相 import：`encode_text`、`AttnObjective`、`CrossAttnObjective` → `core/optimization/attention.py`；`_shift`、`channel_shift_p95`、`channel_shift_max` → `core/color/shift.py`；`load_images`、`write_rows` → `core/io.py` 的 `load_dataset_images`、`write_sorted_csv`；CPU 測試（stub UNet）。 |
| `958081b`、`de12528` | color、style 的 vendor 快照（core commit `2097c77`／`39ca57a`，core 內容相同）。 |
| `2097c77` | color 自足化：`color_defence.py` 拆為 `immunization_color.method` 與 `cli.generate_color_defenses`（argparse 抽為 `build_parser`）；四支共用流程改為呼叫 core pipelines；`paths.py` → `layout.py`；shell 入口改由 color 根執行；佇列改為包裝 vendor 內的通用 `queue_worker.sh`，以 `queue_job.sh`、`queue_validate.sh`、`queue_depends.sh` 注入 pilot／def／chain／readout／fid 語法。 |
| `f01d985` | style 自足化：`style_prompt_defence.py` 拆為 `immunization_style.method` 與 `cli.generate_style_prompt_defenses`；讀數的未防禦分母與資料根改為本專案預設；排程改用 vendor 內的 `run_with_gpu_lease.sh`。 |
| `88ead52` | 行為修正：`--patience` 說明字串中未跳脫的 `%` 使 `generate_style_prompt_defenses --help` 拋 `ValueError`（原 `style_prompt_defence.py` 即有此問題）。 |
| `0390596` | color、style 的 README、STATUS 與 DESIGN 路徑更新（color 的 STATUS 由原 `lab/HANDOFF.md` 改寫）。 |
| `22195d8` | baseline 與 core 文件中其餘 `lab/` 引用改指 color 專案或標明已刪除的 commit。 |

### 檔案對照（程式、設定、文件）

| 舊 | 新 |
|---|---|
| `lab/.gitattributes` | `color/.gitattributes` |
| `lab/.gitignore` | `color/.gitignore` |
| `lab/HANDOFF.md` | `color/STATUS.md` |
| `lab/code/color_defence.py` | `color/src/immunization_color/method.py` |
| `lab/code/defence_fidelity.py` | `color/src/immunization_color/cli/measure_defense_fidelity.py` |
| `lab/code/edit_preflight.py` | `color/src/immunization_color/cli/run_edits.py` |
| `lab/code/purify_run.py` | `color/src/immunization_color/cli/apply_purifiers.py` |
| `lab/code/edit_displacement.py` | `color/src/immunization_color/cli/measure_edit_displacement.py` |
| `lab/code/edit_retention.py` | `color/src/immunization_color/cli/measure_purified_displacement.py` |
| `lab/code/paths.py` | `color/src/immunization_color/layout.py` |
| `lab/code/validate_job.py` | `color/src/immunization_color/cli/validate_queue_job.py` |
| `lab/code/style_prompt_defence.py` | `style/src/immunization_style/method.py` |
| `lab/code/style_prompt_readout.py` | `style/src/immunization_style/cli/measure_style_prompt_edits.py` |
| `lab/data/color_lpips_ref.csv` | `color/data/color_lpips_ref.csv` |
| `lab/docs/DESIGN.md` | `color/docs/DESIGN.md` |
| `lab/docs/STYLE_PROMPT.md` | `style/docs/DESIGN.md` |
| `lab/docs/paper/neucom_134591.json` | `style/docs/references/neucom_134591.json` |
| `lab/results/displacement.csv` | `color/results/displacement.csv` |
| `lab/results/retention.csv` | `color/results/retention.csv` |
| `lab/results/fidelity.csv` | `color/results/fidelity.csv` |
| `lab/scripts/defence_cmd.sh` | `color/scripts/generate_condition.sh` |
| `lab/scripts/arm_chain.sh` | `color/scripts/evaluate_condition.sh` |
| `lab/scripts/readout.sh` | `color/scripts/measure_condition_results.sh` |
| `lab/scripts/queue_worker.sh` | `color/scripts/run_queue.sh` |
| `lab/scripts/style_prompt_round.sh` | `style/scripts/run_style_prompt_jobs.sh` |
| `lab/tests/test_color_conditions.py` | `color/tests/test_color_conditions.py` |
| `lab/tests/test_queue_validation.py` | `color/tests/test_queue_validation.py` |
| `lab/tests/test_readout_script.py` | `color/tests/test_measure_condition_results.py` |
| `lab/tests/test_style_readout_inputs.py` | `style/tests/test_style_readout_inputs.py` |
| `lab/tests/test_style_selection.py` | `style/tests/test_style_selection.py` |

`lab/results/exp/<讀數>_<條件>.csv` → `color/results/variants/<條件>/<讀數>.csv`（三個 simple 條件 × displacement／fidelity／retention）。

### 遠端 CSV 與產物目錄對照（第 7 項 CSV 路徑改寫與第 10 項遠端搬移使用）

舊位置相對遠端 `~/image-immunization/lab/runs/`。CSV 已收入 `results/` 的同名子樹；影像與其餘產物於第 10 項搬入各專案 `artifacts/` 的同名子樹。

| 舊 | CSV（已入版控） | 產物 |
|---|---|---|
| `defence/<c>/` | `color/results/defenses/<c>/` | `color/artifacts/defenses/<c>/` |
| `defence_shards/<c>/<img>/` | `color/results/defense_shards/<c>/<img>/` | `color/artifacts/defense_shards/<c>/<img>/` |
| `defence_pilot/<c>/<img>/` | — | `color/artifacts/defense_pilots/<c>/<img>/` |
| `edit_defended/<c>/` | `color/results/defended_edits/<c>/` | `color/artifacts/defended_edits/<c>/` |
| `purified/<c>/` | `color/results/purified/<c>/` | `color/artifacts/purified/<c>/` |
| `edit_purified/<c>/<pur>/` | `color/results/purified_edits/<c>/<pur>/` | `color/artifacts/purified_edits/<c>/<pur>/` |
| `edit_preflight/`（原為指向主表 `runs/edit_preflight/` 的連結） | — | `color/artifacts/undefended_edits/`，內容與 `baseline/artifacts/undefended_edits/` 相同 |
| `edit_purified/undefended/`（原為連結） | — | `color/artifacts/purified_edits/undefended/`，內容與 baseline 同名目錄相同 |
| `state/`、`queue/<q>/`、`logs/` | — | `color/runtime/state/`、`color/runtime/queues/<q>/`、`color/runtime/logs/`；不搬舊完成狀態 |
| `style_prompt_<輪>/<工作>/` | `style/results/defenses/<輪>/<工作>/` | `style/artifacts/defenses/<輪>/<工作>/` |
| `style_prompt_<輪>_edit/` | `style/results/edits/<輪>/` | `style/artifacts/edits/<輪>/` |
| 主表 `runs/edit_preflight/ip2p_si18/`（style 讀數的分母） | — | `style/artifacts/undefended_edits/ip2p_si18/`，或以 `--undefended` 指定 |
| `specs/style_prompt_cls_*.txt`、各輪 `jobs.spec` | 未入版控 | 協定端決定是否收入 `style/configs/` |

`<輪>` 為 `r11`、`r13`、`cls_p_noedit`、`cls_p_snow`。完成標記由 `<條件>.defence.done` 改名為 `<條件>.defense.done`。

### 驗證

| 指令 | 結果 |
|---|---|
| `python -m pytest -q -p no:cacheprovider core/tests` | 172 passed、21 deselected |
| `baseline`、`color`、`style` 各自 `python -m pytest -q -p no:cacheprovider tests` | 71、40、14 passed |
| 將 `color/`、`style/` 各自單獨複製到 scratch 後同一指令，及 `python vendor/scripts/export_vendor.py . --check` | 40、14 passed；lock 一致 |
| 各 CLI `PYTHONPATH=src:vendor python -m immunization_<專案>.cli.<名稱> --help` | color 7 支、style 3 支全部成功 |
| `bash -n color/scripts/*.sh style/scripts/*.sh` | 通過 |
| 132 份遠端 CSV 與搬入位置逐位元比對 | 一致 |
| `git ls-files lab/` | 0 個檔案 |

原 lab 測試 56 項的去向：GPU 租約 3 份（15 項）已在 core；其餘移入 color（條件指令、佇列驗收、讀數退出碼）與 style（選點、讀數輸入）。原 queue `launch` 單元測試由 core 的佇列整合測試涵蓋，color 改為測試注入的相依與 `FID_ARMS` 預檢。

### 未完成與待裁定

1. color 與 style 需要主表的未防禦編輯（見上表）；第 10 項切換遠端時由協調端決定複製或以符號連結提供。
2. style 各輪的工作清單不在 repo 中；若要保存，協調端取回後放入 `style/configs/`。
3. 各專案的 vendor 快照對應不同 core commit（baseline `85663a7`；color、style `2097c77`／`39ca57a`，core 內容相同）。baseline 不使用第 5 項新增的 core 模組，未更新。
4. 第 4 項紀錄的未完成事項 1–3、5、6 仍適用（CSV 路徑欄於第 7 項改寫；舊續跑 CSV 的 `protocol_id` 含絕對路徑）。

## 第 6 項：建立 `/archive`

- 基準：`00264ac`。commit 見下表（前 3 個為第 4、5 項的修正）。

| commit | 內容 |
|---|---|
| `32f6045` | 行為修正（第 4、5 項）：三個專案 `.gitignore` 的 `artifacts/`、`runtime/` 未錨定，同時排除了 `vendor/immunization_core/artifacts/` 與 `runtime/` 兩個子套件，先前推送的 vendor 快照缺這 5 個檔案，乾淨 clone 無法匯入 `immunization_core`。規則改為 `/artifacts/`、`/runtime/`、`/trials/`、`/.pytest_cache/`，補入缺少的檔案（與各 `vendor.lock.json` 一致）。 |
| `cd9b97f` | `export_vendor.py` 在匯出後以 `git check-ignore` 檢查，有檔案會被排除即失敗。 |
| `e846124` | 三個專案重新匯出 vendor，統一對應 core commit `cd9b97f`。 |
| `487e6d7` | 純改名：`anti-purification/`（1,545 檔）→ `archive/anti-purification/`、`frequency-phase/`（13 檔）→ `archive/frequency-phase/`、`HANDOFF.md`、`COLOUR_LINE.md` → `archive/`；內部不改名、不改內容。 |
| `42676af` | 根 `.gitignore` 的 `*/.tmp/`、`*/.pytest_tmp_*/` 改為 `**/`，使搬入 archive 後的巢狀 pytest 暫存目錄仍被排除；baseline、core 文件中的來源路徑改指 `archive/anti-purification/`。 |

### 驗證

| 指令 | 結果 |
|---|---|
| 比對 `487e6d7~1` 與 `487e6d7` 的 `ls-tree` blob（`anti-purification/`↔`archive/anti-purification/`、`frequency-phase/`↔`archive/frequency-phase/`、兩份根文件） | 1,545／13／2 個檔案全部相同 |
| 自本地 repo `git clone` 一份乾淨副本（修正後），執行 `core/tests` 與三個專案的 `tests` | 172 passed／21 deselected；71、40、14 passed |
| 搬移後於工作目錄重跑同樣四組測試 | 同上 |

根目錄現為 `README.md`、`archive/`、`baseline/`、`color/`、`core/`、`style/`（`docs/` 為空目錄，git 不追蹤，於第 13 項建立）。

### 未完成與待裁定

1. 根 `README.md` 仍是舊內容，連結指向已不存在的 `anti-purify/` 與已移入封存區的 `anti-purification/`；依計畫於第 13 項重寫。
2. `archive/anti-purification/` 內的程式、測試與文件仍以原位置為準（例如 `tests/test_metrics_union_failure.py` 引用已移出的 `main_table/code`）；依規則不修改，封存區不承諾可執行。
3. 協調端本機若有 `anti-purification/` 下未追蹤的資料（`runs/` 影像、`.tmp/` 等），合併本分支時 git 不會搬動它們，需在本機一併移到 `archive/anti-purification/`。

## 第 7 項：CSV 改寫

- 基準：`18aa70a`。commit：`b35bb2b`（改寫與驗證工具）、`9614c38`（改寫結果）、`c6d4b4e`（程式與文件中的 `colour_curve_ours` → `color_curve`），另加本紀錄。
- 範圍：`baseline/results`、`color/results`、`style/results` 下全部 228 份 CSV（84／84／60 份，38,938 列）；`archive/` 不動。

### 改寫內容

| 類別 | 規則 | 格數 |
|---|---|---|
| 路徑欄（`data`、`data_root`、`input_png`、`output_png`、`png`、`defended_png`、`undefended_png`、`source_png`、`original_png`、`defence_png`、`reference_png`） | 去除遠端根 `/nfs/home/nelson0314/image-immunization/` 或本機根 `C:/image-immunization/[anti-purification/]`，再依產物角色的前綴規則改為相對所屬專案根的路徑；未知前綴即中止 | 16,674 |
| 識別值 | `colour_curve_ours` → `color_curve`（color 的四份 fidelity 表 `anchor_source` 欄） | 32 |
| 新欄 | `baseline/results/ultraedit/{displacement,retention}.csv` 在 `scenario` 後加入 `editor`＝`ultraedit`（`scenario=ip2p` 為借用欄位，原值不改） | 2 張表 |

前綴規則全文見 `archive/migration/rewrite_results.py`；baseline 的規則即第 4 項的產物目錄對照，另加 `runs/edit_defended_aligned/` → `artifacts/defended_edits_aligned/`、`runs/edit_purified_aligned/` → `artifacts/purified_edits_aligned/`（第 4 項對照表遺漏，第 10 項搬移時一併處理），以及 sweep 影像目錄：`main_table/images/{flux_preview,flux_preview_g2,flux_preview_truecfg}/` → `artifacts/sweeps/flux/{guidance_3p5,guidance_2p0,true_cfg_3p5}/`、`sdedit_preview/`、`sdedit_preview_sd21/`、`sdedit_preview_sd21base_sweep/`、`sdedit_guidance_sweep_sd15/` → `artifacts/sweeps/sdedit/{sd15_strength,sd21_v_prediction,sd21_base_strength,sd15_guidance}/`、`sd_family/<批次>/` → `artifacts/sweeps/<與 results/sweeps 同名>/`。color／style 的規則即第 5 項的產物目錄對照。

未改寫：非路徑欄（含 `dir_norm`、`direction` 一類數值欄、`model`／`victim` 的模型 ID、`spec_source`／`modification_note` 等說明文字）、欄名（含 `defence_png`、`free_*`、`D_*`、`DT_*`、`disp_*`）、其他識別值（`dia_pt`、`r11`、`ip2p_si18` 等，依診斷報告 4.7）、主表幾何淨化分區欄的數值（第 12 項）。

### 驗證（`rewrite_results.py` 逐表執行，任一條件不符即中止；結果寫入 `archive/migration/csv_rewrite_report.json`）

1. 改寫前 csv 解析後再寫出與原檔逐位元相同（全部 LF），確保改寫只動目標欄。
2. 列數不變；表頭除新增的 `editor` 外不變。
3. 鍵集合不變（鍵欄取 `condition`、`arm`、`scenario`、`image`、`prompt_index`、`purifier`、`variant`、guidance／strength、`style`、`pairing` 中存在者）。
4. 非路徑欄逐值相同；唯一例外是識別值對照，且改寫後值等於對照結果。
5. 96 份可追溯到遷移前 manifest（`8bcaae0`）的表：改寫前的 blob 與 manifest 相同，表頭與列數相同。其餘 132 份為遠端數值 CSV，第 5 項已與暫存副本逐位元比對。
6. 改寫後 `baseline`、`color`、`style` 的測試：71、40、14 passed。

### 未完成與待裁定

1. style 的 `r11`／`r13` 參照編輯表（`edits/r11/ref_cool_grade/`、`edits/r13/ref_p_snow/`）的 `input_png` 指向已刪除的輪 `pilot`、`r12_p_snow`，依同一規則改寫為 `artifacts/defenses/pilot/…`、`artifacts/defenses/r12_p_snow/…`；這些產物在遠端已刪除時即為失效參照（原值同樣失效）。
2. 續跑 CSV 的 `protocol_id`（雜湊值）無法改寫；第 4 項紀錄第 2 點仍適用。
3. 第 4 項紀錄第 1 點（CSV 路徑欄為舊值時兩支讀數不可用）已由本項解決：路徑欄現在相對 baseline 根，與程式的解析方式一致。

## 第 8 項：程式風格、文件與 trials 機制

使用者裁定（第 8 項範圍）：CLI 參數全面統一為 `--data-root`、`--output-dir`、`--output-csv` 等，不保留舊參數別名，遠端腳本於第 10 項由協調端改寫；七道淨化與防禦條件改為單一設定檔正本；`import_defense_artifacts` 加輸入雜湊 manifest、`measure_additional_metrics` 區分必要與選配指標，既有 CSV 需遠端影像的欄位以補值腳本於第 10 項補回，不留兩種 schema。分五段提交。

### 第 1 段：trials 機制（`4f0d0dc`、`c043345`）

- `core/scripts/trial.sh`（經 vendor 分發）：`new <名稱>` 建立 `trials/<名稱>/` 與 README 樣板；`promote <名稱>` 要求 `trials/` 以外沒有未提交變更後刪除 trial；`drop <名稱>` 要求 `docs/TRIALS.md` 已有該名稱的一列，再以 `ssh $TRIAL_REMOTE rm -rf $TRIAL_REMOTE_ROOT/trials/<名稱>` 刪除遠端副本並刪除本機；兩個環境變數未設定時拒絕，只刪本機須明確給 `--local-only`。名稱只允許小寫英數字與底線。
- baseline、color、style 各加 `docs/TRIALS.md`（說明與空表），README 目錄表列出；`/trials/` 已由各專案 `.gitignore` 排除。
- 驗證：`core/tests/test_gpu_scripts.py` 新增 2 項（假 git 專案與 ssh stub）；core 相關測試 25 passed；三個專案 71／40／14 passed。
- 協調端：遠端使用時設定 `TRIAL_REMOTE="-p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw"`、`TRIAL_REMOTE_ROOT=<遠端專案根>`（第 10 項確定路徑）。

### 第 2 段：協定與條件正本（`c36db56`…`04a7cdd`）

| 正本 | 讀取方式 | 取代的副本 |
|---|---|---|
| `core/src/immunization_core/purifiers/protocol.json`（identity＋七道，順序與強度） | Python：`purifiers.protocol.PURIFIERS`、`label()`、`purifier_labels()`；shell：`python -m immunization_core.purifiers.protocol --exclude-identity` | `protocol.py` 的常數、`color/scripts/evaluate_condition.sh` 與 `baseline/scripts/evaluate_color_condition.sh` 的 `for PUR in …`、`validate_queue_job.PURIFIERS` |
| `baseline/configs/conditions.yaml`（求解族、spec、求解端文字條件與出處、是否屬主表、主表列序） | `immunization_baseline.conditions`；`python -m immunization_baseline.conditions [--main-table｜--solvable]` | `generate_defenses` 的 `PGD_SPECS`、`SOLVER_PROMPT`、`CONTENT_CONDITIONS`、`DCT_CONDITIONS`、`FEEDFORWARD_CONDITIONS`；`run_ultraedit_edits.CONDITIONS`；`measure_flux_displacement.CONDITIONS` |
| `color/configs/conditions.yaml`（條件 → 額外參數） | `python -m immunization_color.conditions <條件>` | `generate_condition.sh` 的 `case` |
| `style/configs/styles.yaml`（風格名 → 指令） | `immunization_style.method.STYLES` | `method.py` 的 `STYLES` 字典 |

行為上的差異（均不改科學協定）：
- shell 鏈的七道淨化執行順序改為協定檔順序（crop_resize0.1、jpeg30、jpeg50、jpeg80、blur1、blur2、rotate15），原為 jpeg50、crop_resize0.1、blur1、rotate15、jpeg30、jpeg80、blur2；各階段的完成標記名稱不變。
- color 的鏈在全部階段完成後寫 `runtime/state/<條件>.chain.done`；佇列的 `WAIT_ARMS` 改等此標記（原等最後一道 `pedit_blur2_ip2p`，其位置隨順序改變）。
- `generate_defenses` 未給 `--conditions` 時的求解順序改為主表列序；`measure_flux_displacement` 的預設條件改為主表十二列（原清單含已退出主表的 `color_curve`、缺 `color`；`results/flux/displacement.csv` 已含 `color`、不含 `color_curve`）。
- `generate_defenses` 啟動時檢查設定檔的常數文字條件與攻擊模組持有的值相同、每個 pgd 條件的 spec 名稱與條件名相同。

驗證：core 174 passed、21 deselected；baseline 72、color 41、style 15 passed（新增 registry 測試各 1 項）；三個專案以 vendor 取得的協定標籤一致。

### 第 3 段：CLI 參數統一（`bc65ff1`…`80a6cf9`）

依使用者裁定，全部 CLI 的路徑類參數依角色重新命名，不保留舊名稱或別名；argparse 的 `dest` 維持原名，程式內部行為不變。命名規則：資料集根 `--data-root`；輸出影像目錄 `--output-dir`；輸出 CSV `--output-csv`（逐檔多份時 `--output-csv-dir`）；單一條件的防禦圖目錄 `--defenses-dir`、多條件根目錄 `--defenses-root`；其餘輸入依產物角色加 `-root`（目錄樹）、`-dir`（單一目錄）或 `-csv`（單一 CSV）。未列出的參數（如 `--images`、`--conditions`、`--scenarios`、`--arm`、`--suffix`、`--steps`、`--cap`、`--limit`、`--runner`、`--validator`、`--depends`）不變。

| CLI | 舊參數 → 新參數 |
|---|---|
| `immunization_core.pipelines.editing` | `--data` → `--data-root`、`--out` → `--output-dir`、`--defended` → `--defenses-dir` |
| `immunization_core.pipelines.purification` | `--defended` → `--defenses-dir`、`--data` → `--data-root`、`--out` → `--output-dir` |
| `immunization_core.pipelines.displacement` | `--defended-root` → `--defended-edits-root`、`--preflight` → `--undefended-edits-root`、`--data` → `--data-root`、`--out` → `--output-csv` |
| `immunization_core.pipelines.retention` | `--purified-root` → `--purified-edits-root`、`--displacement` → `--displacement-csv`、`--data` → `--data-root`、`--out` → `--output-csv` |
| `immunization_baseline.cli.check_edit_completion` | `--data` → `--data-root`、`--defended` → `--defenses-dir`、`--out` → `--output-dir` |
| `immunization_baseline.cli.generate_defenses` | `--out` → `--output-dir`、`--data` → `--data-root` |
| `immunization_baseline.cli.import_defense_artifacts` | `--run` → `--source-dir`、`--data` → `--data-root`、`--out` → `--output-dir` |
| `immunization_baseline.cli.measure_additive_transfer` | `--out` → `--output-csv`、`--data` → `--data-root`、`--defenses` → `--defenses-root`、`--aligned-defenses` → `--aligned-defenses-root`、`--results` → `--results-dir` |
| `immunization_baseline.cli.measure_flux_displacement` | `--out` → `--output-csv`、`--data` → `--data-root`、`--edits-csv-root` → `--edits-csv-dir` |
| `immunization_baseline.cli.measure_off_target_changes` | `--edits` → `--edits-csv`、`--out` → `--output-csv`、`--data` → `--data-root` |
| `immunization_baseline.cli.run_flux_edits` | `--data` → `--data-root`、`--defenses` → `--defenses-root`、`--defended` → `--defenses-dir`、`--out` → `--output-dir`、`--out-csv` → `--output-csv` |
| `immunization_baseline.cli.run_ultraedit_edits` | `--root` → `--output-dir`、`--defenses` → `--defenses-root`、`--purified-edits` → `--purified-edits-root`、`--results` → `--output-csv-dir` |
| `immunization_baseline.cli.sweep_editor_parameters` | `--data` → `--data-root`、`--out` → `--output-dir`、`--out-csv` → `--output-csv` |
| `immunization_baseline.cli.sweep_sdedit_parameters` | `--data` → `--data-root`、`--out` → `--output-dir`、`--out-csv` → `--output-csv` |
| `immunization_color.cli.generate_color_defenses` | `--out` → `--output-dir`、`--data` → `--data-root`、`--lpips-ref` → `--lpips-ref-csv` |
| `immunization_color.cli.measure_defense_fidelity` | `--root` → `--defenses-root`、`--out` → `--output-csv` |
| `immunization_color.cli.validate_queue_job` | `--project` → `--project-root` |
| `immunization_style.cli.generate_style_prompt_defenses` | `--out` → `--output-dir`、`--data` → `--data-root` |
| `immunization_style.cli.measure_style_prompt_edits` | `--edits` → `--edits-root`、`--undefended` → `--undefended-edits-dir`、`--data` → `--data-root`、`--out` → `--output-csv`、`--ref-edits` → `--reference-edits-root` |
| `immunization_baseline.cli.run_edits` | 同 `immunization_core.pipelines.editing` |
| `immunization_baseline.cli.measure_edit_displacement` | 同 `immunization_core.pipelines.displacement` |
| `immunization_baseline.cli.measure_purified_displacement` | 同 `immunization_core.pipelines.retention` |
| `immunization_baseline.cli.apply_purifiers` | 同 `immunization_core.pipelines.purification` |
| `immunization_color.cli.run_edits` | 同 `immunization_core.pipelines.editing` |
| `immunization_color.cli.measure_edit_displacement` | 同 `immunization_core.pipelines.displacement` |
| `immunization_color.cli.measure_purified_displacement` | 同 `immunization_core.pipelines.retention` |
| `immunization_color.cli.apply_purifiers` | 同 `immunization_core.pipelines.purification` |
| `immunization_style.cli.run_edits` | 同 `immunization_core.pipelines.editing` |
| `core/scripts/run_with_gpu_lease.sh` | `--workdir` → `--work-dir`、`--env` → `--env-file` |
| `core/scripts/queue_worker.sh` | `--workdir` → `--work-dir`、`--state` → `--state-dir`、`--logs` → `--log-dir`、`--env` → `--env-file` |

專案內已同步：`baseline/scripts/evaluate_color_condition.sh`、`color/scripts/{generate_condition,evaluate_condition,measure_condition_results,queue_job,queue_validate,run_queue}.sh`、`style/scripts/run_style_prompt_jobs.sh`、各 CLI 包裝的預設參數、測試與文件。第 10 項由協調端改寫遠端家目錄腳本時，依上表替換。

驗證：core 174 passed（21 deselected）；baseline 72、color 41、style 15 passed；三個專案全部 CLI 的 `--help` 成功；`bash -n` 全部 shell 通過。

### 第 4 段：匯入 manifest、必要與選配指標（`3c712c2`、`ff0d37a`、`33f70bf`）

**`import_defense_artifacts`**：新增必填 `--source-settings <方法設定紀錄>`（color 條件為 color 專案該條件目錄的 `results.csv`）。匯入階段先寫 `<--output-dir>/import_manifest.json`（條件、variant、norm、budget、來源目錄、方法設定檔路徑與 SHA-256、逐張的來源防禦圖／原圖／輸出防禦圖路徑與 SHA-256），任何輸入缺少即中止、不寫 manifest；保真量測階段的 `results_all.csv` 每列新增 `source_sha256`、`original_sha256`、`defended_sha256`、`source_settings`、`source_settings_sha256`。`baseline/scripts/evaluate_color_condition.sh` 已傳入 `--source-settings "$color_defenses/results.csv"`。

**`measure_additional_metrics`**：
| stage | 必要（任何錯誤即中止） | 選配（後端無法建立時留空並記原因） |
|---|---|---|
| fidelity | `fid_fsim`、`fid_delta_e00`、`fid_mse` | — |
| displacement | `disp_fsim`、`disp_mse` | — |
| retention | `disp_purified_fsim` | — |
| aesthetic | — | `aes_*` 七項（pyiqa） |
| vmaf | — | `vmaf`（含 libvmaf 的 ffmpeg） |

選配指標只在建立後端時捕捉 `ImportError`、`OSError`、`RuntimeError`、`ValueError`，原因寫入 `unavailable_metrics`（`<欄名>: <原因>`，以 `; ` 分隔；全部可用時為空字串）；建立後的計算錯誤一律中止。每列新增 `device`（本工具固定 CPU）。

**既有 CSV 的補值**（`archive/migration/backfill_result_schema.py`）：
- 已於本機執行 `additional-metrics`：`additional_metrics/` 五份表補 `device=cpu`，aesthetic 與 vmaf 補 `unavailable_metrics`（既有 104／6,240 列的選配指標皆有值，故為空字串）；列數、鍵與原有欄位逐值不變，欄序與程式輸出一致。
- **待協調端於第 10 項遠端版面切換後執行** `import-hashes`，把 `baseline/results/defense_color.csv` 補成新 schema 並在遠端寫出 `baseline/artifacts/defenses/color/import_manifest.json`：

  ```bash
  cd <遠端 repo 根>
  python archive/migration/backfill_result_schema.py import-hashes \
      --source-settings color/artifacts/defenses/color/results.csv
  git add baseline/results/defense_color.csv   # 提交補值後的表
  ```

  前提：`baseline/artifacts/color_import/<影像>__color__defended.png`（指向 color 防禦圖）、`baseline/artifacts/defenses/color/<影像>__color__def.png`、`baseline/data/portraits/<類>/<影像>.png` 與上述設定檔都存在；任一缺少即中止且不寫檔。腳本驗證列數、鍵集合與原有欄位不變。補值前 `defense_color.csv` 為唯一仍是舊 schema 的表。

驗證：baseline 78 passed（新增匯入 2 項、選配指標 4 項）；`import-hashes` 以替身檔案在 scratch 目錄模擬成功一次、缺一張防禦圖時中止（結束碼 1）。

### 第 5 段：文件與 docstring 清理（`74cce42`…`f327f3e`）

依 RESTRUCTURE_AUDIT §5.1、§5.2、§5.5 清理 core、baseline、color、style；`archive/` 內部不改。

- **baseline 參考文件**（`docs/reference/*.md`、`docs/EVALUATION.md`）：指向本專案的舊路徑改為現行位置（`src/baselines/X.py` → `src/immunization_baseline/attacks/X.py`、`src/metrics/X.py` → `vendor/immunization_core/metrics/X.py`、`src/purify/ops.py` → `vendor/immunization_core/purifiers/operators.py`、`main_table/results/defence_*.csv` → `results/defense_*.csv`、`main_table/images/defence_portraits/` → `artifacts/defenses/`、淨化設定 → core `purifiers/protocol.json`）；仍在封存區的檔案指向 `archive/anti-purification/` 或 `archive/frequency-phase/`；已不在 repo 的檔案標為「原 `…`（不在 repo 內）」。上游 repo 的路徑不改。
- **稽核 §5.5 逐項**：AUDIT_DANP §0 改為查證範圍與來源版本；AUDIT_DAYN 的接入段標為歷史提案並寫明實際入口 `generate_defenses` 的 `dayn` 分支；AUDIT_DIA「目前仍可存取」改為查證快照；AUDIT_INPAINTING_METHODS §4 綁定 2026-08-05 下載批次；AUDIT_MIST_DIFFVAX 增補段與原始查證分開；AUDIT_PURIFIERS 的 404 限於查證日期、`DESIGN_2026-08-05.md` 標為不在 repo 的歷史文件、主表七道以協定檔為準；AUDIT_SIFM「本輪沒有改」改為整合入口；BASELINE_ALIGNMENT §4–5 標為頻域／相位研究線的歷史協定。
- **用語**：自有程式 docstring、註解與文件中的「本輪」「本次」「目前」「先前」「我們」「為什麼…」改為穩定敘述（本專案、修正前、用途、理由）；程式執行期語意（目前裝置、本次執行的 arm、這次前向）保留。附日期與數值的「實測」紀錄保留。
- **color `measure_defense_fidelity`**：docstring 移除十二臂描述；說明 `budget` 欄為沿用的文字標籤（`ΔE00 16 / 臉 8`），不反映 `generate_color_defenses` 的實際上限（整圖 ≤ 32、臉框與膚色同色 ≤ 16），實際上限以各條件 `results.csv` 為準。CSV 欄位與值不變。
- **拼法**：`measure_additive_transfer.defence_png()` 改為 `defense_png()`（CSV 欄 `defence_png` 保留）；style 佇列訊息改為 `defense`；`results/aligned/README.md` 的表名改為 `defense_<條件>.csv`。`pipelines.editing` 寫出的 `defence` 欄與 `import_defense_artifacts` 寫入 CSV 的 `spec_source` 值屬既有 schema／資料，不改。
- vendor 重新匯出（baseline、color、style），`export_vendor.py --check` 通過。

驗證：`core` 174 passed（21 deselected）、`baseline` 78 passed、`color` 41 passed、`style` 15 passed；改動的 `.py` 皆通過 `py_compile`。

### 第 8 項完成狀態

五段皆已推送。留給第 10 項（協調端）：遠端腳本改用新 CLI 參數（第 3 段對照表）、遠端執行 `backfill_result_schema.py import-hashes`（第 4 段）。未處理：`archive/` 內文件（依規則不改）、style `docs/` 的 r11／r13／cls 設定對照與 PDF 依賴清單（需要遠端 job spec 與原 PDF 來源，列入第 13 項前的待確認事項）。

## 第 9 項：環境與自足驗證（`f618c0d`…）

開始前已將 `origin/main`（含 `d3168ee`）快轉併入本分支。

### 換行（`f618c0d`）

根目錄新增 `.gitattributes`：`* text=auto eol=lf`；影像（png、jpg、jpeg、gif、webp、avif、bmp、tif、tiff）、PDF、壓縮檔與權重檔標為 `binary`。`git add --renormalize .` 未改變任何索引內容（索引中原本就沒有 CRLF 文字檔）。`archive/anti-purification/.gitattributes`、`color/.gitattributes`、`style/.gitattributes` 保留。

驗證：以 `core.autocrlf=true`（Windows 簽出設定）clone，工作目錄 CRLF 檔 0 個；在該 clone 內 `export_vendor.py --check` 三個專案通過；2,245 個版控檔的 SHA-256 與 blob 相同（見下方獨立驗證）。

### Windows 可攜性（`c8987cb`）

- `color/tests/test_color_conditions.py`：替身腳本改為 `exec "<Path(sys.executable).as_posix()>" "$@"`（加引號、POSIX 形式）。
- `core/tests/test_gpu_scripts.py`：傳給 bash 的腳本與目錄參數改為 `as_posix()`；`test_command_exit_releases_owned_lease` 原本比較 bash `pwd` 的輸出與 Python 路徑字串（Windows 上 `pwd` 為 `/c/...` 形式），改為由 `sys.executable`（POSIX 形式）寫出 `os.getcwd()`，再以 `Path.resolve()` 比較。
- `color/tests` 其餘傳給 bash 的腳本路徑同樣改為 `as_posix()`。`PATH` 前置替身目錄仍以 `os.pathsep` 串接（Git Bash 會轉換繼承的 Windows `PATH`）。

本環境為 Linux，未能在 Windows 上實跑；Linux 上 core 25／color 41 項通過。

### CSV 欄名與識別值改名（`2e348fa`、`d5b3150`、`e9418c9`）

使用者裁定一律改名。對照：

| 種類 | 舊 | 新 | 範圍 |
|---|---|---|---|
| CSV 欄名 | `defence_png` | `defense_png` | `baseline/results/additive_transfer.csv`（1 表） |
| CSV 欄名 | `defence` | `defense` | color `defended_edits`／`purified_edits`（32 表）、style `edits`（20 表）的 `preflight.csv` |
| CSV 欄名 | `deltaE00_skin_colour` | `deltaE00_skin_color` | color `defenses`／`defense_shards` 的 `results.csv`（36 表） |
| CSV 值 | `diffvax.py::immunise`（`solver_prompt_source`） | `diffvax.py::immunize` | `baseline/results/defense_diffvax.csv` 8 格；`configs/conditions.yaml` 同步 |
| CSV 值 | `../scripts/`（`spec_source`） | `archive/anti-purification/scripts/` | `baseline/results/defense_color.csv` 8 格 |
| 寫出欄的程式 | `pipelines.editing` 的 `"defence"`、`measure_additive_transfer` 的 `"defence_png"`、`generate_color_defenses` 的 `"deltaE00_skin_colour"` | 對應新欄名 | 讀這些欄的程式：`core/tests/test_edit_pipelines.py` |
| 自有識別字 | `skin_colour_support`、`skin_centre`、限制名 `"skin_colour"`、`centre` | `skin_color_support`、`skin_center`、`"skin_color"`、`center` | `immunization_color` |
| 自有識別字 | `diffvax.immunise`、`danp.normalise_attention` | `immunize`、`normalize_attention` | `immunization_baseline.attacks` |
| 自有識別字 | `grey()`、`--objective enc_grey` | `gray()`、`enc_gray` | `immunization_style`（`enc_grey` 不出現在任何 CSV） |
| 預設值 | `import_defense_artifacts --condition` 預設 `colour` | `color` | 與主表條件名一致 |

`archive/migration/rename_csv_columns.py` 只改寫表頭與上述兩欄：89 表只換表頭列（其餘位元組不變），2 表改值；逐表驗證列數、欄數與未改欄位逐值相同。`--check` 模式在改寫後回傳 0。遠端 `artifacts/` 內的 `preflight.csv`（`defence` 欄）與 `results.csv`（`deltaE00_skin_colour` 欄）由協調端在第 10 項於遠端 repo 根執行：

```bash
python archive/migration/rename_csv_columns.py --check color/artifacts style/artifacts baseline/artifacts   # 列出待改
python archive/migration/rename_csv_columns.py color/artifacts style/artifacts baseline/artifacts
```

保留不改：`archive/` 內部、描述封存檔名的字串（`immunise.py`、`__immunised.png`）、core `PROVENANCE.md` 與 `pipeline_source_manifest.json` 中的原始檔名。

### style 工作清單與論文出處（`82689e0`）

- `archive/migration/remote_style_specs/` 的四份清單移為 `style/configs/jobs/{r11,r13,cls_p_noedit,cls_p_snow}.spec`：每份加三行檔頭（輪的設定摘要、原遠端檔名、用法），其餘內容逐位元相同；暫存目錄已刪除。
- 核對：244 個與 `results.csv` 同名的設定值全部相同（6 處只差布林寫法 `True`／`1`）；每列可由現行 `generate_style_prompt_defenses.build_parser()` 解析；`style/tests/test_job_specs.py` 檢查每個結果輪都有清單、清單列與 `results/defenses/<輪>/` 的工作目錄一一對應、風格名存在於 `configs/styles.yaml`。
- `style/docs/references/README.md`：以 DOI `10.1016/j.neucom.2026.134591`、PII `S0925-2312(26)01989-2`、EID、刊期日期記錄出處，欄位來源為 `neucom_134591.json`；第一作者 Wang 與卷號 702 不在 metadata 內，沿用 `DESIGN.md` 的既有記載（本環境沒有 PDF，未能再核對）。`DESIGN.md` 的「根目錄 PDF」改為 DOI 與此檔。PDF 不入版控。

### 依賴與鎖定（`a5ab24f`、`c589b30`、`d7b1d01`）

- 依賴宣告：以 AST 掃描各專案 `src/`、`vendor/`、`tests/` 的第三方 import，對照 pyproject。補上 `sentencepiece`、`protobuf`（`MetricSuite` 以 `AutoProcessor` 載入 SigLIP，其 tokenizer 需要這兩項；`siglip_pair` 為位移與保留量的必要欄）；core 的 `metrics` extra 同步。`opencv-contrib-python`、`lpips` 只用於不在主表協定內的算子（CLAHE、AdverseCleaner、IMPRESS），列為三個專案的 `purifiers` extra。`guided_diffusion`（DiffPure）與 `torch_xla`（UltraEdit pipeline 的可選分支）不宣告，缺席時前者明確失敗、後者不走該分支。
- 鎖定：新增 `core/scripts/freeze_env.py`（已匯入各專案 `vendor/scripts/`）。在專案根以該專案的直譯器執行即寫出 `requirements.lock`（檔頭記 Python、平台、torch 與 CUDA 版本；內容為 `pip freeze --all`，排除本 repo 的 `immunization-*` 套件）；`--check` 在環境與鎖定檔不符時結束碼 1。`core/tests/test_freeze_env.py` 2 項。
- **待協調端於第 10 項在遠端實際執行環境產生並提交**（本沙箱的套件版本不是產生結果的環境，不能作為鎖定依據）：

  ```bash
  cd <遠端 repo 根>/baseline && source scripts/env.sh && "$PY" vendor/scripts/freeze_env.py
  cd <遠端 repo 根>/color    && source scripts/env.sh && "$PY" vendor/scripts/freeze_env.py
  cd <遠端 repo 根>/style    && source scripts/env.sh && "$PY" vendor/scripts/freeze_env.py
  cd <遠端 repo 根>/core     && python scripts/freeze_env.py      # 以 core 測試所用的直譯器
  git add */requirements.lock
  ```

  三個專案若共用同一個遠端環境，三份鎖定檔內容相同，仍各自入庫以維持專案自足。提交後由第 13 項在各專案 README 記錄鎖定檔。

### 獨立目錄驗證（`677b6d1`；報告 `archive/migration/standalone_verification.json`）

`archive/migration/verify_standalone.py` 對 `677b6d1`：各專案以 `git archive` 單獨解到 repo 外的空目錄（不含其他專案），只設 `PYTHONPATH=src[:vendor]`，並設 `HF_HUB_OFFLINE=1`、`CUDA_VISIBLE_DEVICES=`。

| 專案 | 檔案（SHA-256 與 blob 相同） | import 模組 | `--help` 通過（未初始化 CUDA） | `bash -n` | pytest |
|---|---|---|---|---|---|
| core | 85 | 41 | 5 | 6 | 176 passed、21 deselected |
| baseline | 232 | 79 | 16 | 9 | 78 passed |
| color | 190 | 54 | 8 | 14 | 41 passed |
| style | 160 | 49 | 3 | 8 | 20 passed |

`immunization_core` 在三個專案副本中皆由副本內的 `vendor/` 載入。模擬 Windows 簽出（`core.autocrlf=true`）的 2,245 個版控檔 SHA-256 全部等於 blob。repo 內直接執行：core 176、baseline 78、color 41、style 20 passed。

### 未完成與待確認

- `requirements.lock` 四份：需遠端執行環境，由協調端於第 10 項產生（指令見上）。
- 遠端 `artifacts/` 的 CSV 欄名改寫：由協調端於第 10 項執行（指令見上）。
- Windows 上的實跑：本環境無法執行，修正依原因分析完成；請協調端在本機 Windows 簽出後執行 `python -m pytest` 於 core 與 color 確認。

### 驗收後修正：佇列重複派工（`da82000`、`21d54bf`）

協調端在 Windows 上回報 `core/tests/test_gpu_scripts.py::test_queue_runs_dependencies_first_and_validates` 連續 3 次失敗，`order` 為 `['first', 'second', 'second']`。

根本原因：`core/scripts/queue_worker.sh` 主迴圈在同一次掃描中先以 `done_` 判定工作未完成，接著執行 `ready`（呼叫外部 `--depends` 指令），最後才檢查 `<工作>.lock`。工作若在這段期間完成，會依序寫入 `.done`、在 EXIT trap 移除鎖；主迴圈據此看到「未完成且無鎖」，取得鎖後再派一次。Windows 上起外部程序較慢，此空檔變長，因而每次重現；遠端的 validator 與 depends 為 Python 程序，同一空檔存在，屬同一缺陷。

修正：取得 `.lock` 後重新判定 `done_` 與 `dead`，已完成或已放棄即釋放鎖、不派送。執行端一律先寫 `.done`／`.GIVEUP` 再釋放鎖，故持有鎖時前一次執行的結果必然可見。

新增 `test_job_finishing_during_readiness_check_is_not_relaunched`：runner 執行期間讓 `--depends` 延遲，固定上述交錯順序。修正前在 Linux 上 3／3 次失敗（`['only', 'only']`），修正後 `test_gpu_scripts.py` 連跑 5 次皆 26 passed。原斷言未放寬，未 skip。

vendor 已重新匯出（baseline、color、style 的 `vendor/scripts/queue_worker.sh`）。測試：core 177（21 deselected）、baseline 78、color 41、style 20 passed。style 的 `run_style_prompt_jobs.sh` 以行程內旗標 `OPT`／`EDT` 記錄已派工作，不經此判斷，不受影響。遠端若有以舊版 `queue_worker.sh` 執行中的佇列，需在第 10 項切換時改用新版。

## 第 10 項：遠端重整（協調端）

- 遠端 `~/image-immunization` 改為 GitHub `main` 的 clone（`git clone --depth 1`，簽出 `b95c6e0`）；之後以 `git pull` 同步，不再以 `git archive` 傳送。
- `archive/migration/remote_migrate.py` 依第 4、5 項的目錄對照，把舊樹的產物以同一 NFS 上的 rename 搬入新樹：baseline 27 組、color 9 組、style 8 組；color、style 讀的未防禦編輯改為各自的複本（原為指向主表的連結）；`baseline/artifacts/purified_edits_aligned/undefended` 改為專案內相對連結 `../purified_edits/undefended`。
  搬遷後容量：`baseline/artifacts` 9.6 GB、`color/artifacts` 814 MB、`style/artifacts` 105 MB、`archive` 4.7 GB。
- 舊研究樹逐檔併入 `archive/anti-purification/`：與版控相同 519 檔、僅換行不同 521 檔、內容不同 13 檔（皆為 repo 版本較新），這三類留在 `~/image-immunization.old`。遠端獨有且未入版控的檔案（約 1,080 份 log／CSV／JSON／txt 與少量影像）依使用者指示刪除。`archive/` 內另有 147 個指向已不存在目錄（例如 `WACV-s4`）的舊連結，依封存規則不修改。
- 遠端 CSV 欄名：`rename_csv_columns.py` 改寫 `artifacts/` 內 290 份表，`--check` 回傳 0。
- `import-hashes` 補值：原匯入中繼目錄已不存在，依原命名由 `color/artifacts/defenses/color/*__color__def.png` 複製重建 `baseline/artifacts/color_import/`（8 張；baseline 與 color 的防禦圖逐位元相同）。`baseline/results/defense_color.csv` 補成新 schema，遠端寫出 `baseline/artifacts/defenses/color/import_manifest.json`。
- `~/env.sh` 移除舊版面的 `PYTHONPATH` 與 `cd` 兩行（備份 `~/env.sh.bak`）；其餘為機器設定（`PY`、`HF_HOME`、`DIFFPURE_CKPT`）。

### 未完成

1. `requirements.lock`：遠端 venv（`~/venvs/wacv`）由 `uv` 建立、沒有 `pip`，`freeze_env.py` 依設計中止。需改為支援 `uv pip freeze --python <直譯器>`（第 13 項），之後由協調端在遠端產生。
2. 租約目錄仍為 `~/lab_leases`（所有取卡入口的共用預設）。

## 第 11 項：Claude 記憶（協調端）

記憶目錄中的舊路徑改為新位置；三份描述舊入口的記憶（根 HANDOFF、lab、main_table）由 `repo-layout-after-restructure` 取代；
`main-table-session-one-card` 改名 `baseline-session-one-card`；取卡競態一份改記為已修（`0b68f9b`、`da82000`）。全部 `[[連結]]` 與索引可解析。記憶不在 repo 內。

## 第 12 項：幾何分區欄更正（協調端）

- basic-1 一張卡（租約 `basic-1-0`）以 `measure_purified_displacement` 重算 11 個條件（`color` 以外）全部七道、4,928 列，約 15 分鐘。
- 比對：全圖欄、`net_gain`、`retained`、`siglip_pair`、`blocked` 與五道非幾何算子的分區欄，4,928 列逐值與原表相同。
- 寫回：`crop_resize0.1`、`rotate15` 的 1,408 列 `disp_purified_subject`／`disp_purified_background`，2,814 個值改變，最大絕對變化 0.10288。
  更正前的表在 `228c59b`。
- `baseline/results/aligned/retention.csv` 的分區欄未重算；當時程式是否含 `purified_mask()` 未查證。
- 發現：`run_with_gpu_lease.sh` 以變數 `ENV_FILE` 保存 `--env-file`，而專案的 `scripts/env.sh` 會 source `$ENV_FILE`；
  `--env-file scripts/env.sh` 因此無限遞迴，bash segfault。本次改傳 `--env-file ~/env.sh` 執行。待第 13 項修正。

## 第 13 項：收尾（`711ec9b`…）

開始前已將 `origin/main`（至 `137cfd0`，含第 10–12 項）併入本分支。

### 程式修正

- **`--env-file` 遞迴**（`711ec9b`）：`run_with_gpu_lease.sh` 與 `queue_worker.sh` 以 `ENV_FILE` 保存 `--env-file`，而各專案 `scripts/env.sh` 在 `ENV_FILE` 有值時 source 它；`--env-file scripts/env.sh` 因此無限遞迴。內部變數改為 `LEASE_ENV_FILE`、`QUEUE_ENV_FILE`；呼叫端自行 export 的 `ENV_FILE`（機器設定，例如 `~/env.sh`）仍由 `env.sh` 讀取。
  回歸測試 3 項（`run_with_gpu_lease.sh` 以專案形式的 env.sh、同時帶機器設定、`queue_worker.sh` 以專案形式的 env.sh）；修正前 3 項皆以 signal 11 結束（結束碼 −11），修正後通過。
- **租約目錄改名**（`83cae3d`）：預設值由 `$HOME/lab_leases` 改為 `$HOME/gpu_leases`，只在 `gpu_policy.sh` 定義（`gpu_lease.sh` 刪除重複定義，改由 source `gpu_policy.sh` 取得）。所有取卡入口（`gpu_lease.sh`、`run_with_gpu_lease.sh`、`queue_worker.sh`、`free_cards.sh` 不讀租約）經此同一定義。新增測試：未設 `LEASE` 時兩支 source 檔皆解析為 `$HOME/gpu_leases`，且 `core/scripts/*.sh` 不含 `lab_leases`。
- **`freeze_env.py` 支援 uv venv**（`7c1e1af`）：直譯器有 pip 時用 `python -m pip freeze --all`；沒有 pip 時用 `uv pip freeze --python <直譯器>`；兩者皆無即中止。鎖定檔檔頭記錄所用工具。以 `uv venv` 建立的無 pip 環境驗證：寫出、`--check` 為 0；安裝一個套件後 `--check` 為 1 並列出差異；`PATH` 無 uv 時中止。新增 3 項單元測試。
- vendor 重新匯出（`069f0c0`）。測試：core 184（21 deselected）、baseline 78、color 41、style 20 passed。

### 文件（`2686646`、`728470a`、`8e82a02`）

- 根 `README.md` 改寫：目錄與入口、專案自足（vendor、env.sh、鎖定檔）、資料保存（results／data 入庫，artifacts／runtime／trials 不入庫）、執行環境。
- 新增根 `CLAUDE.md`：共通規則（範圍、書面用語、命名、不設判準與科學協定、程式與測試、GPU 與租約、暫時性嘗試）。`archive/anti-purification/CLAUDE.md` 依封存規則不改；根 `CLAUDE.md` 註明 `archive/` 內文件不適用於現行專案。
- `core/STATUS.md` 改寫為範圍、現況、已知限制與規則；`core/README.md` 更新租約目錄、`export_vendor.py`、`freeze_env.py` 與 `--env-file` 說明，刪除流程字句與 `.tmp/codex_audit` 路徑。
- `baseline/STATUS.md`：已知限制新增一條，`results/aligned/retention.csv` 的 `crop_resize0.1`、`rotate15` 共 1,280 列（10 個條件 × 2 道 × 64 格）的 `disp_purified_subject`／`disp_purified_background` 未以 `purified_mask()` 重算、數值未改，全圖欄不受影響；遠端位置、租約目錄、`requirements.lock` 狀態更新。
- `color/STATUS.md`、`style/STATUS.md`：未防禦分母改記為本專案 `artifacts/` 內的複本（第 10 項已搬入），style 工作清單位置改為 `configs/jobs/`，租約目錄更新；三份 STATUS 的共通規則改指向根 `CLAUDE.md`，保留專案專屬規則。
- 根 `.gitignore`：改寫過時註解（原指向不存在的 `non-additive-frequency/`），規則保留；新增 `/*.pdf`（根目錄的論文 PDF 不入版控）。`git ls-files -ci --exclude-standard` 為 0（沒有版控檔被排除）。

### 殘留舊名查核

範圍：根 `README.md`、`CLAUDE.md`、`core/`、`baseline/`、`color/`、`style/` 的版控文字檔，排除 `vendor/`（core 的副本）與出處文件（`core/docs/PROVENANCE.md`、`PIPELINE_BEHAVIOR.md`、`*manifest.json`）。
樣式：`lab_leases`、`lab/`、`main_table/`、`anti-purify`、`colour`、`defence`、`immunis`、`WACV`、`LAB_CAP`、`run_on_card`、`codex_audit`、`*_aligned.csv`、流程字眼（`第 N 項`、`重整`、`round<N>`），以及已刪除或改名的腳本名（`arm_chain`、`defence_cmd`、`readout.sh`、`color_row_chain`、`flux_full_queue`、`style_prompt_round`、`metrics_union`、`passthrough_readout` 等）。

修正：`results/aligned/README.md` 的 `displacement_aligned.csv`／`retention_aligned.csv`／`edit_purified/undefended` 改為現行檔名與 `artifacts/purified_edits/undefended`；`AUDIT_MIST_DIFFVAX.md` 的 `immunise` 與 `src.baselines.diffvax` 改為 `immunize` 與 `immunization_baseline.attacks.diffvax`。

保留（出處或外部名稱）：「原 `lab/...`」「原 `main_table/...`」加 commit 的出處註記；CSV 值與設定中「原為 `colour_curve_ours`」的改名紀錄；`archive/anti-purification/scripts/immunise.py` 及其輸出檔名 `__immunised.png`（封存腳本的實際檔名，`import_defense_artifacts` 讀取該格式）；文獻標題中的 colour；`test_device_contract.py` 刻意設定 `WACV_ALLOW_TF32=1` 以驗證舊變數不再生效；`test_gpu_scripts.py` 檔頭的來源檔名；公式敘述中的「第 0 項／第 1 項」（batch 索引）與清單項次。

### 協調端要在遠端執行的指令

**1. 租約目錄改名**（必須在沒有任何工作持有租約時執行；兩台主機共用 NFS 家目錄，只執行一次）：

```bash
cd ~/image-immunization
# 前提：沒有持有中的租約、沒有 .guard 鎖，且新目錄尚不存在
find ~/lab_leases -mindepth 1 -maxdepth 1 ! -name .capacity -print   # 必須無輸出
[ ! -e ~/gpu_leases ] || { echo "~/gpu_leases 已存在，先確認其內容"; exit 1; }
mv ~/lab_leases ~/gpu_leases
git pull --ff-only                      # 取得預設為 ~/gpu_leases 的工具
bash baseline/vendor/scripts/free_cards.sh   # 確認工具可讀新目錄
cat ~/gpu_leases/.capacity 2>/dev/null       # 既有全局授權值隨目錄保留
```

`git pull` 之後才啟動新工作；`~/image-immunization.old` 與 `archive/` 內的舊腳本仍預設 `~/lab_leases`，不得再用於派工。

**2. 產生 `requirements.lock`**（在實際執行環境；`~/env.sh` 設定 `PY` 指向 `~/venvs/wacv` 的直譯器，該 venv 沒有 pip，須 `uv` 在 `PATH` 上）：

```bash
cd ~/image-immunization && git pull --ff-only
command -v uv                                            # 必須有輸出
for p in baseline color style; do
  (cd "$p" && export ENV_FILE=~/env.sh && source scripts/env.sh && "$PY" vendor/scripts/freeze_env.py) || break
done
(cd core && source ~/env.sh && "$PY" scripts/freeze_env.py)
head -4 */requirements.lock                               # 檔頭應記錄 torch 與 CUDA 版本、uv pip freeze
for p in baseline color style; do (cd "$p" && export ENV_FILE=~/env.sh && source scripts/env.sh && "$PY" vendor/scripts/freeze_env.py --check); done
git add core/requirements.lock baseline/requirements.lock color/requirements.lock style/requirements.lock
git commit -m "Lock the remote execution environment for each project"
```

三個專案共用同一個 venv 時，四份鎖定檔的套件列相同，仍各自入庫以維持專案自足。提交後在各專案 README 的「執行」一節記錄 `requirements.lock` 與 `freeze_env.py --check` 的用法。

### 未完成

- 上述兩組遠端指令（協調端）。
- `results/aligned/retention.csv` 的幾何分區欄重算（需 GPU；是否重算由使用者決定），已列於 `baseline/STATUS.md` 已知限制。

## 驗收後處理（協調端）

- Codex 唯讀驗收 `f89cac2`：未發現 P0，列 7 項 P1、11 項 P2，另有三題待裁定。使用者裁定：等失真臂重算；style 的 `r11`、`r13`、`cls_p_noedit`、`cls_p_snow` 四組刪除；
  卡數規則只保留「使用者未說明時全局合計 6 張」一條，其餘以使用者口頭派發為主。
- 遠端：`~/lab_leases` 已改名 `~/gpu_leases`（改名前為空），遠端樹同步至 `f89cac2`。
- 等失真臂 `results/aligned/retention.csv`：basic-1 一張卡以 `purified_mask()` 重算 10 個條件全部七道、4,480 列，逐值與原表相同（含 1,280 列幾何分區欄），數值未改，`baseline/STATUS.md` 的限制改為已查證。

## 驗收修正：Codex 審查（`RESTRUCTURE_REVIEW.md`）與使用者裁定（`d92417a`…`9ae0011`）

開始前已將 `origin/main`（至 `7caa13a`）併入本分支。遠端現況依協調端告知：租約目錄已改名為 `~/gpu_leases`，遠端樹已同步到 `f89cac2`。

### 使用者裁定

1. 等失真臂的幾何分區欄：協調端已重算，4,480 列逐值相同，本輪不處理。
2. style 四組結果刪除（`d92417a`、`d84f364`）：刪除前在 `style/docs/TRIALS.md` 各記一列（另補三列只存在於原 DESIGN 結果表的設定），來源 commit `7caa13a`；刪除 `configs/jobs/*.spec`、`results/{defenses,edits}/{r11,r13,cls_p_noedit,cls_p_snow}/`、`tests/test_job_specs.py`；DESIGN 移除結果與資料位置兩節，改寫入本專案自己的攻擊端協定；STATUS 寫明本專案沒有保存結果。
3. 卡數規則（`df68acb`、`105862b`）：根 `CLAUDE.md` 只留「使用者沒有說明或說明不清時，全局合計上限 6 張」；佔卡門檻、他人 compute app、各線預設卡數等條文自 `CLAUDE.md`、各 STATUS、README 刪除。`measure_free_gpus.sh` 的偵測門檻保留為工具行為，只在其檔頭與 `core/README.md` 的工具說明中出現。
4. Windows 測試（`e153005`）：`test_default_lease_directory_is_shared_by_all_tools` 改為在 `$LEASE` 內建立標記檔，由 Python 確認其位於 `tmp_path/gpu_leases`，不比對 bash 回報的路徑字串。
5. `requirements.lock` 由協調端在本輪合併後產生（指令見下）。

### P1

| 項目 | 修正 | 驗證 |
|---|---|---|
| style 排程完成判定 | `run_style_prompt_jobs.sh` 重寫（`7f48f44`）：各階段結束碼寫入 `runtime/logs/<工作>.rc`；新 CLI `evaluate_job_outputs`（原暫名 `check_job_outputs`）驗收防禦（逐影像一列＋防禦圖）、編輯（影像 × 指令鍵集合與 PNG）、讀數（strength × 影像 × 指令）；既有輸出須與 `<防禦目錄>/job.spec` 的設定相同才沿用，不同即中止；清單缺 `ref` 或風格不一致在啟動時中止；任一階段失敗記為失敗、不重派，結束碼 1，全部通過才印 `_DONE` | `style/tests/test_job_runner.py` 5 項；舊腳本 5 項皆失敗（失敗情境無限等待而逾時） |
| color 階段標記 | `immunization_color.stages`（`f6c3680`）：標記內容為摘要（各 CLI 自身 parser 解析後的設定，含 seed、步數、s_t、s_i、影像子集；淨化協定；各輸入檔 SHA-256）；摘要相符且輸出通過驗收才略過，否則重跑；分片合併寫同一標記；`evaluate_queue_job chain` 核對全部階段標記；合併的 `results.csv` 改原子寫入 | `color/tests/test_stage_markers.py` 6 項 |
| CSV 原子寫入 | `immunization_core.io.write_rows_atomic`（`a513b10`）：同目錄暫存檔、fsync、`os.replace`，例外時刪除暫存檔；`write_csv`、`write_sorted_csv` 與 baseline 四支自行截斷寫檔的 CLI 改用它（`e52fdfa`）。`run_flux_edits` 為附加寫入，不截斷既有列，維持 | `core/tests/test_io_atomic.py` 5 項，舊 writer 3 項失敗 |
| trials 升格 | `run_trial_lifecycle.sh`（原 `trial.sh`，`d8949f9`）：`promote` 需 `trials/<名稱>/PROMOTED` 列出已提交的目的檔（限 src、configs、results、scripts、tests、docs），紀錄目的檔、SHA-256、commit 於 `docs/TRIALS.md` 的「升格紀錄」 | `core/tests/test_gpu_scripts.py` 5 項 trials 測試；舊腳本 4 項失敗 |
| trials 遠端刪除 | 遠端根須為不含 `..` 的絕對路徑，參數以 `printf %q` 傳遞；遠端核對 `pyproject.toml` 專案名、`trials` 與目標都不是符號連結、解析後路徑、目標存在，刪除後確認不存在；遠端完成才刪本機 | 同上（身份不符、路徑不存在、相對路徑、符號連結各有案例） |
| style 參照相依 | 四組結果刪除後不再有缺參照的讀數；排程在啟動時要求 `ref` 工作 | 同 style 排程 |
| color 匯入出處 | `configs/conditions.yaml` 的 `color` 條目新增 `spec_source`、`solver_prompt`、`solver_prompt_source`（主表匯入條件必填）；`import_defense_artifacts --condition` 必填且須為 imported 條件，出處欄取自條目，封存 `immunise.py` 輸入模式移除（`e3109a3`）。`defense_color.csv` 8 列的三欄以 `correct_import_provenance.py` 更正，其餘欄逐值不變（`1547d50`） | `test_conditions.py`、`test_import_defense_artifacts.py` 新增 4 項 |

更正前的出處值：`spec_source` 為 `archive/anti-purification/scripts/paper_baseline.py 的 color 臂`；`solver_prompt_source` 為「無文字條件：三個項都不經過 text encoder（設定檔的 assert_no_instructions 擋下含指令的設定）」。color 專案 `results/defenses/color/results.csv` 等表中的同名欄是當時產生器寫出的原始紀錄，未改。

### P2

- trials 紀錄完整性與遠端殘留：見上表（`drop` 要求 ledger 已提交、五欄皆填、結論來源不指向 trial；`promote` 也刪遠端副本）。
- 命名 `r11`／`r13`：隨裁定 2 刪除。
- color README（`d6ec9fb`）：範例改經 `run_queue.sh` 或 `run_with_gpu_lease.sh`；註明 `evaluate_condition.sh`、`measure_condition_results.sh` 為已持有租約時的內部入口；`--workdir` 改 `--work-dir`；STATUS 的 `--out` 改 `--output-csv`。新增 `core/tests/test_documented_commands.py`：文件中 core 工具名稱之後的選項須為該工具實際接受的選項（舊 README 的 `--workdir` 會被抓出）。
- 共通規則衝突：隨裁定 3 解決。
- 文件路徑（`dbafcc6`）：`docs/_audit_*.md` 改指 `docs/reference/AUDIT_*.md`（抽查章節與行號與現行檔一致）；`_audit_dia_apa.md` 為較早版本，改指 `AUDIT_DIA.md` 並不再引用行號；`BASELINE_CANDIDATES.md` 改指封存位置並註明篩選時主表為六個條件。
- style 文件自足：DESIGN 寫入攻擊端協定（模型、步數、s_t、s_i、種子、指令、資料、分母、讀數定義）。
- 用語：style 結果結論（含「沒有關聯」「Codex 診斷」）隨結果一節移入 TRIALS 的數值紀錄；color DESIGN「撐得過淨化」改為以淨化後讀數界定；`run_flux_edits` 首段、color `pilot` 說明改寫。
- STATUS 分工：三份 STATUS 不再重複根規則；「沒有工作在跑」改為以遠端租約與 `runtime/` 狀態為查核依據。
- `requirements.lock`：仍未產生（需遠端執行環境）。

### 命名規範（使用者追加）

- 根 `CLAUDE.md`「命名」補成完整規範（`f4318bb`）：大小寫、拼法、程式入口動詞、結果與掃描的目錄、試驗、測試、縮寫；Python 私有模組可用單一底線開頭；語言與工具慣例名（`src`、`cli`、`io`、`env`、`__init__`）不在縮寫限制內。
- `core/tests/test_repository_naming.py`：掃描四個專案的版控檔名（不含 `vendor/`），檢查大小寫、美式拼法、流程與順序用語、測試檔名、入口動詞；附已知違規與合規樣本。對 `7caa13a` 的樹找出 55 個違規，現行為 0。縮寫規則未自動化。
- 改名（`ae898eb`；引用逐一更新，封存區不改，source manifest 只改 `destination`）：

| 原名 | 新名 |
|---|---|
| `baseline/.../cli/check_edit_completion.py` | `evaluate_edit_completion.py` |
| `color/.../cli/validate_queue_job.py` | `evaluate_queue_job.py` |
| `style/.../cli/check_job_outputs.py` | `evaluate_job_outputs.py` |
| `core/scripts/export_vendor.py` | `generate_vendor_snapshot.py` |
| `core/scripts/freeze_env.py` | `generate_requirements_lock.py` |
| `core/scripts/free_cards.sh` | `measure_free_gpus.sh` |
| `core/scripts/queue_worker.sh` | `run_queue_worker.sh` |
| `core/scripts/trial.sh` | `run_trial_lifecycle.sh` |
| `color/scripts/queue_job.sh`、`queue_validate.sh`、`queue_depends.sh` | `run_queue_job.sh`、`evaluate_queue_job.sh`、`generate_queue_dependencies.sh` |
| `core/tests/test_purify_new_ops.py`、`test_purify_cr.py`、`test_freeze_env.py` | `test_purifier_operators.py`、`test_purify_crop_resize_chain.py`、`test_generate_requirements_lock.py` |
| `baseline/data/targets/MIST.png` | `mist.png`（程式與訊息同步；文件中上游 repo 的原檔名 `MIST.png` 保留） |
| `color/data/color_lpips_ref.csv` | `color_lpips_reference.csv` |
| `baseline/results/ADDITIVE_TRANSFER.md` | `baseline/docs/ADDITIVE_TRANSFER.md` |

- 結果與產物樹對應（`bc47c2b`、`ef9fbab`）：`results/ultraedit/edits/<條件>.csv` → `results/ultraedit/edits_<條件>.csv`；`results/sweeps/<編輯器>/<名稱>.csv` → `results/sweeps/<編輯器>/<變因>/<名稱>.csv`；baseline `layout.py` 的 `FLUX_EDITS`、`ULTRAEDIT_EDITS`、`ALIGNED_DEFENSES` 改為 `artifacts/flux/edits`、`artifacts/ultraedit/edits`、`artifacts/aligned/defenses`。29 表 6,206 個路徑格以 `restructure_result_layout.py` 只改前綴並逐表驗證。
- CSV 欄名（`e368f8f`、`2c5fe30`）：以 `rename_csv_columns.py` 改寫 41 表表頭：`D_/P_/DT_lpips_*` → `disp_/predicted_disp_/residual_disp_lpips_*`、`D_csv` → `displacement_csv_lpips_full`、`siglip_pair_T`／`blocked_T` → `siglip_pair_residual`／`blocked_residual`、`deltaE00*` → `delta_e00*`；style 讀數程式的 `D_`、`C_`、`D_pair_` 欄改為 `disp_`、`edit_change_`、`disp_reference_`；`MetricSuite.full()` 的鍵改 `delta_e00`。`ADDITIVE_TRANSFER.md` 的 D、P、D_T 代號改為描述性名稱（`17c2cd6`）。

### 驗證

core 238 passed（21 deselected）、baseline 82、color 47、style 20；`generate_vendor_snapshot.py --check` 三個專案通過；`restructure_result_layout.py --check` 與 `rename_csv_columns.py --check` 皆為 0。獨立目錄驗證（`verify_standalone.py`，`e13af6d`）：core 222（21 deselected；獨立副本沒有其他專案的文件可掃描，參數化案例較少）、baseline 82、color 47、style 20 passed，import、`--help`、`bash -n` 與檔案雜湊皆通過；模擬 Windows 簽出的 2,193 個版控檔 SHA-256 與 blob 相同。

### 協調端要在遠端執行的指令

**0. 同步**：`cd ~/image-immunization && git pull --ff-only`（取得改名後的工具；`~/image-immunization.old`、`archive/` 內的舊腳本名不得再用於派工）。

**1. 刪除 style 四組影像產物**：

```bash
cd ~/image-immunization/style
for g in r11 r13 cls_p_noedit cls_p_snow; do
  rm -rf -- "artifacts/defenses/$g" "artifacts/edits/$g"
done
rm -f -- runtime/logs/r11_* runtime/logs/r13_* runtime/logs/cls_p_noedit_* runtime/logs/cls_p_snow_*
```

**2. baseline 產物樹與結果樹對應**（先確認沒有 baseline 工作持有租約）：

```bash
cd ~/image-immunization
python archive/migration/restructure_result_layout.py --remote-commands > /tmp/restructure_layout.sh
bash /tmp/restructure_layout.sh            # 搬移並重建 aligned/purified_edits/undefended 的相對連結
python archive/migration/restructure_result_layout.py --paths-only baseline/artifacts   # artifacts 內 CSV 的路徑欄
python archive/migration/restructure_result_layout.py --paths-only baseline/artifacts --check   # 應為 0
```

**3. 遠端 artifacts 內 CSV 的欄名**：

```bash
python archive/migration/rename_csv_columns.py --check baseline/artifacts color/artifacts style/artifacts
python archive/migration/rename_csv_columns.py baseline/artifacts color/artifacts style/artifacts
```

**4. color 階段標記**：既有的 `color/runtime/state/<條件>.<階段>.done` 是空檔，與新摘要不符，下次執行 chain 會重跑各階段。若確認既有輸出來自現行設定，可在執行前以
`cd color && source scripts/env.sh && "$PY" -m immunization_color.stages write <條件> <階段>` 逐一補寫（階段：`defense`、`edit_ip2p`、`purify`、`pedit_<淨化>_ip2p`；影像不滿 8 張的條件加 `--images …`）；輸出未通過驗收時該指令會拒絕寫入。

**5. 產生 `requirements.lock`**（`~/venvs/wacv` 沒有 pip，`uv` 須在 `PATH` 上）：

```bash
cd ~/image-immunization && command -v uv
for p in baseline color style; do
  (cd "$p" && export ENV_FILE=~/env.sh && source scripts/env.sh && "$PY" vendor/scripts/generate_requirements_lock.py) || break
done
(cd core && source ~/env.sh && "$PY" scripts/generate_requirements_lock.py)
for p in baseline color style; do (cd "$p" && export ENV_FILE=~/env.sh && source scripts/env.sh && "$PY" vendor/scripts/generate_requirements_lock.py --check); done
git add core/requirements.lock baseline/requirements.lock color/requirements.lock style/requirements.lock
git commit -m "Lock the remote execution environment for each project"
```

## 驗收修正後的遠端操作（協調端）

- 遠端同步至 `73eb706`；style 四組（`r11`、`r13`、`cls_p_noedit`、`cls_p_snow`）的 `artifacts/defenses`、`artifacts/edits` 與 `runtime/logs` 已刪除。
- `restructure_result_layout.py --remote-commands` 產生的搬移已執行，`aligned/purified_edits/undefended` 重建為 `../../purified_edits/undefended`；`--paths-only --check` 回傳 0（artifacts 內 CSV 路徑格 0 處需改）。
- `rename_csv_columns.py` 改寫 artifacts 內 36 份表，`--check` 回傳 0。
- `requirements.lock`：遠端 `~/venvs/wacv`（Python 3.11.15、torch 2.13.0+cu126、CUDA 12.6）以 `uv pip freeze` 產生，四份內容相同（106 個套件），三個專案 `--check` 回傳 0。
- color 既有階段標記未補寫（`immunization_color.stages write`），下次執行 chain 會依新摘要重跑。
- Windows 本機測試（`73eb706`）：baseline 82、color 47 passed；core 234 passed、4 failed，style 16 passed、4 failed。
  core 的 4 項為 `run_trial_lifecycle.sh` 測試：`TRIAL_REMOTE_ROOT` 只接受 `/` 開頭，Windows 暫存路徑為 `C:/…`（3 項），以及建立符號連結需要 Windows 權限（WinError 1314，1 項）。
  style 的 4 項為 `test_job_runner.py`，`run_style_prompt_jobs.sh` 在 120 秒逾時內未結束。遠端（Linux）未執行這兩組測試。
