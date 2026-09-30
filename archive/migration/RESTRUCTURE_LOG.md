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
