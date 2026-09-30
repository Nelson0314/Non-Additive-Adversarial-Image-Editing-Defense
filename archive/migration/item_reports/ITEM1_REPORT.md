# 第 1 項可靠性修正驗收紀錄

程式修改與 CPU 驗證已完成；commit 尚未建立。沙箱將 `.git` 設為唯讀，`git add` 與 `git commit` 均因無法建立 `.git/index.lock` 而失敗。HEAD 維持 `8bcaae0f4a0131811e28a91d67edcb0112cdf4a9`，index 未變更；未 push，未執行遠端或 GPU 工作。

## 基準

修改前逐一核對 `archive/migration/pre_migration_manifest.json` 的 1,707 個檔案：Git blob、工作目錄 SHA-256 均一致。HEAD 相對該 manifest 僅新增 `archive/migration/pre_migration_manifest.json` 與 `archive/migration/snapshot_manifest.py`。原有未追蹤檔案與研究 PDF 未修改。

## 行為修正與預定提交

每列各有一份可獨立提交的 patch，順序及完整檔案清單見 `item1_patches/series.json`。所有 commit 欄位目前均為「未建立：Git 目錄唯讀」。文件另列一份 patch，與行為修正分開。

| Patch | 主要檔案 | 修正與英文 commit message |
|---|---|---|
| `readout_exit.patch` | `lab/scripts/readout.sh` | 保留各階段退出碼；失敗後停止且不印完成。`Propagate readout stage failures` |
| `queue_completion.patch` | `lab/scripts/queue_worker.sh`、`lab/code/validate_job.py` | 完成狀態須通過產物、列數、唯一鍵及必要欄位驗收；既有 `.done` 重新驗收；缺 `FID_ARMS` 拒絕派工；佇列未完成回傳失敗。`Validate queue outputs before marking jobs complete` |
| `editor_resume.patch` | `anti-purification/main_table/code/{edit_flux_preview,edit_ultraedit_full,resume_state}.py` | 續跑驗證完整協定摘要、輸入雜湊、唯一鍵及產物路徑；舊 schema、缺圖、設定不符及未登錄產物明確報錯，保留原始列；UltraEdit 原子更新 CSV。`Validate editor protocols and artifacts before resuming` |
| `color_completion.patch` | `anti-purification/main_table/code/{color_row_chain.sh,check_edit_complete.py}` | 依預期編輯格、協定欄位、影像驗收與雜湊 manifest 判定完成；部分輸出不得因目錄存在而略過。`Validate color edit completion before skipping stages` |
| `style_feasible.patch` | `lab/code/style_prompt_defence.py` | `feasible` 與 `selected_feasible` 採所選更新的實際限制判定；另記 `selection_policy`，保留選點政策及輸出行為。`Record feasibility independently from style selection policy` |
| `required_metrics.patch` | 兩份 `metrics_union.py`、`anti-purification/scripts/readout_panel.py` | 已宣告的必要指標初始化失敗直接使階段失敗，保留例外與 traceback。`Fail metric stages when required models cannot initialize` |
| `atomic_leases.patch` | `lab/scripts/{gpu_lease,run_on_card,queue_worker}.sh`、`color_row_chain.sh` | 共用 NFS `mkdir` 鎖涵蓋容量檢查、空卡檢查、取租約及釋放；token 防止釋放其他工作的租約。`Acquire and release GPU leases atomically across schedulers` |
| `global_capacity.patch` | `lab/scripts/{gpu_policy,gpu_lease,run_on_card,queue_worker,style_prompt_round}.sh`、`color_row_chain.sh`、`anti-purification/scripts/free_cards.sh` | 單一預設 6 張，計算所有主機、session 與工作的租約；明確參數或 `LAB_CAP` 覆寫授權值，CLI 優先；所有取卡入口使用共用容量。`Enforce a shared global GPU capacity with a six-card default` |
| `image_dimensions.patch` | `anti-purification/src/utils/io.py` | 明給尺寸時同時檢查寬高；插值設定未變。`Check both image dimensions before resizing` |
| `style_empty.patch` | `lab/code/style_prompt_readout.py` | 計算指標前驗證指定組別與參照格，空組別明確報錯且不覆寫結果。`Reject missing style readout groups before computing metrics` |
| `gpu_policy_docs.patch` | `HANDOFF.md`、`COLOUR_LINE.md`、`anti-purification/CLAUDE.md`、`anti-purification/main_table/STATUS.md`、`lab/HANDOFF.md` | 五卡敘述改為使用者逐次授權、未明確指定時全局預設 6 張，並記錄共用租約與覆寫介面。`Document the user-authorized global GPU capacity policy` |

## 驗證

- 主回歸：`python -m pytest lab/tests anti-purification/main_table/tests anti-purification/tests/test_io_size.py anti-purification/tests/test_metrics_union_failure.py anti-purification/tests/test_readout_panel.py -q -p no:cacheprovider`，115 passed。
- 新增完整 `run_on_card.sh` 入口測試 2 項：GPU 查詢與 CUDA 檢查均使用 stub，成功與失敗指令皆保留退出碼並釋放租約；2 passed。加上主回歸共 117 項不同測試通過。
- queue 完成狀態整合測試 2 項包含在主回歸內；工作程序退出碼為 0、產物驗收失敗時不得寫 `.done`。
- 所有修改的 Python 通過 AST 語法檢查；所有修改及新增 `.sh` 通過 `bash -n`，且不含 CR。
- `git diff --check` 通過。活動檔案未殘留原五卡上限敘述。
- 11 份 patch 已依序套用到 `.tmp/codex_audit/` 中的獨立基準副本，重建 37 個任務檔案；內容與工作目錄逐一一致。未修改 Git index。

## 限制與交接

- 實際 commit 是唯一尚未完成的交付步驟。`item1_patches/commit_item1.ps1` 會先核對 HEAD、空 index、37 個檔案及 patch 的雜湊，再以 `git apply --cached` 逐項建立 10 個行為 commit 與 1 個文件 commit；不重寫工作目錄、不納入原有未追蹤檔案、不 push。腳本尚未執行，須由可寫 `.git` 的執行環境操作。
- 未執行 GPU、SSH、遠端部署或 NFS 實機測試；租約競爭以本機獨立 Bash 程序和 stub 驗證。
- 沒有協定摘要的歷史 FLUX／UltraEdit CSV 不會被自動改寫或視為可續跑；須使用獨立的新 CSV 與影像輸出。部分 color 編輯輸出會停止並保留證據，不直接覆寫。
- `.capacity` 保存目前的明確全局授權，所有取卡入口重新讀取；未設定時採共用預設值，可用 `LAB_CAP=default` 清除明確授權。排程自身的較低限制不能提高全局容量。
- 本次未處理第 2 項以後的搬移、命名、CSV 改寫或研究數值更正。
