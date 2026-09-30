# style：現況與接續指引

設計見 `docs/DESIGN.md`，執行方式見 `README.md`。

## 現況

- 本專案沒有保存結果：只保留方法程式（`src/`、`scripts/`）、設定（`configs/styles.yaml`）、資料與文件。
  已刪除的實驗（`xattn_chaos_cool_grade`、`paper_setting_xattn_chaos_p_snow`、`paper_classifier_p_noedit`、
  `paper_classifier_p_snow` 等）的設定、關鍵數字與可取回數值 CSV 的 commit 記錄於 `docs/TRIALS.md`。
- 研究方向由使用者決定。
- 讀數的未防禦分母為 baseline 未防禦編輯的複本，位於本專案的 `artifacts/undefended_edits/ip2p_si18`；另以 `--undefended-edits-dir` 指定。
- 本專案在遠端位於 `~/image-immunization/style`；`requirements.lock` 未入庫，由遠端執行環境以 `vendor/scripts/generate_requirements_lock.py` 產生。

## 規矩

- 訓練不得使用任何編輯指令（評估指令與自選替代指令皆不可），文字只可用空字串或類別詞；不可加性雜訊；只用主種子。
- 讀數以同一組實驗內名為 `ref` 的工作（lr 0、1 更新的未最佳化風格圖）為風格參照；清單檔須包含該工作且所有工作與它同一風格，否則排程在啟動時中止。
- 共通規則見根目錄 `CLAUDE.md`。
