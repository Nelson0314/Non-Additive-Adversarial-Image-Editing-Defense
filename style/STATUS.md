# style：現況與接續指引

設計與結果見 `docs/DESIGN.md`，執行方式見 `README.md`。

## 現況

- 保留的輪：`r11`、`r13`、`cls_p_noedit`、`cls_p_snow`；其餘舊輪、預覽與其程式已刪除。
- 研究方向待使用者決定；沒有工作在跑。
- 讀數的未防禦分母為 baseline 未防禦編輯的複本，位於本專案的 `artifacts/undefended_edits/ip2p_si18`；另以 `--undefended-edits-dir` 指定。
- 各輪的工作清單在 `configs/jobs/<輪名>.spec`，與 `results/defenses/<輪名>/` 一一對應（`tests/test_job_specs.py`）。
- 本專案在遠端位於 `~/image-immunization/style`；`requirements.lock` 未入庫，由遠端執行環境以 `vendor/scripts/freeze_env.py` 產生。

## 規矩

- 訓練不得使用任何編輯指令（評估指令與自選替代指令皆不可），文字只可用空字串或類別詞；不可加性雜訊；只用主種子。
- 取卡經 `vendor/scripts/`，租約目錄 `~/gpu_leases/`；全局卡數由使用者逐次授權，未說明或說明不清時預設 6 張，所有 session、主機與排程合計。第三個參數為明確授權值，第四個參數只限制最佳化派工。
- 遠端腳本不可就地覆寫。
- 共通規則（命名、書面用語、不設判準、commit、LF）見根目錄 `CLAUDE.md`。
