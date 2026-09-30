# style：現況與接續指引

設計與結果見 `docs/DESIGN.md`，執行方式見 `README.md`。

## 現況

- 保留的輪：`r11`、`r13`、`cls_p_noedit`、`cls_p_snow`；其餘舊輪、預覽與其程式已刪除。
- 研究方向待使用者決定；沒有工作在跑。
- 讀數的未防禦分母原取自主表的未防禦編輯（遠端 `runs/edit_preflight/ip2p_si18`）；遠端版面切換（重整第 10 項）時需放入本專案的 `artifacts/undefended_edits/ip2p_si18`，或以 `--undefended` 指定。
- 各輪的工作清單（原遠端 `specs/style_prompt_cls_*.txt` 與各輪目錄內的 `jobs.spec`）未入版控。

## 規矩

- 訓練不得使用任何編輯指令（評估指令與自選替代指令皆不可），文字只可用空字串或類別詞；不可加性雜訊；只用主種子。
- 取卡經 `vendor/scripts/`；全局卡數由使用者逐次授權，未說明或說明不清時預設 6 張，所有 session、主機與排程合計。第三個參數為明確授權值，第四個參數只限制最佳化派工。
- `.sh` 必須是 LF；遠端腳本不可就地覆寫。
- 命名不含日期、流水號或順序詞；commit message 用英文。
- 不設判準：數據與圖並列為止。
