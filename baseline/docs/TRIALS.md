# 已刪除的暫時性嘗試

暫時性嘗試放在不入版控的 `trials/<名稱>/`（`bash vendor/scripts/trial.sh new <名稱>`）。成功時把需要的程式、設定與結果搬入 `src/`、`configs/`、`results/` 並提交，再執行 `trial.sh promote <名稱>`；失敗時先在下表寫一列，再執行 `trial.sh drop <名稱>`（同時刪除遠端副本）。

| 名稱 | 試了什麼 | 設定 | 關鍵數字 | 結論來源 |
|---|---|---|---|---|
