# 已刪除的暫時性嘗試

暫時性嘗試放在不入版控的 `trials/<名稱>/`（`bash vendor/scripts/trial.sh new <名稱>`）。採用時把需要的程式、設定與結果搬入 `src/`、`configs/`、`results/` 等目錄並提交，在 `trials/<名稱>/PROMOTED` 逐行列出目的檔，再執行 `trial.sh promote <名稱>`：目的檔須已提交，升格紀錄（目的檔、SHA-256、commit）寫入本檔末的「升格紀錄」。不採用時先在下表寫一列並提交（五欄皆須填寫，未量測寫「未量測」；結論來源不可指向 trial 本身），再執行 `trial.sh drop <名稱>`。兩者都經 `TRIAL_REMOTE`、`TRIAL_REMOTE_ROOT` 一併刪除遠端副本；只處理本機時明確給 `--local-only`。

| 名稱 | 試了什麼 | 設定 | 關鍵數字 | 結論來源 |
|---|---|---|---|---|
