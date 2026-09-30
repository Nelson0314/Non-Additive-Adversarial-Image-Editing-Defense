# 求解端的批間變異

同一組影像（`data/portraits` 八張）、同一組原生設定，**只換 PGD 的種子**，
量求解端重跑之間的差距。只產防禦圖，不跑編輯評測。

## 為什麼量這個

`../../../HANDOFF.md` §5 記著「IP2P 的編輯端是可重現的，求解端不是」：編輯端
同一顆種子連跑三次逐位元相同，而求解端的 GPU 非決定性仍在。但**求解端的重跑帶
從來沒有量過**——沒有這個帶寬，兩個條件的失真差多少才算差得出來就說不清楚。

Mist 的隨機性有明確來源：兩項損失每次前傳都含 VAE 後驗抽樣，加上擴散時間步與
雜訊兩重隨機，原始碼自己 `seed_everything(23)`（`src/baselines/mist.py` docstring）。
其餘三個條件的隨機性來源不同，一併量。

## 跑了什麼

四個便宜條件（`mist`、`dia_r`、`dct_shield`、`dct_shield_y`）× 八張人像 ×
種子 1 與 2。種子 0 的同一組在 `runs/defence_portraits/<條件>/`，
**那一批就是這個比較的第三顆種子**，不必重跑。

    <條件>/seed_<n>/{image}__{條件}__def.png
    <條件>/seed_<n>/results_seed<n>.csv

設定、prompt 的處置與 CSV 欄位同 `runs/defence_portraits/README.md`；
`--seed` 之外沒有任何一欄不同。

## 位置

排在第五張卡佇列的最後一段（`runs/defence_portraits/_launch/worker_cheap_queue.sh`
階段五），前面四個階段的產物不依賴它。砍掉這一段不影響人像與動物那兩批。
