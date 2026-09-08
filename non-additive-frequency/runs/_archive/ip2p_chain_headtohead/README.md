# 服從率驗收 → 相位 θ=1.30 → DCT ε=1.0 → 抗淨化，七段串接的一次跑批

**與 `ip2p_mainline` 無關。** 原名 `ip2p_mainline_partial`，那個名字看起來像
主線的半成品，實際上是更早的一條七段串接批次（見 `chain.log`），資料集與
工作點都不同（150 列、`--edit-steps 100`）。改名以免誤讀。

`chain.log` 逐段記了起訖時間；`check_*` 是每一段的驗收輸出；`dct_*.csv` 與
其餘 CSV 是各段的量測結果。這些數字沒有進入 `RESULTS.md`，保留的理由是
量測結果不可重現（`docs/DECISIONS.md` 的資料保全條）。
