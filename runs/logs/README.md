# 執行 log 保存

容器刪除前由 `/work/nelson0314/*.log` 拉回。保留理由：部分數字只存在於 log，沒有對應的 CSV。

**`e0d.log` / `e0dL.log` 是學習率掃描的唯一來源。** `scripts/e0d_lr_sweep.py` 在 `main()` 結尾才寫 CSV，而 site P 那一輪的 process 在 site L 跑到一半時被中止（為了套用修正後的損失函數重跑），因此 `runs/e0d/lr_sweep.csv` 從未產生。`docs/NIGHT_RUN_2026-07-29.md` §3.3 引用的五個 site P 學習率數字，來源即為 `e0d.log`。

| log | 對應執行 |
|---|---|
| `e0.log` | E0 成本掃描（僅 UNet checkpoint） |
| `e0v.log` | E0 成本掃描（加 VAE checkpoint） |
| `e0c2.log` | E0c 重建地板，t_max × k_inv 掃描 |
| `e0d.log` | E0d 學習率掃描，site P 五個值（**唯一來源**） |
| `e0dL.log` | E0d 學習率掃描，site L 三個值（修正損失後重跑） |
| `pilot.log` | 首次 real-SD 試跑，暴露 lr=0.05 發散 |
| `e2.log` | E2/E3 主網格 36 格 |
| `e2la.log` | site LA 對照 6 格 |
| `e2phi0.log` | site L 於 φ=0 的對照 6 格 |
| `e4.log` | E4 注入倍率掃描 8 格 |
| `e5.log` | E5 三點對比，site LA scale=10、5 張圖 |
| `e6P.log` | E6 訓練步數，site P 150 步 |
| `e6LA.log` | E6 訓練步數，site LA 100 步（**未完成**，容器刪除時中止於約 step 30） |
