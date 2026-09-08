# 相位軸與幅度軸各自是否必要——等失真三方對照

**這些數字尚未進入 `docs/RESULTS.md`**，等使用者裁定。

13 張人眼確認服從的影像（清單見 `runs/obedience_audit/recognisability_verdict.csv`）、
`latent_norm` 損失、1000 步、`--quantile 0`、`--freq-weight jpeg_luma`
`--freq-weight-power 0.25`、`--hop 8`。驅動是 `scripts/axis_necessity.sh`
與 `scripts/gain_reach_extension.sh`。

三個條件：

| 條件 | 是什麼 |
|---|---|
| `phase` | 純相位，`gain_ratio = 0`，`theta_max` 封頂在 π |
| `gain_only` | 凍結相位、只學幅度，上界不封頂 |
| `phase_gain` | 兩者，即現行主線 |

兩組：`a_*` 是非加性（`--spectral-floor 0`）、`b_*` 是含加性下限
（`--spectral-floor 0.04`）。加法項的強度不隨半徑變，故半徑在兩組裡都只驅動
乘法那一半。

**比較一律在等失真上做**，用 `scripts/matched_distortion_table.py` 內插；
固定半徑下並排讀只是沿強度軸移動。擋下數是 `clip_sim < 0.8445` 的格數，
帶 ±2/13 的重跑雜訊（`docs/RESULTS.md`），差異小於該值時要看位移。
