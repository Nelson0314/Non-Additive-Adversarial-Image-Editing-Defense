# 粗網格 × EOT：兩個有效的機制能不能相加

派工 `scripts/immunise_field.py`，設定 `configs/immunise_field_coarse_eot.json`，
由 `scripts/queue_coarse_eot.sh` 排程，**跑完整四道淨化**。

## 為什麼把這兩個放在一起

`runs/field_grid/` 量到粗網格買自然度：6×6 網格（間距 85 px）在 96 px 預算下
產物自然，而 16×16 網格在同樣的位移量下五官被拉散。
`runs/field_eot/` 量到 EOT 買格數，但它的最強一檔 `eot_crop_16` 用的是
16×16 網格，`task_env_weather_114555` 的鼻嘴區是壞的。

兩者限制的東西不同——粗網格限制位移的**空間頻率**，EOT 改的是目標的**期望值**
——所以機制不衝突，相加有理由成立。

## 量到什麼（四個變體同批，彼此可比）

| 變體 | EOT | identity | jpeg | blur | crop | 產物 |
|---|---|---|---|---|---|---|
| `coarse_eot_96_none` | 無（批內對照） | 1 | 2 | 2 | 3 | `114555` 邊緣 |
| **`coarse_eot_48_crop`** | 裁切 | **6** | **6** | **7** | **7** | **看過的三張都自然** |
| `coarse_eot_96_both` | 裁切＋模糊，成本兩倍 | 7 | 7 | 7 | 7 | `114555` 壞 |
| `coarse_eot_96_crop` | 裁切 | 9 | 9 | 10 | **15** | `114555` 壞 |

**EOT 抽的算子族越窄，峰值越高且落在它訓練過的那一道。** `_crop` 在 crop 上是
15、其餘三道 9–10，criterion 中位 0.454 對 0.812。**族越寬，曲線越平但整體較低**：
`_both` 四道齊平 7，而且四道都不超過 `_crop`——多花一倍成本買不到東西。

**`coarse_eot_48_crop` 是目前唯一同時滿足「產物自然」與「四道淨化全部 ≥ 6」的
設定。** 較小的預算讓粗網格守得住自然度（`114555` 走 43.0 px 仍自然，而 96 px
那兩檔在 37–54 px 就壞了），EOT 把強度補回來——同預算無 EOT 的
`runs/field_coarse_margin/` 的 `coarse_margin_48` 只有 3/25。

## 這一批也量到重跑變異有多大

`coarse_eot_96_none` 與 `runs/field_coarse_margin/` 的 `coarse_margin_96`、
`runs/field_grid/` 的 `grid_06` 設定逐欄相同（`knobs`／`caps`／`lr`／`free` 全部
驗過）。三次獨立跑的過半穿透是 **13、13、1**，代理讀數同向偏弱
（0.493、0.497、0.653）。

**後果：任何「某設定是 N/25」的敘述都不可靠，跨批次的對照全部作廢。**
同一批次內的比較仍然有效——上表就是。所以**每一批都要自帶對照**。
詳見 `docs/EVALUATION.md`。
