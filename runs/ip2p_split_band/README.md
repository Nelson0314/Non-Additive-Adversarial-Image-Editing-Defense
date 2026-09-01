# 加性項與乘性項分帶：軟性存活加權是唯一動得了模糊的東西

**尚未進入 `docs/RESULTS.md`**，等使用者裁定。兩張影像（盆栽人
`task_attr_mod_color_11699`、瑪利歐 `task_attr_mod_color_6205`），規模只夠當
篩選，不足以支撐結論。

## 這一批問什麼

本方法的擾動有兩半：乘性（相位旋轉＋幅度增益，改動量正比於原圖自己的振幅
`|S_b(ω)|`）與加性（`--spectral-floor`，不受該限制，佔可用預算的 **67.6%**，
見 `runs/ip2p_residual_signature/allowed_budget_gini.csv`）。先前兩半**共用**
同一個徑向帶通，所以壓低 `--r-max` 去遷就模糊時，乘性那一半的未淨化強度會
一起被削掉（實測掉四成以上）。

`--floor-r-min` / `--floor-r-max` / `--floor-survival` 讓加性那一半有自己的帶
與自己的定價。四個點：硬切上界 0.35 與 0.55、軟性的期望存活加權、以及兩者疊加。

## 未淨化的讀數

全部 `stop_reason = max_steps`、`stopped_at = 6000`——**沒有一個收斂**，固定
抽樣評估在上限處仍在下降。以下數字因此是下界。

| 條件 | 位移 | DISTS | PSNR | RMS | L∞ |
|---|---|---|---|---|---|
| `sb_surv`（軟性存活加權，不切邊） | 0.6587 | **0.2475** | **19.41** | **0.1075** | 0.968 |
| `sb_f55`（加性壓到 r ≤ 0.55） | 0.6732 | 0.3567 | 16.31 | 0.1531 | 0.987 |
| `sb_f35_surv` | 0.6708 | 0.4030 | 14.12 | 0.1968 | 0.963 |
| `sb_f35`（加性壓到 r ≤ 0.35） | 0.6652 | 0.4137 | 14.04 | 0.1988 | 0.988 |

**硬切帶是失真災難。** 這正是 `_build_floor_price` 的 docstring 事先寫下的
代價：總預算被固定住，同樣的 L1 總量壓進更少頻格會抬高每格定價，而 **L1 相等
不等於 L2 相等**，集中之後實際失真上升。DISTS 0.40、PSNR 14 不可用。
`block = 32` 的格點裡 `r ≤ 0.4` 只佔約 12%，這個比例解釋了幅度。

**三個硬切點撤掉。**

## 抗淨化：`sb_surv` 是目前的最佳點

扣空白地板的淨增益，佔可達範圍（可達 = 0.772 − 地板，`runs/readout_ceiling/`）。
地板沿用 `runs/ip2p_ig_loss/purify/floor_all.csv`——同兩張影像、同六個算子、
同 3 seed，而地板只與算子有關、與條件無關。

| 條件 | DISTS | identity | jpeg75 | jpeg30 | blur σ1 | blur σ2 | crop 10% |
|---|---|---|---|---|---|---|---|
| **`sb_surv`** | 0.2475 | **84.4%** | **84.7%** | **67.0%** | **52.4%** | **25.2%** | 61.5% |
| `ig_f08_eot`（前一個最佳） | 0.2289 | 82.2% | 84.6% | 61.0% | 43.9% | 21.0% | **62.0%** |

六欄贏五欄。多付 8% 失真，換到 **blur σ2 +20%、blur σ1 +19%**；歸一化成
每單位 DISTS 之後仍是 +11% 與 +10%。

**這是至今唯一動得了模糊 σ2 的東西。** 同一晚試過而沒有動它的有八條：換頻帶
（1.28×）、加強度（radius 8 飽和）、換損失、加性下限、寬 EOT、`--r-max` 硬切、
存活加權加在**相位**閘上、`--floor-gate complement`。

## 機制

`--floor-survival blur12` 把期望存活振幅

    w(ω) = (1 + Σ_{σ∈{1,2}} exp(−2π²σ²f²)) / 3          值域 (0, 1]

乘到**加性項的價目表**上。它是軟性傾斜不是切邊：含 identity 那一項，所以最低
只到 1/3，未淨化那一側不會被掏空。

高斯模糊在頻域乘的是實正數，**構造上不可能改相位**——它抽掉的是載體
（`phase_shift_amp.png` 那一組殘差熱圖：模糊 σ=1 的相位在 r < 0.6 幾乎完好，
而該處的能量存活率已掉到 0.05）。所以對模糊唯一能做的是別把預算放在它拿得走
的地方，而加性項是唯一能自由選擇放在哪裡的那一半。

## 未完成

`sb_surv` 停在上限而非收斂。`runs/ip2p_sbsurv_long/` 由它的 `__w.pt` 續跑
（`scripts/sbsurv_converge.sh`），上限再加 6000 步、patience 8 × 400 步。
**收工後要做的兩件事**：

1. `~/venvs/wacv/bin/python scripts/band_allocation_table.py --src
   runs/ip2p_sbsurv_long/purify`（先跑
   `bash scripts/purify_split_band.sh "<卡>" "sb_surv_long" nofloor`，
   並把 `runs/ip2p_ig_loss/purify/floor_all.csv` 複製過去）。
2. 讀 `results.csv` 的 `stop_reason`／`stopped_at`：停在 `early_stop` 才算收斂，
   仍是 `max_steps` 就照實記為未收斂。

**續跑這條路徑在今晚之前是壞的**（`--resume-weights` 對所有參數化都會當機，
而且「呼叫端先 reset 再載」的修法會靜默從零開始練），修正與測試見同日的
commit `Make --resume-weights actually resume`。
