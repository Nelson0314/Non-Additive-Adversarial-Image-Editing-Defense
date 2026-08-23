# 待裁定

**這一頁不是結論。** 每一筆都是量過但**尚未進入 [RESULTS.md](RESULTS.md) 或
[DECISIONS.md](DECISIONS.md)** 的東西，等使用者裁定之後才搬過去或刪掉。
證據路徑一律相對於 `runs/`。

## 一、證據完整，可以裁定

### 否決 `--floor-gate watson`
亮度遮蔽 × 對比遮蔽（Watson 1993）。等 DISTS 0.1447 下位移 0.6735 對 uniform
的 0.7012，掉 4%；DISTS 0.1188 下 0.6330 對 0.6756，掉 6%。原因可量：它把
66.7%／51.8% 的預算送進「乘法可達量高」的那一半，也就是乘法已經在動的地方。
**證據**：`ip2p_floor_gate`、`ip2p_residual_signature/allowed_budget_gini.csv`

### 加法下限佔了可用預算的 67.6%，並吃掉大部分空間選擇性
逐區塊的可用預算（**構造允許在哪裡花**，不是最佳化實際花在哪裡）：乘法那一半
單獨的 Gini 是 0.531，開了 `--spectral-floor 0.04` 之後合計掉到 0.163，而
DCT-Shield 由構造是 0。空間選擇性是本方法對它的主要構造差異。
**證據**：`ip2p_residual_signature/allowed_budget_gini.csv`

### 三條指紋差異在下限開著時仍然成立
逐區塊能量 Gini 差 4–6 倍、半 Nyquist 以上能量佔比 0.60 對 0.95、殘差沒有鎖在
8×8 格點上（1.27 對 1.78）。**分離最小的是 phase_share**，下限把它由 0.607
壓到 0.427。
**證據**：`ip2p_residual_signature/signature.csv`

### 人眼頭對頭打平，但失真多付 32%
純相位＋下限 r=π（DISTS 0.1286）與 DCT-Shield ε=1.8（DISTS 0.0976）都是
**11 擋下、1 邊緣、1 失敗**。等效果（位移 0.70）時前者要付 0.1279、後者 0.0930。
共同覆蓋的失真區間上效率是 6.08–6.27 對 7.05（86–89%）。
**證據**：`ip2p_axis_necessity/recognisability_phase_floor.csv`、
`ip2p_dct_band_extend/recognisability_dct_e18.csv`、
`ip2p_axis_necessity/equal_effect_anchor.csv`、`efficiency_overlap.csv`

### 兩軸都必要——但只在非加性設定下
純相位因 θ 封頂在 π，失真最高只到 DISTS 0.0950，而工作點在 0.1377：**它到不了
失真帶**，整條曲線擋下 0/13。純幅度到得了，但等失真下輸給兩者合用 10%。
**證據**：`ip2p_axis_necessity`

### 三個淨化算子壞掉的方式各不相同
模糊是**能量**問題（方向存活率每帶 0.88–1.00，半 Nyquist 以上能量剩 0.3%）；
JPEG 是**打散**問題（能量比 1 還大但方向由 0.68 掉到 0.05）；裁切縮放**兩者
都不是**——能量留 51–99%，對原網格的方向 0.000，對算子自己搬過的同一擾動卻是
0.995–0.996。擾動原封不動地通過了，只是被搬走。
**後果**：裁切的弱點不可能靠選頻帶解決（沒有一帶的方向存活率高於 0.02），
只有幾何上的不變性有機會。
**證據**：`ip2p_residual_signature/band_transfer.csv`

## 二、證據不完整，等批次

| 待裁定 | 目前指向 | 缺什麼 |
|---|---|---|
| 含加性主線要不要關掉 `--gain-ratio` | 等效果 0.65 與 0.70 兩個錨點都顯示純相位較便宜（0.0956 對 0.1051、0.1279 對 0.1435） | `ip2p_phase_floor`（把純相位＋下限用 floor 推到帶上緣） |
| 要不要採用 `--floor-gate complement` | 效果上幾乎免費（位移 −0.4%、PSNR +0.28 dB），但**讓可用預算更均勻**（0.163 → 0.078），在空間選擇性上是往 DCT-Shield 靠 | `complement_rank`（把分配推到 75/25） |
| 只有加性下限、相位與幅度都不動 | 從未跑過。程式已補（`floor_only`），先前被一道在加性下限存在**之前**寫的自由度檢查擋住 | `ip2p_floor_only` |
| `r_min` 要不要離開 0.12 | FND-042 已記「拉高一致更好」，保留 0.12 的理由（既有批次的基準）早已不成立 | `ip2p_band_lower_bound` |
| `--freq-weight-power` 0.25 → 0.35 | RESULTS 自己註明要先補等失真掃描 | `ip2p_pricing_power` |
| 要不要採用 `--theta-budget` | 新實作，出處 arXiv:2602.06577，處理 FND-038 | `ip2p_theta_budget` |
| 隨機化幾何 EOT 能不能救裁切 | band_transfer 指出這是唯一可能有效的方向 | `ip2p_eot_geometry` |
| `--quantile` 單獨的效果 | 從未量過 | `ip2p_quantile_sweep` |

## 三、必須先修正的一項

**本方法在 JPEG 上的優勢暫時不可引用。** `ip2p_purify_headtohead` 的 Y-only
點用的是 `--q-alg 0.95`，而 DCT-Shield 的抗 JPEG 設定是 §6.3 的 Y-only ＋
`Q_alg = 0.85`；它的保證是單向的（補充材料 D.4），0.95 由構造就擋不住品質 75
的壓縮。`RESULTS.md` 早已記過 SDEdit 線上 `dct_shield_y` 在 jpeg75 拿 +0.5185
而本方法 +0.1349，並明寫「頭對頭表不可只放 base 變體」。
**要跑**：`scripts/dct_antijpeg_configs.sh`。

## 四、工程

遠端有數個 `RESULTS.md` 引用為證據、卻從未入版控的 `runs/` 目錄
（`ip2p_pricing_strength`、`ip2p_overlap_sweep`、`ip2p_untested_knobs`、
`ip2p_band_calibration`、`ip2p_luma_only`、`ip2p_perimage_budget` 等）。依
`DECISIONS.md`「不可重現的量測結果一律保留」，它們應該補進來。
