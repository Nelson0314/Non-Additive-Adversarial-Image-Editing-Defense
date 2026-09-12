# 評測

## 指標

程式在 `src/metrics/suite.py`，欄位名與方向在 `src/metrics/standard.py`。

兩欄的意思：**保真**是防禦圖對原圖，**防禦**是編輯後對未防禦編輯。

| | 指標 | 保真 | 防禦 |
|---|---|---|---|
| 感知 | LPIPS | ↓ | ↑ |
| | DISTS | ↓ | ↑ |
| | FID | ↓ | ↑ |
| 訊號 | PSNR | ↑ | ↓ |
| | SSIM | ↑ | ↓ |
| | VIFp | ↑ | ↓ |
| 色彩 | CIEDE2000 | ↓ | ↑ |
| 語意 | CLIP / SigLIP | 照報，不作判準 | |
| 身分 | 人臉餘弦 | ↑ | ↓ |

## 淨化

算子在 `src/purify/ops.py`。`identity` 不可排除，它是保留率的分母。

| 算子 | 設定 |
|---|---|
| `identity` | — |
| `blur` | 高斯 σ=1.0 |
| `jpeg` | 品質 75 / 30 |
| `crop_resize` | 每邊裁 10%，bicubic 升回 |
| `jpeg_then_resize` | C&R 串接 |
| `noise` / `quantize` | — |
| `rotate` | 隨機 ±10°，雙線性、補零 |
| `adverse_cleaner` | 導向濾波 |
| `impress` | 1000 步 Adam |
| `gridpure` / `fdpure` | 超參數論文未載，本專案指定 |

要看的比較是**淨增益**：

```
淨增益 = metrics(編輯(原圖), 編輯(防禦圖))
       − metrics(編輯(淨化(原圖)), 編輯(淨化(防禦圖)))
```

驅動 `scripts/phase_retention.py`，只讀已存的防禦圖，不重跑攻擊。

## 同一設定重跑不會得到同一張防禦圖

`runs/field_coarse_margin/` 的 `coarse_margin_96` 與 `runs/field_grid/` 的
`grid_06` 的 `knobs`／`caps`／`lr` 逐欄相同，兩批之間沒有改過程式。產出的防禦圖
仍然明顯不同：

| 量 | `coarse_margin_96` | `grid_06` |
|---|---|---|
| 逐像素平均絕對差 | 24.7 / 255 | 同左 |
| 代理身分項 | 0.408 | 0.502 |
| 臉框位移 CVaR99 | 47.91 px | 40.85 px |

（以 `task_env_weather_114555` 為例。）**兩張圖的自然度判斷也相反**：一張看起來
自然，另一張的鼻頰有一片拉扯。

來源不是亂數種子——載體初始化是零場、探針噪聲與取樣鏈都走明確的
`torch.Generator(...).manual_seed(...)`。來源是 **GPU kernel 本身的非決定性**：
repo 裡沒有任何地方設 `torch.use_deterministic_algorithms` 或
`torch.backends.cudnn.deterministic`，而 bf16 下的 conv 與 attention 會依當下的
記憶體與負載選不同的演算法，歸約順序因此不同。

怎麼讀數字
────────────────────────────────────────────────────────────────────
- **大的差異仍然可信。** 0/25 對 21/25 不可能來自這個變異。
- **25 格的總計是穩的。** 同一設定的兩次獨立跑（`coarse_margin_96` 與
  `grid_06`），過半穿透都是 13/25。
- **逐張的讀數不穩。** 兩個獨立的估計：同一張影像上代理身分項差 0.09
  （`coarse_margin_96` 對 `grid_06`）；五張的中位差 0.16
  （`runs/field_ladder/` 的 `flow_08` 是 0.695，`runs/field_objective/` 的
  `obj_baseline` 同設定是 0.533）。**後者大於目標函數那一批五個變體彼此的差距**
  （0.533–0.681），所以那一批的比較不能只看代理項。
- **單次跑出來的 2/25 與 3/25 之間沒有順序可言。**
- **「某個設定的產物自不自然」不是設定本身的性質。** 同一設定重跑可能落在
  自然度判斷的兩側，所以逐張看圖的結論要綁在**那一張影像檔**上，不是綁在變體名。

要消掉它就得設決定性旗標並接受變慢，或者同一設定重跑數次、報分布而不是報單點。
兩者都還沒做。
