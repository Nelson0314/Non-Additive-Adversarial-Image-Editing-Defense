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
