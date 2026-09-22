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
**下表是模組裡各算子的預設設定，不是主表跑的那一組**；主表實跑的七道與
它們的強度見下一節。

| 算子 | 設定 |
|---|---|
| `identity` | — |
| `blur` | 高斯 σ=1.0 |
| `jpeg` | 品質 75 / 30 |
| `crop_resize` | 每邊裁 10%，bicubic 升回 |
| `jpeg_then_resize` | C&R 串接 |
| `noise` / `quantize` | — |
| `rotate` | 角度 `rotate_angle()`，雙線性、補零。`ROTATE_FIXED = True` 時取固定角度，非隨機 |
| `adverse_cleaner` | 導向濾波 |

## 主表實跑的七道算子

`results/retention.csv` 的 `purifier` 欄只有這七個值，強度由
`code/purify_run.py` 的 `PURIFIERS` 指定：

| `purifier` | 算子 | 強度 | 幾何類 |
|---|---|---|---|
| `blur1`、`blur2` | 高斯模糊 | σ = 1.0、2.0 | 否 |
| `jpeg80`、`jpeg50`、`jpeg30` | JPEG | 品質 80／50／30 | 否 |
| `crop_resize0.1` | 每邊裁 10%，升回 512² | 0.1 | 是 |
| `rotate15` | 繞中心旋轉 | **15°，固定** | 是 |

`identity` 不在這七個之內：保留率的分母是同一格的未淨化位移
（`disp_plain`，取自 `results/displacement.csv`），不是 `identity` 那一列。

### `rotate15` 的角度：三個來源寫的是三個值

| 來源 | 值 |
|---|---|
| FaceLock 原值 `ROTATE_DEGREES_FACELOCK`（`src/purify/ops.py`） | 10.0 |
| 本檔上一節的算子表，改正前的寫法 | 隨機 ±10° |
| **實跑**：`code/purify_run.py` 的 `PURIFIERS` 給 `strength=15.0`，且 `src/purify/ops.py` 的 `ROTATE_FIXED = True` | **固定 15.0** |

實跑值與 FaceLock 原值不同，且不是隨機而是固定角度，**引用 `rotate15` 的任何
讀數都要連這一點一起引用**。`src/purify/ops.py` 的函式名 `rotate_random` 在
`ROTATE_FIXED = True` 下已不描述它的行為；該檔在主線目錄，不在本目錄的範圍內。

幾何類的兩道（`crop_resize0.1`、`rotate15`）另有一個影響分區讀數的已知限制，
見下一節。

### 幾何類的分區讀數：遮罩曾經沒有跟著變換

`crop_resize0.1` 與 `rotate15` 改掉取景，淨化後的圖裡主體已不在原來的像素座標
上。`code/edit_retention.py` 原本把遮罩載入一次後對七道算子重用，於是這兩道的
分區是用**未變換的遮罩**切出來的。

| 範圍 | 狀態 |
|---|---|
| `results/retention.csv` 的 `disp_purified_subject`、`disp_purified_background` | 受影響，**1,536 列**（`crop_resize0.1` 768 ＋ `rotate15` 768） |
| 同檔的 `disp_purified`、`net_gain`、`retained` | 不受影響，三者都由全圖 LPIPS 算，不吃遮罩 |
| 非幾何的五道（`jpeg30/50/80`、`blur1/2`）的分區 | 不受影響，那些算子不動座標 |
| `results/metrics_retention_union.csv` | 不受影響，該檔只有全圖 FSIM |

**程式已修**：`edit_retention.purified_mask()` 把主體遮罩送過與影像同一個
`Purifier`（強度取自 `purify_run.PURIFIERS`，不另寫一組），插值後以 0.5 重新
二值化。順序是先翻極性再變換，使 `rotate` 補零的黑角落在背景側；反過來會把
黑角算成主體。實測遮罩歸屬的改變量：`crop_resize0.1` 為全幅的 9.3%–13.6%、
`rotate15` 為 11.1%–13.8%（`man_00`、`woman_02` 兩張）。

**已交付的 `results/retention.csv` 那 1,536 列仍是舊遮罩算出來的**，重算需要
LPIPS（`piq`）的一次完整重跑：

    python code/edit_retention.py --purified-root images/edit_purified         --displacement results/displacement.csv --out results/retention.csv

引用那兩欄在幾何類上的值時要連這一點一起引用。

**不進外部比較的算子**，理由各不相同，四個都要寫清楚：

| 算子 | 為什麼不進比較 |
|---|---|
| `grayscale` / `gray_world` | 色彩正規化。它們是**診斷用的上界**，不是攻擊者會做的事。留在紀錄裡是因為量到一件事：它們洗掉隨機對照卻洗不掉最佳化解（`rand_a` −46%，最佳化臂 gain −0.02 到 +0.05） |
| `diffpure` / `impress` | 使用者裁定移出本階段的淨化集 |
| `gridpure` / `fdpure` | 超參數論文未載，本專案指定；設定沒有出處就不當比較項 |

要看的比較是**淨增益**：

```
淨增益 = metrics(編輯(原圖), 編輯(防禦圖))
       − metrics(編輯(淨化(原圖)), 編輯(淨化(防禦圖)))
```

驅動 `scripts/phase_retention.py`，只讀已存的防禦圖，不重跑攻擊。

## 同一設定重跑不會得到同一張防禦圖

兩次獨立跑同一份設定（`knobs`／`caps`／`lr`／`free` 逐欄驗過相同，兩批之間沒有
改過程式），產出的防禦圖仍然明顯不同：逐像素平均絕對差 24.7/255，代理身分項
0.408 對 0.502，臉框位移 CVaR99 47.91 px 對 40.85 px。**兩張圖的自然度判斷也
相反**：一張看起來自然，另一張的鼻頰有一片拉扯。

來源不是亂數種子——載體初始化、探針噪聲與取樣鏈都走明確的
`torch.Generator(...).manual_seed(...)`。來源是 **GPU kernel 本身的非決定性**：
repo 裡沒有任何地方設 `torch.use_deterministic_algorithms` 或
`torch.backends.cudnn.deterministic`，而 bf16 下的 conv 與 attention 會依當下的
記憶體與負載選不同的演算法，歸約順序因此不同。

怎麼讀數字
────────────────────────────────────────────────────────────────────
- **大的差異仍然可信。** 0/25 對 21/25 不可能來自這個變異。
- **總計會大幅變動，跨批次的比較因此不成立。** 同一份設定三次獨立跑，過半穿透
  的格數量到 **13、13、1** 與 **5、1、4** 兩組。13、13、1 那一組的代理讀數也
  同向偏弱（0.493、0.497、0.653），所以不是評估端的雜訊，是解本身不同。
  **後果：任何「某設定是 N/25」的敘述都不可靠，跨批次的對照一律作廢。**
- **逐張的讀數不穩。** 同一張影像上代理身分項的兩個獨立估計差 0.09；五張的中位
  差 0.16，**大於目標函數那一批五個變體彼此的差距**（0.533–0.681），所以那種
  比較不能只看代理項。
- **單次跑出來的個位數格數之間沒有順序可言。**
- **「某個設定的產物自不自然」不是設定本身的性質。** 同一設定重跑可能落在自然度
  判斷的兩側，所以逐張看圖的結論要綁在**那一張影像檔**上，不是綁在變體名。

**同一批次內的比較仍然有效**，因為它們共用同一次的 GPU 條件。所以**每一批都要
自帶對照**，不可以借別批的。

上述證據來自已經從本樹移除的批次，數字保留在此作為現行做法的理由。要消掉這件事
就得設決定性旗標並接受變慢，或者同一設定重跑數次、報分布而不是報單點。兩者都
還沒做。
