# 顏色族的天花板：把合法的顏色位移推到自然的極限

派工 `scripts/color_ceiling.py`，設定 `configs/color_ceiling.json`。
自變數是**臂 × 顏色位移（ΔE00）**，其餘一切固定。三個防禦臂**不訓練**——
問的是「這個顏色載體能走到的地方，攻擊還成不成立」。

`runs/objective_pilot/` 已經量到最佳化在這個載體上幾乎不值錢（整圖載體上投影
後盒角的位移 0.3626，最佳化後 0.3098；兩者都在可達集合內部，見
`carrier_objectives.box_corner_init`），所以先把可達集合本身量清楚。

## 設定

| 項目 | 值 |
|---|---|
| 影像 | `data/ncf_manifest.json` 的五張 |
| 指令類 | 衣物改色、頭部配件 `put a hat on the person`、背景 `change the background to a snowy street` |
| 載體分工 | 衣物指令用衣物支撐，頭部與背景指令用整圖 |
| 配色 | ADE20K 真實色彩分布庫，`ade093_apparel_00`（`region_palette` 的第二塊用 `_04`） |
| ΔE00 目標 | 6.3（`objective_pilot` 的現行工作點）、20、30 |
| 色相旋轉角 | 90 度（`chroma_isometric` 專用） |
| 權重圖低通 | `blur_sigma` 24（`region_palette` 專用） |
| 攻擊 | InstructPix2Pix，`steps=100 s_T=7.5 s_I=1.5`，種子 17001／27001／47001 |
| 精度 | fp32（fp16 下官方 pipeline 的 `edit()` 會在 `vae.encode` 拋 Half/float 不符） |

## 四個臂

| 臂 | 構造 |
|---|---|
| `chroma_bounded` | `ChromaAffineParam(max_gain=1)`，色度映射的奇異值只設上界 |
| `chroma_isometric` | 同上但投影到 O(2) 並指定色相旋轉角，奇異值恰為 1 |
| `region_palette` | `RegionPaletteParam`，衣物與其補集各一組配色，權重圖重度低通 |
| `undefended` | 未防禦的攻擊，每個（影像, 指令類, 種子）都有 |

`undefended` 不可省：種子層的攻擊失敗會讓整批讀數變成雜訊，逐種子要有對照。

## 兩件量測上的事，會影響怎麼讀這張表

**一、等失真錨點取支撐加權的 ΔE00，不是全圖平均。** 全圖平均會被支撐面積稀釋：
衣物只佔一兩成像素，整塊完全換色，全圖平均也只有 4.5–5.2，於是 6 以上的目標
全部不可達，而整圖濾鏡在同一個數字上輕鬆到 24——兩者對不起來。支撐加權問的是
「載體真正作用的地方顏色走了多遠」。全圖那個數字仍照報，欄名 `final_deltaE00`，
錨點那個是 `support_deltaE00`。

**二、`protected_max_abs` 才是「受保護像素有沒有被動到」。**
`final_outside_support_max_abs`（沿用 `gate_row`）用 `(1 - w)` 加權，羽化帶上
`0 < w < 1` 就會非零——那是羽化，不是違規。`protected_max_abs` 只看 `w` 恰為 0
的像素，量到的是 0.0（189219 個像素，`task_env_weather_70149` 的衣物支撐）。

## 量測欄位

欄名的合約在 `scripts/color_ceiling.py` 的 `COLUMNS`，沿用
`runs/objective_pilot/` 的字彙，兩批才並列得起來。四組：

- 幅度：`delta_e_target`、`amplitude`、`delta_e_reached`、`support_deltaE00`
- 主讀數：`edit_lpips`（位移）
- 主體錨定身分：`subject_id_orig`／`subject_id_def`／`subject_id_drop`、
  三張圖的臉數、`subject_box_iou_*`
- 外觀：`final_psnr`、`final_dists`、`final_deltaE00`、
  `final_hf_rgb_total`、Lab 三通道、`protected_max_abs`

身分讀數走 `subject_identity_row()`，**不是** `embed()`：後者取畫面上面積最大的
臉，編輯輸出多長一張臉時量到的是別人。

不可達的目標會解到同一個幅度（1.0），渲染出逐位元相同的防禦圖；那些格子的
攻擊按幅度快取、不重跑，列仍逐目標各一列，`delta_e_reached` 標明可不可達。

## 高頻約束不是構造保證

等距只在 Lab 的 (a,b) 平面上成立。`hf_ratio_rgb_total` 隨影像內容與旋轉角變動：
`task_env_weather_70149` 上 0–135 度是 0.966–0.980、180 度 1.062，而均勻隨機圖
30 度就 1.079、90 度峰值 1.338。所以逐列照量，不設門檻、不擋工作。
`tests/test_chroma_rotation.py` 把這件事釘住。

## 冒煙測試量到的（CPU，`task_env_weather_70149`，不含攻擊）

支撐加權的 ΔE00，兩格：

| 格 | 臂 | dE 6.3 | dE 20 | dE 30 |
|---|---|---|---|---|
| 衣物 | `chroma_bounded` | a 0.385 | 18.33（未達標） | 同左 |
| 衣物 | `chroma_isometric` | a 0.367 | 19.78（未達標） | 同左 |
| 衣物 | `region_palette` | a 0.397 | 17.38（未達標） | 同左 |
| 配件 | `chroma_bounded` | a 0.533 | 13.34（未達標） | 同左 |
| 配件 | `chroma_isometric` | a 0.341 | **20.00** | 20.13（未達標） |
| 配件 | `region_palette` | a 0.487 | 12.34（未達標） | 同左 |

最大幅度上的外觀（配件那三格）：`chroma_bounded` hf 0.867／PSNR 20.88、
`chroma_isometric` hf 1.035／PSNR 15.97、`region_palette` hf 0.885／PSNR 21.42。
