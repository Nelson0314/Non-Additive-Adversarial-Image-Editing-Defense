# 目標函數 pilot：五張人像 × 三類指令

四個臂共用同一支訓練迴圈、同一組攻擊種子、同一份量測，差別只有目標函數與起點。
派工 `scripts/objective_pilot.py`，設定 `configs/objective_pilot.json`。

## 設定

| 項目 | 值 |
|---|---|
| 影像 | `data/ncf_manifest.json` 的五張 |
| 指令類 | 衣物（改顏色）、頭部配件 `put a hat on the person`、背景 `change the background to a snowy street` |
| 衣物載體 | `NCFColorParam`，`radius 0.2`、`epsilon_lab 5`、`whiten`，支撐取 ATR 衣物並挖掉 MTCNN 臉框 |
| 整圖載體 | `ReColorAdvParam`，CIELUV 3D LUT `16×32×32`、`radius 0.06`、平滑正則權重 1.0 |
| 最佳化 | Adam 1000 步，步長為各載體半徑的 2%，每步投影 |
| 攻擊 | InstructPix2Pix，`steps=100 s_T=7.5 s_I=1.5`，保留種子 17001／27001／47001 |
| 精度 | fp32（fp16 下官方 pipeline 的 `edit()` 會在 `vae.encode` 拋 Half/float 不符） |

## 目錄

| 目錄 | 內容 |
|---|---|
| `gate/` | 未防禦的攻擊，45 格（15 × 3 種子），主體錨定讀數 |
| `random_edge/` | 載體推到可達集合角點後**不最佳化** |
| `latent_norm/` | `‖E(x')‖₂` |
| `latent_norm_edge/` | 同上，起點在可達集合邊界 |
| `cfg_shift/` | 完整三分支引導預測對影像條件的依賴 |
| `smoothness_sweep/` | 平滑正則權重掃描，單格 200 步 |
| `_superseded_no_smoothness/` | ReColorAdv 平滑正則未接上、`radius 0.5` 的批次 |

逐格 `*__trace.csv` 是每 50 步一次固定抽樣的目標值與外觀讀數。
影像（邊界圖、防禦圖、編輯輸出）依 repo 規則不入版控，由已記錄的參數與種子重跑。

## 量測

臂層 CSV 每列一個（影像, 指令類, 評估種子），欄位分三組：

- `gate_*`：載體推到約束邊界、**不最佳化**時的外觀讀數
- `final_*`：訓練完的防禦圖上，同一組讀數
- 其餘：編輯輸出上的主體錨定身分與臉數

外觀讀數含高頻殘差比（`highfreq_report`，RGB 與 Lab 分開）、PSNR、DISTS、
CIEDE2000、防禦圖上的主體身分、支撐外最大絕對差。
`gate_hf_below_reference` 與 `gate_id_above_reference` 是與參照值的相對位置，
不是通過與否，也不影響流程。

## 中位數

45 列／臂。

| 臂 | 防禦圖 PSNR | 高頻比 | ΔE00 | 編輯輸出的主體身分降幅 |
|---|---|---|---|---|
| `random_edge` | 26.49 | 1.170（0.72–1.44） | 6.31 | +0.0077 |
| `latent_norm` | 24.65 | 0.843（0.70–0.93） | 6.24 | +0.0005 |
| `latent_norm_edge` | 24.74 | 0.863（0.70–0.95） | 6.20 | +0.0039 |
| `cfg_shift` | 25.30 | 0.792（0.65–0.92） | 4.81 | +0.0009 |

`cfg_shift` 的訓練目標 1000 步降幅中位 7.2%。衣物那五格的
`final_subject_identity` 是 1.0——支撐挖掉臉框，臉的像素逐位元未動。

45 格裡有 4 格的未防禦攻擊本身把主體換掉（`subject_id_orig < 0.55`），
集中在 `task_env_weather_70149` 與 `task_obj_remove_284852` 的頭部配件指令。

## 機時

fp32、RTX 3090：單次攻擊編輯（100 步）21 秒；`latent_norm` 0.171 秒/步、
`cfg_shift` 0.778 秒/步。

## `_superseded_no_smoothness/` 是什麼

該批的整圖載體用 `radius 0.5` 且**未接上 ReColorAdv 原文的平滑正則**
（原文目標是對抗損失加平滑項）。量到的外觀是 `hf_ratio_rgb_total` 5.7–14.1、
PSNR 9.6–11.9。批次中止，資料保留。
