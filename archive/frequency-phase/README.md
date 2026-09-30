# 頻域與相位重參數化：已結束

這條線的方法是把擾動放進**影像的頻域相位與殘差結構**，而不是放進像素上的加性雜訊。
**方向已結束**，本目錄只保留它獨有的程式。

## 為什麼在這裡

這條線原本整批歸檔在 `non-additive-frequency/`（見 commit `3d770e1`：
「The frequency and phase reparameterisation direction is concluded. Its code,
docs and numerical records move to non-additive-frequency/ intact」），
但後續的抗淨化主線又長在同一個目錄裡，兩條線混在一起。現在主線改名為
`anti-purification/`，這條線的程式抽出來放在這裡，兩者平行。

歸檔當時就已經丟掉可重生的產物（報告頁、抓回來的防禦圖與淨化圖、彙整的 JSON），
因為產生它們的參數與種子都有紀錄；不可重現的 CSV 留下。之後的整理又移走了它的
文件與設定檔。**所以本目錄沒有 `runs/`、沒有 `configs/`、沒有 `docs/`——
那些內容在 commit `3d770e1` 之前的 git 歷史裡。**

## 內容

| 路徑 | 是什麼 |
|---|---|
| `src/residual/texture_rephase.py` | `PhaseResidual`——這條線的主載體，相位重定相 |
| `src/residual/spectral_split.py` | 頻帶切分 |
| `src/residual/latent_inject.py` | 去噪側殘差注入 |
| `src/residual/lowrank.py`、`lora_weights.py` | 低秩與 LoRA 形式的殘差模塊 |
| `src/residual/perceptual_weight.py` | 知覺加權 |
| `src/residual/base.py`、`composite.py` | 殘差模塊的統一介面與組合 |
| `src/purify/freq_grid.py` | `gridpure_real`／`fdpure_real` 兩個頻域淨化器 |
| `scripts/phase_retention.py` | 相位留存率讀數 |
| `scripts/freq_baselines_run.py` | 頻域 baseline 的跑批入口 |

`src/residual/base.py` 的介面判準：模塊以「能力」而非「型別」對外表達——像素側殘差
實作 `pixel_residual`，去噪側殘差實作 `eps_hook`，兩者的預設實作皆回傳 `None`
表示不提供。優化器只呼叫 `module.parameters()`，不需要知道自己在優化外積向量還是
LoRA 矩陣。

## 與主線的兩個牽連

1. **`anti-purification/src/defense/param_pgd.py`**（相位殘差的 PGD 驅動）留在主線，
   因為主線的 `scripts/dct_shield_run.py` 要用它的 `fit_to_budget`，而
   `dct_shield`／`dct_shield_y` 是主表十二個條件裡的兩個。它對
   `PhaseResidual` 的 import **已改成延遲載入**，所以主線不會因為這條線被搬走而
   整個模組載不起來；只有真的去實例化 `PhaseParam` 時才需要本目錄在 `sys.path` 上。
2. **`anti-purification/src/purify/ops.py`** 在函式內延遲載入 `freq_grid`，用於
   `gridpure`／`fdpure`。這兩個算子**不在主表的七道淨化裡**
   （identity、crop_resize 0.1、jpeg 30/50/80、blur 1/2、rotate 15），
   所以主線不會走到那條路徑。要用的話把本目錄加進 `sys.path`。

## 要跑這條線的話

本目錄沒有自己的 `environment.yml`、測試與設定檔。程式依賴主線的
`src/utils/`、`src/metrics/` 等模組，所以實際執行時需要把
`anti-purification/` 與本目錄一起放進 `sys.path`。
