# CLIP-IQA、FID 與 KID

程式：`scripts/measure_distribution_metrics.py`（遠端 CPU）。對象為等失真臂（`results/aligned/`）與 color 的 `color3d`
（雙角度色度旋轉版，影像取自 color trial，只讀）。

## 端別與集合

| 端別 | 淨化 | 受測集合 A | 參照集合 B |
|---|---|---|---|
| `distortion` | — | 防禦圖 | 原圖 |
| `displacement` | `none` | 防禦後編輯（ip2p，四句） | 未防禦編輯（同影像、同句） |
| `displacement` | 七道算子 | 淨化後防禦圖的編輯 | 同一淨化的未防禦編輯 |
| `displacement` | `vae_trans` | VAE-Trans 淨化後防禦圖的編輯 | VAE-Trans 淨化後原圖的編輯（color trial 產物） |

## 指標

- CLIP-IQA：pyiqa `clipiqa`，逐張；`clipiqa_a`、`clipiqa_b` 為 A、B 的平均。協定與 `measure_additional_metrics` 的 `aes_clipiqa` 相同。
- FID、KID：piq.FID、piq.KID（三次多項式核，n_subsets 100，subset_size 為較小集合的張數）；特徵為 piq 的 InceptionV3
  （pytorch-fid 權重 pt_inception-2015-12-05，pool3 2048 維，輸入縮放到 299）。`n_a`、`n_b` 為兩集合的張數。
- 小樣本下（portraits：失真端 8 對 8 張，位移端 32 對 32 格）FID 的共變異數估計秩不足，數值隨樣本數變動大。
- KID 在 portraits 全部為負值：A、B 逐張配對（同影像、同句），無偏估計的交叉項含配對項，加上 n ≤ 32，估計值偏負；數值不作距離解讀。
- MCIQA（arXiv:2609.14495）未量測：作者的 GitHub 儲存庫（ARBEZ-ZEBRA/MCIQA）只有 README、補充材料與 Kaggle 資料集連結，
  無推論程式、無模型權重、無授權條款（2026-10-27 查閱）。

## 檔案

| 檔案 | 內容 |
|---|---|
| `portraits_distribution.csv` | data/portraits 8 張；等失真 11 臂與 `color3d`，每條件 10 組（失真端 1、位移端淨化前 1、七道 7、VAE-Trans 1） |
| `portraits_clipiqa.csv` | 同上，逐張 CLIP-IQA |
