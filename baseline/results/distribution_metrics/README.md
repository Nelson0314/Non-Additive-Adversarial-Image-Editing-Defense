# CLIP-IQA、FID 與 KID

程式：`scripts/measure_distribution_metrics.py`（遠端 CPU）。對象為等失真臂（`results/aligned/`）與 color 的 `color3d`
（雙角度色度旋轉版，影像取自 color trial，只讀）；資料組為 data/portraits 8 張與 Pexels 人像（見「Pexels 人像」一節）。

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
| `pexels_distribution.csv` | Pexels 人像；Pexels 等失真 8 臂與 `color3d_pexels`（列中 `condition=color3d`），每條件 9 組（失真端 1、位移端淨化前 1、七道 7） |
| `pexels_clipiqa.csv` | 同上，逐張 CLIP-IQA（只含受測集合 A 的影像） |

## Pexels 人像

- 程式與 portraits 相同（`--dataset pexels`）；特徵以 `--shard` 平行抽取後合併計算，合併時缺特徵的影像當場補算。
  合併計算於 2026-10-29 14:00 開始、14:27 完成，遠端 CPU。
- 只取 A、B 兩側檔案都存在的格。`n_images` 為該組的影像數，`n_a`、`n_b` 為格數（位移端每張 4 句）。
- 無 VAE-Trans 組：Pexels 尚無 VAE-Trans 淨化後的未防禦編輯（分母）。
- `color3d`：306 張全數納入，9 組皆為 `n_images` 306（失真端 306 對 306 張，位移端 1,224 對 1,224 格）。
  影像取自 color trial `artifacts_pexels/`（`defense_shards/color3d_pexels/`、`edits_by_image/color3d_pexels/`），
  未防禦編輯與淨化分母為同目錄的 `undefended_edits/ip2p_si18/`、`purified_edits/undefended/`。
- 等失真 8 臂：影像取自 `artifacts/pexels_portraits/aligned/`（第一批與 `sets/<影像組>/`），合併計算時仍由佇列推進，
  各列只含當時已完成的影像。`n_images` 依條件與淨化不同：`dct_shield`、`dct_shield_y`、`dia_pt`、`dia_r`、`sifm` 全部 9 組為 147；
  `mist` 147–166；`dayn` 50–147；`danp` 42–50。同一條件內 `n_images` 不同的列，影像集合不同。
- KID 為負值的組：81 組中 43 組，全部屬於等失真臂（`danp` 8、`dct_shield_y` 7、`dct_shield` 6、`mist` 6、`dayn` 5、`dia_r` 4、`sifm` 4、
  `dia_pt` 3）；`color3d` 9 組皆為正。A、B 同樣逐張配對（同影像、同句），配對項的成因見「指標」一節的 KID 說明。
