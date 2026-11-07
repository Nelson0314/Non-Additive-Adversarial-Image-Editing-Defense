# 分布指標的計算裝置：CPU 與 GPU 特徵的比對

`scripts/measure_distribution_metrics.py` 的 `--device cuda`（commit 3ad91e71）只改變 CLIP-IQA 與 InceptionV3 特徵的計算裝置：
同一權重（piq InceptionV3 pool3，pt_inception-2015-12-05）、同一預處理與影像集合，影像逐張送入，關閉 TF32；FID、KID 由特徵在 CPU 以 float64 計算。
Pexels 人像的主表、CAPGuard 2.5 倍、BlurGuard 與 VIBE-DistilledCFG 的 FID、KID、CLIP-IQA 均以 GPU 特徵重算並取代 CPU 版。

## 比對

`comparison_groups.csv`：逐組（條件 × 端 × 淨化）的 CPU 值、GPU 值、差（GPU − CPU）與相對差；`comparison_summary.csv`：
每條件的 none、seven（七道平均）、vae_trans、distortion 欄。

| 資料 | 組數 | FID 最大絕對差 | KID 最大絕對差 | CLIP-IQA 最大絕對差 | 排名改變的欄 |
|---|---|---|---|---|---|
| 主表（等失真 9 臂＋`color3d_pexels`＋未防禦，306 張，ip2p） | 100 | 0 | 1e-06 | 3e-05 | 無 |
| CAPGuard 2.5 倍（`color3d_scale2p5_pexels`，306 張，ip2p） | 10 | 0 | 1e-06 | 2e-05 | 無 |
| BlurGuard（306 張，ip2p） | 10 | 0 | 1e-06 | 2e-05 | 無 |
| VIBE-DistilledCFG（100 張，5 條件） | 45 | 0.001 | 1e-06 | 5e-05 | 無 |

FID 以小數 3 位、KID 以 6 位、CLIP-IQA 以 5 位記錄，表中差值以此精度計。排名為同一欄內各條件依 FID 或 KID 由小到大的順序。
重算後的彙整與 CPU 版逐格比對，改變的只有分布指標欄與 `device` 欄：主表 `kid_seven_ci95` 最大差 1e-05、`clipiqa_*` 最大差 3e-05；
VIBE-DistilledCFG 100 張 `fid_seven` 最大差 0.00014。

KID 的子集抽樣使用未設種子的 `torch.randperm`；n_a＝n_b 時 subset_size＝n，每個子集即全部特徵的重排，KID 與抽樣順序無關。

## 速度

VIBE-DistilledCFG 100 張的一組（400 對 400 張）：GPU（RTX 3090）含載入 63 s；CPU 8 執行緒於 basic-2 約 4 分、於高負載的 basic-1 約 10 分。
主表 137,088 張特徵於單卡 2.2 小時。

## 來源

- 重算與取代：遠端 `runtime/pexels_portraits/fid_cuda/`（`run_fid_cuda.sh`、`replace_with_cuda.sh`、`compare_fid_devices.py`、`diff_summaries.py`）。
- CPU 版的完整副本與 sha256：遠端 `~/backup/baseline/fid_cpu_2026-11-07/`（`SHA256SUMS_static.txt`、`SHA256SUMS_vibe100.txt`）。
