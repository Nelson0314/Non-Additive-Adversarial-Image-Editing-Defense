# FLUX.1-Kontext，512×512 協定（Pexels 人像）

資料版本 `flux_512`：ip2p 上求得的防禦圖直接送 FLUX.1-Kontext-dev 編輯，生成與條件圖皆為 512×512。
產生程式 `scripts/evaluate_pexels_flux_512.sh`（每張影像一次載入，`run_flux_edit_jobs`），佇列工作 `flux_512_<影像>`。

## 範圍

- 影像：color trial `pexels_subsets/flux.txt` 的 50 張。
- 條件：`undefended`（分母）、等失真 `photoguard_encoder`、`dct_shield`、`dct_shield_y`、`mist`、`color3d_pexels`。
- 每條件 {淨化前、VAE-Trans、七道} × 4 句 = 36 格，每張 216 格。
- 讀數（位移、保留率、SigLIP、CLIP-S、CLIP-I、ArcFace、編輯端 SSIM／PSNR、FID）由 baseline 量測，彙整表放在本目錄。

## 讀數

程式與 UltraEdit 50 張（`scripts/evaluate_pexels_ultraedit_readouts.sh`）相同，位移與保留率改用 FLUX 版面的
`measure_flux_displacement`、`measure_flux_retention`，以 `--resolution 512` 讀圖（其餘指標原本即以 512² 讀圖）。

| 步驟 | 程式 | 輸出（遠端 `runtime/pexels_portraits/flux_512/readouts/`） |
|---|---|---|
| 逐格表 | `scripts/generate_pexels_flux_512_cell_tables.py` | `edits_<條件>.csv`（每條件 1,800 格）、`plain/`、`pairs.csv` |
| 全部讀數 | `scripts/evaluate_pexels_flux_512_readouts.sh`（佇列工作 `flux_512_readouts`，GPU 1 張；50 張完成前結束碼 4） | `displacement.csv`、`retention.csv`、`semantic_identity.csv`、`clip_score.csv`、`arcface_pair.csv`、`distribution/`、`edit_fidelity.csv` |
| 彙整 | `scripts/generate_pexels_flux_512_summary.py`（欄與區間算法同 `generate_pexels_main_table_summary.py`） | `summary.csv`、`per_purifier.csv`，複製到本目錄 |

防禦圖 LPIPS 取 ip2p 主表的 `fidelity.csv`（同一批防禦圖，只取這 50 張）。

## 與 1024 版（`results/flux/` 與遠端 `artifacts/pexels_portraits/transfer/flux/`）的差異

| 項目 | 1024 版 | 512 版 |
|---|---|---|
| 生成尺寸 | `max_area=1024²`，管線把 `height=width=512` 改為 1024×1024 | `max_area=512²`，512×512 |
| 條件圖 | `_auto_resize=True`，放大到最接近的 Kontext 偏好解析度（約 1MP） | `_auto_resize=False`，維持 512×512 |
| 協定摘要 | 不含 `generation_size` | 含 `generation_size: 512`，續跑紀錄與 1024 版分開 |

其餘設定相同：guidance 3.5、28 步、transformer 與 T5 為 4-bit NF4、輸入以 512² 讀入、`prompts.yaml` 的 ip2p 四句、種子 20260812。
兩版的輸出解析度不同，讀數不可混合平均。
