# FLUX.2 [klein] 4B 蒸餾版跨編輯器讀數（Pexels 人像）

ip2p 流程求得的防禦圖與淨化後輸入直接送 FLUX.2 [klein] 4B 蒸餾版（`black-forest-labs/FLUX.2-klein-4B`，bf16，4 步、guidance 1.0，512²）編輯，
不以本編輯器重新求解。指令為 `configs/prompts/ultraedit_templates.json` 的 add 句型四句，主種子 20260812。

## 範圍

- 影像：`images.txt` 的 306 張；`crosseditor100/` 為其中前 100 張（`crosseditor100/images.txt`，與 `../vibe_distilled/crosseditor100/images.txt` 相同）。
- 條件：`undefended`（分母）、等失真 `photoguard_encoder`、`dct_shield`、`mist`，BlurGuard（`blurguard`），CAPGuard 2.5 倍（`color3d_scale2p5_pexels`）。
- 每條件 {淨化前、七道、VAE-Trans} × 4 句 = 36 格，每張 216 次編輯。

## 檔案

- `summary.csv`：306 張的逐條件彙整，欄同 `../vibe_distilled/summary.csv`；`per_purifier.csv`：逐淨化。
- `crosseditor100/summary.csv`、`crosseditor100/per_purifier.csv`：同一程式在 100 張上的彙整。

## 產生方式

- 編輯：`scripts/evaluate_pexels_nonsd_editor.sh klein <影像>`（`scripts/run_nonsd_edits.py`）。
- 100 張讀數：`EDITOR_NAME=klein scripts/evaluate_pexels_flux_512_readouts.sh`；306 張讀數：`scripts/run_pexels_readout_parts.sh`
  （base → 語意 4 片 → finish_gpu → finish_cpu）；彙整 `scripts/generate_pexels_flux_512_summary.py`。
- FID、KID、CLIP-IQA 的特徵以 GPU 計算（`measure_distribution_metrics.py --device cuda`，兩版皆然）；CPU 與 GPU 特徵的比對見 `../distribution_device/README.md`。
- 306 張版中屬於前 100 張的格，與 100 張版逐值相同：語意與身分 21,600 列、CLIP-S 21,600 列、ArcFace(E_u, E_d) 18,000 列、
  位移 2,000 列、保留率 16,000 列、SSIM／PSNR 18,000 列，差異格數皆為 0。
