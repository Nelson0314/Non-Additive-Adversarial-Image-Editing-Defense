# VIBE-DistilledCFG 跨編輯器讀數（Pexels 人像）

ip2p 流程求得的防禦圖與淨化後輸入直接送 VIBE-DistilledCFG（`iitolstykh/VIBE-Image-Edit-DistilledCFG`，20 步，512²）編輯，
不以本編輯器重新求解。指令為 `configs/prompts/ultraedit_templates.json` 的 add 句型四句，主種子 20260812。

## 範圍

- 影像：`images.txt` 的 306 張；`crosseditor100/` 為其中前 100 張（`crosseditor100/images.txt`，選取方式見 `docs/HANDOVER.md`「資料」）。
- 條件：`undefended`（分母）、等失真 `photoguard_encoder`、`dct_shield`、`mist`，BlurGuard（`blurguard`），CAPGuard 2.5 倍（`color3d_scale2p5_pexels`）。
- 每條件 {淨化前、七道、VAE-Trans} × 4 句 = 36 格，每張 216 次編輯。

## 檔案

- `summary.csv`：306 張的逐條件彙整，欄同 `flux_512/summary.csv`（`disp_{none,seven,vae_trans}`、`disp_{ssim,psnr}_*`、`retained_*`、
  `siglip_*`、`clip_i_*`、`clip_s_*`、`arcface_*`、`fid_*`、`kid_*`、`defense_lpips`，附 `_std`、`_ci95`）；`per_purifier.csv`：逐淨化。
- `crosseditor100/summary.csv`、`crosseditor100/per_purifier.csv`：同一程式在 100 張上的彙整。

## 產生方式

- 編輯：`scripts/evaluate_pexels_nonsd_editor.sh vibe_distilled <影像>`（`scripts/run_nonsd_edits.py`）。
- 讀數：`EDITOR_NAME=vibe_distilled scripts/evaluate_pexels_flux_512_readouts.sh`；彙整 `scripts/generate_pexels_flux_512_summary.py`。
  306 張的語意與身分讀數分 4 個 GPU 工作計算後合併（遠端 `runtime/pexels_portraits/vibe_distilled/readouts_parts.sh`）。
- 306 張版中屬於前 100 張的格，與 100 張版逐值相同：語意與身分 21,600 列、CLIP-S 21,600 列、ArcFace(E_u, E_d) 18,000 列、
  位移 2,000 列、保留率 16,000 列、SSIM／PSNR 18,000 列，差異格數皆為 0。
- FID、KID、CLIP-IQA 的特徵以 GPU 計算（`measure_distribution_metrics.py --device cuda`）；與 CPU 特徵的比對見 `../distribution_device/README.md`。
  100 張版另有 CPU 讀數，兩者 FID 最大絕對差 0.001。
