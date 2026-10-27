# 跨模型移植：以替代編輯器的 VAE 求解 encoder 類 baseline

## 用途

主表的 baseline 以 SD v1.4 求解，ip2p、SD2.1-base 與 SD-inpainting 共用同一個 VAE 編碼器；SD3（UltraEdit）與
FLUX.1-Kontext 的 VAE 不同。本目錄記錄損失只經過 VAE 編碼器的條件改以該編輯器自己的 VAE 求解、再由該編輯器編輯的結果。

## 條件與求解

| 條件 | 損失 | 與原方法的差異 |
|---|---|---|
| `photoguard_encoder` | ‖E(x+δ).mean‖₂ | 無 |
| `dct_shield`、`dct_shield_y` | ‖E(x′)‖₂（DCT 係數域） | 無 |
| `mist_textural` | ‖E(x+δ) − E(y)‖² 之和（y 為 Mist 的目標影像 `data/targets/mist.png`） | 只保留 textural 項（Mist 原始碼 `mist_v3.py` 的 mode 1）；論文主設定為 fused（mode 2，textural 減 1e4 倍 semantic 項），semantic 項需要 SD 的去噪器，換成其他編輯器的 VAE 後無法計算。eps、步數、步長與目標影像同 `mist` |

- 求解：`generate_defenses --vae-model <編輯器> --vae-resolution <邊長>`。SD3 為 `BleachNick/SD3_UltraEdit_freeform` 的 VAE、512；
  FLUX 為 `black-forest-labs/FLUX.1-Kontext-dev` 的 VAE、1024（輸入以 bicubic antialias 縮放後編碼）。各條件取原生預算，未對齊防禦圖 LPIPS。
- 編輯：SD3 為 UltraEdit（guidance 2.5、image guidance 1.5、`add` 句型，與 `results/ultraedit/` 同協定）；FLUX 為 FLUX.1-Kontext-dev
  （與 `results/flux/` 同協定，輸出 1024）。data/portraits 8 張、四句、主種子。
- 淨化：七道（`apply_purifiers`）與 VAE-Trans（`apply_vae_trans`，color 專案訓練的 encoder）。未防禦的未淨化與七道淨化後編輯
  沿用 `results/ultraedit/`、`results/flux/` 全表的產物；VAE-Trans 的未防禦編輯另行產生（`*/purified_edits/vae_trans/edits_undefended.csv`、
  `sd3/edits_undefended.csv`）。
- 執行腳本：遠端 `runtime/cross_model/run_cross_model_job.sh`、`run_cross_model_lane.sh`（不入版控）。

## 檔案

| 檔案 | 內容 |
|---|---|
| `<sd3｜flux>/additional_metrics/fidelity.csv` | 防禦圖對原圖（PNG、CPU） |
| `<sd3｜flux>/displacement.csv` | 未淨化的編輯結果 LPIPS（`disp_lpips_full`），32 格／條件，GPU |
| `<sd3｜flux>/retention.csv` | 七道與 VAE-Trans 後的編輯結果 LPIPS（`disp_purified`）與保留率，256 格／條件，GPU |
| `<sd3｜flux>/semantic_identity.csv` | 指令方向 CLIP-S、身分與影像語意相似度（未淨化、七道、VAE-Trans），GPU |
| `<sd3｜flux>/edits_<條件>.csv`、`purified_edits/` | 逐格編輯紀錄 |
| `<sd3｜flux>/vae_trans/fidelity_<條件>.csv` | VAE-Trans 淨化前後的 PSNR 與 LPIPS |

## 結果（8 張平均；七道為七個算子的平均）

| 編輯器 | 條件 | 防禦圖 LPIPS | 編輯結果 LPIPS 淨化前 | 七道後 | VAE-Trans 後 | 保留率（七道） | 保留率（VAE-Trans） |
|---|---|---|---|---|---|---|---|
| SD3 | `photoguard_encoder` | 0.478 | 0.530 | 0.322 | 0.066 | 0.609 | 0.125 |
| SD3 | `dct_shield` | 0.480 | 0.528 | 0.252 | 0.061 | 0.478 | 0.114 |
| SD3 | `dct_shield_y` | 0.578 | 0.603 | 0.336 | 0.091 | 0.558 | 0.151 |
| SD3 | `mist_textural` | 0.596 | 0.609 | 0.437 | 0.304 | 0.718 | 0.501 |
| FLUX | `photoguard_encoder` | 0.407 | 0.410 | 0.274 | 0.105 | 0.684 | 0.265 |
| FLUX | `dct_shield` | 0.444 | 0.423 | 0.258 | 0.121 | 0.620 | 0.294 |
| FLUX | `dct_shield_y` | 0.610 | 0.491 | 0.339 | 0.158 | 0.712 | 0.332 |
| FLUX | `mist_textural` | 0.519 | 0.472 | 0.411 | 0.322 | 0.879 | 0.691 |

保留率為逐格「淨化後 ÷ 淨化前」的平均。FLUX 的編輯結果 LPIPS 在 1024 解析度上計算，與 512 的讀數不並排比較解析度效應。
