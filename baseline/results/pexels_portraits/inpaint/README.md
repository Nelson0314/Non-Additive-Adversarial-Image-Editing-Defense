# SD inpainting 跨編輯器讀數（Pexels 人像）

ip2p 流程求得的防禦圖與淨化後輸入直接送 Stable Diffusion inpainting（`runwayml/stable-diffusion-inpainting`，SD 1.5 inpainting）編輯，
不以本編輯器重新求解：SD1.4（各 baseline 的求解目標）、SD1.5 inpainting 與 ip2p 的 VAE 逐張量相同（最大差 0）。
編輯設定：diffusers `StableDiffusionInpaintPipeline`、fp32、strength 1.0、50 步、guidance 7.5、主種子 20260812、512²；遮罩為
`data/pexels_portraits/masks/`（CLIPSeg `CIDAS/clipseg-rd64-refined` 主體分割，門檻 0.30、膨脹 4 px，白＝重繪＝主體的補集），輸出合成回
遮罩外的原輸入。指令為 `data/pexels_portraits/prompts.yaml` 的 `edits.inpaint` 四句（"A {man/woman} in a hospital"、"… in the gym"、
"… in a snowy street"、"… sitting next to a dog"，句型取自 DiffusionGuard arXiv 2410.05694）。

## 範圍

- 影像：`images.txt` 的 306 張。
- 條件：`undefended`（分母）、等失真 `photoguard_encoder`、`dia_pt`、`dct_shield`、`sifm`，BlurGuard（`blurguard`，306 張版），
  ImmuLUT（`color3d_scale2p5_pexels`）。
- 淨化：none、`jpeg50`、`blur2`、`vae_trans`；每條件 4 淨化 × 4 句 = 16 格，每張 112 次編輯。

## 檔案

- `summary.csv`：306 張的逐條件彙整，每種淨化各一欄、不做跨淨化平均：`disp_<淨化>`（編輯結果 LPIPS，piq VGG，E_u 與 E_d 經同一淨化）、
  `disp_{ssim,psnr}_<淨化>`、`clip_s_<淨化>`、`retained_<淨化>`、`siglip_<淨化>`、`clip_i_<淨化>`（E_d 對 E_u 的影像相似度）、
  `arcface_pair_<淨化>`（ArcFace(E_u, E_d)，`_missing` 為偵測不到臉的格數）、`fid_*`、`kid_*`、`defense_lpips`（ip2p 主表的防禦圖 LPIPS）；
  逐格平均，`_std` 為逐張值的影像間標準差，`_ci95`＝1.96·s／√n。

## 產生方式

- 編輯：`scripts/evaluate_pexels_inpaint_editor.sh <影像>`（`scripts/run_inpaint_edits.py`），每格約 7.7 秒（RTX 3090）。
- 讀數：`scripts/evaluate_pexels_inpaint_readouts.sh`（base、sem 2 片、finish_gpu、finish_cpu）；彙整 `scripts/generate_pexels_inpaint_summary.py`。
  逐格表在遠端 `runtime/pexels_portraits/inpaint/readouts/`，影像在 `artifacts/pexels_portraits/inpaint/`（含合成前的 `__raw.png`）。
