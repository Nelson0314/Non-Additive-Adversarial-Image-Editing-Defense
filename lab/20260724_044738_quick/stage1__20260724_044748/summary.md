# stage1 摘要（實測值）

- smoke=False（smoke 為流程驗證，數值不具比較意義）
- 校準檢查點：pg_enc / pg_diff 之 **sdedit（img2img）** 列須與 SPEC §2.7 DAYN Table 1（引用值）之 Encoder / Diffusion 欄比對，接近方可確認設定正確。（論文核驗：Table 1 為 image editing（img2img）情境、20 seeds 平均；inpainting 於 DAYN 僅質性比較，勿以 inpaint 列比對。）
- **placeholder 資料集**：結果與真實資料集不可直接比較。

## 有 DAYN 錨點之條件：sdedit（img2img，主要條件）

校準比對限本表之 pg_enc / pg_diff 列。

| method | model | edit | psnr | ssim | vifp | fsim | lpips | clip |
|---|---|---|---|---|---|---|---|---|
| pg_enc | CompVis/stable-diffusion-v1-4 | sdedit | 11.3684 | 0.2864 | 0.1056 | 0.5312 | 0.8942 | 0.3108 |
| advdiff | CompVis/stable-diffusion-v1-4 | sdedit | 12.3207 | 0.4878 | 0.1694 | 0.6939 | 0.7993 | 0.2923 |
| apa | CompVis/stable-diffusion-v1-4 | sdedit | 15.9471 | 0.6530 | 0.4320 | 0.8226 | 0.4502 | 0.2574 |
| hybrid | CompVis/stable-diffusion-v1-4 | sdedit | 13.9946 | 0.5037 | 0.2924 | 0.7159 | 0.5916 | 0.2881 |
