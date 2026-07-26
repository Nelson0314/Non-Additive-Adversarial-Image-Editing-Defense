# stage1 摘要（實測值）

- smoke=False（smoke 為流程驗證，數值不具比較意義）
- 校準檢查點：pg_enc / pg_diff 之 **sdedit（img2img）** 列須與 SPEC §2.7 DAYN Table 1（引用值）之 Encoder / Diffusion 欄比對，接近方可確認設定正確。（論文核驗：Table 1 為 image editing（img2img）情境、20 seeds 平均；inpainting 於 DAYN 僅質性比較，勿以 inpaint 列比對。）

## 有 DAYN 錨點之條件：sdedit（img2img，主要條件）

校準比對限本表之 pg_enc / pg_diff 列。

| method | model | edit | psnr | ssim | vifp | fsim | lpips | clip |
|---|---|---|---|---|---|---|---|---|
| pg_enc | CompVis/stable-diffusion-v1-4 | sdedit | 14.5458 | 0.5007 | 0.1552 | 0.6960 | 0.4826 | 0.3019 |
| pg_diff | CompVis/stable-diffusion-v1-4 | sdedit | 15.3688 | 0.5159 | 0.1484 | 0.6981 | 0.4858 | 0.3029 |
| advdiff | CompVis/stable-diffusion-v1-4 | sdedit | 14.8493 | 0.4757 | 0.1132 | 0.6785 | 0.5315 | 0.3104 |
| apa | CompVis/stable-diffusion-v1-4 | sdedit | 17.1569 | 0.6225 | 0.2097 | 0.7693 | 0.3888 | 0.3000 |
| hybrid | CompVis/stable-diffusion-v1-4 | sdedit | 18.7793 | 0.7086 | 0.2608 | 0.8121 | 0.3166 | 0.3044 |
