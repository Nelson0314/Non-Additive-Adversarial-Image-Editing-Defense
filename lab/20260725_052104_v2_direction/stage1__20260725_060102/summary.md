# stage1 摘要（實測值）

- smoke=False（smoke 為流程驗證，數值不具比較意義）
- 校準檢查點：pg_enc / pg_diff 之 **sdedit（img2img）** 列須與 SPEC §2.7 DAYN Table 1（引用值）之 Encoder / Diffusion 欄比對，接近方可確認設定正確。（論文核驗：Table 1 為 image editing（img2img）情境、20 seeds 平均；inpainting 於 DAYN 僅質性比較，勿以 inpaint 列比對。）

## 有 DAYN 錨點之條件：sdedit（img2img，主要條件）

校準比對限本表之 pg_enc / pg_diff 列。

| method | model | edit | psnr | ssim | vifp | fsim | lpips | clip |
|---|---|---|---|---|---|---|---|---|
| pg_enc | CompVis/stable-diffusion-v1-4 | sdedit | 14.6199 | 0.5047 | 0.1586 | 0.6972 | 0.4749 | 0.3014 |
| advdiff | CompVis/stable-diffusion-v1-4 | sdedit | 14.9156 | 0.4935 | 0.1243 | 0.6852 | 0.5103 | 0.3115 |
| apa | CompVis/stable-diffusion-v1-4 | sdedit | 18.1278 | 0.6871 | 0.2470 | 0.7970 | 0.3386 | 0.2998 |
| hybrid | CompVis/stable-diffusion-v1-4 | sdedit | 16.8567 | 0.6148 | 0.1866 | 0.7573 | 0.3995 | 0.3032 |
