"""Stable Diffusion 封裝（STRUCTURE.md §1）。

職責：
- 載入 SD pipeline（模型名稱由 config 提供，不得硬編碼；
  本地開發用 hf-internal-testing/tiny-stable-diffusion-pipe，
  TWCC 上換成 CompVis/stable-diffusion-v1-4 等真實模型）
- 暴露 vae / unet / text_encoder 與 cross-attention map（DAYN 式 (2) 所需）
- 提供可微分之 img2img（PhotoGuard diffusion attack、SPEC §3.3 所需）
- 裝置一律經 src/utils/device.py 取得，禁用 .cuda()
"""
