# TWCC_CHECKLIST.md — 上 TWCC 前的準備清單

## 下載清單（TWCC 端須預先下載）

- [ ] `256x256_diffusion_uncond.pt`（約 2 GB）— GrIDPure 淨化用之 pixel-space 無條件
      guided diffusion（ImageNet 256×256）。
      來源：`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`
      （GrIDPure README 指定，沿用 DiffPure）
- [ ] `CompVis/stable-diffusion-v1-4`（HuggingFace）— 保護生成與評測模型
- [ ] `stabilityai/stable-diffusion-2-base`（HuggingFace）— 評測模型
- [ ] DAYN 測試集（向作者索取；見 data/README.md）

## 環境

- [ ] conda env（參考本機 `wacv`：Python 3.11；TWCC 踩坑紀錄見 memory：
      `conda tos accept`、`PIP_USER=0`、`PYTHONNOUSERSITE=1`）
- [ ] 執行 `python -m src.utils.device` 確認 CUDA 偵測正常並記錄裝置資訊至 NOTES.md
- [ ] 將 configs 中模型名稱由 tiny 測試模型換回真實 SD（僅改 config，不改程式碼）

## T2 校準（PhotoGuard vs DAYN Table 1）

- [ ] 依 SPEC §8 第 6–8 項掃描參數組合（epsilon_scale × target_latent，必要時加 norm=l2），
      找出與 DAYN Table 1 Encoder/Diffusion 欄對齊之設定
