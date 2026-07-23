# TWCC_CHECKLIST.md — 上 TWCC 前的準備清單

## 下載清單（TWCC 端須預先下載）

- [ ] `256x256_diffusion_uncond.pt`（約 2 GB）— GrIDPure 淨化用之 pixel-space 無條件
      guided diffusion（ImageNet 256×256）。
      來源：`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`
      （GrIDPure README 指定，沿用 DiffPure）
- [ ] `CompVis/stable-diffusion-v1-4`（HuggingFace）— 保護生成與評測模型
- [ ] `stabilityai/stable-diffusion-2-base`（HuggingFace）— 評測模型
- [ ] DAYN 測試集（向作者索取；見 data/README.md）
      - 備援方案（論文核驗發現）：DAYN §4.3 載明其測試集為「以 diffusion model
        生成之 150 張影像、3 類物件」；若索取未果，可依 supplementary 設定
        以 SD V1.4 自行生成同規格資料集（結果仍標記非原始資料）

## 環境

- [ ] conda env（參考本機 `wacv`：Python 3.11；TWCC 踩坑紀錄見 memory：
      `conda tos accept`、`PIP_USER=0`、`PYTHONNOUSERSITE=1`）
- [ ] 執行 `python -m src.utils.device` 確認 CUDA 偵測正常並記錄裝置資訊至 NOTES.md
- [ ] 將 configs 中模型名稱由 tiny 測試模型換回真實 SD（僅改 config，不改程式碼）

## T2 校準（PhotoGuard vs DAYN Table 1）

- [ ] 依 SPEC §8 第 6–8 項掃描參數組合（epsilon_scale × target_latent，必要時加 norm=l2），
      找出與 DAYN Table 1 Encoder/Diffusion 欄對齊之設定

## 首次執行驗證（preflight 產出，詳見 PREFLIGHT.md 第 [3] 節）

- [ ] **假設 1 驗證（L_ref 基準）**：5 張影像各跑 PhotoGuard encoder 與 diffusion attack，
      比較兩者 LPIPS(protected, original) 平均。差距 ≤10% → 沿用 encoder 為 L_ref；
      >10% → 回報並重新決定（stage0 已內建此檢核，`--skip-pg-diff` 勿用於正式執行）
- [ ] **假設 3（遮罩）**：目前 inpaint 用 placeholder 中央方形遮罩
      （config edit.inpaint_mask），結果與真實資料不可直接比較；
      DAYN 資料集（含遮罩）到手後，所有 inpaint 實驗**須重跑**
- [ ] **DDIM inversion 重建品質**（「錨定原圖」核心前提，tiny 模型無法驗證）：
      真實 SD 上 5 張影像 inversion（10 步與 50 步）後直接去噪，量 PSNR/LPIPS(重建, 原圖)。
      預期 PSNR > 25、LPIPS < 0.1；不符 → 非加性「語意一致」前提受損，
      須改用部分 inversion（GC 式）或提高步數
- [ ] **scheduler 混用影響**（SPEC §8 第 7 項）：同 5 張影像，DDIM inversion 後分別以
      手動 DDIM（匹配）與 PNDM 採樣重建，比較重建品質；差距顯著 → 維持匹配 DDIM 並記錄，
      官方混用組合列消融
- [ ] **cross-attention 擷取**：真實 SD v1.4/v2.0 上執行 capture_cross_attention，
      確認擷取層數 >0、attention map 解析度合理（v2.0 之 attn processor 結構可能不同）；
      失敗 → 檢查 diffusers 版本之 processor 類名並更新 _CaptureAttnProcessor
- [ ] **GPU 記憶體峰值**：各方法單張 protect 記錄 peak_memory_mb()（腳本已內建輸出欄位），
      對照本地 CPU 相對量（pg_diff 最重）；OOM → 見 PREFLIGHT.md 決策樹降級路徑
- [ ] **fp16 可用性**：編輯（無梯度）以 fp16 跑一張並與 fp32 比對指標差異（<1% 可用）；
      保護（含梯度）維持 fp32，除非記憶體不足才試 fp16 並檢查梯度 NaN
- [ ] **編輯 pipeline 之 eta=1**：SPEC §2.2 eta=1 僅對 DDIM 類 scheduler 有意義；
      diffusers img2img 預設 PNDM 會忽略 eta。確認評測編輯所用 scheduler，
      若 T2 校準對不上 DAYN，改 DDIMScheduler（eta=1）為第一個排查項
