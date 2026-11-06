# 論文結構表

對象為 `docs/paper/`（CVPR 2027，正文上限 8 頁，含圖表；參考文獻與 LLM Usage Statement 不計頁）。方法名 CAPGuard（`\capg`），論文版本為 2.5 倍角度上界（`color3d_scale2p5_pexels`）；2.5 倍讀數到齊前，表與正文的 CAPGuard 數字為 1.0 倍並以紅字標示。待改清單見 `docs/PAPER_HANDOVER.md` 第 5 節。

編譯狀態：共 11 頁，正文含 Conclusion 至第 8 頁，參考文獻自第 8 頁起，末頁為暫代數字的紅字註解。圖表：Fig. 1 teaser、Fig. 2 方法總覽、Fig. 3 失真—位移、Fig. 4 淨化分析；Table 1 主表（含 UltraEdit、FLUX 區塊）、Table 2 消融（`tab:ablation`）。

| 節 | 內容 | 現況 | 狀態 |
|---|---|---|---|
| 標題 | CAPGuard: Color-Based Anti-Purification Image Immunization against Diffusion-Based Editing | 定稿 | 完成 |
| Fig. 1 teaser | man_60、helmet；防禦圖列與 blur σ=2 淨化後編輯列；6 法含 BlurGuard，CAPGuard 2.5 倍 | 已放入正文 | 完成 |
| Abstract | 背景、痛點、CAPGuard 全名、淨化後位移倍數 | 倍數 1.34 to 3.23，下端依 BlurGuard 100 張，紅字 | 待 BlurGuard 306 |
| 1 Introduction | 背景、加性擾動、淨化研究、CAPGuard、自然度參數化、比較協定；貢獻 3 點 | 貢獻第二條倍數同摘要 | 待 BlurGuard 306 |
| 2 Related Work | 免疫（含同期 SPIN、SRIM、Copying as Defense）、淨化、色彩對抗樣本、美學評估；Fig. 2 方法總覽 | 已寫 | 完成 |
| 3 Method | 威脅模型、3D LUT、平滑色度旋轉、目標、最佳化 | 上界極角 < 25°、色相 < 7.5°，求解每張中位 9.5 分鐘（231 張）；Fig. 2 以 man_60 的 2.5 倍 LUT 重繪 | 完成 |
| Fig. 3、Table 1 | 失真—位移雙面板；主表 | CAPGuard 為 2.5 倍 306 張；BlurGuard（100 張）與跨編輯器區塊為紅字 | 待 BlurGuard 306、跨編輯器 2.5 倍 |
| 4.1 Setup | 資料、攻擊（ip2p、UltraEdit、FLUX 512 設定）、淨化、十個加性方法、讀數 | Pexels 選取規則與授權 `\TODO` | 部分 |
| 4.2 Robustness to purification | 主結果、跨淨化區間、VAE-Trans、美學、FID | CAPGuard 為 2.5 倍；BlurGuard 紅字 | 待 BlurGuard 306 |
| 4.3 Distortion and displacement | Fig. 3 | 擬合與偏離值不含 BlurGuard，紅字 | 待 BlurGuard 306 |
| 4.4 What purification removes | Fig. 4（a 分組直方圖、b–d 強度掃描 8 張） | 正文存續比為 2.5 倍；Fig. 4 的 CAPGuard 為 1.0 倍、b–d 為 8 張，紅字 | 待讀數 |
| 4.5 Editors with different encoders | Table 1 下段 | UltraEdit 50、FLUX 34 張紅字 | 待讀數 |
| 4.6 Ablations | Table 2：A–E，4 欄 | A 部分、D 暫代、其餘未跑 | 待讀數 |
| 5 Limitations | 限制、負面社會影響與緩解 | 已寫；配件計數 8 張紅字 | 部分 |
| 6 Conclusion | 方法、主要結果、限制 | ip2p 數字為 2.5 倍；BlurGuard 與跨編輯器紅字 | 待 BlurGuard 306、跨編輯器 |
| LLM Usage Statement | 工具、受影響部分、使用程度 | 已寫（Claude Code） | 待使用者審閱 |
| 補充材料 | 協定細節、防禦圖 PSNR／SSIM／NIMA／NIQE、逐道淨化表 | 未編入 PDF；DISTS、色彩對齊後失真為 `\TODO` | 部分 |
| References | 無 URL；executor 核實 | — | 完成 |
