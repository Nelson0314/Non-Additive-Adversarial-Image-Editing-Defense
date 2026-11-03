# 論文結構表

對象為 `docs/paper/`（CVPR 2027，正文上限 8 頁，含圖表；參考文獻與 LLM Usage Statement 不計頁）。「Ours」指 `color3d`（雙角度色度旋轉：以黑點為原點、灰軸為極軸，極角 < 10°、方位角 < 3°，截斷前；損失只有目標）。現有頁數為 Tectonic 編譯結果的估計，雙欄版面下以欄位置換算，誤差約 ±0.2 頁。每次修改 tex 後同步本表。

編譯狀態：共 9 頁，正文與 Conclusion 至第 8 頁，參考文獻自第 8 頁起。正文圖表：Fig. 1 方法總覽、Fig. 2 失真—位移、Fig. 3 淨化分析；Table 1 主表、Table 2 美學分數、Table 3 跨編輯器、Table 4 消融。

資料範圍（executor 定案）：主表、存續比、強度掃描用 Pexels 306 張；跨編輯器 UltraEdit 306 張、FLUX 約 50 張固定子集；消融的隨機對照與單角 6° 用 306 張，逐項懲罰與單一常數上界約 100 張。數字由 baseline／color 陸續送來（約 11/12–13 到齊），到齊時改 caption 的張數。

| 節 | 小節 | 應含內容 | 現有內容 | 狀態 | 頁數預算（現有） |
|---|---|---|---|---|---|
| 標題 | teaser | 各防禦淨化前後的編輯 | 暫時移出（`main.tex` 中 `\input{fig/teaser}` 為註解）；`fig/teaser.pdf` 由 `docs/paper_tools/generate_teaser_figure.py` 產生 | 移除 | 0（0） |
| Abstract | — | 背景、加性擾動的痛點、filter-like 色彩映射、淨化後保留的編輯位移 | 定稿，結果句一個 `\NUM`；不含高頻主張 | 部分 | 0.3（0.3） |
| 1 Introduction | 背景、加性擾動、淨化研究、本方法、自然度參數化、比較協定 | — | 已寫；一個 `\NUM` | 部分 | 1.0（1.0） |
| | 貢獻 | 3 點：方法 → 4.2；淨化後倍數 → 4.2；共同協定下的分析 → 4.3、4.4 | 已寫；倍數為 `\NUM` | 部分 | |
| 2 Related Work | 免疫、淨化、色彩對抗樣本、美學與品質評估 | 段末收回限制或寫選擇理由 | 已寫 | 完成 | 0.7（0.8） |
| | Fig. 1 方法總覽 | (a) 參數化、(b) 色彩旋轉、(c) 最佳化 | `fig/method_overview.pdf`（TikZ，`docs/paper_tools/method_overview/`）；(b) 右為 woman_00 color3d LUT 的實際位移；同色像素 (228,85)、(423,378) → 同一輸出 | 完成 | |
| 3 Method | 3.1–3.5 | 威脅模型、3D LUT、平滑色度旋轉（式 1–5）、目標（式 6）、最佳化 | 已寫 | 完成 | 1.3（1.1） |
| | Fig. 2、Table 1、Table 2 | 失真—位移雙面板圖；主表；美學分數表 | Fig. 2 為 306 張；Table 1 為 306 張，防禦圖端 PSNR、SSIM 與 VAE-Trans、CLIP-S、CLIP-I、FID 欄為 `\TBD`；Table 2（tab:aesthetics）為 306 張 | 部分 | |
| 4 Experiments | 4.1 Setup | 資料、攻擊、淨化、baseline、讀數 | 已寫；Pexels 選取規則與授權為 `\TODO` | 部分 | 3.2（2.9） |
| | 4.2 Robustness to purification | 主結果敘述 | 淨化前、七道、最弱一道已填；VAE-Trans 兩處與美學比較為 `\NUM` | 部分 | |
| | 4.3 Distortion and displacement before purification | Fig. 2a | 已寫（306 張；8 張的掃描與 DISTS／PieAPP 結論另註） | 完成 | |
| | 4.4 What purification removes | Fig. 3（a 存續比 306 張；b–d 強度掃描 8 張） | 已寫；VAE-Trans 存續比為 `\NUM` | 部分 | |
| | 4.5 Editors with different encoders | Table 3（兩編輯器上下排） | 8 張已填；CLIP-S、CLIP-I、FID 欄為 `\TBD` | 部分 | |
| | 4.6 Ablations | Table 4 | 8 張已填；待換定案張數 | 部分 | |
| 5 Limitations | — | 位移與編輯失敗的差異、未評估色彩反推攻擊、未做受測者實驗、求解時間、負面社會影響 | 前四項已寫；社會影響 `\TODO` | 部分 | 0.4（0.3） |
| 6 Conclusion | — | 方法、主要結果、意義 | `\TODO` | 未寫 | 0.2（0.1） |
| LLM Usage Statement | — | 工具、受影響部分、使用程度（不計頁） | `\TODO` | 未寫 | 不計 |
| 補充材料 | Additional metrics | DISTS、色彩對齊後失真（PSNR、SSIM 在 Table 1） | `\TODO`；CLIP-IQA 與 FID 表已撤下；未編入 PDF | 未寫 | 不計 |
| References | — | 無 URL；全部核實 | executor 已核實（含 radford2021clip、hessel2021clipscore）；未引用條目已移除 | 完成 | 不計 |

正文預算合計 8.0 頁。圖表與程式：`docs/paper_tools/generate_paper_figures.py`（遠端 `~/paper_figures/` 執行）。
