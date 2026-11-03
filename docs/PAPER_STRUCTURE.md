# 論文結構表

對象為 `docs/paper/`（CVPR 2027，正文上限 8 頁，含圖表；參考文獻與 LLM Usage Statement 不計頁）。「Ours」指 `color3d`（雙角度色度旋轉：以黑點為原點、灰軸為極軸，極角 < 10°、方位角 < 3°，截斷前；損失只有目標）。現有頁數為 Tectonic 編譯結果的估計，雙欄版面下以欄位置換算，誤差約 ±0.2 頁。每次修改 tex 後同步本表。

編譯狀態：共 10 頁，正文與 Conclusion 至第 8 頁，參考文獻自第 8 頁起，第 10 頁為暫代數字的紅字註解。正文圖表：Fig. 1 方法總覽、Fig. 2 失真—位移、Fig. 3 淨化分析；Table 1 主表（含 LAION、MUSIQ）、Table 2 遷移、Table 3 消融。內容依全規模實驗清單（`docs/PAPER_HANDOVER.md` 第 7 節）撰寫，未達全規模的數字以 `\prelim` 標紅。

資料範圍（使用者 11/3 定案）：所有表與圖皆為 Pexels 306 張、ip2p 訓練、全協定；消融為逐項拿掉 pipeline 的一部分（A–E，見 `docs/PAPER_HANDOVER.md` 第 3 節）。

| 節 | 小節 | 應含內容 | 現有內容 | 狀態 | 頁數預算（現有） |
|---|---|---|---|---|---|
| 標題 | teaser | 各防禦淨化前後的編輯 | 暫時移出（`main.tex` 中 `\input{fig/teaser}` 為註解）；`fig/teaser.pdf` 由 `docs/paper_tools/generate_teaser_figure.py` 產生 | 移除 | 0（0） |
| Abstract | — | 背景、加性擾動的痛點、filter-like 色彩映射、淨化後保留的編輯位移 | 定稿，結果句為 1.37–2.60 倍；不含高頻主張 | 部分 | 0.3（0.3） |
| 1 Introduction | 背景、加性擾動、淨化研究、本方法、自然度參數化、比較協定 | — | 已寫 | 完成 | 1.0（1.0） |
| | 貢獻 | 3 點：方法 → 4.2；淨化後倍數 → 4.2；共同協定下的分析 → 4.3、4.4 | 已寫；倍數同摘要 | 完成 | |
| 2 Related Work | 免疫、淨化、色彩對抗樣本、美學與品質評估 | 段末收回限制或寫選擇理由 | 已寫 | 完成 | 0.7（0.8） |
| | Fig. 1 方法總覽 | (a) 參數化、(b) 色彩旋轉、(c) 最佳化 | `fig/method_overview.pdf`（TikZ，`docs/paper_tools/method_overview/`）；(b) 右為 woman_00 color3d LUT 的實際位移；同色像素 (228,85)、(423,378) → 同一輸出 | 完成 | |
| 3 Method | 3.1–3.5 | 威脅模型、3D LUT、平滑色度旋轉（式 1–5）、目標（式 6）、最佳化 | 已寫 | 完成 | 1.3（1.1） |
| | Fig. 2、Table 1 | 失真—位移雙面板圖（含 BlurGuard、Ours 多預算點）；主表（兩個 panel，數字旁小字為 95% 信賴區間半寬） | Fig. 2 為 306 張、無 BlurGuard 與多預算點；Table 1 為 306 張，只有 BlurGuard 列為紅字 | 部分 | |
| 4 Experiments | 4.1 Setup | 資料、攻擊、淨化、baseline、讀數 | 已寫；Pexels 選取規則與授權為 `\TODO` | 部分 | 3.2（2.9） |
| | 4.2 Robustness to purification | 主結果敘述 | 已寫；VAE-Trans 加性範圍與 FID 句為紅字 | 部分 | |
| | 4.3 Distortion and displacement before purification | Fig. 2a | 已寫（306 張；8 張的掃描與 DISTS／PieAPP 結論另註） | 完成 | |
| | 4.4 What purification removes | Fig. 3（a 存續比、b–d 強度掃描，306 張，DCT-Shield-Y） | 已寫；b–d 與其正文數字為 8 張、DCT-Shield 的紅字；VAE-Trans 存續比為紅字 | 部分 | |
| | 4.5 Editors with different encoders | Table 2：ip2p 求解的等失真臂直接遷移，兩編輯器上下排 | 全部紅字（加性為原生預算 8 張、Ours 為 8 張） | 部分 | |
| | 4.6 Ablations | Table 3：上半逐項拿掉 pipeline 的一部分，下半為自然度的其他施加方式 | 完整 pipeline 與 A 的未淨化為 306 張；其餘紅字；結果段 `\TODO` | 部分 | |
| 5 Limitations | — | 位移與編輯失敗的差異、未評估色彩反推攻擊、未做受測者實驗、求解時間、負面社會影響 | 前四項已寫；社會影響 `\TODO` | 部分 | 0.4（0.3） |
| 6 Conclusion | — | 方法、主要結果、意義 | `\TODO` | 未寫 | 0.2（0.1） |
| LLM Usage Statement | — | 工具、受影響部分、使用程度（不計頁） | 已寫：Claude Code；程式、圖表腳本、文字 | 待使用者審閱 | 不計 |
| 補充材料 | Additional metrics | NIMA、NIQE 表（`tab:supp_aesthetics`）；DISTS、色彩對齊後失真 | 美學表已寫；其餘 `\TODO`；未編入 PDF | 未寫 | 不計 |
| References | — | 無 URL；全部核實 | executor 已核實（含 radford2021clip、hessel2021clipscore）；未引用條目已移除 | 完成 | 不計 |

正文預算合計 8.0 頁。圖表與程式：`docs/paper_tools/generate_paper_figures.py`（遠端 `~/paper_figures/` 執行）。
