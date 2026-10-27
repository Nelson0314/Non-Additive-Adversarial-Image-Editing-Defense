# 論文結構表

對象為 `docs/paper/`（CVPR 2027，正文上限 8 頁，含圖表；參考文獻與 LLM Usage Statement 不計頁）。「Ours」指 `color3d`（雙角度色度旋轉：以黑點為原點、灰軸為極軸，極角 < 10°、方位角 < 3°，截斷前；損失只有目標）。現有頁數為 Tectonic 編譯結果的估計，雙欄版面下以欄位置換算，誤差約 ±0.2 頁。每次修改 tex 後同步本表。

編譯狀態：共 7 頁，正文至第 6 頁。

| 節 | 小節 | 應含內容 | 現有內容 | 狀態 | 頁數預算（現有） |
|---|---|---|---|---|---|
| 標題、Fig. 1 | teaser | 原圖 → 防禦 → 淨化 → 編輯，對照加性方法 | 佔位（`fig:teaser`）；草案在 color 遠端 `trials/lut3d_defense_and_naturalness/results/paper_figures/teaser_woman_00_p2_jpeg50_dct_shield.png`（DCT-Shield 對 Ours、JPEG-50、helmet），待使用者確認 | 部分 | 0.5（0.4） |
| Abstract | — | 背景、加性擾動的痛點、filter-like 色彩映射、淨化後保留的編輯位移 | 定稿，結果句一個 `\NUM` | 部分 | 0.3（0.3） |
| 1 Introduction | 背景與免疫的發展 | 生成模型與隱私問題、PhotoGuard 起的研究線 | 已寫 | 完成 | 1.0（1.0） |
| | 加性擾動的共同形式 | PGD 加性擾動、逐像素、高頻 | 已寫；高頻證據為 `\TODO`（頻譜暫不入正文） | 部分 | |
| | 淨化研究的發現 | JPEG、noisy upscaling、VAE 淨化、CAT、評估研究 | 已寫 | 完成 | |
| | 本方法 | filter-like、逐影像 3D LUT、無空間結構 | 已寫 | 完成 | |
| | 自然度由參數化保證 | 色度旋轉的形式；逐瑕疵懲罰會衝突 | 已寫 | 完成 | |
| | 比較協定 | 失真—位移直線、等失真比較 | 已寫 | 完成 | |
| | 自然度量測 | 色彩對齊後失真、參照帶、受測者實驗 | 已寫 | 完成 | |
| | 貢獻 | 每點對應一個實驗 | 3 點（方法 → 4.2；淨化後倍數 → 4.2；自然度評估 → 4.6，受測者實驗未做） | 部分 | |
| 2 Related Work | 擴散編輯免疫 | 加性方法、DiT 方法、抗淨化方法；段末收回限制 | 已寫，段末收回「仍是像素格上的加性圖樣」；SIFM、DANP、DAYN 為 `\TODO` | 部分 | 0.7（0.8） |
| | 淨化 | 淨化與破解研究；段末收回 | 已寫，段末收回「只證明可移除、各自預算」 | 完成 | |
| | 色彩對抗樣本 | ReColorAdv、AdvCF、NCF 等；段末收回 | 已寫，段末說明取用 ReColorAdv 的載體、改用旋轉約束 | 完成 | |
| | 色彩改變的評估 | LPIPS、DISTS、CIEDE2000、CD-Flow、SCOOTER；段末收回 | 已寫，段末說明 PSNR／SSIM／DISTS 的色調偏差與本文的指標選擇 | 完成 | |
| 3 Method | Fig. 2 方法總覽 | 參數化、編碼器目標、淨化與編輯 | `fig:overview`，原始碼置於第 2 節開頭以浮動到第 3 頁頂端；精緻版由 executor 派人重畫中 | 部分 | 1.3（1.1） |
| | 3.1 Threat model | 白盒、外掛、訓練不用指令 | 已寫 | 完成 | |
| | 3.2 LUT carrier | 33³ LUT、三線性內插、與位置無關 | 已寫；淨化與 LUT 近似可交換的論證為 `\TODO` | 部分 | |
| | 3.3 Smooth color rotations | 旋轉場、tanh 軟飽和、極角 < 10°、方位角 < 3°（截斷前） | 已寫，color 已核對與程式一致 | 完成 | |
| | 3.4 Objective | VAE 編碼器後驗平均差的平方平均 | 已寫 | 完成 | |
| | 3.5 Optimization | 初始化、Adam、步數、學習率排程、交付步、求解時間 | 已寫 | 完成 | |
| 4 Experiments | 導覽段 | 各小節回答什麼 | 已寫，依新順序 | 完成 | 3.2（2.4） |
| | 4.1 Setup | 資料、攻擊、淨化、baseline、讀數 | 已寫；讀數改為 LPIPS＋美學／LPIPS＋SigLIP，傳統指標移補充材料；Pexels 為 `\TODO` | 部分 | |
| | 4.2 抗淨化（主結果） | 等失真主表：防禦圖 LPIPS、LAION、編輯 LPIPS 淨化前／七道平均／最低一道／VAE-Trans、保留率、SigLIP；淨化後散布圖 | 主表已填（baseline `results/aligned/condition_summary.csv`；color `results/paper_table/summary.csv`、`arms_vae_siglip.csv`）；散布圖佔位 | 部分 | |
| | 4.3 淨化前平面與原生預算 | 直線擬合、原生預算表 | 原生預算表已填（baseline `results/condition_summary.csv`）；平面圖佔位 | 部分 | |
| | 4.4 淨化移除了什麼 | 擾動存續比、淨化強度掃描 | 兩表有加性方法與 color3d_orig；Ours 未量（存續比需 CPU、強度掃描需 GPU 新跑） | 部分 | |
| | 4.5 其他編輯器 | FLUX.1 Kontext、UltraEdit、SD inpainting | color3d_orig 的 inpainting 數字；baseline FLUX encoder 類已完成、SD3 讀數中；Ours 待求解 | 部分 | |
| | 4.6 感知品質 | 美學、色彩對齊後失真、參照帶、受測者實驗 | color3d_orig 數字；受測者實驗未做 | 部分 | |
| | 4.7 Ablations | 自然度形式、旋轉上界、載體、目標、隨機對照 | 消融軸表；數字為 `\TODO` | 部分 | |
| 5 Limitations | — | 位移與編輯失敗的差異、色調／白平衡淨化未評估、負面社會影響與緩解 | 前兩項部分；社會影響無 | 部分 | 0.4（0.2） |
| 6 Conclusion | — | 方法、主要結果、意義 | `\TODO` | 未寫 | 0.2（0.1） |
| LLM Usage Statement | — | 工具、受影響部分、使用程度（不計頁） | `\TODO` | 未寫 | 不計 |
| References | — | 無 URL；`main.bib` 的 TODO 條目逐條核實 | 41 條，其中 28 處標 TODO 未核實 | 部分 | 不計 |

正文預算合計 8.0 頁。
