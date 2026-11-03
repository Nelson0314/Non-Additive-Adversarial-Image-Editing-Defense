# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、進行中的工作與規則，供冷啟動接手。數字一律連同協定引用。各節的逐條狀態見 `docs/PAPER_STRUCTURE.md`。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認。交接當下的編號：

| pane | 標題 | 角色 | 寫入範圍 |
|---|---|---|---|
| 5 | paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、本檔 |
| 2 | executor | 專案管理、派工、文獻管理；`main.bib` 的唯一核實者 | `docs/EXECUTOR_HANDOVER.md` 等 |
| 3 | color | 本方法的實驗與讀數 | `color/` |
| 4 | baseline | 加性方法、主表、跨模型 baseline 的讀數 | `baseline/` |

- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n；完成一節後回報 executor（訊息 ≤ 4–8 行，依 executor 要求）。
- 不派實驗、不改其他專案的檔案。
- `main.bib`：新增條目可以直接用 Edit 插入（不要整檔重寫），加入後立即通知 executor 核實；executor 正在改 bib 時不動它。現有條目已全部核實、無 URL／DOI 欄。
- 本機不跑計算（Tectonic 編譯除外）；圖在遠端畫，看圖時抓回 scratchpad。

## 2. 檔案與工具

| 路徑 | 內容 |
|---|---|
| `docs/paper/` | 唯一的 tex 正本，整個目錄（`build/` 除外）即 Overleaf 上傳內容。CVPR 2027 author kit v1（`cvpr.sty`、`ieeenat_fullname.bst`、`preamble.tex` 為官方檔，不修改）。`main.tex` 另載入 `cuted`、`currfile`、`caption`、`colortbl`、`multirow`，定義 `\NUM`、`\TBD` 與表列底色 `oursrow`；`\ifdefined\XeTeXrevision` 區塊只在 Tectonic 下以 TeX Gyre 字型取代 `times`。teaser（`\input{fig/teaser}`）與補充材料（`\input{sec/X_suppl}`）在 `main.tex` 中為註解 |
| `docs/paper/fig/` | `method_overview.pdf`（Fig. 1）、`distortion_displacement.pdf`（Fig. 2）、`purification.pdf`（Fig. 3）；`teaser.tex`、`teaser.pdf` 為暫時移出的 teaser |
| `docs/paper_tools/run_paper_preview.py` | Tectonic 編譯 `docs/paper/`，成功後複製到桌面 `C:\Users\nelso\OneDrive\桌面\paper.pdf`；`--watch` 為存檔即重編。說明見同目錄 `README.md` |
| `docs/paper_tools/method_overview/` | Fig. 1 的 TikZ 原始檔 `method_overview.tex`；資料圖塊（`lattice_*.tex`、`disk_*`）由 `generate_method_overview_panels.py` 在遠端產生後複製進來，於該目錄執行 `tectonic -X compile method_overview.tex`，再把 PDF 複製到 `docs/paper/fig/` |
| `docs/paper_tools/generate_paper_figures.py` | 產生 Fig. 2、Fig. 3。Pexels 輸入由 `import_pexels_figure_inputs.py` 從 baseline `runtime/pexels_portraits/` 轉出（遠端 `~/paper_figures/in_pexels/`）；Fig. 3b–d 的強度掃描仍用 8 張的 `retention.csv` |
| `docs/paper_tools/generate_teaser_figure.py` | teaser 與同版面的例子（`--image`、`--prompt`、`--name`），遠端執行 |
| `docs/paper_tools/experiment_overview/` | 內部用實驗總覽（第 6 節） |
| `docs/PAPER_STRUCTURE.md` | 結構表（節 → 應含內容 → 現況 → 狀態 → 頁數）；每次改 tex 後同步 |

## 3. 方法現況

`color3d`（論文的 Ours）：33³ RGB LUT，色空間平滑的色度旋轉。以黑點為原點、灰軸為極軸，自由參數 Θ 經 σ = 4 格點的 3D 高斯濾波成旋轉場，極角（彩度）變化 < 10°、方位角（色相）變化 < 3°（截斷到 RGB 色域前），‖c‖ 不變；損失只有 ip2p VAE 編碼器後驗平均差的平方平均；Adam (0.5, 0.99)、1800 步、學習率餘弦 3e-4 → 1e-5（前 2/3）；每張 8.5–9 分鐘（RTX 3090）。程式在 color trial `generate_color3d_defenses.py`（250–300 行為旋轉）。

消融用的其他版本：`color3d_orig`（逐項懲罰、PHR）、`color3d_phr`（單一常數上界）、`color3d_adaptive_set`（可行集合投影）、`color3d_cone_theta12`（色相夾角錐投影）、`color3d_smooth_angle6`（單角 6°）、`color3d_smooth_polar9_hue0`／`_hue3`、`color3d_random_matched`（同類隨機旋轉，逐張對齊 Ours 的防禦圖 LPIPS）。

## 4. 論文各節現況與待改處

正文共 9 頁 PDF，正文與 Conclusion 到第 8 頁，參考文獻自第 8 頁起。圖表：Fig. 1 方法總覽、Fig. 2 失真—位移、Fig. 3 淨化分析；Table 1 主表、Table 2 美學分數、Table 3 跨編輯器、Table 4 消融。

| 節 | 現況 | 待改 |
|---|---|---|
| Abstract | 產品介紹式；結果句一個 `\NUM`（淨化後保留位移相對加性方法的倍數） | 倍數定義與「at matched distortion」待使用者決定 |
| 1 Introduction | 已寫；貢獻 3 點；一個 `\NUM` | 貢獻第二條同樣有「at matched distortion」 |
| 2 Related Work | 四段 | — |
| 3 Method | 威脅模型、3D LUT、平滑色度旋轉（式 1–5）、目標（式 6）、最佳化；Fig. 1 | |Δφ| < 3° 為格點上的保證，像素經三線性內插後可能超出，是否註明待使用者決定 |
| 4.1 Setup | Pexels 306 張用於 Table 1、2、Fig. 2、Fig. 3a；8 張 portraits 用於 Fig. 3b–d、Table 3、4；Metrics 段已含 CLIP-S、CLIP-I、FID | Pexels 選取規則與授權為 `\TODO` |
| 4.2 Robustness to purification | Table 1 已填（306 張），VAE-Trans、CLIP-S、CLIP-I、FID 欄為 `\TBD`；正文 3 個 `\NUM` | 等 baseline／color 的讀數 |
| 4.3 Distortion and displacement before purification | Fig. 2（306 張，18 點，1.037x＋0.049，r＝0.967；Ours 高於線 0.009） | — |
| 4.4 What purification removes | Fig. 3a 為 306 張，Fig. 3b–d 為 8 張 | VAE-Trans 存續比為 `\NUM`；強度掃描的 Pexels 版本（P2）由使用者暫停 |
| 4.5 Editors with different encoders | Table 3：兩編輯器上下排（8 張），CLIP-S、CLIP-I、FID 欄為 `\TBD` | 遷移性實驗讀數與張數 |
| 4.6 Ablations | Table 4（8 張） | 定案張數的讀數 |
| 5 Limitations | 已寫 | 負面社會影響與緩解為 `\TODO` |
| 6 Conclusion、LLM Usage Statement | `\TODO` | LLM Usage 需使用者提供工具清單 |
| 補充材料 `sec/X_suppl.tex` | CLIP-IQA 與 FID 表已撤下（使用者 11/3 決定）；未編入 PDF | PSNR、SSIM、DISTS、色彩對齊後失真 |

待使用者決定的事項：
1. 摘要 `\NUM` 的倍數定義。
2. 摘要與貢獻是否保留「at matched distortion」（Pexels 上 Ours 的防禦圖 LPIPS 0.211，加性方法 0.251–0.318）。
3. 論文重心是否由「對編碼器最佳化」轉向「參數化本身」（同類隨機旋轉與最佳化在 8 張上相近）。
4. Fig. 1 色度盤箭頭用深灰或 teal；|Δφ| < 3° 在像素層是否註明。

## 5. 已收到的數字與位置

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15，VAE-Trans 另列。

| 內容 | 用在哪裡 | 來源 |
|---|---|---|
| Pexels 306 張：10 條件的防禦圖 LPIPS、美學、編輯 LPIPS（未淨化、七道、最弱一道）、保留率、SigLIP、ArcFace | Table 1、Table 2、4.2 | baseline 遠端 `runtime/pexels_portraits/summary/main_table.csv`（欄定義見 `scripts/generate_pexels_main_table_summary.py`）、`per_purifier_displacement.csv` |
| Pexels 失真—位移點與擾動存續比 | Fig. 2、Fig. 3a | 同目錄 `interim/fig3_points.csv`（原生預算）、`followup/sets/*/results/aligned/survival_seven.csv` 與 `followup/results/color3d_all/survival_seven.csv` |
| 8 張的淨化強度掃描 | Fig. 3b–d | baseline `results/sweeps/ip2p/purifier_strength/retention.csv` |
| 跨編輯器（8 張） | Table 3 | baseline `results/cross_model/`；color trial `results/cross_model/{flux,sd3}/` |
| 消融（8 張） | Table 4 | color trial `README.md`、`results/color3d_random_matched/` |

## 6. 進行中的工作：實驗總覽 PDF

- 原始檔：`docs/paper_tools/experiment_overview/experiment_overview.tex`；成品：`C:\Users\nelso\OneDrive\桌面\experiment_overview.pdf`。
- 編譯：在該目錄執行 `tectonic -X compile experiment_overview.tex`，再把 `experiment_overview.pdf` 複製到桌面同名檔。中文字型為 xeCJK＋Microsoft JhengHei；圖直接取 `docs/paper/fig/`。
- 規格（executor 代使用者派工，照字面）：論文中每個實驗一節，依論文順序（主表、Fig. 3、Fig. 4、Table 2、Table 3、CLIP-IQA、FID／KID）；每節只有「設定」「組別」條列、與論文同格式的圖表、中文說明（caption 譯文）；不加引言、結論或製作說明。
- 使用者的要求：表格不標紅字；各欄最佳值用粗體（依欄的 ↑／↓）；設定與組別只寫必要資訊，組別直接列出方法名稱。
- 現況：七節都已填數字，無空格。主表加性方法為 8 張（標 b，n＝8），Ours 有 Pexels 讀數的格為 306 張（標 a），其餘為 8 張。
- 下一步：baseline 與 color 的 Pexels 讀數到齊後，替換主表對應格與張數標記，重新判定各欄粗體，重新編譯並覆蓋桌面檔。

## 7. 還在等的數字

| 來源 | 內容 | 用途 |
|---|---|---|
| baseline／color | Pexels 的 VAE-Trans（編輯 LPIPS、SigLIP、存續比） | Table 1、4.2、4.4 |
| baseline／color | CLIP-S、CLIP-I、編輯端 FID（Pexels 主表與遷移表） | Table 1、Table 3 |
| color | 遷移性實驗（11/4 20:00 收一次；FLUX 張數到時再改 caption） | Table 3 |
| color | 消融定案張數的讀數 | Table 4 |

## 8. 寫作規則

- 正文英文，學術、精確；避免冒號式揭示、問答式鋪陳、「not X but Y」、Moreover／Notably 類連接詞、比喻、口語、無出處的主張；不寫時間相依字眼；數字附協定。
- 摘要寫背景、痛點、本方法特色與預期效果，不寫 baseline、方法細節或「We observe」句型。
- 失真—位移直線是對所有方法的觀察（含本方法），寫成「等失真下可比」，不寫成對立。
- 方法節以式子表達，先講選擇、再講用意；不寫章節導覽句。
- 表格固定字級、不用自動縮放；caption 一兩行，定義與解讀寫在正文；指標標 ↑／↓、各欄最佳粗體、Ours 列底色。
- 不下「方法成立／不成立」的判定。不做色調／白平衡類淨化（Limitations 已寫）。
- 正文上限 8 頁，參考文獻與 LLM Usage Statement 不計；全文不放 URL。
- commit message 用英文；push 由使用者執行；論文檔目前未入版控（`docs/` 為 untracked）。
