# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、進行中的工作與規則，供冷啟動接手。數字一律連同協定引用。各節的逐條狀態見 `docs/PAPER_STRUCTURE.md`。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認，不沿用本表或記憶中的編號。

| 標題 | 角色 | 寫入範圍 |
|---|---|---|
| paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、本檔 |
| executor | 專案管理、派工、文獻管理；`main.bib` 的唯一核實者；代使用者轉達決定 | `docs/EXECUTOR_HANDOVER.md` 等 |
| color | 本方法的實驗與讀數 | `color/` |
| baseline | 加性方法、Pexels 主表、跨模型 baseline 的讀數 | `baseline/` |

- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n；完成一件事後回報 executor，訊息行數依 executor 要求（通常 ≤ 3–6 行）。
- 不派實驗、不改其他專案的檔案。
- `main.bib`：新增條目以 Edit 插入（不整檔重寫），加入後立即通知 executor 核實；未引用的條目交 executor 處理，不自行刪除。現有條目已全部核實、無 URL／DOI 欄。
- 本機不跑計算（Tectonic 編譯除外）：資料彙整、畫圖、PDF 轉 PNG 都在遠端 basic-1（`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`）；看圖時抓回 session 的 scratchpad。

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
| Abstract | 產品介紹式；結果句為「at matched distortion …1.37 to 2.60 times」：Ours 七道淨化平均編輯 LPIPS 0.291 除以各加性方法的 0.212（DAYN）至 0.112（DCT-Shield），Pexels 306 張、ip2p，不含 VAE-Trans | — |
| 1 Introduction | 已寫；貢獻 3 點；第二條與摘要同一倍數，指向 4.2 | — |
| 2 Related Work | 四段 | — |
| 3 Method | 威脅模型、3D LUT、平滑色度旋轉（式 1–5）、目標（式 6）、最佳化；Fig. 1 | |Δφ| < 3° 為格點上的保證，像素經三線性內插後可能超出，是否註明待使用者決定 |
| 4.1 Setup | Pexels 306 張用於 Table 1、2、Fig. 2、Fig. 3a；8 張 portraits 用於 Fig. 3b–d、Table 3、4；Metrics 段已含 CLIP-S、CLIP-I、FID | Pexels 選取規則與授權為 `\TODO` |
| 4.2 Robustness to purification | Table 1 已填（306 張），防禦圖端 PSNR、SSIM 與 VAE-Trans、CLIP-S、CLIP-I、FID 欄為 `\TBD`；正文 3 個 `\NUM` | 等 baseline／color 的讀數 |
| 4.3 Distortion and displacement before purification | Fig. 2（306 張，18 點，1.037x＋0.049，r＝0.967；Ours 高於線 0.009） | — |
| 4.4 What purification removes | Fig. 3a 為 306 張，Fig. 3b–d 為 8 張 | VAE-Trans 存續比為 `\NUM`；強度掃描的 Pexels 版本（P2）由使用者暫停 |
| 4.5 Editors with different encoders | Table 3：兩編輯器上下排（8 張），CLIP-S、CLIP-I、FID 欄為 `\TBD` | 遷移性實驗讀數與張數 |
| 4.6 Ablations | Table 4（8 張） | 定案張數的讀數 |
| 5 Limitations | 已寫 | 負面社會影響與緩解為 `\TODO` |
| 6 Conclusion、LLM Usage Statement | `\TODO` | LLM Usage 需使用者提供工具清單 |
| 補充材料 `sec/X_suppl.tex` | CLIP-IQA 與 FID 表已撤下（使用者 11/3 決定）；未編入 PDF | DISTS、色彩對齊後失真（PSNR、SSIM 已移入 Table 1） |

使用者 11/3 的決定：摘要倍數寫區間（逐一除以各加性方法）；保留「at matched distortion」；Fig. 1 色度盤箭頭用深灰（`disp` 樣式 `black!62`，現行即是）；論文重心等 306 張的同類隨機對照（`color3d_random_matched`）讀數到齊後再定。

待使用者決定的事項：
1. |Δφ| < 3° 在像素層是否註明。
2. LLM Usage Statement 的工具清單。

## 5. 已收到的數字與位置

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15，VAE-Trans 另列。

| 內容 | 用在哪裡 | 來源 |
|---|---|---|
| Pexels 306 張：10 條件的防禦圖 LPIPS、美學、編輯 LPIPS（未淨化、七道、最弱一道）、保留率、SigLIP、ArcFace | Table 1、Table 2、4.2 | baseline 遠端 `runtime/pexels_portraits/summary/main_table.csv`（欄定義見 `scripts/generate_pexels_main_table_summary.py`）、`per_purifier_displacement.csv` |
| Pexels 失真—位移點與擾動存續比 | Fig. 2、Fig. 3a | 同目錄 `interim/fig3_points.csv`（原生預算）、`followup/sets/*/results/aligned/survival_seven.csv` 與 `followup/results/color3d_all/survival_seven.csv` |
| 8 張的淨化強度掃描 | Fig. 3b–d | baseline `results/sweeps/ip2p/purifier_strength/retention.csv` |
| 跨編輯器（8 張） | Table 3 | baseline `results/cross_model/`；color trial `results/cross_model/{flux,sd3}/` |
| 消融（8 張） | Table 4 | color trial `README.md`、`results/color3d_random_matched/` |

## 6. 交付物與進行中的工作

- **桌面 `paper.pdf`**：`run_paper_preview.py` 每次成功編譯後自動覆寫；Overleaf 由使用者整個上傳 `docs/paper/`，不另做壓縮包。
- **簡報素材**：不再維護（executor 11/3 轉達），不重出 PNG。遠端 `~/paper_figures/deck_assets/` 保留舊版。
- **實驗總覽 PDF**：`docs/paper_tools/experiment_overview/experiment_overview.tex`，成品為桌面 `experiment_overview.pdf`（於該目錄 `tectonic -X compile`，再複製）。內容停在 8 張與 Pexels 混合的版本，表格不標紅字、各欄最佳粗體；使用者未再要求更新。

## 7. 還在等的數字

| 來源 | 內容 | 用途 |
|---|---|---|
| baseline／color | Pexels 的防禦圖端 PSNR、SSIM（防禦圖對原圖） | Table 1 |
| baseline／color | Pexels 的 VAE-Trans（編輯 LPIPS、SigLIP、存續比） | Table 1、4.2、4.4 |
| baseline／color | CLIP-S、CLIP-I、編輯端 FID（Pexels 主表與遷移表） | Table 1、Table 3 |
| color | 遷移性實驗（11/4 20:00 收一次；FLUX 張數到時再改 caption） | Table 3 |
| color | 消融定案張數的讀數 | Table 4 |

## 8. 寫作規則

- 正文英文，學術、精確；避免冒號式揭示、問答式鋪陳、「not X but Y」、Moreover／Notably 類連接詞、比喻、口語、無出處的主張；不寫時間相依字眼；數字附協定。
- 摘要寫背景、痛點、本方法特色與預期效果，不寫 baseline、方法細節或「We observe」句型。
- 失真—位移直線是對所有方法的觀察（含本方法），寫成「等失真下可比」，不寫成對立。
- 方法節以式子表達，先講選擇、再講用意；不寫章節導覽句。
- 表格：固定字級、不用 `\resizebox`；Method 欄獨立（`\multirow`），表頭依「對象 → 指標 → 淨化條件」分層；指標標 ↑／↓，各欄最佳粗體（Ours 與 Ours (transfer) 分開比），Ours 列底色 `oursrow`；七道淨化平均一律寫 Purified$^{*}$ 並在 caption 註明七道名稱，不寫 7-mean；最弱一道寫 Weakest$^{\dagger}$（平均最低的那一道淨化，不是逐格取最小）；每個指標的比較條件寫進 caption（例如 ArcFace 是同一淨化下兩張編輯的嵌入餘弦）。
- 圖：與論文同字體（Times 系：matplotlib 用 STIXGeneral＋stix，TikZ 用 texgyretermes），印刷尺寸 ≥ 7 pt；配色全文一致：加性方法紅 `#A94438`、本方法 teal `#1F6F7C`、飽和度／θ 藍 `#2F5DA8`、色相／φ 橘 `#C9781A`、參考與凍結元件灰；caption 只寫看什麼與結論，示意與實際資料要分清楚。
- 不下「方法成立／不成立」的判定。不做色調／白平衡類淨化。不提「加性擾動偏高頻」（無頻譜證據，頻譜圖已撤）。
- 正文上限 8 頁，參考文獻與 LLM Usage Statement 不計；全文不放 URL。
- commit message 用英文；內容修改、清理、文件更新分開 commit；push 由使用者或 executor 執行。工作樹與其他 session 共用，在 main 上 commit，不切 branch。

## 9. 工作方式

- **改論文**：改 `docs/paper/` 後執行 `python docs/paper_tools/run_paper_preview.py`；檢查頁數、`build/main.log` 的 Overfull 與未定義引用。看排版時把 `build/main.pdf` scp 到遠端 `/tmp`，以 `mutool draw -q -r 150 -o page.png main.pdf <頁>` 轉 PNG 抓回看。
- **填表**：先把 baseline 的 CSV scp 到 scratchpad，用 Python 依欄名產生表格列與粗體，不手抄數字；改 tex 用 Python 時一律寫成 scratchpad 內的 `.py` 檔並用 `assert s.count(old) == 1` 確認唯一替換（heredoc 內的反斜線與 f-string 容易出錯）。
- **畫圖**：腳本改在本機，`scp` 到遠端 `~/paper_figures/in/` 執行，輸出抓回 `docs/paper/fig/`。Fig. 1 的 TikZ 在本機 Tectonic 編譯，資料圖塊在遠端產生。畫之前先確認讀者要記住的一件事；使用者偏好先看清楚的版面草圖，再上實際資料，圖面要乾淨、少字、標籤不壓線與框。
- **示意與實際資料**：圖中標示的像素、數值必須來自既有產物並在腳本頂端註明出處；示意部分在 caption 註明放大或非比例。
- **數字來源**：Pexels 主表與分布指標都由 baseline 彙整在 `runtime/pexels_portraits/summary/`（`main_table.csv` 欄定義在 `baseline/scripts/generate_pexels_main_table_summary.py` 開頭）；`distribution_groups.csv` 的 `clipiqa_b` 為未防禦編輯。color 的 `metrics_<條件>.csv` 曾有過期版本（Mist），兩邊數字不一致時以 baseline 為準並請其確認。

## 10. 近期決定（使用者，經 executor 轉達）

- 11/3：不用 CLIP-IQA；新增 CLIP-S（編輯對指令文字）、CLIP-I（防禦後編輯對未防禦編輯）、編輯端 FID；防禦圖對原圖的 FID 不報；KID 不放（Pexels 90 組中 32 組為負）。
- 11/3：teaser 暫時移出論文（檔案保留）；Fig. 2、Fig. 3a 改用 Pexels 306 張；Table 1 表頭重排、ArcFace 寫明比較條件；Table 3（遷移）各欄最佳粗體。
- 11/4 20:00 收一次遷移性實驗，FLUX 可能不足 50 張，caption 張數屆時更新。
- Fig. 1（方法總覽）的版面已由使用者逐輪定稿：(a) 參數化、(b) 球面上的色彩旋轉（含 woman_00 實際 LUT 的色度盤）、(c) 最佳化與同色像素；色度盤箭頭用深灰或 teal 尚待確認。
- 11/3：主表的 SSIM、PSNR 為防禦圖端（防禦圖對原圖），列在 Table 1 的 Protected 組；Ret.、ArcFace、CLIP-S、CLIP-I、FID 的定義改在 caption 以名稱標示，不用註記符號（容納兩欄的表寬）。簡報不再維護。
