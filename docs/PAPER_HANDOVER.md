# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、待辦與規則，供冷啟動接手。數字一律連同協定引用。接手順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔第 1、2、5、6、7 節 → 第 4 節現況。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認，不沿用記憶中的編號。

| 標題 | 角色 | 寫入範圍 |
|---|---|---|
| paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、本檔；Overleaf 專案（第 2 節） |
| executor | 進度、派工、轉達使用者決定、`main.bib` 核實、代為 push、ISR 等新實驗的協調 | `docs/EXECUTOR_HANDOVER.md`、`docs/isr_pilot/` 等 |
| color | ImmuLUT（條件名 color3d 系列）的實驗、程式與讀數 | `color/` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 [klein] 4B、inpainting）、主表讀數、Fig. 1 與 Fig. 4 的圖檔 | `baseline/` |

- executor 有權直接對 paper 下令（使用者授權），其指示視同使用者指示。例外：**Overleaf 同步只在使用者於 paper pane 親自下令時執行**。
- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n。完成一件事後回報 executor，附 commit 號；回報不超過要求的行數，先講結果。
- 不派實驗、不改其他專案的檔案。push 由 executor 執行。
- `main.bib`：新增條目以 Edit 插入，加入後立即通知 executor 核實；不刪除未引用條目，列給 executor。文獻原標題不改。
- 本機只做 Tectonic 編譯與依 CSV 產生表格的小腳本；彙整、畫圖、PDF 轉 PNG 在遠端 basic-1（`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`，Python 用 `~/venvs/wacv/bin/python`）。看圖時抓回 scratchpad。
- git index 與其他 session 共用：`git mv`／`git rm` 暫存後要立即 commit，否則會被其他 session 的 commit 一併帶走（`59f47556` 即含 paper 的改名與刪檔）。

## 2. Overleaf 協作

- 專案：Overleaf `[CVPR 2027] Color-based Image Immunization`（`6ad193f8288983a2f6af5d50`）。iamshuai（指導教授）與 Lynn 在線上直接編修。
- **repo 與 Overleaf 必須逐字一致。** 同步前先在編輯器頁面計算 14 個文字檔的雜湊，與上次同步點比對；不一致表示有人在線上修改，先以三方合併（基底為上次同步點）收進 repo。
- **文字檔一律在 Overleaf 編輯器內修改，不上傳覆蓋**（覆蓋會失去編輯歷史）：`generate_overleaf_ops.py` 產生詞級修改清單，於頁面以 CodeMirror transaction 套用，前後比對雜湊，重新載入再驗證一次。步驟與頁面程式碼見 `docs/paper_tools/README.md`「Overleaf 同步」。PDF 圖檔無法在編輯器修改，以上傳覆寫。
- 瀏覽器：使用者以 `/chrome` 連線 Claude in Chrome；只操作自己開的分頁群組。擴充功能會擋下看似網址參數的輸出，此時改用截圖。線上 PDF 檢視器預設不顯示時，按 "Switch to PDF"；頁面文字層需捲動到該頁後才能讀取。
- 線上若有 repo 沒有的檔案（現有 `sec/[old] related.tex`），不動。

## 3. 方法與定調

- 方法名 **ImmuLUT**（巨集 `\capg`，定義在 `main.tex`；2026-11-12 由 CAPGuard 改名，`5bf6f6f7`）。巨集名、條件名（`color3d_scale2p5_pexels` 等）與遠端資料目錄沿用舊名，不改。
- 標題（iamshuai 定）：「When Noise Fades, Color Remains: Purification-Resistant Image Immunization via Color Mapping」。
- 論文版本：2.5 倍角度上界 `color3d_scale2p5_pexels`（極角 < 25°、色相 < 7.5°），33³ RGB LUT、灰軸為極軸、σ = 4 格點高斯平滑、‖c‖ 不變、損失為 ip2p VAE 編碼器後驗平均差、Adam 1800 步、每張中位 9.5 分鐘（RTX 3090，306 張）。
- 受保護的圖一律寫 immunized image（正文、caption、表頭）；未防禦寫 undefended。圖內文字由 baseline 重出時一併統一（Fig. 1、Fig. 4 已改）。
- **主張**：施加在色彩上的擾動在淨化後被保留，加性擾動被淨化移除。

## 4. 現況（2026-11-17）

### 4.1 同步狀態
- repo 與 Overleaf 最後逐字一致於 `677f2f54`（同步前線上 14 個文字檔皆等於前一同步點；文字檔以編輯器詞級修改套用，重新載入後雜湊相符；`fig/teaser.pdf` 上傳覆寫；線上編譯 10 頁、0 錯誤，第 9 頁由 LLM Usage Statement 開始）。下次同步以 `677f2f54` 為三方合併基底。
- 其後 repo 對 `docs/paper/` 的修改：`6f163b5d`（`X_suppl_purification.tex` caption 的 defended → immunized），尚未同步。
- **頁數以 Overleaf 編譯為準**：Overleaf 用 pdfLaTeX＋Times，本機 Tectonic 用 TeX Gyre 替代字型，斷行不同（Fig. 3 寬 1.0\textwidth 時本機 Conclusion 收在第 8 頁、線上溢到第 9 頁 4 行；故 `1ada1c6f` 改為 0.85\textwidth）。本機預覽只作參考。

### 4.2 結構與圖表
正文 8 頁，Conclusion 在第 8 頁；第 9 頁起為 LLM Usage Statement 與參考文獻（依 CVPR 2027 Author Guidelines，LLM 聲明不計入 8 頁）。表格 caption 一律在表格上方（author kit `sec/2_formatting.tex`），圖 caption 在圖下方。

| 節 | 內容 | 撰寫者 |
|---|---|---|
| Abstract | 含淨化後位移倍數「1.32 to 3.25 times」（表 1 Purified* 欄，ImmuLUT .364 對 BlurGuard .275、DCT-Shield .112） | Lynn／教授；倍數句由 paper 加入 |
| 1 Introduction | 背景、加性擾動、淨化、ImmuLUT、貢獻 | Lynn／教授 |
| 2 Related Work | 含 Fig. 2 方法總覽（`fig/method_overview.pdf`） | Lynn／教授 |
| 3 Method | 3.1 Threat Model、LUT 參數化、3.3 Bounded Angular Displacement、目標；表 1 主表原始碼在 `3_method.tex` | Lynn；3.3 兩句為 paper |
| 4.1 Setup | Data and Editing Setup、Compared Baselines、Purification Operations | Lynn |
| 4.2 | Quantitative（表 1）、ISR（表 2，306 張人工複核版）、Distortion–Effectiveness（Fig. 3）、Qualitative（Fig. 4 woman_111 police suit） | Lynn；Qualitative 段為 paper |
| 4.3 Cross-Model | 表 3（單欄、只列平均、欄名 Purif.*／VAE）；段尾指向 supp 的 inpainting 結果 | Lynn；末句為 paper |
| 4.4 Persistence | Fig. 5（擾動存續） | Lynn |
| 4.5 Ablation Study | 數字在 supp（`tab:ablation`、`fig:ablation_appearance`） | Lynn |
| 5 Limitations、6 Conclusion | 一段連貫敘事；無未來展望 | Lynn／教授 |
| 7 LLM Usage Statement | Claude Code、Claude Sonnet 5.5 評審並由作者抽樣複核、teaser 人像由 FLUX.2 [klein] 4B 生成 | paper |
| supp（`X_suppl.tex`，`main.tex` 中 `\input` 註解掉） | 協定細節、消融表與外觀圖、補充指標表、逐道淨化表（`X_suppl_purification.tex`，由腳本產生） | paper |

圖：Fig. 1 teaser（Mist、DIA-PT、ImmuLUT，JPEG 30，生成人像 man_x 殭屍）、Fig. 2 方法總覽、Fig. 3 失真—位移（(a) 兩軸自 0.2 起）、Fig. 4 質性圖、Fig. 5 擾動存續。

### 4.3 待使用者決定
1. tex 中的舊文註解（`3_method` 45 行、`4_experiments` 30 行、`1_intro` 13 行等）是否刪除；先問 Lynn 是否保留，刪除後需同步 Overleaf。
2. 消融段數字（使用者指示先不動）。
3. supp 的 inpainting 節：baseline 讀數完成後補寫（`baseline/results/pexels_portraits/inpaint/`）。
4. `generate_main_table.py` 仍輸出含 VIBE／klein 下段的舊版表 1，與現行表 1（只有 InstructPix2Pix）與表 3 不符；重跑前需先改寫。

## 5. 寫作規範（使用者 2026-11-06 至 11-17 的決定）

- **Lynn 與教授（iamshuai）寫的文字一律不改**，論調與 paper 定調不同也不改、不提建議；只有使用者或 executor 明確指定某一句時才改，且只改被指定的句子。被 Lynn 註解掉的內容視為不存在，不補回。
- 用詞：persist／survive／is preserved；位移寫成與未防禦編輯的差異（edit difference）。
- 不寫 ISR 絕對高低的判語；改寫成與其他方法的比較。
- 區間一律用 ± 半寬；差值寫成 %。
- 正文每段只留趨勢與一至兩個代表數字，其餘交給表格。
- 內部分析不入正文：灰階 LPIPS 比值、LPIPS 對 ISR 的 AUC、複核前後的變化、單一標註者。評審寫成「Claude Sonnet 5.5 with human review of a sample of the judgments」。
- VAE-Trans 描述為「a pretrained VAE-based denoiser」，不寫訓練細節。
- caption 簡短，協定細節放正文；方法名在圖表標「ImmuLUT (Ours)」，正文不加。
- 表格：固定字級、不用 `\resizebox`；各區塊最佳平均粗體；本方法列底色 `oursrow`。圖：STIX／texgyretermes；加性方法紅 `#A94438`、本方法 teal `#1F6F7C`；Fig. 5 用 Okabe-Ito（其他加性灰、BlurGuard 藍 `#0072B2`、DCT-Shield-Y 橙 `#E69F00`）。
- 暫代數字用 `\prelim{值}{註解}`、`\prelimnote{註解}`，未寫內容用 `\TODO{}`；投稿前設 `\prelimfinaltrue`。
- 正文上限 8 頁；全文不放 URL；commit message 英文，內容、搬檔與文件分開 commit。

## 6. 檔案與工具

| 路徑 | 內容 |
|---|---|
| `docs/paper/` | tex 正本（CVPR 2027 author kit；`cvpr.sty`、`ieeenat_fullname.bst`、`preamble.tex` 不修改）。`main.tex` 定義 `\capg`、`\prelim`、`\prelimnote`、`oursrow` |
| `docs/paper_tools/run_paper_preview.py` | Tectonic 編譯，成功後複製到桌面 `C:\Users\nelso\OneDrive\桌面\paper.pdf` |
| `docs/paper_tools/generate_per_purification_table.py` | supp 逐道淨化表；三組輸入合併（9 臂、BlurGuard 306 張、ImmuLUT 2.5 倍），重出結果與現行 `X_suppl_purification.tex` 逐字相同（`6f163b5d`） |
| `docs/paper_tools/generate_main_table.py` | 表 1（`--immulut <color3d_scale2p5/main_table.csv>`）；見 4.3 第 4 項 |
| `docs/paper_tools/generate_paper_figures.py` | Fig. 3（`distortion_displacement.pdf`）、Fig. 5（`purification.pdf`） |
| `docs/paper_tools/import_pexels_figure_inputs.py` | 把 baseline 的 Pexels 讀數轉成 `generate_paper_figures.py` 的輸入（遠端 `~/paper_figures/in_pexels/`） |
| `docs/paper_tools/generate_pexels_teaser_figure.py` | 舊版 Fig. 1；現行 teaser 由 baseline 產生（`baseline/artifacts/teaser_paper_format/`） |
| `docs/paper_tools/generate_method_overview_panels.py`、`method_overview/` | Fig. 2 |
| `docs/paper_tools/generate_overleaf_ops.py` | Overleaf 同步（第 2 節） |

## 7. 工作方式

- 每次修改：Edit 改 tex → `python docs/paper_tools/run_paper_preview.py` → 檢查 `build/main.log` 的 Overfull、未定義與重複 label → 以 `\color{blue}` 標出改動另編一版 PDF 放 scratchpad 開給使用者 → commit → 回報 executor。頁數最後以 Overleaf 編譯確認。
- 改 tex 或腳本用 Edit，或在 scratchpad 以 Write 建 `.py`（`assert s.count(old) == 1`）。不要用 bash heredoc 寫含反斜線的 Python（`\\t` 會變成 tab、`\\` 會變 `\`）。
- 覆蓋遠端腳本前先 diff，確認遠端沒有 repo 以外的修改；一律先 scp 成 `.tmp` 再 `mv`。
- Fig. 3、Fig. 5 重出（遠端 `~/paper_figures`）：`~/venvs/wacv/bin/python in/generate_paper_figures.py --points in_pexels/points_blurguard.csv --aligned in_pexels/condition_summary_blurguard.csv --color-summary in_pexels/color_summary_scale2p5.csv --survival in_pexels/perturbation_survival_scale2p5.csv --out-dir <目錄>`，再 scp 兩個 PDF 到 `docs/paper/fig/`。重出後與現行圖逐像素比對，確認只改了預期之處。
- supp 逐道淨化表重出（遠端 `baseline/runtime/pexels_portraits/`）：`--per-purifier summary/per_purifier.csv blurguard_306/per_purifier.csv summary/color3d_scale2p5/per_purifier.csv --main-table summary/main_table.csv blurguard_306/summary.csv summary/color3d_scale2p5/main_table.csv --out <檔>`。
- 方法圖：`docs/paper_tools/method_overview/method_overview.tex` 於該目錄以 `tectonic -X compile` 編譯，PDF 複製到 `docs/paper/fig/`。
- supp 檢查：在 scratchpad 複製 `docs/paper/`，取消 `main.tex` 中 `\input{sec/X_suppl}` 的註解後編譯。

## 8. 讀數來源

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15；VAE-Trans 另列；306 張 Pexels 人像。

| 內容 | 來源 |
|---|---|
| 9 臂主表 | baseline 遠端 `runtime/pexels_portraits/summary/main_table.csv`、`per_purifier.csv` |
| ImmuLUT 2.5 倍 | baseline 遠端 `runtime/pexels_portraits/summary/color3d_scale2p5/`（`main_table.csv`、`per_purifier.csv`） |
| BlurGuard 306 張 | baseline 遠端 `runtime/pexels_portraits/blurguard_306/`（`summary.csv`、`per_purifier.csv`） |
| ISR（表 2） | `docs/isr_pilot/audit206/summary_audited.txt`、`paired_difference_audited.txt`（306 張 × 4 句，人工複核） |
| VIBE、klein | `baseline/results/pexels_portraits/vibe_distilled/`、`klein_distilled/` |
| 消融 | `X_suppl.tex` 的 `tab:ablation`（200 張） |
| 求解時間 | color trial `logs/*.log`（中位 9.53、p90 10.20 分鐘，n = 306） |
