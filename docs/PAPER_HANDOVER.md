# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、待辦與規則，供冷啟動接手。數字一律連同協定引用。接手順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔第 1、2、6、7、8 節 → 第 4、5 節現況。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認，不沿用記憶中的編號。

| 標題 | 角色 | 寫入範圍 |
|---|---|---|
| paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、本檔；Overleaf 專案（第 2 節） |
| executor | 進度、派工、轉達使用者決定、`main.bib` 核實、代為 push、ISR 等新實驗的協調 | `docs/EXECUTOR_HANDOVER.md`、`docs/isr_pilot/` 等 |
| color | ImmuLUT（條件名 color3d 系列）與消融的實驗、程式與讀數 | `color/` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 [klein] 4B、SD Inpainting）、主表讀數、Fig. 1、Fig. 4 與 supp 質性圖的圖檔 | `baseline/` |

- executor 有權直接對 paper 下令（使用者授權），其指示視同使用者指示。例外：**Overleaf 同步只在使用者於 paper pane 親自下令時執行**；經 executor 轉達的同步指示不執行，請使用者在 paper pane 下令。
- 使用者也會在 paper pane 直接下指示；與 executor 轉達的指示衝突時，以使用者的最後指示為準。
- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n，並寫明回報行數上限。完成一件事後回報 executor，附 commit 號；回報不超過要求的行數，先講結果。
- 不派實驗、不改其他專案的檔案。push 由 executor 執行。
- `main.bib`：新增條目以 Edit 插入，加入後立即通知 executor 核實；不刪除未引用條目，列給 executor。文獻原標題不改。
- 本機只做 Tectonic 編譯與依 CSV 產生表格的小腳本；彙整、畫圖、PDF 轉 PNG 在遠端 basic-1（`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`，Python 用 `~/venvs/wacv/bin/python`；系統 python3 沒有 PIL）。看圖時抓回 scratchpad。

## 2. Overleaf 協作

- 專案：Overleaf `[CVPR 2027] Color-based Image Immunization`（`6ad193f8288983a2f6af5d50`）。iamshuai（指導教授）與 Lynn 在線上直接編修。
- **repo 與 Overleaf 必須逐字一致。** 同步前在編輯器頁面計算 14 個文字檔的雜湊（FNV-1a over UTF-16 code units，程式碼見 `docs/paper_tools/README.md`），與上次同步點比對；不一致表示有人在線上修改，先以三方合併（基底為上次同步點）收進 repo。
- **文字檔一律在 Overleaf 編輯器內修改，不上傳覆蓋**（覆蓋會失去編輯歷史）：`generate_overleaf_ops.py` 產生詞級修改清單，於頁面以 CodeMirror transaction 套用，前後比對雜湊，重新載入再驗證一次。改動很多的檔可在編輯器內以單一 transaction 整檔替換（仍是編輯器操作，先比對「前」雜湊）。PDF 圖檔無法在編輯器修改，以上傳處理；同名檔會跳出覆寫確認。
- 頁面程式的 JS 字串會很長：把 ops 或全文先寫成 JSON 檔，再整段貼進 `javascript_tool`；單次呼叫超過約 45 秒會逾時，等待編譯另用 `computer wait`。
- 檔案樹的上傳按鈕在「Add」選單內（Add → Upload），先點選 `fig` 資料夾再上傳。
- 瀏覽器：使用者以 `/chrome` 連線 Claude in Chrome；只操作自己開的分頁群組。擴充功能會擋下看似網址參數的輸出，此時改用截圖。線上 PDF 檢視器預設不顯示時，按 "Switch to PDF"；頁面文字層需捲動到該頁後才能讀取。
- 線上若有 repo 沒有的檔案（現有 `sec/[old] related.tex` 與根目錄的 `build` 資料夾），不動。

## 3. 方法與定調

- 方法名 **ImmuLUT**（巨集 `\capg`，定義在 `main.tex`；由 CAPGuard 改名，`5bf6f6f7`）。巨集名、條件名（`color3d_scale2p5_pexels` 等）與遠端資料目錄沿用舊名，不改。
- 標題（iamshuai 定）：「When Noise Fades, Color Remains: Purification-Resistant Image Immunization via Color Mapping」。
- 論文版本：2.5 倍角度上界 `color3d_scale2p5_pexels`（極角 < 25°、色相 < 7.5°），33³ RGB LUT、灰軸為極軸、σ = 4 格點高斯平滑、‖c‖ 不變、損失為 ip2p VAE 編碼器後驗平均差、Adam 1800 步（lr 3e-4 → 1e-5）、每張中位 9.5 分鐘（RTX 3090，306 張）。
- 受保護的圖一律寫 immunized image（正文、caption、表頭）；未防禦寫 undefended。
- **主張**：施加在色彩上的擾動在淨化後被保留，加性擾動被淨化移除。

## 4. 現況

### 4.1 同步狀態
- repo 與 Overleaf 最後逐字一致於 `ba5a8de8`：同步前線上只有 `X_suppl.tex` 與 `7ec63889` 不同（羅婈修訂兩處，已以三方合併收入 `74988b9e`）；`X_suppl.tex` 以詞級修改套用；新上傳 `fig/inpaint_woman_118_snowy_street.pdf`，覆蓋 `ablation_appearance.pdf`、`inpaint_man_26_dog.pdf`、`inpaint_woman_14_hospital.pdf`、`inpaint_man_81_dog.pdf`。線上編譯 16 頁、0 錯誤。下次同步以 `ba5a8de8` 為三方合併基底。線上的 `fig/inpaint_woman_39_gym.pdf` 已不被引用，保留未刪。補充材料 PDF：basic-1 以 pdflatex 編譯整份（16 頁，Conclusion 在第 8 頁）後擷取第 11–16 頁，存為桌面 `paper_supp.pdf`（6 頁）。
- `7ec63889` 同步時：同步前線上 14 個文字檔皆等於 `04ab6494`；`X_suppl.tex`（inpainting ISR 段與表 `tab:supp_inpaint_isr`、兩張質性圖）與 `7_llm_usage.tex`（ip2p 有人工複核、inpainting 無）以詞級修改套用；新上傳 `fig/inpaint_woman_39_gym.pdf`、`fig/inpaint_man_81_dog.pdf`。線上編譯 16 頁、0 錯誤（正文未改）。下次同步以 `7ec63889` 為三方合併基底。
- `04ab6494` 同步時：同步前線上 14 個文字檔皆等於 `f0cd3cee`，`4_experiments.tex` 以詞級修改套用（消融段 LAION、MUSIQ 換成 306 張版），線上編譯 14 頁、0 錯誤。
- `f0cd3cee` 同步時：線上 14 個文字檔皆等於前一同步點 `677f2f54`；`main.tex`（啟用 `\input{sec/X_suppl}`，supp 接在正文後）與 `X_suppl_purification.tex` 以詞級修改套用，`X_suppl.tex` 整檔替換；新上傳 `fig/inpaint_man_26_dog.pdf`、`fig/inpaint_woman_14_hospital.pdf`。線上編譯 14 頁、0 錯誤：正文第 1–8 頁、Conclusion 在第 8 頁，第 9 頁起為 LLM Usage Statement 與參考文獻，supp 接在其後。
- repo 有而線上沒有的檔：`sec/X_suppl_parked.tex`（未被 `\input`）、`fig/qualitative_woman_171_police.pdf`、`fig/qualitative_woman_173_bowtie.pdf`、`fig/qualitative_man_91_police.pdf`（只被 parked 檔引用）；baseline 送來而依使用者指示不引用的 inpainting 圖 `fig/inpaint_woman_138_hospital.pdf`、`inpaint_man_51_hospital.pdf`、`inpaint_man_07_gym.pdf`、`inpaint_woman_146_snowy_street.pdf`、`inpaint_woman_87_snowy_street.pdf`、`inpaint_woman_27_dog.pdf`（只在 repo，未上傳）。
- 線上根目錄的 `build/` 只含 `main.pdf`（編譯產物，未被引用）；executor 轉達刪除，刪除動作被 Claude Code 權限檢查擋下，現仍保留。
- **頁數以 Overleaf 編譯為準**：Overleaf 用 pdfLaTeX＋Times，本機 Tectonic 用 TeX Gyre 替代字型，斷行不同（Fig. 3 寬 1.0\textwidth 時本機 Conclusion 收在第 8 頁、線上溢到第 9 頁 4 行；故 `1ada1c6f` 改為 0.85\textwidth）。本機預覽只作參考。

### 4.2 正文結構與圖表
正文 8 頁，第 8 頁無剩餘行。表格 caption 在表格上方（author kit `sec/2_formatting.tex`），圖 caption 在圖下方。

| 節 | 內容 | 撰寫者 |
|---|---|---|
| Abstract | 含淨化後位移倍數「1.32 to 3.25 times」（表 1 Purified* 欄，ImmuLUT .364 對 BlurGuard .275、DCT-Shield .112） | Lynn／教授；倍數句由 paper 加入 |
| 1 Introduction | 背景、加性擾動、淨化、ImmuLUT、貢獻 | Lynn／教授 |
| 2 Related Work | 含 Fig. 2 方法總覽（`fig/method_overview.pdf`） | Lynn／教授 |
| 3 Method | Threat Model、LUT 參數化、3.3 Bounded Angular Displacement、目標；表 1 主表原始碼在 `3_method.tex` | Lynn；3.3 兩句為 paper |
| 4.1 Setup | Data and Editing Setup、Compared Baselines、Purification Operations | Lynn |
| 4.2 | Quantitative（表 1）、ISR（表 2，306 張人工複核版）、Distortion–Effectiveness（Fig. 3）、Qualitative（Fig. 4 woman_111 police suit） | Lynn；Qualitative 段為 paper |
| 4.3 Cross-Model | 表 3（單欄、只列平均、欄名 Purif.*／VAE）；段尾指向 supp 的 inpainting 結果 | Lynn；末句為 paper |
| 4.4 Persistence | Fig. 5（擾動存續） | Lynn |
| 4.5 Ablation Study | 文字段，數字表在 supp | Lynn |
| 5 Limitations、6 Conclusion | 一段連貫敘事；無未來展望 | Lynn／教授 |
| 7 LLM Usage Statement | Claude Code、Claude Sonnet 5.5 評審並由作者抽樣複核、teaser 人像由 FLUX.2 [klein] 4B 生成 | paper |

圖：Fig. 1 teaser（Mist、DIA-PT、ImmuLUT，JPEG 30，生成人像 man_x 殭屍）、Fig. 2 方法總覽、Fig. 3 失真—位移（(a) 兩軸自 0.2 起，無斷軸記號）、Fig. 4 質性圖、Fig. 5 擾動存續。

### 4.3 supp 結構（`sec/X_suppl.tex`）
使用者指示 supp 只保留兩節，其餘「先記著、不寫」。

- **A. Ablation Study**（`sec:supp_ablation`）：開頭段寫共用求解器；`Component Variants` 與 `Alternative Color Mappings` 兩段逐組寫設定（A1 隨機 Θ、A2 σ = 0、A3 無角度上界、A4 單一 25° 上界；B1 隨機色相／飽和度，以二分對齊 ±0.005、34 張達不到目標；B2 最佳化色相／飽和度、B3 3×3 仿射、B4 無約束 33³ LUT，只留防禦圖 LPIPS 在 ImmuLUT ±0.015 內的檢查點）；`Effect of Optimization and Geometric Constraints`、`Comparison with Alternative Color Mappings` 兩段為結果，數字取自表 `tab:ablation`。圖 `fig:ablation_appearance`（`fig/ablation_components.pdf`、`fig/ablation_appearance.pdf`，viewport 裁切）。
  - 表 `tab:ablation`：306 張，各列用自己的 n 與均值（Full 306、A4 305、B2 303、B3 305，其餘 306；不可行影像寫在 caption：A4 man_33，B2 woman_44／82／129，B3 man_99）。來源 color 遠端 `~/image-immunization/color/trials/lut3d_defense_and_naturalness/results/ablation_pexels306/summary.csv`（subset `all`）。變體設定的出處見同 trial 的 `generate_color3d_ablation_*`、`generate_color_baseline_*` 與 `conds/`。
  - 成對差表已依使用者指示刪除；資料在同目錄 `paired_differences.csv`（共同 301 張，paired bootstrap 10,000 次）。
- **B. Cross-Editor Evaluation on SD Inpainting**（`sec:supp_inpainting`）：`Experimental Setup`（`runwayml/stable-diffusion-inpainting`，與 ip2p 共用 VAE、不重新求解；CLIPSeg 主體遮罩的補集，只重繪背景並合成回原圖；50 步、guidance 7.5、strength 1.0、主種子；DiffusionGuard 四句；PhotoGuard、DIA-PT、DCT-Shield、SIFM、BlurGuard 等失真臂；none、JPEG 50、blur σ2、VAE-Trans）與 `Quantitative and Qualitative Comparison`。表 `tab:supp_inpainting`（來源 `baseline/results/pexels_portraits/inpaint/summary.csv`，306 張完整）。圖 `fig:supp_inpaint_man_26`（dog、VAE-Trans）、`fig:supp_inpaint_woman_14`（hospital、JPEG 50）；baseline 在 27 張候選中只找到這兩張符合「加性方法淨化後出現指令內容、ImmuLUT 沒有」，gym 與 snowy street 沒有符合者。
- **`sec/X_suppl_parked.tex`**（不被 `\input`，內容與數字保留，要放回時直接取用）：
  1. Protocol details：資料挑選六步驟、VIBE／FLUX 設定、淨化實作細節、等失真校準（8 張校準集上平均防禦圖 LPIPS 等於 0.3344，套用到 306 張）。
  2. Baseline provenance：各加性方法原生預算、ε 乘數、實作來源（來源 `baseline/docs/reference/BASELINE_PROVENANCE.md`；DAYN、DANP、DCT-Shield 的步長與範數標為 --，待核）。
  3. Additional metrics：防禦圖品質表（BlurGuard 列取自 `baseline/results/pexels_portraits/blurguard/defense_quality.csv`）、編輯端指標表（SSIM、PSNR、CLIP-I、CLIP-S、FID、KID，七道平均）、`\input{sec/X_suppl_purification}` 的兩張逐道淨化表。
  4. Immunization success rate：306 張完整表、各指令表、失敗標籤表、成對差（含 McNemar）、Claude Sonnet 5.5 評審與人工複核流程（來源 `docs/isr_pilot/audit206/summary_audited.txt`、`paired_difference_audited.txt`、`agreement.txt`、`docs/isr_pilot/rubric.md`）。
  5. Evaluation validity：未防禦編輯失敗比例（依淨化與依指令）。
  6. More qualitative examples：woman_171 police（VAE-Trans）、woman_173 bowtie（blur σ2）、man_91 police（VAE-Trans），圖檔已在 `fig/qualitative_*.pdf`。

### 4.4 待完成與待決
1. OpenReview 補充材料上傳截止 2026-11-23 19:59（臺灣時間）。補充材料 PDF 需從 supp 單獨產出（正文與 supp 分開上傳時，依 `main.tex` 中 author kit 的註解刪除 supp 頁）。
2. 正文 4.5 消融段的數字已依使用者指示換成 306 張版（Full LAION 5.34／MUSIQ 5.31、B3 5.27／4.81，`04ab6494`）。
3. tex 中的舊文註解（`3_method` 約 45 行、`4_experiments` 約 30 行、`1_intro` 約 13 行）是否刪除；先問 Lynn，刪除後需同步 Overleaf。
4. `generate_main_table.py` 仍輸出含 VIBE／klein 下段的舊版表 1，與現行表 1（只有 InstructPix2Pix）、表 3 不符；重跑前需先改寫。
5. supp inpainting 節若要更多質性圖，需放寬挑選條件，向 baseline 提出。

## 5. 寫作規範

### 5.1 全文
- **Lynn 與教授（iamshuai）寫的文字一律不改**，論調與 paper 定調不同也不改、不提建議；只有使用者或 executor 明確指定某一句時才改，且只改被指定的句子。被 Lynn 註解掉的內容視為不存在，不補回。
- 用詞：persist／survive／is preserved；位移寫成與未防禦編輯的差異（edit difference）。
- 不寫 ISR 絕對高低的判語；改寫成與其他方法的比較。
- 區間一律用 ± 半寬；差值寫成 %。
- 正文每段只留趨勢與一至兩個代表數字，其餘交給表格。
- 內部分析不入正文：灰階 LPIPS 比值、LPIPS 對 ISR 的 AUC、複核前後的變化、單一標註者。評審寫成「Claude Sonnet 5.5 with human review of a sample of the judgments」。
- VAE-Trans 描述為「a pretrained VAE-based denoiser」，不寫訓練細節。
- caption 簡短，協定細節放正文；方法名在圖表標「ImmuLUT (Ours)」，正文不加。
- 表格：固定字級、不用 `\resizebox`；有數字且有方向的欄位，最佳值粗體（計數表、預算來源表等沒有「最佳」的表不加）；本方法列底色 `oursrow`。圖：STIX／texgyretermes、印刷字級 ≥ 7 pt（數學上下標除外）；加性方法紅 `#A94438`、本方法 teal `#1F6F7C`；Fig. 5 用 Okabe-Ito（其他加性灰、BlurGuard 藍 `#0072B2`、DCT-Shield-Y 橙 `#E69F00`）。
- 暫代數字用 `\prelim{值}{註解}`、`\prelimnote{註解}`，未寫內容用 `\TODO{}`；投稿前設 `\prelimfinaltrue`。
- 正文上限 8 頁；全文不放 URL（GitHub repo 名也不寫，只寫 commit 號）；commit message 英文，內容、搬檔與文件分開 commit。

### 5.2 supp（使用者在 paper pane 的決定）
- 寫作風格與正文一致：we、現在式；節標題與段名用 Title Case，段名比照正文 4.5（Effect of …、Comparison with …）；結果段以「As shown in Tab. …」開頭；圖說以「Edits with the prompt … on …」開頭。
- 與正文重複的內容不寫（指令、ip2p 設定、淨化清單、等失真 LPIPS 範圍、ISR 定義等），supp 只寫正文沒有的細節與結果。
- 版面依 CVPR 模板雙欄，不逐節換頁（逐節換頁會變成多頁只有幾行字）；單欄小表併成一個 `table*` 並排，以免單獨占頁。supp 開頭放寬浮動體參數，只作用於 supp。
- 節號用字母（A、B…）。
- 消融要逐組寫清楚設定；inpainting 寫設定、結果與質性圖。

## 6. 檔案與工具

| 路徑 | 內容 |
|---|---|
| `docs/paper/` | tex 正本（CVPR 2027 author kit；`cvpr.sty`、`ieeenat_fullname.bst`、`preamble.tex` 不修改）。`main.tex` 定義 `\capg`、`\prelim`、`\prelimnote`、`oursrow`，並 `\input{sec/X_suppl}` |
| `docs/paper/sec/X_suppl.tex`、`X_suppl_parked.tex` | supp 現行兩節；暫置的六節（第 4.3 節） |
| `docs/paper/sec/X_suppl_purification.tex` | 由腳本產生的逐道淨化表，只被 parked 檔引用 |
| `docs/paper_tools/run_paper_preview.py` | Tectonic 編譯，成功後複製到桌面 `C:\Users\nelso\OneDrive\桌面\paper.pdf` |
| `docs/paper_tools/generate_per_purification_table.py` | 逐道淨化表；三組輸入合併（9 臂、BlurGuard 306 張、ImmuLUT 2.5 倍），每欄最佳值粗體 |
| `docs/paper_tools/generate_main_table.py` | 表 1（`--immulut <color3d_scale2p5/main_table.csv>`）；見 4.4 第 4 項 |
| `docs/paper_tools/generate_paper_figures.py` | Fig. 3（`distortion_displacement.pdf`）、Fig. 5（`purification.pdf`） |
| `docs/paper_tools/import_pexels_figure_inputs.py` | 把 baseline 的 Pexels 讀數轉成 `generate_paper_figures.py` 的輸入（遠端 `~/paper_figures/in_pexels/`） |
| `docs/paper_tools/generate_method_overview_panels.py`、`method_overview/` | Fig. 2（TikZ，`tectonic -X compile`） |
| `docs/paper_tools/generate_overleaf_ops.py` | Overleaf 同步（第 2 節） |

| 圖檔 | 產生者與來源 |
|---|---|
| `fig/teaser.pdf`、`fig/purification_example.pdf` | baseline（`baseline/artifacts/teaser_paper_format/teaser_zombie_compact_mist.pdf`、`baseline/artifacts/pexels_portraits/purification_example/woman_111_p1.pdf`） |
| `fig/method_overview.pdf` | paper，`docs/paper_tools/method_overview/` |
| `fig/distortion_displacement.pdf`、`fig/purification.pdf` | paper，`generate_paper_figures.py` |
| `fig/ablation_components.pdf`、`fig/ablation_appearance.pdf` | color（man_37 消融外觀） |
| `fig/inpaint_man_26_dog.pdf`、`fig/inpaint_woman_14_hospital.pdf` | baseline（`baseline/artifacts/pexels_portraits/inpaint_example/man_26_p3_vae_trans.pdf`、`woman_14_p0_jpeg50.pdf`） |
| `fig/qualitative_*.pdf` | baseline（`baseline/artifacts/pexels_portraits/purification_example/woman_171_p1_vae_trans.pdf`、`woman_173_p3.pdf`、`man_91_p1_vae_trans.pdf`） |

## 7. 工作方式

- 每次修改：Edit 改 tex → `python docs/paper_tools/run_paper_preview.py` → 檢查 `build/main.log` 的 Overfull、未定義與重複 label → 需要時以 `\color{blue}` 標出改動另編一版 PDF 放 scratchpad 開給使用者 → commit → 回報 executor。頁數最後以 Overleaf 編譯確認。
- 改 tex 或腳本用 Edit，或在 scratchpad 以 Write 建 `.py`（`assert s.count(old) == 1`）。**不要用 bash heredoc 寫含反斜線的 Python**：`\r`、`\t` 會被解成控制字元（`\renewcommand` 曾變成 CR＋`enewcommand`），`\\` 會變 `\`。
- 覆蓋遠端腳本前先 diff，確認遠端沒有 repo 以外的修改；一律先 scp 成 `.tmp` 再 `mv`。
- git index 與其他 session 共用：`git mv`／`git rm` 暫存後立即 commit，否則會被其他 session 的 commit 一併帶走（`59f47556` 即含 paper 的改名與刪檔）。`git commit -- <paths>` 只提交列出的路徑，已暫存的其他變更會留在 index。
- Fig. 3、Fig. 5 重出（遠端 `~/paper_figures`）：`~/venvs/wacv/bin/python in/generate_paper_figures.py --points in_pexels/points_blurguard.csv --aligned in_pexels/condition_summary_blurguard.csv --color-summary in_pexels/color_summary_scale2p5.csv --survival in_pexels/perturbation_survival_scale2p5.csv --out-dir <目錄>`，再 scp 兩個 PDF 到 `docs/paper/fig/`。重出後與現行圖逐像素比對（`mutool draw` 轉 PNG 後以 PIL `ImageChops.difference`），確認只改了預期之處。
- 逐道淨化表重出（遠端 `baseline/runtime/pexels_portraits/`）：`--per-purifier summary/per_purifier.csv blurguard_306/per_purifier.csv summary/color3d_scale2p5/per_purifier.csv --main-table summary/main_table.csv blurguard_306/summary.csv summary/color3d_scale2p5/main_table.csv --out <檔>`。
- 方法圖：`docs/paper_tools/method_overview/method_overview.tex` 於該目錄以 `tectonic -X compile` 編譯，PDF 複製到 `docs/paper/fig/`。
- 看版面：遠端 `mutool draw -F txt` 逐頁找 `Table n.`／`Figure n.` 判斷浮動體所在頁；`mutool draw -F stext` 可讀字級。

## 8. 讀數來源

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15；VAE-Trans 另列；306 張 Pexels 人像。

| 內容 | 來源 |
|---|---|
| 9 臂主表 | baseline `results/pexels_portraits/main_table/main_table.csv`（遠端 `runtime/pexels_portraits/summary/` 同值）、`per_purifier.csv` |
| ImmuLUT 2.5 倍 | baseline `results/pexels_portraits/color3d_scale2p5/`（`main_table.csv`、`per_purifier.csv`） |
| BlurGuard 306 張 | baseline `results/pexels_portraits/blurguard/`（`summary.csv`、`per_purifier.csv`、`defense_quality.csv`） |
| ISR（表 2） | `docs/isr_pilot/audit206/summary_audited.txt`、`paired_difference_audited.txt`（306 張 × 4 句，人工複核） |
| VIBE、klein | `baseline/results/pexels_portraits/vibe_distilled/`、`klein_distilled/` |
| SD Inpainting | `baseline/results/pexels_portraits/inpaint/summary.csv` |
| 消融（supp，306 張） | color 遠端 trial `results/ablation_pexels306/`（`summary.csv`、`paired_differences.csv`、`failed.csv`） |
| 求解時間 | color trial `logs/*.log`（中位 9.53、p90 10.20 分鐘，n = 306） |
