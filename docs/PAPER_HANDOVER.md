# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、待辦與規則，供冷啟動接手。數字一律連同協定引用。`docs/PAPER_STRUCTURE.md` 的逐節狀態未隨 2026-11-07 之後的修改更新，以本檔第 3 節與第 10 節為準。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認，不沿用記憶中的編號。

| 標題 | 角色 | 寫入範圍 |
|---|---|---|
| paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、本檔；Overleaf 專案（第 2 節） |
| executor | 進度、派工、轉達使用者決定、`main.bib` 核實、代為 push、ISR 等新實驗的協調 | `docs/EXECUTOR_HANDOVER.md`、`docs/isr_pilot/` 等 |
| color | ImmuLUT（條件名 color3d 系列）的實驗、程式與讀數 | `color/` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 klein）、主表讀數 | `baseline/` |

- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n。完成一件事後回報 executor，附 commit 號。
- 不派實驗、不改其他專案的檔案。
- `main.bib`：新增條目以 Edit 插入，加入後立即通知 executor 核實；不刪除未引用條目，列給 executor。
- 本機只做 Tectonic 編譯與依 CSV 產生表格的小腳本；彙整、畫圖、PDF 轉 PNG 在遠端 basic-1（`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`，Python 用 `~/venvs/wacv/bin/python`）。看圖時抓回 scratchpad。
- 使用者也會透過 color 或 executor 轉達文字修改；以使用者的最後指示為準。

## 2. Overleaf 協作

- 專案：Overleaf `[CVPR 2027] Color-based Image Immunization`（`6ad193f8288983a2f6af5d50`），由 `docs/paper/` 在 2026-11-06 上傳（基底 commit `51fd87d7`）。iamshuai（指導教授）與 Lynn 在線上直接編修。
- **repo 與 Overleaf 必須逐字一致。** 兩邊都有修改時，以三方合併處理（基底為上次同步點）。repo 最近一次與 Overleaf 完全一致：commit `3be20a56`（2026-11-17，見第 5 節），以雜湊比對確認。線上原文可在編輯器頁面以 JS 讀出 CodeMirror 內容後組成 JSON、以 Blob 下載到「下載」資料夾取得（頁面輸出有長度上限）。
- **線上修改一律在 Overleaf 編輯器內進行，不上傳或覆蓋檔案**（覆蓋會失去編輯歷史）。工具與步驟見 `docs/paper_tools/README.md`「Overleaf 同步」：`generate_overleaf_ops.py` 產生詞級修改清單，於編輯器頁面以 CodeMirror transaction 套用，前後比對雜湊，重新載入再驗證一次。
- 瀏覽器：使用者先以 `/chrome` 連線 Claude in Chrome；只能操作自己開的分頁群組。擴充功能會擋下看似網址參數的輸出，此時改用截圖，不繞過過濾。
- 線上任何修改前先讓使用者看渲染結果：在 scratchpad 複製 `docs/paper/`，以 `\color{blue}` 標出修改處編譯，遠端 `mutool draw -r 170` 轉 PNG，以 `Start-Process` 開給使用者看。使用者確認後才改 Overleaf 與 repo。

## 3. 方法與定調

- 方法名 **ImmuLUT**（`\capg`，定義在 `main.tex`；2026-11-12 由 CAPGuard 改名，commit `5bf6f6f7`）。正文不再展開全名。檔名、條件名（`color3d_scale2p5_pexels` 等）與程式不改名。
- 標題（iamshuai 定）：「When Noise Fades, Color Remains: Purification-Resistant Image Immunization via Color Mapping」。
- 論文版本：2.5 倍角度上界 `color3d_scale2p5_pexels`（極角 < 25°、色相 < 7.5°），33³ RGB LUT、灰軸為極軸、σ = 4 格點高斯平滑、‖c‖ 不變、損失為 ip2p VAE 編碼器後驗平均差、Adam 1800 步、每張中位 9.5 分鐘（RTX 3090，306 張）。
- **主張**：施加在色彩上的擾動在淨化後被保留，加性擾動被淨化移除。主表證明的是「保留」，不是「編輯失敗」。
- 3.3 Bounded Angular Displacement（commit `4bb1f3d7`，依 Lynn 回饋）：只用球座標敘述，不提切平面；Δθ = s(v_θ)、Δφ = s(v_φ / r)、r = sin(θ+Δθ)/cos Δθ 寫在式中；不寫色彩學理由；方位步造成的極角二階誤差（φmax = 7.5° 時約 0.9%）正文不提。程式為 color trial `generate_color3d_defenses.py` 的 `SmoothAngleLut.lut()`。

## 4. 現況（改名後 HEAD `5bf6f6f7`，編譯 10 頁）

Conclusion 在第 8 頁，Table 3（消融）在第 9 頁，正文含圖表超過 8 頁。

| 位置 | 現況 |
|---|---|
| Abstract | 只改了方法名那一句；結尾仍是「1.32 to 3.23 times」（使用者指示先不改） |
| 1 Introduction | 第 3–5 段依新定調改寫；貢獻三點（觀點、方法、評估），第三點含 100 張子集的失敗率 17.0%／16.5% |
| 2 Related Work | Lynn 的文字未動；Fig. 2 caption 的 (b) 左側已改為描述可達區域（commit `cbb4e8cb`） |
| 3 Method | 3.3 如上；Fig. 3（存續比）的 figure 原始碼放在 `3_method.tex`、排在 Fig. 4（失真—位移）之前，使圖號與引用順序一致 |
| 4 Experiments | 開頭一句概述；設定三段 Data and attack／Purification methods／Compared methods；4.1 Results 五段 Perturbation survival（Fig. 3）→ Edit displacement after purification（Table 1）→ Distortion and displacement（Fig. 4）→ Edit failure rate（Table 2）→ Editors with different encoders；4.2 Ablation study 內文為 `\TODO`，Table 3 保留 |
| Table 2（ISR） | 100 張 × 4 句、人工複核後；區間以 ± 半寬表示；底列為 ImmuLUT − BlurGuard |
| Table 3（消融） | Full、A1–A5、B1、B4 的 n、防禦圖 LPIPS、LAION、MUSIQ、四種淨化的編輯 LPIPS 已填；B2（最佳化色相／彩度）、B3（3×3 affine）未跑；使用者決定消融不做 ISR 評審，ISR 欄已移除（commit `45eb1664`） |
| 5 Limitations | 一段連貫敘事：保留不提高保護水準（不淨化時失敗率不高於加性方法）、只在 ip2p 上量、色彩校正（gray-world、Lab 正規化）的擾動存續比與編輯 LPIPS 照實報、註明未評 ISR 與加性方法（commit `45eb1664`）、可見色偏、濫用與緩解 |
| 6 Conclusion | 三句：方法、擾動與編輯位移被保留（主表數字）、失敗率 17.0%／16.5% 對加性 2.6–12.2%；無未來展望句 |
| 7 LLM 聲明 | 含「Claude Sonnet 5.5 評審、作者抽樣複查」，指向 `sec:exp_isr` |
| 圖 | Fig. 1 teaser 為 man_60（已換新方法名）；Fig. 2 方法圖 (b) 左側重畫為球面上的可達區域（commit `dccdfd7f`、`cbb4e8cb`、`4bb1f3d7`，使用者 11/12 定稿），圖中沒有方法名；Fig. 3、4 已用新方法名重產 |

## 5. 同步狀態

- **Overleaf 與 repo 逐字一致的最後一點：commit `3be20a56`**（2026-11-17，Fig. 2 圖說縮短，2_related 詞級套用；此前 `750ac6a7`：K 取代晶格 33、Fig. 1／Fig. 2 圖說移除實驗細節、Setup 補 K、σ、角度上限與 ip2p 設定；2_related、4_experiments、fig/teaser.tex 詞級套用，fig/method_overview.pdf 上傳覆蓋。此前 `04b9930d`（線上 0_abstract、1_intro、3_method、4_experiments（第 4 章重新編排）、5_discussion、6_conclusion 的修改直接收進 repo（`8030cc25`；repo 自上次同步未改這些檔）；7_llm_usage 改引 `tab:isr`（原 `sec:exp_isr` 標籤已被註解）並同步上線；`fig/teaser.tex` 手貼版經雜湊確認一致；全部 12 個文字檔重新載入後雜湊相符，線上編譯 11 頁無錯誤）。此前 `e7b18690`（2026-11-16：教授 11/15 在 Overleaf 改的 1_intro、2_related（僅刪註解）、main.bib 以三方合併收進 repo，200 張消融的 4_experiments 以詞級修改同步上線；此前 `a2da3035`（2026-11-14，check 報告修正、ISR 更正數字、main.bib 標題大括號與作者／頁碼補正以詞級修改套用；此前 `f4b26f1b`（2026-11-13；`8ce868c0` 全面同步後，依序同步 man_37 消融圖（`cc4e1323`）、woman_23 範例圖、新版 teaser 與 LLM 聲明（`f4b26f1b`，`fig/teaser.tex` 亦以詞級修改套用，`fig/teaser.pdf` 覆蓋、`fig/purification_example.pdf` 新增））。文字七檔（`main.tex`、`0_abstract`–`5_discussion`）以編輯器詞級修改套用，重新載入後雜湊與 repo 相同；`fig/` 的 `ablation_appearance.pdf`、`ablation_components.pdf`（新增）與 `teaser.pdf`、`method_overview.pdf`、`purification.pdf`、`distortion_displacement.pdf`（覆蓋）以上傳處理；線上重新編譯 10 頁、無錯誤。
- 此次同步前，Lynn 在 Overleaf 改寫了 `1_intro.tex`（舊文保留為註解）與 `3_method.tex` 第 3、10、19、40、69 行，已以基底 `453a196e` 三方合併進 repo（commit `7f179414`）。下次合併以 `3be20a56` 為基底。
- **同步 Overleaf 必須由使用者在 paper pane 直接下令**；經 executor 轉達的同步指示會被權限分類器擋下。套用前先比對線上雜湊（工作目錄可能是 CRLF，先轉 LF 再算），不一致就停下回報。
- push 由 executor 執行。

## 6. 進行中與待決定

進行中：
1. **Teaser**：woman_75 × helmet 的紅圈放大版草稿已做（腳本 `--highlight 256 105 95 lower-right`，遠端 `~/paper_figures/out_teaser_draft/teaser_woman_75_highlight.pdf`，圖中方法名仍是舊名，採用時需重產）。放大後本方法那格頭頂有一圈薄的帽形痕跡。executor 轉達「使用者已選定、可換入」，但使用者在 paper pane 直接說「teaser 先別放」；`fig/teaser.pdf` 仍是 man_60。換入前向使用者確認。man_17、woman_24 在 blur σ2 下未防禦編輯也沒有安全帽，不適用現行版面。
2. **Table 3**：B2、B3 由 color 補跑，數字經 executor 轉來再填；不做 ISR 評審；B1 的防禦圖 LPIPS 0.236 低於 Full 的 0.299（部分影像到不了目標），逐張清單待 color 提供；共同子集排除規則使用者未表態。
3. **4.2 內文**：`\TODO`。數據齊備後依序寫 Design → 元件的作用 → 簡單顏色映射；B 組若同樣被保留，支持主張，本方法的貢獻落在約束帶來的自然度與可達失真。
4. **壓回 8 頁**：Table 3 落在第 9 頁。可行做法：Table 3 縮欄或改單欄、精簡 Lynn 的 Method 段落（需使用者同意）。
5. **摘要**：線上摘要由使用者以新方法名上傳；結尾數字句與新定調不一致，使用者指示先不改。

待使用者決定：teaser 案例；3.3 改寫後「φ 上界較緊」已無理由句，是否補；de Ridder 1995、Fedorovskaya 1997 兩條 bib 已無引用；`docs/PAPER_STRUCTURE.md` 未隨改寫更新。

## 6a. 寫作規範（使用者 2026-11-06 至 11-12 的決定）

- 用詞：persist／survive／is preserved；不用 prevents／blocks／effective／outperforms（除非附數字與區間）。位移寫成「與未防禦編輯的差異」（edit difference）。
- **不寫 ISR「低」「modest」「in absolute terms」這類絕對高低的判語**；改寫成與其他方法的比較（「不高於加性方法」）。
- 區間一律用 ± 半寬；差值寫成 %。
- 有利與不利的結果都要寫，但正文每段只留趨勢與一至兩個代表數字，其餘交給表格；不逐項複述表中數字。
- 內部分析不入正文：灰階 LPIPS 比值、LPIPS 對 ISR 的 AUC、606 格與複核前後的變化、單一標註者。評審寫成「Claude Sonnet 5.5 with human review of a sample of the judgments」。
- Conclusion 不把 BlurGuard 獨立列出，併入加性方法的範圍；不寫未來展望。
- Limitations 寫成一段連貫敘事，不是逐句列點。
- VAE-Trans 描述為「a pretrained VAE-based denoiser」，不寫訓練細節。
- 被 Lynn 註解掉的內容（3.1 指標定義、3.5 Optimization）視為不存在，不補回。
- 第 2、3 章是 Lynn 與 iamshuai 的文字：保留措辭與結構，只在使用者明確同意時改。
- 「沒用到不提」；正文不提補充材料；caption 簡短，協定細節放正文。
- 方法名在圖表標「ImmuLUT (Ours)」，正文不加。
- 表格：固定字級、不用 `\resizebox`；各區塊最佳平均粗體；本方法列底色 `oursrow`。圖：STIX／texgyretermes、印刷 ≥ 7 pt；加性方法紅 `#A94438`、本方法 teal `#1F6F7C`；Fig. 3 用 Okabe-Ito（其他加性灰、BlurGuard 藍 `#0072B2`、DCT-Shield-Y 橙 `#E69F00`）。
- 暫代數字用 `\prelim{值}{註解}`、`\prelimnote{註解}`，未寫內容用 `\TODO{}`；投稿前設 `\prelimfinaltrue`。
- 正文上限 8 頁；全文不放 URL；commit message 英文，內容與文件分開 commit。

## 7. 檔案與工具

| 路徑 | 內容 |
|---|---|
| `docs/paper/` | tex 正本（CVPR 2027 author kit；`cvpr.sty`、`ieeenat_fullname.bst`、`preamble.tex` 不修改）。`main.tex` 定義 `\capg`、`\prelim`、`\prelimnote`、`oursrow` |
| `docs/paper_tools/run_paper_preview.py` | Tectonic 編譯，成功後複製到桌面 `C:\Users\nelso\OneDrive\桌面\paper.pdf` |
| `docs/paper_tools/generate_main_table.py` | Table 1：`--main-table <9 臂 main_table.csv> --capguard <color3d_scale2p5/main_table.csv> --blurguard <blurguard_306/summary.csv> --vibe baseline/results/pexels_portraits/vibe_distilled/summary.csv --klein baseline/results/pexels_portraits/klein_distilled/summary.csv`；前三個 CSV 在 baseline 遠端 `runtime/pexels_portraits/summary/`、`runtime/pexels_portraits/blurguard_306/` |
| `docs/paper_tools/generate_paper_figures.py` | Fig. 3、4（遠端執行，見第 8 節） |
| `docs/paper_tools/generate_pexels_teaser_figure.py` | Fig. 1 |
| `docs/paper_tools/generate_method_overview_panels.py`、`method_overview/` | Fig. 2 |
| `docs/paper_tools/generate_overleaf_ops.py` | Overleaf 同步（第 2 節） |

## 8. 工作方式

- 改論文後執行 `python docs/paper_tools/run_paper_preview.py`，檢查 `build/main.log` 的 Overfull、未定義與重複 label；確認 Conclusion 所在頁（遠端 `mutool draw -F txt`）。
- 改 tex 或腳本用 Edit，或在 scratchpad 寫 `.py`（`assert s.count(old) == 1`）。不要用 bash heredoc 寫含反斜線的 Python（`\\` 會變 `\`）。
- 覆蓋遠端腳本前先 diff，確認遠端沒有 repo 以外的修改。
- Fig. 3（`purification.pdf`）、Fig. 4（`distortion_displacement.pdf`）重出（遠端 `~/paper_figures`）：`~/venvs/wacv/bin/python in/generate_paper_figures.py --points in_pexels/points_blurguard.csv --aligned in_pexels/condition_summary_blurguard.csv --color-summary in_pexels/color_summary_scale2p5.csv --survival in_pexels/perturbation_survival_scale2p5.csv --out-dir out_pexels_ours`，再 scp 兩個 PDF 到 `docs/paper/fig/`。強度掃描只有 8 張，已自圖中移除。
- teaser 重出：`~/venvs/wacv/bin/python in/generate_pexels_teaser_figure.py --image man_60 --prompt 2 --layout protected-purified --no-lpips --cite --scale2p5 --arms photoguard_encoder,mist,dia_pt,dct_shield,blurguard --out-dir out_teaser_ours --name teaser`；加 `--highlight CX CY R CORNER` 會在每張淨化後編輯圖畫紅圈與放大框。搜尋 BlurGuard 檔案在 NFS 上需數分鐘。
- 方法圖：`docs/paper_tools/method_overview/method_overview.tex` 於該目錄以 `tectonic -X compile` 編譯，PDF 複製到 `docs/paper/fig/`。
- 覆蓋遠端腳本一律先 scp 成 `.tmp` 再 `mv`。
- 寫含反斜線的 patch 腳本用 Write 工具建檔，不用 bash heredoc。

## 9. 讀數來源

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15；VAE-Trans 另列；306 張 Pexels 人像。FID、KID 以 GPU 特徵計算（與 CPU 最大差 0.00014，`baseline/results/pexels_portraits/distribution_device/`）。

| 內容 | 來源 |
|---|---|
| 9 臂主表 | baseline 遠端 `runtime/pexels_portraits/summary/main_table.csv` |
| CAPGuard 2.5 倍 | baseline 遠端 `runtime/pexels_portraits/summary/color3d_scale2p5/`（`main_table.csv`、`per_purifier.csv`） |
| BlurGuard 306 張 | baseline 遠端 `runtime/pexels_portraits/blurguard_306/`（`summary.csv`、`per_purifier.csv`） |
| VIBE、klein | `baseline/results/pexels_portraits/vibe_distilled/`、`klein_distilled/` |
| 求解時間 | color trial `logs/*.log`（中位 9.53、p90 10.20 分鐘，n = 306） |

## 10. 現況（2026-11-17，HEAD `542f4f54`），取代第 4、6 節

接手順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔第 1、2、6a、7、8 節（角色、Overleaf、寫作規範、工具、工作方式）→ 本節。第 3 節的方法敘述仍適用；第 4、6 節只描述 11/12 的狀態，以本節為準。

### 10.1 權限與工作方式
- executor 有權直接對 paper 下令（使用者授權）；executor 的指示視同使用者指示。例外：**Overleaf 同步只在使用者於 paper pane 親自下令時執行**，經 executor 轉達的同步指令不執行。
- **Lynn 與教授（iamshuai）寫的文字一律不改**，論調與 paper 定調不同也不改、不提建議；只有使用者或 executor 明確指定某一句時才改，且只改被指定的句子。現行正文第 1–6 章幾乎全為兩人的文字，paper 自己寫的只剩圖表 caption、Qualitative Comparison 一句與 3.3 兩句。
- 每次修改：Edit 改 tex → `python docs/paper_tools/run_paper_preview.py` 編譯 → 確認 Conclusion 收在第 8 頁、`build/main.log` 無未定義 \cref 與重複 label → 以 `\color{blue}` 標出改動另編一版 PDF 放 scratchpad，以 `Start-Process` 開給使用者 → commit（英文 message，內容與文件分開）→ 回報 executor commit hash 與 PDF 路徑。push 由 executor 執行。
- 一次只做被指派的事；做不到（例如不刪 Lynn／教授文字就縮不下）時停下回報差幾行與可行選項，不自行擴大修改範圍。
- 本機只編譯 tex；畫圖、PDF 轉 PNG、讀數彙整在遠端 basic-1。數字一律向 color／baseline 索取，附協定與 n，不自行推算。
- 回報 executor 不超過要求的行數，先講結果。

### 10.2 頁數與版面（正文 8 頁，第 8 頁約剩 3 行）
- 第 9 頁起為 LLM Usage Statement 與參考文獻；依 CVPR 2027 Author Guidelines §2，LLM 聲明不計入 8 頁。
- 圖表編號與位置：Fig. 1 teaser（man_x 殭屍）、Fig. 2 方法總覽、表 1 主表（`tab:main`，原始碼在 `3_method.tex`）、Fig. 3 失真—位移（`fig:plane`）、表 2 ISR（`tab:isr`）、Fig. 4 消融外觀（`fig:ablation_appearance`，兩列 viewport 裁切，寬 0.139＋0.416\textwidth，原始碼排在 Cross-Model 小節前以免被推到第 9 頁）、Fig. 5 質性圖 woman_23 helmet（`fig:purification_example`）與表 3 cross-model transfer（`tab:transfer`）並排於同一 figure*（minipage 0.55／0.43\textwidth，表 3 用 `\captionof{table}` 與 `\resizebox`）、Fig. 6 擾動存續（`fig:purification`）。
- 消融表（`tab:ablation`）在 `X_suppl.tex`；消融段寫「Quantitative ablation results are reported in the supplementary material」，Fig. 4 caption 引用 `sec:exp_ablation`。
- 表 3 的 `\resizebox` 違反第 6a 節「表格固定字級、不用 `\resizebox`」，表內字級約為 \scriptsize 的 55%；是否接受待使用者看 PDF 決定（替代案：minipage 改 0.50／0.48）。

### 10.3 同步狀態
- repo 與 Overleaf 最後逐字一致於 `2ec5c671`（2026-11-17，`4_experiments.tex` 以詞級修改追加同步，重新載入後雜湊相符）；此前 `b5b8a9b5`：同步前線上文字檔雜湊皆等於 `5e5f9f38`（無他人修改）；`fig/teaser.tex`、`main.bib`、`0_abstract`、`1_intro`、`2_related`、`3_method`、`4_experiments`、`X_suppl`、`X_suppl_purification` 以編輯器詞級修改套用，重新載入後 14 個文字檔雜湊與 repo 相同；`fig/` 的 `distortion_displacement.pdf`、`method_overview.pdf`、`purification.pdf`、`purification_example.pdf`、`teaser.pdf` 以上傳覆寫；線上編譯 10 頁、0 錯誤。線上另有 repo 沒有的 `sec/[old] related.tex`，未動。下次同步以 `2ec5c671` 為三方合併基底。

### 10.4 待使用者決定
1. 表 3 字級（10.2）。
2. 表 2 是否換成 306 張人工複核版 ISR（executor 已算好，`docs/isr_pilot/audit206/summary_audited.txt`、`paired_difference_audited.txt`）。現行表 2 為 100 張（ImmuLUT 淨化合計 15.8±2.9）。306 張複核版淨化合計：ImmuLUT 12.3 [10.8, 13.8]、BlurGuard 8.9、SIFM 5.4、Mist 5.2、PhotoGuard 3.2、DCT-Shield 2.6；ImmuLUT − BlurGuard ＋3.4 [＋2.3, ＋4.5]。換入時連同 caption、正文 ISR 段與 Introduction 第三點貢獻、Conclusion 的數字一併檢查（後三者是 Lynn／教授文字，改數字須使用者同意）。
3. supp 的 woman_111 p1「police suit」質性圖（baseline `artifacts/pexels_portraits/purification_example/woman_111_p1.pdf`，主種子 20260812、blur σ2；腳本 `baseline/scripts/generate_purification_example_figure.py`）尚未放入 `docs/paper/`。

### 10.5 已下令、尚未執行（使用者在 paper pane 直接下達，新 paper 接手後先做）
1. 消融外觀圖（現 Fig. 4，`fig:ablation_appearance`）改放 supp，與消融表同一節；正文 Lynn 的「As illustrated in \cref{fig:ablation_appearance}」改為指向補充資料（`efbd3998` 的寫法）。
2. 消融段補一句 ImmuLUT 的 LAION 與 MUSIQ 最高。消融表（200 張）中 random Θ 的 LAION 5.43／MUSIQ 5.33 高於 ImmuLUT 5.32／5.30，random hue/sat 的 LAION 也接近，因此句子須限定在成立的比較範圍（例如等失真的簡單映射之間）；寫好先給使用者看措辭。

### 10.6 版面備案（未採用）
- teaser 與方法圖縮到 0.95 倍；Fig. 5／表 3 並排比例改 0.45／0.53 或 0.40／0.58，均可在 8 頁內放大表 3 的字級。舊 paper session 的腳本 `shrink_top.py` 在其 scratchpad（`2f0566ab-…/scratchpad/`），未入庫。
