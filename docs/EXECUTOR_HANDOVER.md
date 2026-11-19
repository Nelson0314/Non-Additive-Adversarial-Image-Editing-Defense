# executor 交接

接手 executor（專案管理、文獻管理、派工與轉達、代為 push）的冷啟動文件。冷啟動順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔 → 專案記憶 `MEMORY.md`（至少 executor-is-paper-pm、executor-owns-main-bib、no-handover-stops-before-85-percent、color-line-current、color-has-gpu-priority、no-local-compute、never-overwrite-running-scripts-on-nfs、paper-targets-cvpr-2027、advisor-feedback-2026-11-08、keep-gpus-busy-on-advisor-direction、advisor-text-is-final）。各專案即時狀態以 `baseline/STATUS.md`、`baseline/docs/HANDOVER.md`、`color/STATUS.md`、`docs/PAPER_HANDOVER.md` 為準。本檔的歷史版本（逐日現況）見 git log。

## 1. 角色與規則

| 角色 | 職責 | 寫入範圍 |
|---|---|---|
| executor | 進度管理、文獻管理（`docs/paper/main.bib` 核實）、派工與轉達、整理成果給使用者、代各 session push | `docs/` 的管理文件、`docs/isr_pilot/`、`main.bib` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 klein、SD inpainting）、主表讀數與彙整、ISR 評審素材、論文圖（Fig. 1、Fig. 4 版面） | `baseline/`、`core/` |
| color | ImmuLUT（條件名 color3d_scale2p5 系列）求解與編輯、逐張讀數、消融 | `color/` |
| paper | 論文正文 `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_HANDOVER.md` | 同左 |

- 使用者授權 executor 直接對各 session 下令；例外是 Overleaf 同步，須使用者在 paper pane 親自下令（只換圖檔時使用者曾授權 executor 自行上傳）。
- pane 編號會變（session 重開或切帳號後常變），每次派工前 `list_panes`，依標題辨認。2026-11-19 交接時：baseline＝1、color＝2、paper＝3、舊 executor＝4（交接後唯讀，標題拼為 excutor）。對方 session 重啟後可能沒收到訊息，派工後以 `read_pane` 確認對方確實開始。
- 排程由使用者決定；使用者未回應時依教授方向自行排下一批（記憶 keep-gpus-busy-on-advisor-direction），刪檔、改協定仍需確認。
- 不因 session 長度停工：唯一換手條件是 context 用量 > 85%。
- push：各 session 回報 commit 後，確認變更檔的 `git ls-files --eol` 為 `i/lf`（二進位檔為 `i/-text`）再 push。`git fetch` 偶爾失敗，可直接 `git push`（非 fast-forward 會被拒）。遠端 repo（basic-1 `~/image-immunization`）以 `git merge --ff-only origin/main` 跟進；執行中腳本被改到時先確認該 `.sh` 沒有在跑。
- 回報使用者：繁體中文，先講結果，數字附協定（資料、張數、設定），不下判定；本機不跑計算（CPU 分析、畫圖、PDF 渲染也送遠端），看圖時把圖抓到 scratchpad 再開給使用者。
- subagent：使用者沒說就不派。
- GPU：全局 ≤ 6 張（兩台 RTX 3090 主機 basic-1／basic-2，NFS 共用 `~`），只取完全空的卡，color 優先。baseline 以 `queue/baseline_cap`、color 以 `queue/color_reserve` 控卡（`baseline/runtime/pexels_portraits/queue/`）。兩台主機常被其他使用者占滿。
- 遠端主機：`nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`（basic-1）／`-p 10102`（basic-2）；遠端 `python` 不存在，用 `python3` 或 `~/venvs/wacv/bin/python`；`pdflatex`、`mutool` 可用。

## 2. 方法與協定

- **ImmuLUT**（原名 CAPGuard；論文巨集 `\capg`）：每張影像一個 K³（K＝33）RGB 3D LUT；自由參數經色空間高斯平滑（σ＝4 格）成旋轉場；以灰軸為極軸，極角 < 25°、色相 < 7.5°（條件名 `color3d_scale2p5_pexels`），‖c‖ 不變；損失只有 ip2p VAE 編碼器後驗平均差；Adam 1800 步，中位 9.53 分鐘／張（RTX 3090）。
- **資料與攻擊**：Pexels 人像 306 張；InstructPix2Pix（text 7.5、image 1.8、50 步、Euler-Ancestral），4 句 FaceLock 配件指令（sunglasses、police suit、helmet、bow tie），主種子 20260812，不跑多種子；最佳化不用任何指令。
- **淨化**：七道（JPEG 80／50／30、blur σ1／2、crop-resize 0.1、rotate15）＋ VAE-Trans。
- **加性對照**：10 個等失真臂（DAYN、DIA-R、DIA-PT、SIFM、DANP、Mist、PhotoGuard encoder、DCT-Shield、DCT-Shield-Y、BlurGuard）。
- **跨編輯器**：VIBE-DistilledCFG、FLUX.2 [klein] 4B（表 3）；SD inpainting（補充資料）。
- **ISR**：條件式（原圖可編輯的格子中，免疫圖編輯失敗的比例），Claude Sonnet 5.5 評審＋人工複核（306 張中 100＋206 張複核，1038 格）；紀錄與程式在 `docs/isr_pilot/`（README、`audit/`、`audit206/`）。

## 3. 論文現況

- CVPR 2027 正文已於 2026-11-17 17:03（台灣時間）上傳 OpenReview（Submission 14011）：PDF 與 Overleaf 一致於 commit `677f2f54`，摘要含 1.32–3.25 倍淨化後位移比。補充材料截止 11/23 19:59（台灣時間）。
- repo 與 Overleaf 的同步點記在 `docs/PAPER_HANDOVER.md`；頁數與 Conclusion 所在頁一律以 Overleaf（pdfLaTeX＋Times）編譯為準，本機 Tectonic 斷行不同。
- 表 2 ISR 為 306 張人工複核版（ImmuLUT 淨化合計 12.3±1.5，對 BlurGuard ＋3.4±1.1 個百分點）。
- 教授 11/08 回饋：(1) ISR 已完成並入正文；(2) 等失真簡單顏色映射在消融（supp）；(3) E(T(x)) 對 T(E(x)) 只有 pilot，未入論文；(4) 色彩校正反向處理未入論文。
- Fig. 4 多色調 ImmuLUT（教授要求）：四輪實驗（固定色調前置、色調初值、強度 0.25／0.5／0.75）在 woman_111 上，不淨化與 blur2 皆保留原衣物的只有與現行版外觀相近的 magenta 初值；暖色起點防禦明顯變弱。2026-11-17 停止，產物在 color trial `artifacts/fig4_tone/`。

## 4. 進行中與已完成（2026-11-19 12:00）

- **inpainting（baseline）**：306 張編輯完成（failed 0），讀數 11/19 06:34 完成：遠端 `baseline/runtime/pexels_portraits/inpaint/readouts/summary.csv`（另有 retention、semantic_identity）。協定：runwayml/stable-diffusion-inpainting，CLIPSeg 主體遮罩之補集（只重繪背景），主種子，4 句指令。編輯位移（不淨化／JPEG50／blur2／VAE-Trans）：PhotoGuard .322／.134／.053／.072、DIA-PT .280／.160／.075／.086、DCT-Shield .283／.109／.044／.061、SIFM .270／.150／.072／.082、BlurGuard .283／.234／.171／.156、ImmuLUT .378／.389／.386／.352。正文 4.3 段已寫「Results on a mask-based inpainting editor, Stable Diffusion Inpainting, are reported in the supplementary material」。`__raw.png`（約 11.3 GB）使用者決定保留。
- **消融擴充到 306 張（color）**：11/17 19:40 開跑，新增 106 張 × 8 列＝848 項（A1–A4、B1–B4；Full 列沿用主表 306 張）。11/18 23:46 時完成 563、失敗 9（全為共用卡被他人占滿的 CUDA OOM：A2 man_85／woman_101／woman_116／woman_117／woman_141、A4 woman_126、B1 woman_181、B2 man_06／man_22）。使用者已核准補跑；color 已排收尾腳本 `runtime/finish_ablation_pexels306.sh`（basic-1 背景）：佇列結束 → 清這 9 項的 failed 標記與 claims → 2 線補跑（仍 OOM 30 分後再試一次）→ 重做 A2、A4、B1、B2 讀數 → 輸出 `results/ablation_pexels306/`（summary、failed、cells、paired_differences common／all）。預計 11/19 下午完成，color 完成後回報 executor，executor 須通知使用者。之後 paper 把正文消融段（目前為 197 張子集的 LAION 5.32／MUSIQ 5.30 等）與 supp 消融表換成 306 張版本——注意正文已交出，正文數字是否改由使用者決定，supp 先換。
- **專案整理（2026-11-17 使用者核可，大部分已完成）**：baseline 刪 11 支腳本、FLUX 逐格表入庫、兩個 trial 與 isr_mllm_judge drop、STATUS／HANDOVER／README 重整；color 遠端刪 17.4 GB、README／STATUS／DESIGN 改寫；paper 交接單一化、supp 逐淨化表產生腳本修正；executor 的 ISR 標註 1038 JSON 併為 `docs/isr_pilot/audit206/human_labels_raw.csv`。遠端合計釋出約 34 GB。core 測試修正（`800e9276`，遠端 416 passed）。三方（本機、GitHub、basic-1 repo）同步於 `800e9276`。
  - **待做（跑完後）**：(1) color trial 升格入庫：ImmuLUT 求解鏈→`color/src`、`scripts/`；論文用結果 CSV（主表、消融 100／200／306、ISR、跨模型、色彩校正淨化、teaser man_x、隨機對照讀數）→`color/results/`（只收 CSV，不收 84 個 `.pt`，文件記遠端路徑）；出圖工具→`scripts/`；舊 ColorMap 若不再被引用就刪；trial 已否決部分（濾鏡、teaser 篩選、多色調、warm-start／skinlock、gore10）寫 TRIALS.md 後刪；論文讀取的舊路徑保留到 paper 改引用。(2) baseline：`generate_pexels_flux_512_{cell_tables,summary}.py` 改名 `generate_pexels_editor_*` 並併入 inpaint summary；`run_pexels_supervisor.sh` 的「最新認領」2 行註解（cron 每 5 分鐘執行，讀數跑完才改）。(3) paper：`generate_main_table.py` 仍產生含 VIBE／klein 下段的舊版表 1，重跑前需改寫；supp 兩張逐淨化表 caption「defended→immunized」需同步 Overleaf（使用者須在 paper pane 下令）。(4) tex 舊文註解（3_method 45 行等）待使用者問 Lynn。
- **多色調 ImmuLUT 搜尋（2026-11-20 使用者指示重啟）**：判準為 blur2、jpeg50、vae_trans 下保留原物件且防禦圖不偏藍，不淨化不要求。woman_111（police suit）上的結果：放寬角度預算無效，因為解受步長限制（max|Θ|≈0.24，未達 tanh 飽和）。學習率放大後解偏向藍色。色調牽引強度與防禦呈連續取捨，pull ≥ 1 時防禦消失。目前最佳為 amber 初值、lr×3、pull 0.3：防禦圖暖橘、臉不藍，三種淨化下與現行 ImmuLUT 同級（只多小臂章或小警徽）；不淨化時換成整套制服。此設定在 woman_154、woman_44、man_00、woman_61 上不成立，四張的現行 ImmuLUT 本身在 vae_trans 下也畫出配件。主表 SigLIP 判定加縮圖挑選與重編結果不一致，候選影像一律以重編判定。拼圖在 color trial `artifacts/fig4_tone/sheets_*`、`artifacts/tone_screen/`，事件記錄在 `logs/night_events.log`；長佇列為 `runtime/tone_long_queue.sh`（woman_111 的 pull 0.25／0.35、sepia、yellowgreen，以及 50 張候選影像 × amber pull0.3）。
- **user study**：使用者決定擱置，supp 不放。

## 5. 補充材料（截止 11/23 19:59 台灣時間）

### 5.1 使用者已核准、交接後立即派工
1. **baseline**：supp 表 `tab:supp_aesthetics` 的 BlurGuard 列（PSNR、SSIM、NIMA、NIQE，現為 `\prelim{--}`）；「Baseline provenance and budget conversion」節內容（各方法原論文預算、等失真縮放係數、實作來源）；inpainting 節的表（上面六方法 × 四淨化的位移，附 SigLIP／ArcFace）與例圖素材。
2. **color**：消融 306 張完成後交出 supp 消融表數字（八列，n、防禦圖 LPIPS、LAION、MUSIQ、四淨化編輯 LPIPS）與成對差。
3. **executor**：ISR 完整表（306 張、各淨化、各指令）與人工複核一致率（`docs/isr_pilot/audit206/agreement.txt`：隨機 114 格 92.1%、不一致對 725 格 68.8%；人工判 ref_fail 199 格、135 個槽位移出分母），以及「Evaluation validity」節要的數字（未防禦編輯失敗的格子比例、阻擋率）。
4. **paper**：在 `X_suppl.tex` 開新節並填入：Inpainting、ISR 完整結果與人工複核、更多質性例子（woman_23 helmet seed 3 舊 Fig. 5 版、man_07 p2 helmet）；填 baseline／color／executor 交來的內容；`main.tex` 中 supp 的 `\input` 目前為註解，supp 另編一份 PDF。
5. **先產出 supp 目錄給使用者看**（節標題與每節內容一行說明、資料來源、負責 session、狀態），使用者確認後再填內容。
6. **inpainting 質性圖**：使用者要先看結果。請 baseline 依 Fig. 4 版面（`baseline/scripts/generate_purification_example_figure.py`）挑 2–3 張 inpainting 例子（未防禦與加性淨化後背景被重繪、ImmuLUT 淨化後背景保留者），出預覽 PNG 給使用者看。

### 5.2 其他待辦
- 兩份獨立審查的低優先項：參考文獻頁碼與卷號補齊、arXiv 條目改引正式版本、Fig. 3／Fig. 5 圖內字級 ≥ 7 pt 與 Fig. 5 灰階可辨、表 2／表 3 比較方法的選擇理由、行文小瑕疵（"On the other hand" 誤用、段尾單字行、專名斷字）。
- OpenReview 的 LLM Usage Declaration 全選 Generate；程式、實驗執行、資料標註實際接近 Agent 等級，使用者未決定是否修改（宣告不給審稿人看）。
- 更多質性例子的篩選經驗：擋得最徹底的例子 ImmuLUT 防禦圖多半偏藍青或粉紅、黃綠背景，使用者不接受；以全圖 ΔE00 排序抓不到臉部色偏，需看原尺寸拼圖。

## 6. 本 session 的經驗與工作方式補充

- **頁數**：Overleaf（pdfLaTeX＋Times）與本機 Tectonic（XeTeX＋TeX Gyre 替代字型）斷行不同，本機 8 頁、線上可能溢出數行。最終檢查用 basic-1 的 `pdflatex`（TeX Live 2023）或直接看 Overleaf；不以本機判斷。
- **CVPR 2027 格式**：author kit（tag `CVPR2027-v1`）`sec/2_formatting.tex` 明文「Table captions should be above tables, while figure captions should be below figures」，現已照做。LLM Usage Statement 必備、不計頁。
- **OpenReview 檢查**：以 Claude in Chrome 在自己的分頁開 `openreview.net/revisions?id=82M6HFkUzi` 讀修訂紀錄（唯讀）；上傳的 PDF 以頁內 `fetch` 取得後比對 SHA-256 與本機檔。編輯表單只讀不送出（按 Cancel）。
- **Overleaf 直接改**：使用者只換圖或改單一數值時曾授權 executor 自行同步：在編輯器頁以 `document.querySelector('.cm-content').cmView.view` 取得 CodeMirror，`dispatch` 修改，改後算 SHA-256 與本機 LF 版比對；圖檔以 Upload 覆寫（選 fig 資料夾）。一般文字同步仍由 paper 在使用者下令後做。
- **派工確認**：session 重啟後訊息會遺失（color 曾整則沒收到）；派工後以 `wait_for_reply` 或 `read_pane` 確認對方已開始，沒有就重送全文。
- **背景監看**：本機記憶體不足時，Claude Code 會停掉閒置的背景指令（等待迴圈、wait_for_reply 背景任務）；長等待改由遠端收尾腳本自動接續並寫事件到 log，或請 session 完成時主動回報，不依賴本機監看。
- **本機不跑計算**：PDF 渲染（`mutool draw`）、LPIPS／CLIP 篩選、JSON→CSV 轉換都在 basic-1 做。遠端 `transformers` 版本的 `get_image_features` 回傳 BaseModelOutputWithPooling，需取 `.pooler_output`。
- **獨立審查**：使用者要求時可派兩個無背景的 general-purpose agent，輸入為遠端渲染的 150 dpi 頁圖、全文文字與檢核表；兩份結果交叉比對，自己看圖驗證後再回報。
- **刪除**：遠端刪除一律先請負責 session 列清單（路徑、大小、理由、保留項），經使用者確認後刪，刪前再核對未被執行中工作使用。trial 刪除前先寫 `TRIALS.md`。

## 7. 文獻

- `main.bib` 歸 executor 核實；期刊名縮寫統一為 IEEE TPAMI／TIFS／TIP。正文引用 CVPR 主會論文 7 篇＋1 篇 workshop。
- 不再被引用的條目保留未刪；來源較弱者為 `doprotect2026`、`zeng2020lut`。
