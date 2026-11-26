# executor 交接

接手 executor（專案管理、文獻管理、派工與轉達、代為 push、ISR 評審）的冷啟動文件。冷啟動順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔 → 專案記憶 `MEMORY.md`（至少 executor-is-paper-pm、executor-owns-main-bib、no-handover-stops-before-85-percent、color-line-current、color-has-gpu-priority、no-local-compute、never-overwrite-running-scripts-on-nfs、paper-targets-cvpr-2027、advisor-text-is-final、keep-gpus-busy-on-advisor-direction、confirm-dispatch-was-received、paper-authors、external-alarm-via-task-scheduler）→ `docs/PAPER_HANDOVER.md` 第 4 節 → `baseline/STATUS.md`、`color/STATUS.md`。本檔的歷史版本見 git log。

## 1. 角色與規則

| 角色 | 職責 | 寫入範圍 |
|---|---|---|
| executor | 進度管理、文獻管理（`docs/paper/main.bib` 核實）、派工與轉達、整理成果給使用者、代各 session push、ISR 評審與計分 | `docs/` 的管理文件、`docs/isr_pilot/`、`main.bib` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 klein、SD inpainting）、主表讀數、ISR 評審素材、論文圖、公開程式碼 `baselines/` | `baseline/`、`core/`、遠端 `~/immulut_release/baselines/` |
| color | ImmuLUT 求解與編輯、讀數、消融、多色調搜尋、公開程式碼主體 | `color/`、遠端 `~/immulut_release/`（`baselines/` 以外） |
| paper | 論文正文與補充材料 `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_HANDOVER.md` | 同左 |

- 使用者授權 executor 直接對各 session 下令；例外是 Overleaf 的任何操作（同步、刪檔），須使用者在 paper pane 親自核准。
- pane 編號常變，派工前 `list_panes` 依標題辨認。使用者重開機或對方 session 重啟、用量耗盡時訊息會遺失（本期發生三次）；派工後以 `read_pane` 確認對方回「收到」，沒有就重送全文。
- push：各 session 回報 commit 後，確認 `.sh` 無 CR 再 push。basic-1 repo 以 `git pull --ff-only` 跟進前，確認 `run_pexels_supervisor.sh` 沒有在跑（`ps -eo args | grep "^bash.*[r]un_pexels_supervisor.sh"`）；若遠端有同名未追蹤檔，比對內容相同後先移開再 pull。
- 回報使用者：繁體中文，先講結果，數字附協定，不下判定。看圖時把圖抓到 scratchpad、自己看過再開給使用者。
- 本機不跑計算（計分、拼圖、PDF 渲染都在 basic-1），本機只編輯、ssh、看圖、合併文字檔。計分程式在本機寫好後 scp 到遠端以 `python3` 執行（遠端 `python3` 沒有 numpy；PIL 用 `~/venvs/wacv/bin/python`）。
- GPU：全局 ≤ 6 張（basic-1／basic-2），只取完全空的卡，color 優先。遠端 `nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`／`-p 10102`。
- 本機關機：使用者要求「做完就關機」時，以工作排程器排一個不經模型的備援關機（`schtasks /Create /TN <名> /SC ONCE ... /TR "shutdown.exe /s /t 120"`），工作提前完成就直接 `shutdown.exe /s /t 120` 並刪除排程。

## 2. 方法、協定與論文現況

- **ImmuLUT**（論文巨集 `\capg`）：每張影像一個 33³ RGB LUT；Θ 經可分離 3D 高斯（σ＝4 格）平滑，以灰軸為極軸，極角 < 25°、方位角 < 7.5°；損失為 ip2p VAE 編碼器後驗平均的均方距離；Adam 1800 步。條件名 `color3d_scale2p5_pexels`。
- **資料與攻擊**：Pexels 人像 306 張；InstructPix2Pix（text 7.5、image 1.8、50 步），4 句 FaceLock 配件指令，主種子 20260812。
- **論文**：CVPR 2027 Submission 14011。正文已於 11/17 上傳，不可再改。補充材料截止延後（教授 11/25 告知「還有四天」）；最終 supp PDF 與程式碼包見第 5 節。Overleaf 最後同步點為 `ba5a8de8`；之後的 repo 改動（supp 大幅增補、`main.tex` 第 77–78 行目錄設定）尚未同步，同步需使用者在 paper pane 核准。
- **ISR 定義（全文一致）**：條件式 ISR，只計同一影像、指令、淨化下未防禦編輯成功的格子（未防禦失敗的格子不計入分母）；防禦後編輯任一標籤失敗即計為擋下。使用者曾說「做錯的不能記進 ISR 分母」，意思就是這個既有定義；把 mismatch 格移出分母的變體是誤解，已刪除（commit `6d43966b`），未進論文。

## 3. 教授 11/25 意見（原文）

> 19:46 Austin Shuai SUPPLEMENTARY 還有四天(賺到XD)
> 可以看一下底下幾項要不要補強
> (1) Supplementary Table 4 已經有完整 ablation，其中 B2 的兩參數映射和 B3 的 affine transformation，在部分 edit LPIPS 上不輸 Full，B3 經 VAE-Trans 後甚至是 0.364，Full 是 0.288。最值得補的是用同一批既有輸出，比較 Full、B2、B3 的實際編輯失敗率與視覺品質，確認我們的優勢是否真的是「保護效果與原圖品質的平衡」。要記得判讀時要把單純留下色偏和真的沒完成指令分開
> (2) 主文 Table 3 的 FLUX.2、VIBE transfer 目前主要靠 LPIPS、SigLIP 和 ArcFace，還不能直接看出編輯指令有沒有被阻止。可以固定一小批相同人像與指令，補原圖編輯、Full、BlurGuard，以及 purification 前後的同案例結果，並沿用 ISR 判準檢查，可以優先整理已生成的圖片，不必重新擴大資料集
> (3) 主文 Table 2 提到人工檢查一部分案例，但沒交代抽了多少、怎麼抽、和 judge 一致多少。supplementary Sec. B 對 SD Inpainting 的分母與 bootstrap 已經寫得比較清楚，也明說尚未人工複核。我們可以把主文 ISR 的完整 judge prompt、模型版本、有效案例數和人工核對結果補齊，再針對 Inpainting 固定抽一小批盲看，要把「judge 與人工在哪些情況不一致」講清楚。
> (4) Main paper Sec. 4.5 的 affine 比較寫 LAION 由 5.32 降到 5.26、MUSIQ 由 5.30 降到 4.80，但 supplementary Table 4 的 Full/B3 是 5.34/5.27、5.31/4.81，主文 Table 1 的 Full 也支持後一組。請確認是哪個版本的結果，統一數字。如果main paper不能改，就只能改supplement
> (5) Supplementary Sec. A 已有我們的 optimizer、步數與 learning rate，但各 baseline 如何調到相近 defense LPIPS 還不夠具體，Sec. B 提到的 scaling factors 也沒有列出。可以的話補每個方法的預算、縮放值、實作版本與指令，以及 VAE-Trans 和 transfer editor 的 checkpoint/推論設定
>
> 19:46 Austin Shuai 另外就是可以幫sup編個目錄比較好讀（附截圖，指令為 \clearpage \appendix \maketitlesupplementary、浮動參數設定、\addtocontents{toc}{\protect\setcounter{tocdepth}{2}} 與 \begingroup\makeatletter\setcounter{tocdepth}{-1}\renewcommand{\contentsname}{Contents}\renewcommand{\l@section}…\renewcommand{\l@subsection}…\footnotesize\tableofcontents\makeatother\endgroup）
> 19:47 Austin Shuai 另外就是codes跟資料也可以一起壓縮成zip放在supplementary，但要注意寫的要跟codes一樣，不然會被抓。至少要拿LLM檢查一下，另外就是注意匿名性～

另：羅婈 11/24 意見——B1／B2 寫成「參數化模型」、避免 hue／saturation 用詞（已改為 rotation and scaling）；inpainting 實驗設定補 matched distortion（已補）；換圖時 Qualitative 敘述一起改（已做）。Overleaf 上羅婈改過的文字一律不動（記憶 advisor-text-is-final 同理）。

## 4. 各項處理狀態

| 項目 | 狀態 | 位置／數字 |
|---|---|---|
| (1) Full、B2、B3 的 ISR 與畫質 | 已量測，未寫入 supp，**待使用者決定是否放** | `docs/isr_pilot/ablation/`（rubric、judgments 400 張 × 20 行、計分程式、summary.txt）。crosseditor100 × 4 句 × 4 淨化，Sonnet 盲評（列序打亂）。條件式 ISR 全部合計：Full 12.4［9.8, 15.1］、B2 19.9［16.3, 23.5］、B3 26.8［22.7, 31.2］；Full−B2 −7.5［−11.9, −3.0］、Full−B3 −14.4［−19.2, −9.4］。配件未出現比例三者 6–8%，B2、B3 較高的 ISR 主要來自 mismatch。「指令完成但帶色偏」Full 84%、B2 74%、B3 59%。Sonnet 自然度（1–5）：未防禦 4.52、Full 1.83、B2 1.36、B3 1.89。畫質自動指標（supp 表 4，306 張）：LAION Full 5.32／B2 5.19／B3 5.26，MUSIQ 5.30／5.13／4.80 |
| (2) FLUX.2、VIBE 的 ISR | 已量測，未寫入 supp，**待使用者決定** | `docs/isr_pilot/transfer/`。50 張 × 4 句 × 4 淨化 × 6 條件。VIBE 全部合計各方法 1.5–2.4%（ImmuLUT 2.1），擋下集中在 police suit；FLUX.2 0–0.3%（ImmuLUT 0.1）。未防禦成功率 VIBE 82–84%、FLUX.2 約 100% |
| (3) ISR 評審細節與 inpainting 盲看複核 | 已寫入 supp | supp D 節：Judging（模型 Claude Sonnet 5.5、model ID claude-sonnet-5-5、評審指令全文，指令中列出方法名那一行依使用者決定改為「Rows R1-R7 correspond to the seven conditions.」）、Human Review（1,038 格、92.1%／68.8%）、Valid Cells、表 8（`tab:supp_isr_full`，正文表 2 的完整版，含 CI 與 n）。supp B 節：inpainting 評審指令全文、Human Review（60 格分層抽樣、13 格未防禦失敗移出分母、47 格中一致 76.6%、不一致 9＋2）、表 5（`tab:supp_inpaint_isr`）已套用人工複核（與 ip2p 同協定）。人工判定只用使用者本人填的 `docs/isr_pilot/inpaint/human_review/answers.csv`；計分 `measure_inpaint_isr_summary.py <key> <answers> <review key>` → `summary_reviewed.txt` |
| (4) 消融數字不一致 | 已改 supp | 表 4 的 Full、B3 兩列 LAION／MUSIQ 改為已投出正文的數字（5.32／5.30、5.26／4.80），以 Full 為基準的差值同步重算；其餘格仍為 306 張完整版 |
| (5) 加性方法細節 | 已寫入 supp（新增 C 節） | Data、Purifications、VAE-Trans、Editors、Matched-Distortion Rescaling、Baseline Settings＋表 7（`tab:supp_provenance`）、Objectives |
| 目錄 | 已加入 | 教授原碼；`##1`／`##2` 在頂層需寫成 `#1`／`#2`；`main.tex` 77–78 行讓正文條目隱藏 |
| 程式碼與資料 zip | supp 包已做（見第 5 節）；GitHub 尚未推送 | 遠端 `~/immulut_release/`（凍結） |

其他本期 supp 改動：B1／B2 改稱 rotation and scaling、Fig. 6 面板標上 A2–B4 代號、A1 寫明 Θ 為單位尺度 N(0,1)、消融補跑到每列 306 張並移除 n 欄、inpainting 質性圖改為加性方法原生預算版（man_26、woman_14、man_81、woman_118）。逐段內容與 commit 見 `docs/PAPER_HANDOVER.md` 與 git log（本期 commit `2512ea6b`、`156d38c1`、`3adfec09` 等）。

## 5. 公開程式碼與補充材料包

- **位置**：遠端 `~/immulut_release/`（不在 repo、無 .git），進度與決定記錄在 `~/immulut_release/RELEASE_NOTES_INTERNAL.txt` 與 `baselines/RELEASE_NOTES_INTERNAL.txt`（此兩檔不上傳）。color 與 baseline 已凍結此目錄，修改需由 executor 列清單交辦。
- **匿名做法（使用者授權 executor 決定）**：GitHub repo `Nelson0314/ImmuLUT`（公開、目前為空）與 supp zip 用同一份完全匿名內容；論文與 supp 不放 repo 連結、README 不提 CVPR 或投稿。依據 CVPR 2027 Author Guidelines：「To preserve anonymity, you cannot cite your public codebase.」，並鼓勵把程式碼放進補充材料。
- **使用者對程式與論文不一致的決定**：Adam betas (0.5, 0.99) 與「前 2/3 步餘弦降到 1e-5、之後固定」只在 README 與註解寫明；損失 −w·d/d0 完整移植（w 來自 `immulut/alignment.py` 的參考目標梯度範數比）；方位角正規化 r＝sin(θ+Δθ)/cos Δθ 完全不動；高斯核 3σ 截斷＋replicate、近黑格點與灰軸處理寫在 README 與註解；A1 公開版補逐張種子、supp 改寫 A1；Mist 程式保留並在 `THIRD_PARTY.md` 註明上游無授權。
- **驗證**：ImmuLUT 三張與既有防禦圖最大差 1 階（根因為 TF32，公開版已關閉）；編輯結果 LPIPS 改在 8 位元 PNG 上量後與主表差 ≤ 0.0003；加性方法差異來自 GPU 非決定性（VAE attention 與 cuDNN），公開版提供 `--deterministic`，論文結果為預設模式。
- **逐行審查進度（使用者要求逐檔對照論文所有細節）**：已審 `immulut/lut.py`、`solve.py`、`ip2p.py`、`io.py`、`alignment.py`、`configs/immulut.yaml`，對照清單與已審結果在 `docs/RELEASE_AUDIT.md`（M1–M25 全部對上）。新發現一處待使用者決定：輸出先轉極角再沿方位角方向加切向量後正規化，最終極角比 θ+Δθ 略大（sin 項最多放大 √(1+tan²Δφ) ≤ 1.0087），方位角上界仍嚴格成立；與 r 算式同一機制，建議比照不改。**未審**：`evaluation/`（約 290 行）、`immulut/color_mappings.py`、`solve_ablation.py`、`scripts/`、`baselines/`（約 2,500 行）、`README.md` 最終版。審查通過後才推 GitHub。
- **補充材料包**：桌面 `C:\Users\nelso\OneDrive\桌面\supplementary.zip`（11.97 MB）＝ `paper_supp.pdf`（repo `3adfec09` 編譯，supp 第 11–19 頁，9 頁）＋ `code/`（凍結版 `~/immulut_release`，49 檔，去除 `__pycache__` 與兩份 RELEASE_NOTES_INTERNAL）。打包時 grep 人名、帳號、主機、路徑、內部檔名、CAPGuard、Claude、日期與中文：僅命中他人論文的「CVPR 2024」引用。上傳 OpenReview 由使用者執行；若之後程式再改，需重新打包。

## 6. 待使用者決定

1. 教授 (1)：消融 ISR 表是否放入 supp（照實結果為 Full 低於 B2、B3）。
2. 教授 (2)：VIBE、FLUX.2 ISR 是否放入 supp。
3. 公開程式碼：極角略大於 θ+Δθ 一事是否處理；其餘檔案審查完成後推 GitHub 的時點。
4. Overleaf 同步（`X_suppl.tex`、`7_llm_usage.tex`、`main.tex`）與 supp 上傳。
5. 多色調 A（色調版 ImmuLUT）暫停於 300 格中 162 格，是否恢復；GPU 目前閒置。
6. Overleaf `build/` 與未引用圖檔是否刪除。

## 7. Sonnet 評審工作方式

- 拼圖由 color／baseline 在遠端產生（列序以 `default_rng(20260812)` 逐張打亂，列標不含條件名，另存 `key.csv`），executor 下載到 scratchpad、10 張一批，以 `Agent`（model sonnet）評審，brief 指定 rubric、批次清單與輸出 CSV，要求只用 Read／Write／Edit。每個 agent 可連續做 2–3 批。
- 評審常偏離 brief（改用 shell 附加、跨批寫入同一檔）；計分前逐檔核對「檔內拼圖＝清單、每張行數正確」。
- 碰到 5 小時或每週用量上限就停，換帳號或重置後依已寫入的 CSV 建立剩餘清單續評。
- 使用者複核的偏差型態（inpainting）：Sonnet 對未防禦編輯接受弱線索、對防禦後編輯在線索微弱或在邊緣、多出合理人物、可見遮罩邊帶時判失敗；兩方向都使 ISR 偏高。
