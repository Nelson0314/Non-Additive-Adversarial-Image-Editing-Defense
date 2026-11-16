# executor 交接

接手 executor（專案管理、文獻管理、派工與轉達、代為 push）的冷啟動文件。冷啟動順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔 → 專案記憶 `MEMORY.md`（至少 executor-is-paper-pm、executor-owns-main-bib、no-handover-stops-before-85-percent、color-line-current、color-has-gpu-priority、no-local-compute、never-overwrite-running-scripts-on-nfs、paper-targets-cvpr-2027、advisor-feedback-2026-11-08、no-agents-unless-asked）。各專案即時狀態以 `baseline/STATUS.md`、`baseline/docs/HANDOVER.md`、`color/STATUS.md`、color trial `trials/lut3d_defense_and_naturalness/HANDOVER.md`、`docs/PAPER_HANDOVER.md`、`docs/PAPER_STRUCTURE.md` 為準。

## 1. 角色與規則

| 角色 | 職責 | 寫入範圍 |
|---|---|---|
| executor | 進度管理、文獻管理（`docs/paper/main.bib` 核實）、派工與轉達、整理成果給使用者、代各 session push | `docs/` 的管理文件、`docs/isr_pilot/`、`main.bib` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 klein）、主表讀數與彙整（含 CAPGuard 的 followup）、ISR 評審素材 | `baseline/` |
| color | CAPGuard（color3d 系列）求解與編輯、逐張讀數、消融、T(E(x)) 對 E(T(x)) 量測 | `color/` |
| paper | 論文正文 `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、`docs/PAPER_HANDOVER.md` | 同左 |

- pane 編號會變（session 重開或切帳號後常變），每次派工前 `list_panes`，依標題辨認。
- `send_to_pane` 送出的訊息在對方 session 額度用盡或重啟時可能未被處理；派工後以 `read_pane` 或遠端狀態確認對方確實開始。
- 排程一律由使用者手動下令：不設自動接續、閘門或自動改卡數的監看；一項完成後回報使用者，等指示再開下一項。使用者明確授權「跑到完成」時才可讓佇列自動接下一段。
- 不因 session 長度停工：唯一換手條件是 context 用量 > 85%。
- push：各 session 回報 commit 後，確認 `git merge-base --is-ancestor origin/main main` 與變更檔的 `git ls-files --eol` 為 `i/lf`（二進位檔為 `i/-text`）再 push。`git fetch` 偶爾卡住，可直接 `git push`（非 fast-forward 會被拒）。提醒對方遠端 pull 前確認被改到的 `.sh` 沒有在執行。
- 回報使用者：繁體中文，先講結果，數字附協定（資料、張數、設定），不下判定；進度用完成數／總數；本機不跑計算（CPU 分析、畫圖也送遠端），看圖時把圖抓到 scratchpad 再用 `Start-Process` 開給使用者；使用者明說才放桌面（`C:\Users\nelso\OneDrive\桌面`）。
- subagent：使用者沒說就不派；派的話一次一個，brief 限制輸出長度。
- GPU：全局 ≤ 6 張（兩台 RTX 3090 主機 basic-1／basic-2，NFS 共用 `~`），只取完全空的卡，color 優先。baseline 以 `queue/baseline_cap`、color 以 `queue/color_reserve` 控卡（`baseline/runtime/pexels_portraits/queue/`）。

## 2. 方法與協定

- **CAPGuard**（Color-based Anti-Purification Image Immunization）：每張影像一個 33³ RGB 3D LUT；自由參數經色空間高斯平滑（σ＝4 格）成旋轉場；以黑點為原點、灰軸為極軸，極角 < 25°、色相 < 7.5°（2.5 倍上界，條件名 `color3d_scale2p5_pexels`），‖c‖ 不變；損失只有 ip2p VAE 編碼器目標；Adam 1800 步，306 張中位 9.53 分鐘／張（RTX 3090）。
- **資料與攻擊**：Pexels 人像 306 張；InstructPix2Pix（text 7.5、image 1.8、50 步），4 句 FaceLock 配件指令，主種子 20260812，不跑多種子；訓練不用任何指令。
- **淨化**：七道（JPEG 80／50／30、blur σ1／2、crop-resize 0.1、rotate15）＋ VAE-Trans。
- **加性對照**：9 個等失真臂（DAYN、DIA-R、DIA-PT、SIFM、DANP、Mist、PhotoGuard encoder、DCT-Shield、DCT-Shield-Y）＋ BlurGuard（事後縮放 0.6071，Pexels 306 張防禦圖 LPIPS 0.2745）。
- **主表**（Table 1）：上段 ip2p 全方法 306 張；下段 VIBE-DistilledCFG 與 FLUX.2 klein 4B 蒸餾版各 306 張，比較 PhotoGuard、DCT-Shield、BlurGuard、Mist、CAPGuard（防禦圖沿用 ip2p 求解者）。所有 FID／KID／CLIP-IQA 已改用 GPU 特徵重算，與 CPU 版在記錄精度下相同（`baseline/results/pexels_portraits/distribution_device/`）。
- **跨編輯器**：UltraEdit 與 FLUX.1 Kontext 已停用（deferred 工作在 baseline 佇列 `jobs_deferred_manual.txt`）。VIBE 與 klein 的結果在 `baseline/results/pexels_portraits/{vibe_distilled,klein_distilled}/`。

## 3. 現況（2026-11-09）

- 論文：Table 1 無紅字；4.5、Conclusion 已換 VIBE／klein。仍有紅字：消融表、4.4 強度掃描（8 張）、Limitations。全文 10 頁、正文止於第 8 頁。
- 遠端：佇列空、無租約、無執行中工作（2026-11-09 12:20 確認）。
- baseline 與 color 的 session 都在待命；兩項為教授回饋開的實驗已由使用者喊停並撤回：顏色正規化淨化（trial 已 drop，commit 21368a32）、雜訊底線（color trial `results/color3d_scale2p5_pexels/edit_lut_commutation/seed_20260813/` 留有 4 張編輯，未計讀數）。
- 遠端 basic-1 有三支 2026-11-03／04 啟動的舊監看迴圈（PID 80936、722084、1857447，只在條件成立時印紀錄，不佔卡）。executor 的終止指令被權限規則擋下，需使用者決定是否清除。

## 4. 指導教授 2026-11-08 回饋（記憶 advisor-feedback-2026-11-08 有原文要點）

| 項 | 內容 | 狀態 |
|---|---|---|
| (1) ISR 與 semantic edit success | 條件式 ISR，比較 No defense、CAPGuard、PhotoGuard、DCT-Shield、BlurGuard（可加 SIFM／SPIN），淨化無／JPEG／blur／VAE-Trans | ip2p pilot 完成，見下 |
| (2) 等失真的簡單顏色 baseline | random HSV、最佳化 hue／saturation、3×3 affine、平滑 LUT；以防禦圖 LPIPS 對齊，報 ISR 與品質，附 budget sweep | 未開始；需 color 寫程式 |
| (3) 是否只改了顏色 | 比較 E(T(x),p) 與 T(E(x,p)) | pilot 完成：isr50 × 4 句、淨化前，LPIPS(T(E(x)), E(T(x))) 平均 ip2p 0.204（編輯結果 LPIPS 0.372）、VIBE 0.207（0.338）；同協定雜訊底線未量（已撤回） |
| (4) 顏色映射能否被反向處理 | Auto Color、白平衡、直方圖正規化 | 使用者已為此解除色彩正規化淨化禁令（限本實驗）；實驗已撤回，未開跑 |

**ISR pilot（第 1 項）**：紀錄與資料在 `docs/isr_pilot/`（README、判定、規則、統計輸出、分指令表）。ip2p、crosseditor100 100 張 × 4 句、主種子、Sonnet 評審，條件式 ISR：

| 條件 | 不淨化 | 淨化後（JPEG 50＋blur σ2＋VAE-Trans） |
|---|---|---|
| PhotoGuard | 42.1% | 3.5% |
| DCT-Shield | 22.0% | 2.4% |
| BlurGuard | 14.2% | 11.3% |
| SIFM | 17.3% | 5.4% |
| Mist | 32.5% | 6.0% |
| CAPGuard 2.5 倍 | 13.6% | 15.6% |

淨化後 CAPGuard 對每個 baseline 的 McNemar p ≤ 0.0033。限制：未人工驗證、部分批次評審標準較寬（見 README）、只有 ip2p。

**相關的其他量測與討論結論**
- 文獻（PhotoGuard、SIFM 等）的 ISR 在原生預算、不淨化下測得；本專案的 PhotoGuard 原生預算 pilot（10 張）不淨化時編輯全部被換成其他場景，JPEG 50 與 VAE-Trans 後恢復正常。ip2p image guidance 降為 1.2 會增加未防禦本身的失敗，不採用。
- SIFM 論文（arXiv 2512.14320）的 ISR：Gemini 2.5 Pro 與 Flash 雙評審、兩者皆判失敗才計，沒有「原圖可編輯」條件，沒有淨化實驗；SD3 表含 PhotoGuard-E／D、EditShield、SA（＝DAYN）、SIFM。
- 開源 MLLM 評審（Qwen2.5-VL-7B、Qwen3-VL-8B）在 50 格試判中一致率低（配件 21／50），Qwen3-VL 疑似讀錯圖、Qwen2.5-VL 從不判劣化；需改為逐題只給必要的圖再試（未做）。
- 測速（RTX 3090、512²）：每次編輯 VIBE-DistilledCFG 0.97 s、klein 蒸餾 1.55 s、HQ-Edit 2.14 s、UltraEdit 6.52 s、SD3 img2img 6.90 s、ip2p 9.85 s；去噪網路單步反傳 klein 0.83 s、VIBE 0.37 s，VIBE 全鏈反傳在 24 GB 放不下；PhotoGuard-D 每張約 6,500 s。若主表改以新編輯器為白箱目標，只打 VAE 的方法可直接移植，Mist 完整版、SIFM、DIA 需改寫，DAYN／DANP 需重新對應注意力。

**給指導教授的回覆草稿**（使用者尚未確認送出）：(1) 現有設定下各方法淨化後 ISR 偏低，將與編輯器、指令設計一起調整；(2) 會補四組顏色 baseline；(3) T(E(x)) 與 E(T(x)) 的 LPIPS 約為編輯結果 LPIPS 的 55–61%，主張改為界定 CAPGuard 改變的面向並以 ISR 補充；(4) 早期版本的灰世界等測試在另一顏色方法上，CAPGuard 會補測。

## 5. 待使用者決定

1. ISR：是否人工驗證（建議 20 張拼圖）、是否重判較寬的批次、是否擴到 306 張或 VIBE／klein，評審用 Sonnet 或改良後的開源模型。
2. 教授第 2、4 項與第 3 項雜訊底線何時重新開跑。
3. 消融是否改為三項（移掉最佳化、移掉平滑、移掉角度約束），以及 Table 3、4.6 的修改。
4. inpainting 是否放補充材料（建議 100 張、4 臂＋CAPGuard，約 60 GPU 小時，排在 ISR 設計定案後）。

已決定（2026-11-09）：主表不換白箱目標編輯器，維持 ip2p；3.3 的 Θ 寫法不改；遠端舊監看迴圈已清除（1857447 已自行結束，80936、722084 由使用者手動終止，已驗證不存在）。

## 6. 論文與文獻

- 標題：CAPGuard: Color-Based Anti-Purification Image Immunization against Diffusion-Based Editing。
- CVPR 2027：摘要 11/10、全文 11/16、補充材料 11/23（皆 AOE，台灣時間隔日 19:59）；正文 8 頁；LLM Usage Statement 必備；不放 URL；匿名；同期工作（2026-09-15 後公開）須引用討論，SPIN、SRIM、Copying as Defense 已在 Related Work。
- 時間表文件：https://claude.ai/artifact/UierLoU8gLjvznRzp9iPmL （Claude Docs 文件，2026-11-06 依手動排程更新；之後的編輯器更換與 ISR 未反映）。
- `main.bib` 已核實至 advpaint2025（AdvPaint, ICLR 2025, Jeon et al.）、alekseenko2026vibe、bfl2026flux2klein、deng2019arcface。不再被引用的 15 條保留未刪；來源較弱的條目為 `doprotect2026`、`zeng2020lut`。新增條目由 paper 通知後核實。
- `baseline/docs/reference/EVALUATION_SET_SIZES.md` 的 SIFM arXiv 編號（2512.14341）與 main.bib（2512.14320）不一致，待 baseline 核對。

## 7. 其他

- 遠端主機：`nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`（basic-1）／`-p 10102`（basic-2），repo `~/image-immunization`；遠端 `python` 不存在，用 `python3` 或 `~/venvs/wacv/bin/python`。
- 簡報不再維護。

## 8. 現況（2026-11-12 00:35），取代第 3、5 節

### 8.1 定調與命名
- 方法名改為 **ImmuLUT**（原 CAPGuard；條件名 `color3d_scale2p5_pexels`、檔名、程式不改）。線上摘要已由使用者以 ImmuLUT 版上傳；標題不變。
- 主張：施加在色彩上的擾動在淨化後被保留，加性擾動被移除。主表（Table 1）證明的是「保留」。不主張防禦有效；用詞 persist／survive／is preserved，不用 prevents／effective／outperforms（除非附數字與 CI）。全文不寫 ISR「低」，寫「不高於加性方法」。
- 內部分析不寫入正文：灰階 LPIPS、LPIPS 對 ISR 的 AUC、定性圖（`docs/isr_pilot/audit/lpips_*`）。
- 第 4 章結構：設定三段（Data and attack／Purification methods／Compared methods）→ 4.1 Results（Perturbation survival［Fig. 3］→ Edit displacement after purification［Table 1］→ Distortion and displacement［Fig. 4］→ Edit failure rate［Table 2］→ Editors with different encoders）→ 4.2 Ablation study（Table 3，內文為紅字 TODO）。第 5 章 Limitations、第 6 章 Conclusion 各一段。
- ISR 評審在正文寫成「Claude Sonnet 5.5 with human review of a sample of the judgments」；不寫 606 格、不寫複核前後變化、不提單一標註者。

### 8.2 已完成的數據（協定：ip2p、crosseditor100 100 張 × 4 句、主種子 20260812）
- 條件式 ISR（人工複核 606 格後，`docs/isr_pilot/audit/`）：不淨化 ImmuLUT 17.0%、PhotoGuard 42.0%、Mist 32.4%、DCT-Shield 21.9%、SIFM 17.6%、BlurGuard 14.8%；淨化後合計（JPEG 50＋blur σ2＋VAE-Trans）ImmuLUT 16.5%、BlurGuard 12.2%、Mist 5.9%、SIFM 5.6%、PhotoGuard 3.5%、DCT-Shield 2.6%；ImmuLUT − BlurGuard 淨化後 +4.3 pp，paired bootstrap 95% CI [+2.0, +6.9]。使用者標註中疑似誤點 4 格（woman_50__p1 vae_trans、woman_22__p3 vae_trans、woman_97__p2 jpeg50／none），未更正。
- 消融（Table 3；防禦圖 LPIPS｜LAION｜MUSIQ-AVA｜編輯 LPIPS 不淨化／JPEG 50／blur σ2／VAE-Trans）：Full 0.299｜5.31｜5.33｜0.361／0.390／0.366／0.295；A1 隨機 Θ 0.045｜5.45｜5.39｜0.088／0.151／0.084／0.064；A2 σ=0 0.408｜5.22｜5.05｜0.468／0.453／0.431／0.331；A3 不設角度界 0.472｜5.20｜4.94｜0.517／0.536／0.525／0.429；A4 單一 25° 0.449｜5.31｜5.16｜0.491／0.524／0.514／0.436；A5 不固定亮度 0.297｜5.32｜5.32｜0.358／0.389／0.364／0.293；B1 隨機色相飽和度 0.236｜5.30｜5.21｜0.297／0.334／0.317／0.254（34/100 張到不了目標失真，`results/color_baseline_random_huesat_pexels/unreachable.csv`）；B4 無約束 LUT 0.300｜5.26｜5.06｜0.390／0.384／0.330／0.315。來源：color trial `results/<條件>/`。A1–A5、B1、B4 已填入 Table 3。
- 色彩校正攻擊（ImmuLUT 對未防禦，`results/color_normalization/`）：gray-world 擾動存續比 0.996、編輯 LPIPS 0.357；Lab 正規化（Reinhard，對齊 306 張原圖統計量）0.970、0.355；淨化前 0.361。未評 ISR、未跑加性方法。正文 Limitations 仍寫「may remove the protection」，與此不符，待使用者決定是否改寫。

### 8.3 進行中
- color：B2（最佳化 hue/saturation，參數尺度 ×10）99/100、1 失敗；B3（3×3 affine，×3）8/100；兩列的 LAION／MUSIQ／VAE-Trans 補充量測在佇列讀數寫出後才開。完成後數字交 paper 填 Table 3。
- color：teaser 激進案例。三張 FLUX.2 klein 生成人像（`baseline/artifacts/synthetic_portraits/teaser_woman_{a,b,c}/`，主種子）已求解 ImmuLUT 並以 `baseline/scripts/run_teaser_edits.py` 編輯（"make this person into a bloody zombie"，另有 `_wound` 變體目錄），輸出 `baseline/artifacts/teaser/`；總覽圖由 `generate_teaser_overview.py` 產生，結果直接回報使用者。裸露類指令 executor 不跑。
- paper：方法名改 ImmuLUT（本地工作目錄有未 commit 的修改：main.tex、0_abstract、1_intro、paper_tools 三支）；teaser 換成 woman_75 紅圈放大版；Overleaf 等使用者指令才同步。

### 8.4 待使用者決定
1. 方法圖（Fig. 2）約束部分的重畫草稿是否採用（已 commit dccdfd7f、cbb4e8cb、4bb1f3d7，含 caption 修改，尚未同步 Overleaf）；色度盤上「θ saturation」的標示是否改。
2. Limitations 的色彩校正句是否依 8.2 的數字改寫；色彩校正攻擊是否補加性方法與 ISR（補充材料 11/23）。
3. 消融各列的 ISR 評審（約 14,400 格，Sonnet）是否做、何時做；「九列皆可行」共同子集排除規則是否同意；4.2 內文何時寫。
4. 壓回 8 頁（目前 11 頁，Conclusion 在第 8 頁、Table 3 在第 9 頁）。
5. teaser 最終案例（woman_75 頭盔版，或生成人像的激進版）。
6. 第 2 章是否對齊新定調；ISR 是否找第二位標註者、是否更正疑似誤點的 4 格。
7. 給指導教授的回覆是否送出。

### 8.4a 更新（2026-11-12 01:00，executor 換手時）
- 各 session：color＝pane 5、paper＝pane 6、baseline＝pane 2；舊 color（pane 1）、舊 paper（pane 4）唯讀。pane 編號會變，派工前先 `list_panes`。
- ImmuLUT 改名已完成並 push（5bf6f6f7；PAPER_HANDOVER 78808d9d）：全文、摘要、第 1 章、Fig. 1／3／4 重產，PDF 無 CAPGuard 字樣；全文 10 頁（Conclusion 第 8 頁、Table 3 第 9 頁）。線上摘要由使用者上傳，與 `sec/0_abstract.tex` 逐字一致。
- Overleaf 同步點仍為 453a196e，之後 7 個 docs/paper commit 未同步；同步需使用者在 paper pane 直接下令。
- teaser 維持 man_60；使用者在 paper pane 說過「teaser 先別放」，woman_75 紅圈版（桌面有副本）與生成人像激進版（`baseline/artifacts/teaser/teaser_woman_{a,b,c}[_wound]`，殭屍與傷口指令，已完成）都待使用者決定；使用者尚未看過激進版總覽圖。裸露類指令 executor 不跑。
- 消融：B2 99/100（woman_82 在 ±0.015 內不可行），主讀數已寫出；B3 10/100（00:54），約每 4 分鐘一張，預計 11/12 早上 6–7 點完成；兩列的 LAION／MUSIQ／VAE-Trans 在 B3 讀數後一起補，color 完成後回報九列表 → 轉 paper 填 Table 3。
- 色彩校正攻擊數字見 8.2；正文 Limitations 的「may remove the protection」仍待使用者決定是否改寫。
- teaser 腳本（baseline，c6c9477b）：`scripts/generate_synthetic_portraits.py`、`scripts/run_teaser_edits.py`、`scripts/generate_teaser_overview.py`；執行前 `export ENV_FILE=~/env.sh PY=~/venvs/wacv/bin/python`。

### 8.6 使用者決定與待辦（2026-11-12 10:00）
- 已決定：方法圖（Fig. 2）草稿與 caption 採納；Limitations 色彩校正句依 8.2 數字改寫；消融不做 ISR 評審，只補 B2／B3 數據填 Table 3；color 持續嘗試視覺效果好的 teaser，失敗再換做法；給指導教授的回覆廢除。
- 3.3 bounded angular displacement 依審閱意見改寫（eq. 6 的 r 定義、rotation 參數化與內文一致、說明 angular bound 的必要性與兩方向的不同處理、省略 tangent plane 細節），paper 負責。
- Overleaf（基底 453a196e）：Lynn 改寫 `1_intro.tex`（舊段落以註解保留於第 23–43 行；第 10 行方法名仍為 Color-based Anti-Purification Image Immunization，未用 `\capg{}`），與 repo 的改名（5bf6f6f7，舊第 9 行）重疊；`3_method.tex` 第 3、10、19、40、69 行有修改，與 repo 的 74–82 行不重疊；其餘檔案與基底一致。
- 待辦：inpainting 編輯器的 user study（補充材料）；防禦圖品質的 user study（補充材料）；主文加入定性圖。

### 8.5 工作方式補充
- executor 的監看：`scratchpad/watch_idle.sh` 每 30 分鐘查兩台空卡與租約數，租約 < 6 且有空卡連續兩次、或佇列跑完即通知；session 重開會中斷，需重啟。使用者原則是卡不要空下來。
- 被派工的 session 也會因用量上限中斷而不回應：派工後以 `read_pane` 或遠端 `~/gpu_leases`、`runtime/done` 確認；遠端沒有行程又沒有租約即表示對方停了。
- 取卡：`run_with_gpu_lease.sh` 只取完全空的卡，需 `export ENV_FILE=~/env.sh PY=~/venvs/wacv/bin/python`（遠端沒有 `python`）。
- 經 executor 轉達的 Overleaf 寫入會被權限規則擋下，需使用者在 paper pane 直接下令。
- 遠端 PDF 工具：`~/isr_audit/pylib`（PyMuPDF，`PYTHONPATH=pylib python3`）；人工標註網頁 https://claude.ai/artifact/UsWeXVkvZmZRNnPySW7zHy ；進度簡報 https://claude.ai/artifact/5zBE9fscPQ9EkCakhtUAfK 。
- 截止：全文 11/16、補充材料 11/23（AOE，台灣時間隔日 19:59）。

## 9. 現況（2026-11-15 晚），取代第 8 節的進行中與待決項目

### 9.1 Session 與分工
- pane 編號會變，派工前以 `list_panes` 依標題辨認。本節寫成時：paper＝pane 1、舊 executor＝pane 2（交接後唯讀）、color＝pane 3、baseline＝pane 4、新 executor＝pane 5。
- check session 已結束；其論文檢查報告的修正已全部合併。
- 同步 Overleaf 必須由使用者在 paper pane 直接下令；executor 轉達會被權限規則擋下。

### 9.2 論文狀態
- 全文截止 11/16 23:59 AOE（台灣時間 11/17 19:59）；補充材料 11/23 AOE（台灣 11/24 19:59）。使用者表示頁數超出不必處理（目前 11 頁）。
- Overleaf 與 repo 一致至 a2da3035；其後 repo 有 dd5ab740（Table 3 與 4.2 改為 200 張消融）尚未同步。教授 11/15 在 Overleaf 改的 1_intro、2_related、main.bib（DANP 改 TIFS 2026）已合併進 repo，線上即為該版。
- 已定案的圖：Fig. 1＝殭屍流程 teaser（man_x、JPEG q30、PhotoGuard／DIA-PT／ImmuLUT）；Fig. 5＝woman_23 helmet（編輯種子 3，7 欄）；Fig. 6＝man_37 消融兩列。
- ISR 更正版：`docs/isr_pilot/audit/judgments_human_corrected.csv`（8 格依圖更正），Table 2 與正文已用更正數字（ImmuLUT 淨化後合計 15.8%，對 BlurGuard +3.6±2.6）。
- 待使用者決定：教授改寫的第三點貢獻用語（substantially more／advantage／improved editing failure rates）與定調不一致；Fig. 4、Table 1／3 的「Edit LPIPS」標頭與正文 edit displacement 不一致（使用者已決定維持現狀）；第 2 章兩句 spatial frequency／spectrum。

### 9.3 實驗進行中
- **濾鏡化 ImmuLUT（教授 11/15 回饋）**：color trial `generate_color3d_filter_defenses.py`、`tools/run_filter_trial.sh`。四種預設濾鏡（warm b*+16 a*+4、cool、vintage、pink；`tools/generate_filter_presets.py`）；臂 a 只套濾鏡、b 濾鏡＋殘差（色相上界 2.5°、極角 25°）、c 膚色區剛體小旋轉（skinrigid）、d 求解後在膚色區混回濾鏡（skinlock）。20 張（crosseditor100 分層抽樣，`pexels_subsets/filter_trial20.txt`）× 新 4 句（helmet、rotting zombie skin、acid、bleeding head wound）已完成；ISR 由 executor 逐格判定：`C:\Users\nelso\AppData\Local\Temp\claude\78c200ab6afda56d\e21819ef-de7d-4cef-9fc0-32a8b8e5e04b\scratchpad\filter_isr\judgments20.csv`（副本在 trial results/filter_trial/prompts20/isr_judgments_executor.csv）。
  - 20 張條件式 ISR（淨化後合計）：ImmuLUT 24.1%、warm c 11.0%、warm d 10.5%、cool c 14.1%、cool d 8.9%；臉部偏移標準差 ImmuLUT 15.6、warm d 8.1、cool d 7.0；防禦圖 LPIPS ImmuLUT 0.279、warm d 0.334、cool d 0.270。結果圖在 trial artifacts/filter_trial/metric_sheets20/、comparison20/。
  - VAE 目標終值濾鏡版高於 ImmuLUT，但 ISR 較低：差距不是最佳化不足，而是目標函數（VAE 距離）與編輯失效脫鉤。偏冷原因：ImmuLUT 把膚色往灰軸／互補色推（極角方向），results/cold_bias/。
- **新任務（2026-11-15 晚派給 color）**：A. 用 FLUX.2 klein 重新生成姿勢自然的人像，跑殭屍等強烈句，目標 warm d 與 cool d 兩版淨化後都擋下、未防禦與 baseline 效果強烈，出 teaser_flow_compact 版面；B. 質性圖改以一種正常濾鏡（warm d 或 cool d）當「我們」欄，同 Fig. 5 版面。先送防禦圖小圖給 executor 篩。
- **inpainting 補充欄（baseline）**：306 張 × 7 條件（未防禦、ImmuLUT、PhotoGuard、DIA-PT、DCT-Shield、SIFM、BlurGuard），SD1.5 inpainting 50 步，不做 ISR。11/15 10:34 為讓卡暫停（queue/PAUSED、baseline_cap 0），完成 101／306；恢復時刪 PAUSED、cap 改回 6，從未完成的 205 張接續（約 8.5 小時）。
- **消融 200 張**：已完成並寫入論文（共同子集 197）；A4 man_33 因他人行程顯存不足未完成，是否補跑待使用者決定。

### 9.4 使用者研究
- 網站 `docs/user_study/site/`，伺服器與 Cloudflare 通道開在本機兩個獨立視窗（標題 survey server／survey tunnel）；網址 https://camcorder-massive-serves-radio.trycloudflare.com（通道重開會變）。
- 現行版：18 張最自然的 ImmuLUT＋2 張一般濾鏡（暖、褪色），每張兩題（願意分享、像一般發文）；前版保存在 pilot/、lpips_bins/。截至 11/15 02:10：2 人完成，ImmuLUT 願意分享 2.08、像一般發文 2.25，濾鏡 3.75／4.75。

### 9.5 工作方式（使用者偏好）
- 圖一律用 `Invoke-Item` 在本機預設看圖程式開，不用小畫家；使用者明說才用 SendUserFile 傳手機。
- 看圖判斷（ISR、防禦圖是否正常）由 executor 親自做；剔除條件：防禦圖背景／衣物偏粉紅洋紅紫、臉偏藍白或過亮、局部色塊、拼貼退化、未防禦編輯不明顯、勾叉與圖不一致。
- 條件式 ISR 只計未防禦編輯成功的格。
- 實驗室使用者 s314511048 會在 basic-1 我們持有租約的卡上開 18 GB 行程，造成顯存不足；只取完全空卡，必要時改用 basic-2。
- 本機記憶體不足時背景指令會被回收；長駐服務改用獨立視窗。

## 10. 現況（2026-11-17 02:00），取代第 9 節的進行中與待決項目

### 10.1 Session 與工作方式
- 本節寫成時：paper＝pane 1、baseline＝pane 2、舊 executor＝pane 3（交接後唯讀）、color＝pane 4；新 executor 以 `list_panes` 辨認。
- 使用者授權 executor 直接對 paper／color／baseline 下令（使用者原話：executor 是主管，有權責），不必先轉回使用者確認；例外是 Overleaf 同步，paper 的規則要求使用者在 paper pane 親自下令，executor 轉達的同步指令會被擋。
- 使用者曾切換 claude.ai 帳號；舊帳號建立的 artifact 新帳號打不開也不能更新。
- 看圖一律看原尺寸拼圖再下結論；以 112 px 縮圖判斷曾把 night_o 誤判為擋下。圖以 `Start-Process` 在本機開；放桌面只在使用者明說時。
- 使用者未回應時機器不可閒置：依教授方向自行排下一批（專案記憶 `keep-gpus-busy-on-advisor-direction`）。teaser 色調標準見專案記憶 `teaser-look-like-optimized-immulut`。
- 使用者明說時可派 subagent；ISR 評審用 Sonnet、effort medium、每個 agent 10 張拼圖、只准 Read／Write。撞到使用上限時，先刪掉行數不足的批次檔再重派新 agent。

### 10.2 論文（全文截止 11/17 19:59 台灣時間；補充 11/23 19:59）
- repo 與 Overleaf 逐字一致於 `b26dfe9f`（同步點記於 `docs/PAPER_HANDOVER.md`）。Lynn 改寫的 abstract、method、experiments、limitations、conclusion 已照收（`8030cc25`）；使用者決定 Lynn 的文字一律不動，包含消融段第一句與 Δφ 公式（舊版有 cosΔθ 項，使用者決定不查）。
- 已完成：圖表與 caption 的 33 改 K（表 4、方法總覽圖）、Fig. 1／Fig. 2 圖說移除實驗細節、Fig. 2 圖說縮短、Setup 段尾補 K=33、σ=4、θ_max=25°、φ_max=7.5° 與 ip2p 設定（`a8bb6596`、`750ac6a7`、`3be20a56`）。
- 使用者須自行把新摘要更新到 OpenReview。
- 待使用者決定 1：Qualitative Comparison 段（目前只有標題）。候選：正文沿用 Fig. 5（woman_23 安全帽，306 張判定中五個 baseline 淨化後全成功、ImmuLUT 全擋下），woman_111 p1「police suit」放 supp（未防禦與 baseline 淨化後整套制服，ImmuLUT 四欄仍為原衣物；缺點是木牆背景偏藍）。拼圖：scratchpad `isr_second100/ip2p_isr_second100_sheets/woman_111__p1.jpg`。剔除者：woman_168／114／167（背景粉紫或換人）、man_44（臉青）、woman_141（半面罩）、man_67（編輯不明顯）。
- 待使用者決定 2：縮回 8 頁。現況正文到第 9 頁中段（第 9 頁有表 4、Fig. 6、Conclusion、LLM 聲明），參考文獻在第 10–11 頁。已提給使用者的計劃，依序做、每步編譯、到 8 頁即停：(1) 第 4 章五段刪鋪陳與重述句，各縮兩到三成；(2) 第 3 章三段合併重複動機，公式不動；(3) 第 1、2 章收緊，教授的第三點貢獻不改；(4) 表 3、圖 3、圖 4 版面；(5) 仍不足時依 Lynn 指示把 Fig. 6 移到 supp（表 4 保留）；(6) 補 Qualitative Comparison 2–3 句；(7) 查 LLM 聲明是否計入 8 頁。改完以藍色標出改動編譯給使用者看，確認後才同步 Overleaf。使用者尚未批准，等回應。
- 表 2 的 ISR 待第二、三批人工複核完成後換成 306 張數字。

### 10.3 ISR 擴充到 306 張
- 第二批 100 張（`baseline/runtime/pexels_portraits/isr_second100/images.txt`，與消融擴充 100 張相同）、第三批 106 張（`isr_third106/`）；拼圖在 baseline `artifacts/pexels_portraits/ip2p_isr_{second100,third106}_sheets/`，協定與 crosseditor100 相同，未用 GPU。
- Sonnet 判定已完成並檢核（11,200 與 11,872 格，無重複、無缺格）。合併檔與結果在 scratchpad `06922550-…/scratchpad/isr_third106/`：`judgments_all306.csv`（第一批人工更正版＋第二、三批評審版）、`summary_judgments_{third106,all306}.txt`、`paired_{third106,all306}.txt`；第二批在 `isr_second100/`。遠端副本在 basic-1 `/tmp/executor_isr2/`（會被清除，應移入 `docs/isr_pilot/` 再入庫，尚未做）。
- 306 張淨化後三種合計條件式 ISR：ImmuLUT 16.2% [14.5, 17.8]、BlurGuard 11.2%、SIFM 5.7%、Mist 5.4%、PhotoGuard 3.4%、DCT-Shield 2.8%；ImmuLUT − BlurGuard ＋5.0 [＋3.4, ＋6.7]（blur2 ＋8.4、VAE-Trans ＋5.8 顯著；JPEG50 ＋1.0、不淨化 −3.0 [−5.8, −0.3]）。
- 人工複核頁（新帳號）：https://claude.ai/artifact/5c7eWE57L3K8tbjhnB1DJC，第二、三批 1038 格（ImmuLUT 與 BlurGuard 判定不同 449 對＋7 條件 × 4 淨化分層隨機 140 格；選格程式 `docs/isr_pilot/generate_isr_audit_cases.py`）。每格只選三項：1 編輯成功、2 編輯失敗（防禦成功）、3 參考編輯本身失敗（排除）。存在頁面 db 的 `labels` 集合（`success`、`ref_fail`、`accessory`），案例對照表 `isr_audit206/audit_key.csv`。使用者標完後：`ref_fail=1` 的槽位（影像 × 指令 × 淨化）整列移出分母，其餘以 `success` 取代評審標籤，算一致率與更正版 ISR（`measure_isr_paired_difference.py` 對總列數的檢查需依實際列數調整）。

### 10.4 teaser
- Fig. 1 現行：man_x 殭屍、JPEG30、PhotoGuard／DIA-PT 等失真臂。原生預算版（`teaser_man_x_zombieskin_native_tg7p5_ig1p8`）中 PhotoGuard、DCT-Shield 不淨化即變多個殭屍，DIA-PT 為唯一「淨化前擋、淨化後敗」者；ImmuLUT 在 blur2、VAE-Trans 下也被編輯，原生預算版不採用。
- 候選（桌面）：man_23 強酸 blur2（臉略偏白、襯衫偏淡藍）、man_35 強酸 jpeg30（背景偏紫、淨化後臉偏青綠）。已否定：暖色起點、skinlock、warmhalf、cool 系、dark／night 批、woman_177（效果太弱）、dark_v、night_v、night_o、dark_d、dark_h。
- color 自動鏈仍可能在跑：暖色背景 18 張（剔除 warm_e、i、k、l、o、p）四句編輯與 gore10 十句血腥指令，拼圖寫到 color trial `artifacts/teaser_filter/{teaser_warm_sheets,gore10_sheets}/`，尚未看圖。

### 10.5 supp 待辦
- user study 只有 2 人回答；inpainting 暫停在 101／306（恢復：刪 `queue/PAUSED`、`baseline_cap` 改回 6，約 8.5 小時）；真實圖片範例（由 306 張判定挑，同 10.2 的篩法）；ISR 完整結果與人工一致率；實驗細節完整版；被移出正文的圖或表；隨機 Θ 消融數字不放論文，留作 rebuttal。
