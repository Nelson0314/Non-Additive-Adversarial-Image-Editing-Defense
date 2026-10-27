# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、背景、現況與規則，供冷啟動接手。內容以 2026-10-22 的狀態為準；數字一律連同協定引用。

## 1. 角色與分工

| pane | 角色 | 寫入範圍 |
|---|---|---|
| 1 executor | 專案管理、文獻管理、派工給 color 與 baseline；不做實驗 | `docs/research_tree.html`、`docs/pseudo_paper.html` |
| 2 color（pane 4 為備援，接手前不寫入） | 本方法（色彩查找表）的實驗與方法設計 | `color/` |
| 3 baseline | 11 個加性方法、主表、等失真臂、跨模型 baseline | `baseline/` |
| 撰寫 session（本文件的讀者） | 論文正文 | `docs/paper/` |

- 需要數據、圖或方法細節時，直接以 multi-agent 的 `send_to_pane` 向 color 或 baseline 索取，並要求附路徑與協定；同時以一行副本告知 executor（pane 1）。
- 不派實驗、不改其他專案的檔案。實驗方向與方法設計由使用者與 color、baseline 直接溝通。
- 新引用的文獻（標題、venue、arXiv）回報 executor，由 executor 登錄。

## 2. 檔案

| 檔案 | 內容 |
|---|---|
| `docs/paper/` | **唯一的 tex 正本**，撰寫只改這裡。CVPR 2026 author-kit（github.com/cvpr-org/author-kit，commit 2917585）的 LaTeX 專案：`main.tex`、`sec/0_abstract.tex`–`6_conclusion.tex`、`X_suppl.tex`、`fig/teaser.tex`、`main.bib`；`cvpr.sty`、`ieeenat_fullname.bst`、`preamble.tex` 為官方檔案，不修改 |
| `C:\Users\nelso\OneDrive\桌面\paper.pdf` | 最新渲染結果，由 `docs/paper_tools/run_paper_preview.py` 自動複製。桌面的 `overleaf_paper\` 與 zip 不再維護 |
| `docs/pseudo_paper.html` | 中文論文骨架：敘事線、各節要點、實驗與狀態、圖表配置、審稿風險（artifact https://claude.ai/artifact/AxbqUjtR2uw4rJWBZA5239 ） |
| `docs/PAPER.md` | 舊的 Markdown 草稿，已由 `docs/paper/` 取代，不再維護 |
| `docs/research_tree.html` | 研究樹與文獻總表（10/16 版，部分狀態已過時） |

`main.bib` 是依專案文獻表寫的暫用版：標 `TODO` 的作者與標題未查證，投稿前須逐條換成出版社或 DBLP 的 BibTeX。

## 3. 背景

- 問題：影像免疫。防禦方在發布前處理影像，使攻擊方以公開權重的擴散編輯器依指令編輯時，編輯結果偏離未防禦時的結果。白盒、外掛（不改攻擊方權重）；防禦方看不到攻擊指令，訓練只用空字串或類別名。
- 既有方法：以 PGD 在像素上最佳化加性、範數有界的擾動。
- 本方法：逐影像最佳化的 3D 色彩查找表（33³ LUT，全域色彩映射），目標為編輯器 VAE 編碼器的潛向量位移；自然度以可行集合加 ADMM 投影約束，約束數值取自真實調色濾鏡（RawTherapee 底片模擬 HaldCLUT）的實測範圍。
- 論文敘事（使用者與指導教授定）：
  1. 既有方法是逐像素加性擾動，偏高頻，淨化（JPEG、blur、crop、VAE 重建）後失效；
  2. 失真—位移平面上各方法落在同一條直線，預算只讓點沿線移動，因此要在等失真下比較；
  3. 淨化分析與抗淨化；
  4. 跨模型泛化；
  5. 共用查找表（universal LUT）作為延伸。
- 「高頻」論點可以寫，但需補證據（擾動徑向功率譜，`sec/1_intro.tex` 已標 TODO）。BlurGuard 經 VAE-Trans 後 0.191，與色彩映射相近，是需要正面處理的競爭者。

## 4. 方法現況

- **最終方法 `color3d`（2026-10-27 正式改名；原條件名 `color3d_smooth_polar10_hue3`）**，取代下方的 smooth_angle6。
  - 做法同 SmoothAngleLut：33³ RGB LUT，σ = 4，損失只有目標，沒有投影、沒有 LPIPS 上下界。
  - 以黑點為原點、灰軸為極軸，分開兩個上界：極角（彩度方向）變化 < 10°，方位角（色相）變化 < 3°；‖c‖ 不變。
  - 使用者看圖判定比黑點 6° 自然。
  - 8 張讀數：
    - 防禦圖：LPIPS 0.296、SSIM 0.942、ΔE00 12.9、DISTS 0.131、ArcFace 0.972。
    - 編輯結果 LPIPS：淨化前 0.374、七道 0.367、最差 crop_resize0.1 0.339、VAE-Trans 0.301；保留率 1.020。
    - 美學：LAION 5.48（原圖 5.48）、BRISQUE 14.1（原圖 16.2）。
    - 膚色色相偏移 33.4°。
  - 擾動 99.8% 能量 < 0.05 週期／像素；淨化後能量 blur2 0.996、VAE-Trans 0.802。
  - 檔案：color trial `results/additional_metrics/color3d/`、`results/perturbation_spectrum/color3d/`。
- 前一個選定：`color3d_smooth_angle6`（2026-10-26），已被取代。
  - 33³ LUT 色空間平滑色度旋轉（SmoothAngleLut），以黑點為原點只轉色度方向，‖c‖ 不變，轉角 < 6°，σ = 4。
  - 沒有投影、沒有 LPIPS 上下界，損失只有目標。綠側原點因膚色偏黃（色相偏移 54°）未採用。
  - 8 張讀數：
    - 防禦圖：LPIPS 0.307、SSIM 0.971、PSNR 24.1、ΔE00 15.5、DISTS 0.138、ArcFace 0.968。
    - 編輯結果 LPIPS：淨化前 0.373、七道 0.389、最差 rotate15 0.323、VAE-Trans 0.329；保留率 1.068。
    - 美學：防禦圖 LAION 5.38（原圖 5.48）。
    - 膚色色相偏移 42.7°。
  - 擾動頻譜：99.8% 能量 < 0.05 週期／像素；blur2 後保留 0.996、VAE-Trans 0.763。
  - 檔案：color trial `results/additional_metrics/color3d_smooth_angle6/`、`results/perturbation_spectrum/color3d_smooth_angle6/`；README〈色空間平滑的色度旋轉〉。
  - 成果簡報內容仍為 green17，尚未改版。
- 前一個選定（已被取代）：`color3d_smooth_angle17_green`，產物已刪除。
  - 自然度由參數化保證：沒有投影、沒有 LPIPS 上下界，損失只有目標。
  - 做法：原點取灰軸往綠方向平移 √(2/3)；自由參數經色空間 3D 高斯模糊（σ = 4 格點）成為旋轉場，只在垂直灰軸的平面內旋轉，亮度不變，轉角 < 17°；中性色只沿黃—藍軸位移。
  - 8 張讀數：防禦圖 LPIPS 0.309；編輯結果 LPIPS 淨化前 0.358、七道 0.375、最差一道 rotate15 0.353、VAE-Trans 0.322；保留率 1.062。
  - 擾動頻譜：99.6% 能量 < 0.05 週期／像素；blur2 後保留 0.994（dct_shield 0.197）。
  - 定義與掃描見 color trial README〈色空間平滑的色度旋轉〉；簡報 https://claude.ai/artifact/VfAGamPdpErLFqNV9iVxdE 。
  - 以下 δ 版改名 `color3d_phr`，為對照。
- 前一版定案：`color3d_phr`（2026-10-24 定案時名為 color3d；原條件名 `principled_unified_rgb_delta30p6`）。論文與簡報中 0.296／0.378／0.363／0.965／0.276 這組數字屬於 color3d_phr。
  - 33³ RGB LUT，目標為 ip2p VAE 編碼器潛向量位移。
  - 約束只有單一常數 δ = 30.6（8 位元 RGB）的三條上界：
    - 影像上相鄰像素的色差變化 ≤ δ；
    - 逐像素到灰軸距離的增加 ≤ δ；
    - 膚色位移 ≤ δ。
  - 另加防禦圖 LPIPS 0.27–0.30；不約束色空間平滑。
  - 8 張讀數：防禦圖 LPIPS 0.296；編輯結果 LPIPS 淨化前 0.378、七道平均 0.363、blur2 0.330、VAE-Trans 後 0.276；保留率 0.965；lut9 對齊後 0.162。
  - 數字與已刪除的候選（參數化、底片包絡、比值與差值版、Lab／YCbCr 版）見 color trial `README.md`〈兩條原則的約束〉。
- 原始 color3d（現名 color3d_orig）（逐條懲罰進損失、PHR 求解）在論文中是「約束形式」的消融：說明逐條懲罰會造成訓練不穩定（man_00 的 180 個檢查點只有 1 個可行），由此引出投影作法。表中標為「Penalty LUT (ablation)」。
- color3d_adaptive_set 為消融。
- 「Ours」改用最終 color3d 的數字；本文件第 5 節中的 color3d_orig 數字屬於消融。

## 5. 已有數據（8 張 portraits、ip2p EulerAncestral 50 步、s_t 7.5、s_i 1.8、主種子 20260812、FaceLock 配件四句）

| 內容 | 數字 | 位置 |
|---|---|---|
| 失真—位移直線 | 38 點合併：編輯結果 LPIPS ≈ 0.865 × 防禦圖 LPIPS ＋ 0.077，r = 0.955；單一方法 r 0.987–1.000；DISTS 軸 r 0.699、PieAPP 軸 0.519 | `baseline/results/distortion_displacement/` |
| 擾動存續比 LPIPS(P(x̃),P(x))/LPIPS(x̃,x) | 七道平均：11 臂 0.288–0.535，color3d_orig 0.838；VAE-Trans：11 臂 0.019–0.092，color3d_orig 0.457 | `baseline/results/aligned/perturbation_survival.csv` |
| 淨化強度掃描 | color3d_orig 全程 0.245–0.340；加性方法在 q10、σ3、×0.25 降到 0.03–0.14 | `baseline/results/sweeps/ip2p/purifier_strength/retention.csv` |
| 等失真編輯位移 | 11 臂淨化前 0.24–0.39、七道 0.12–0.24、VAE-Trans 0.02–0.07；color3d_orig 0.341／0.302／0.185 | `baseline/results/aligned/`、color trial `results/comparison/cuda/` |
| inpainting | color3d_orig 0.343／0.288–0.323／0.241 | `baseline/results/color_conditions/` |
| 隨機 LUT 對照 | 同失真隨機 LUT 經 VAE-Trans 後 0.196；最佳化比隨機高 32–61% | color trial `README.md`「消融與對照」 |
| 感知指標 | lut9 對齊後 color3d_orig LPIPS 0.137（11 臂 0.282–0.358）；調色濾鏡參照帶；LAION、MUSIQ、CLIP-IQA、ImageReward、HPSv3、PickScore | color trial `results/perceptual_quality/`、`results/human_aligned_metrics/` |
| VAE 編碼器 | ip2p、SD inpainting、SD 2.1-base 權重相同；FLUX、SD3 各不同 | color 回報 |
| 評估效度 | 等失真點 32 格中 19 格所有方法都未擋下配件 | `C:\Users\nelso\OneDrive\桌面\evaluation_issues_report.md` |

進行中：約束候選 A／B／C 的先驗（color）；FLUX／SD3 上 encoder 類 baseline 與 mist_textural（baseline）。

## 6. 開工前必讀

專案：
- 根目錄 `CLAUDE.md`、`README.md`；`baseline/STATUS.md`、`color/STATUS.md`。
- `baseline/docs/reference/BIBLIOGRAPHY.md`、`BASELINE_PROVENANCE.md`。
- color trial 的 `README.md`、`HANDOVER.md`、`RELATED_WORK_NOTES.md`、`LITERATURE_COLOR_CONSTRAINTS.md`、`design_notes/principled_constraint_proposal.md`。
- `docs/pseudo_paper.html`。

寫作參考文獻：
- 須讀摘要、引言與相關工作：
  - BlurGuard（NeurIPS 2025）https://arxiv.org/html/2511.00143
  - Hönig et al.（ICLR 2025）https://arxiv.org/html/2406.12027
  - ReColorAdv（NeurIPS 2019）https://ar5iv.labs.arxiv.org/html/1906.00001
  - PhotoGuard（ICML 2023）https://ar5iv.labs.arxiv.org/html/2302.06588
  - FaceLock（CVPR 2025）https://arxiv.org/html/2411.16832
  - DiffusionGuard（ICLR 2025）https://arxiv.org/html/2410.05694
  - Natural Color Fool（NeurIPS 2022）https://ar5iv.labs.arxiv.org/html/2210.02041
- 寫法反例：EditShield https://arxiv.org/html/2311.12066

## 7. 寫作規則

- 正文英文。人類學術敘事，避免下列寫法：
  - 冒號式揭示、問答式鋪陳、「not X but Y」句型；
  - Moreover／Notably 類連接詞；
  - 比喻、口語、無出處的主張。
- 數字附協定；不寫時間相依字眼。
- 不下「方法成立／不成立」的判定；數據完整陳述。
- 不做色調／白平衡類淨化（使用者決定），在 Limitations 明寫。
- commit message 用英文；push 由使用者執行。

## 8. 第一項工作（使用者指定）

0. **編修與預覽流程**：手動上傳 Overleaf 不順，先解決這一步。
   - 若使用者的 Overleaf 方案有 Git 整合（需付費方案或學校授權），可設定以 git 直接推送到 Overleaf 專案。這需要使用者提供專案的 git 網址與 token，token 不得寫入任何入庫檔案。
   - 若沒有，寫一個本地渲染小工具：建議用 Tectonic（單一執行檔，自動取得 LaTeX 套件），一個指令就能把桌面工作副本編成 PDF，並在檔案變更時自動重編。工具與使用說明放在 `docs/paper_tools/`。
   - 先向使用者確認有沒有 Overleaf Git，再動手。
1. **Abstract**：參照同領域論文，摘要不放結果數據與分析數字。改寫為：問題與領域現況 → 既有方法的侷限 → 本文的想法與方法 → 評估範圍。
2. **Introduction**：降低白話程度，刪去日常情境的描寫（例如以一句指令改圖的例子）。改從研究者的立場，以此領域的研究現況開場：免疫方法的發展、加性擾動的共同形式、淨化研究的發現、比較協定的問題，再引出本文。
3. 第 2 節以後暫不處理。完成後同步桌面副本，並向使用者回報修改摘要。
