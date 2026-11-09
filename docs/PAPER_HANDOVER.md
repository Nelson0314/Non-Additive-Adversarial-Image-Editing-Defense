# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、待辦與規則，供冷啟動接手。數字一律連同協定引用。`docs/PAPER_STRUCTURE.md` 的逐節狀態未隨 2026-11-07 之後的修改更新，以本檔第 4、5 節為準。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認，不沿用記憶中的編號。

| 標題 | 角色 | 寫入範圍 |
|---|---|---|
| paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、本檔；Overleaf 專案（第 2 節） |
| executor | 進度、派工、轉達使用者決定、`main.bib` 核實、代為 push、ISR 等新實驗的協調 | `docs/EXECUTOR_HANDOVER.md`、`docs/isr_pilot/` 等 |
| color | CAPGuard 的實驗、程式與讀數 | `color/` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 klein）、主表讀數 | `baseline/` |

- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n。完成一件事後回報 executor，附 commit 號。
- 不派實驗、不改其他專案的檔案。
- `main.bib`：新增條目以 Edit 插入，加入後立即通知 executor 核實；不刪除未引用條目，列給 executor。
- 本機只做 Tectonic 編譯與依 CSV 產生表格的小腳本；彙整、畫圖、PDF 轉 PNG 在遠端 basic-1（`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`，Python 用 `~/venvs/wacv/bin/python`）。看圖時抓回 scratchpad。
- 使用者也會透過 color 或 executor 轉達文字修改；以使用者的最後指示為準。

## 2. Overleaf 協作

- 專案：Overleaf `[CVPR 2027] Color-based Image Immunization`（`6ad193f8288983a2f6af5d50`），由 `docs/paper/` 在 2026-11-06 上傳（基底 commit `51fd87d7`）。iamshuai（指導教授）與 Lynn 在線上直接編修。
- **repo 與 Overleaf 必須逐字一致。** 兩邊都有修改時，以三方合併處理（基底為上次同步點）。repo 最近一次與 Overleaf 完全一致：commit `d152ea8c`（2026-11-10；第 4–7 章以編輯器詞級修改套用、`fig/purification.pdf` 以上傳覆蓋），以雜湊比對確認。
- **線上修改一律在 Overleaf 編輯器內進行，不上傳或覆蓋檔案**（覆蓋會失去編輯歷史）。工具與步驟見 `docs/paper_tools/README.md`「Overleaf 同步」：`generate_overleaf_ops.py` 產生詞級修改清單，於編輯器頁面以 CodeMirror transaction 套用，前後比對雜湊，重新載入再驗證一次。
- 瀏覽器：使用者先以 `/chrome` 連線 Claude in Chrome；只能操作自己開的分頁群組。擴充功能會擋下看似網址參數的輸出，此時改用截圖，不繞過過濾。
- 線上任何修改前先讓使用者看渲染結果：在 scratchpad 複製 `docs/paper/`，以 `\color{blue}` 標出修改處編譯，遠端 `mutool draw -r 170` 轉 PNG，以 `Start-Process` 開給使用者看。使用者確認後才改 Overleaf 與 repo。

## 3. 方法現況

- 標題（iamshuai 於 Overleaf 修改）：「When Noise Fades, Color Remains: Purification-Resistant Image Immunization via Color Mapping」。方法名仍為 CAPGuard（`\capg`，全名 Color-based Anti-Purification Image Immunization）。
- 論文版本：2.5 倍角度上界 `color3d_scale2p5_pexels`（極角 < 25°、色相 < 7.5°），33³ RGB LUT、灰軸為極軸的色度旋轉、σ = 4 格點高斯平滑、‖c‖ 不變、損失為 ip2p VAE 編碼器後驗平均差、Adam 1800 步、每張中位 9.5 分鐘（RTX 3090，306 張）。
- 角度上界的實作（color trial `generate_color3d_defenses.py`，`SmoothAngleLut.lut()`）：先沿 e_θ 在切平面走 tan Δθ，Δθ = s_{θmax}(v_θ)；該點離球心 1/cos Δθ、離灰軸 r = sin(θ+Δθ)/cos Δθ；再沿方位切向加位移，Δφ = s_{φmax}(v_φ / r)；最後正規化回 ‖c‖。方位步會使極角再增加一個二階小量（φmax = 7.5° 時 cos Δφ ≥ 0.991），論文未寫。論文 3.3「Bounded Angular Displacement」依此改寫（commit `1d4f619a`），不寫 r 的公式、不寫具體角度值；使用者指定的說法見第 6 節。

## 4. 現況（編譯 11 頁）

**正文超過 8 頁**：Conclusion 在第 9 頁，主要因 Lynn 改寫的 Method 較長。

| 位置 | 現況 |
|---|---|
| Abstract | 倍數 1.32 to 3.23（CAPGuard Purified* 編輯 LPIPS 0.364 ÷ 各加性方法；下端 BlurGuard 0.275、上端 DCT-Shield 0.113），黑字 |
| 1 Introduction | 貢獻第二條同摘要；第 4 行只引用比較過的方法；編輯器引用 VIBE、FLUX.2 klein |
| 2 Related Work | Lynn 改寫：Immunization（含 SPIN、SRIM、Copying as Defense 三篇同期工作）、Countermeasures（開頭定義淨化）、Color Transformations；美學段已註解掉。舊文字以 `%` 保留 |
| 3 Method | Lynn 改寫：開頭 overview、3.1 威脅模型（**指標定義段被註解掉**）、3.2 3D LUT-based Parameterization（K×K×K，未寫 K = 33）、3.3 Geometry-Constrained（Gray-Axis Rotation、Smooth Rotation Field、Bounded Angular Displacement）、3.4 Overall Immunization Objective；**3.5 Optimization 整節被註解掉**（Adam、學習率、512²、求解時間不在正文）。5 個公式 label 已改為 eq:rotation、eq:field、eq:tangent、eq:increments、eq:protected |
| Table 1 | 上段 ip2p 306 張全部方法；下段 VIBE-DistilledCFG 與 FLUX.2 [klein] 4B 各 306 張（Mist、PhotoGuard、DCT-Shield、BlurGuard、CAPGuard）；全部黑字；CAPGuard 列標「(Ours)」 |
| Fig. 1–4 | Fig. 1 teaser（man_60、helmet、caption 由 iamshuai 改寫）；Fig. 2 方法總覽（man_60 的 2.5 倍 LUT）；Fig. 3 失真—位移（19 點擬合 r = 0.968，CAPGuard 在線上方 0.006）；Fig. 4 (a) 2.5 倍存續比，(b–d) 仍為 8 張、1.0 倍（紅字註記） |
| 4.1–4.5 | 4.1 Metrics 寫「定義見 3.1」，但 3.1 的定義已被註解掉；4.5 為 VIBE、klein 結果 |
| Table 2 消融 | 完整方法列為 2.5 倍；A（隨機 Θ）為 1.0 倍上界、紅字；B–E 未跑 |
| 紅字殘留 | `4_experiments.tex` 8 處（消融表、Fig. 4 b–d）、`5_discussion.tex` 1 處（Limitations 計數） |
| 補充材料 | `X_suppl.tex` 未編入 PDF，仍提 UltraEdit／FLUX |

## 5. 待辦

1. **正文壓回 8 頁**：需與使用者討論刪減處（Method 篇幅、被註解的舊文字不佔版面）。
2. **指標定義**：3.1 的定義段被 Lynn 註解掉，4.1 仍指向 3.1；需使用者決定放回 3.1 或移到 4.1。
3. **3.5 Optimization**：實作細節目前不在正文；需使用者決定是否恢復或精簡後放回。
4. **紅字**：消融表 B–E 與 A 的 2.5 倍版本、Fig. 4 (b–d) 的 306 張 2.5 倍強度掃描、Limitations 計數，都等使用者決定後續實驗。
5. **教授 11/08 回饋**（executor 協調，見 `docs/EXECUTOR_HANDOVER.md`）：ISR 指標、簡單顏色 baseline、E(T(x)) 與 T(E(x)) 比較、顏色反向處理。結果出來後可能需要新增正文或表格。
6. `main.bib` 中不被引用的條目 15 條，已列給 executor，未刪。
7. `docs/PAPER_STRUCTURE.md` 需依第 4 節更新。

## 6. 使用者的寫作偏好與決定（2026-11-06 至 11-09）

- 「沒用到不提」：Related Work 與 Introduction 只引用正文有比較或使用的方法；CVPR 2027 規定 2026-09-15 後公開的同期工作須引用並討論，所以保留 SPIN、SRIM、Copying as Defense（已核對上線日期皆在 10 月）。
- 正文不提補充材料；指標只寫正文有數字的（LPIPS、SigLIP、ArcFace、FID、保留率；防禦圖 LPIPS、LAION、MUSIQ）。
- Related Work 第一段結尾：多數方法的保護是加性擾動或像素格上的空間結構，對各種淨化的穩健性可能不一；不逐篇寫「某某沒評估淨化」。
- 方法名在圖表標「CAPGuard (Ours)」，正文不加。
- 跨編輯器：UltraEdit、FLUX.1 Kontext 停用，改為 VIBE-DistilledCFG 與 FLUX.2 [klein] 4B（306 張、add 句型四句、512²）。
- 3.3 Θ 的寫法不改；主表維持 ip2p，不換白箱目標編輯器。
- Bounded Angular Displacement 段的定案寫法：先定義 s_a，並說明 a、t（單位球切向位移）、tanh 壓切向長度、arctan 轉角度、t → 0 時 s_a(t) ≈ t；兩式 Δθ = s(v_θ)、Δφ = s(v_φ / r) 寫成同一形式；r 只寫成「極角位移後該點到灰軸的距離」，不寫公式、不說「沿緯線圈」（程式是沿切向加位移）；不寫 25°／7.5°；不寫「so that the bound φmax applies to the hue angle …」那句。
- 修改要保留原作者（Lynn、iamshuai）的措辭與敘事結構，只補必要內容；改稿前先給渲染預覽。
- 正文英文、學術語氣；避免冒號式揭示、「not X but Y」、口語、比喻；不寫時間相依字眼；數字附協定。
- 表格：固定字級、不用 `\resizebox`；各區塊最佳平均粗體；CAPGuard 列底色 `oursrow`。
- 圖：論文字體（STIX／texgyretermes）、印刷 ≥ 7 pt；加性方法紅 `#A94438`、CAPGuard teal `#1F6F7C`。
- 暫代數字用 `\prelim{值}{註解}`、`\prelimnote{註解}`；投稿前設 `\prelimfinaltrue`。
- 正文上限 8 頁；全文不放 URL；commit message 英文，內容與文件分開 commit，push 由 executor 執行。

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
- Fig. 3／4 重出（遠端 `~/paper_figures`）：`~/venvs/wacv/bin/python in/generate_paper_figures.py --points in_pexels/points_blurguard.csv --aligned in_pexels/condition_summary_blurguard.csv --color-summary in_pexels/color_summary_scale2p5.csv --survival in_pexels/perturbation_survival_scale2p5.csv --out-dir out_pexels_ours`（Fig. 4 只有存續比一格；強度掃描只有 8 張，已自圖中移除），再 scp 兩個 PDF 到 `docs/paper/fig/`。
- teaser 重出：`~/venvs/wacv/bin/python in/generate_pexels_teaser_figure.py --image man_60 --prompt 2 --layout protected-purified --no-lpips --cite --scale2p5 --arms photoguard_encoder,mist,dia_pt,dct_shield,blurguard --out-dir out_teaser_ours --name teaser`。

## 9. 讀數來源

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15；VAE-Trans 另列；306 張 Pexels 人像。FID、KID 以 GPU 特徵計算（與 CPU 最大差 0.00014，`baseline/results/pexels_portraits/distribution_device/`）。

| 內容 | 來源 |
|---|---|
| 9 臂主表 | baseline 遠端 `runtime/pexels_portraits/summary/main_table.csv` |
| CAPGuard 2.5 倍 | baseline 遠端 `runtime/pexels_portraits/summary/color3d_scale2p5/`（`main_table.csv`、`per_purifier.csv`） |
| BlurGuard 306 張 | baseline 遠端 `runtime/pexels_portraits/blurguard_306/`（`summary.csv`、`per_purifier.csv`） |
| VIBE、klein | `baseline/results/pexels_portraits/vibe_distilled/`、`klein_distilled/` |
| 求解時間 | color trial `logs/*.log`（中位 9.53、p90 10.20 分鐘，n = 306） |
