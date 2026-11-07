# 論文撰寫交接

本文件交代論文撰寫 session 的角色、檔案、現況、待辦與規則，供冷啟動接手。數字一律連同協定引用。各節的逐條狀態見 `docs/PAPER_STRUCTURE.md`。

## 1. 角色與分工

pane 編號會變動，送訊前一律以 multi-agent 的 `list_panes` 依標題確認，不沿用本表或記憶中的編號。

| 標題 | 角色 | 寫入範圍 |
|---|---|---|
| paper | 論文撰寫（本文件的讀者） | `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、本檔 |
| executor | 專案管理、派工、文獻管理；`main.bib` 的唯一核實者；代使用者轉達決定；代為 push | `docs/EXECUTOR_HANDOVER.md` 等 |
| color | 本方法（CAPGuard）的實驗與讀數 | `color/` |
| baseline | 加性方法、Pexels 主表、跨編輯器讀數 | `baseline/` |

- 需要數字、圖或方法細節時，以 `send_to_pane` 向 color 或 baseline 索取，要求附路徑、協定與影像數 n；完成一件事後回報 executor（附 commit 號，訊息行數依 executor 要求）。
- 不派實驗、不改其他專案的檔案。
- `main.bib`：新增條目以 Edit 插入（不整檔重寫），加入後立即通知 executor 核實；未引用的條目交 executor 處理。
- 本機不跑計算（Tectonic 編譯與依 CSV 產生表格的小腳本除外）：資料彙整、畫圖、PDF 轉 PNG 都在遠端 basic-1（`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`，Python 用 `~/venvs/wacv/bin/python`；`python3` 無 pandas）；看圖時抓回 session 的 scratchpad。

## 2. 檔案與工具

| 路徑 | 內容 |
|---|---|
| `docs/paper/` | 唯一的 tex 正本（Overleaf 上傳整個目錄，`build/` 除外）。CVPR 2027 author kit v1（`cvpr.sty`、`ieeenat_fullname.bst`、`preamble.tex` 不修改）。`main.tex` 定義 `\capg`（方法名 CAPGuard，改名只改此處）、`\prelim`、`\prelimnote`（第 8 節）與表列底色 `oursrow`；`\input{fig/teaser}` 已啟用；補充材料 `\input{sec/X_suppl}` 為註解（未編入 PDF） |
| `docs/paper/fig/` | `teaser.pdf`＋`teaser.tex`（Fig. 1）、`method_overview.pdf`（Fig. 2）、`distortion_displacement.pdf`（Fig. 3）、`purification.pdf`（Fig. 4） |
| `docs/paper_tools/README.md` | 各腳本用途與執行位置 |
| `run_paper_preview.py` | Tectonic 編譯，成功後複製到桌面 `C:\Users\nelso\OneDrive\桌面\paper.pdf` |
| `generate_main_table.py` | Table 1（`3_method.tex` 的 `tab:main`）：本機執行，`--main-table <baseline summary/main_table.csv> --blurguard baseline/results/pexels_portraits/blurguard/summary.csv`；UltraEdit、FLUX 區塊的數字寫在腳本內並註明來源 |
| `generate_per_purification_table.py` | 補充材料逐道淨化表 `sec/X_suppl_purification.tex`（`--per-purifier`、`--main-table`） |
| `generate_paper_figures.py` | 失真—位移圖與淨化分析圖，遠端 `~/paper_figures/` 執行（指令見第 9 節）；`--ours-extra` 畫第二個 CAPGuard 操作點 |
| `import_pexels_figure_inputs.py` | 把 baseline Pexels 讀數轉成上一個腳本的輸入（`~/paper_figures/in_pexels/`） |
| `generate_pexels_teaser_figure.py` | teaser 與候選排序（`--rank`）；`--layout protected-purified --no-lpips --cite --scale2p5 --arms photoguard_encoder,mist,dia_pt,dct_shield,blurguard` 為現行版面 |
| `generate_teaser_overview.py` | 多組 teaser 候選疊成一張總覽（選圖用） |
| `method_overview/`、`generate_method_overview_panels.py` | 方法總覽圖的 TikZ 與資料圖塊 |

遠端 `~/paper_figures/`：`in/`（上傳的腳本與 8 張舊輸入 `retention.csv`）、`in_pexels/`（`points.csv`、`condition_summary.csv`、`color_summary.csv`、`perturbation_survival.csv`、`perturbation_survival_native_dct_shield_y.csv`）、`in/retention_native_dct_shield_y.csv`、`out_pexels/`、`out_teaser/`、`out_teaser_round3/`（7 組候選與 `teaser_overview.png`）。

## 3. 方法現況

方法名 CAPGuard（Color-based Anti-Purification Image Immunization）。標題：「CAPGuard: Color-Based Anti-Purification Image Immunization against Diffusion-Based Editing」。

使用者定案（11/6）：論文只保留 2.5 倍角度上界的版本 `color3d_scale2p5_pexels`（極角 < 25°、色相 < 7.5°，數值待與 color 的程式核對）；1.0 倍（`color3d_pexels`，極角 < 10°、色相 < 3°）從正文與表圖移除。其餘設定：33³ RGB LUT、灰軸為極軸的色度旋轉、σ = 4 格點高斯平滑、‖c‖ 不變、損失只有 ip2p VAE 編碼器後驗平均差、Adam 1800 步、每張約 9.4 分鐘（單張 RTX 3090，1.0 倍的實測）。

消融（使用者定案）：A 拿掉最佳化、B 拿掉平滑（σ＝0）、C 拿掉角度上界、D 不分極角／色相、C＋D、E 拿掉 ‖c‖ 限制；只量防禦圖（LPIPS、LAION、MUSIQ）與淨化前編輯 LPIPS，306 張，以 2.5 倍為基準重跑。逐項懲罰與可行集合投影已移除。

## 4. 現況（編譯 11 頁；正文含 Conclusion 止於第 8 頁）

| 位置 | 現況 |
|---|---|
| Fig. 1 teaser | man_60、指令 helmet；上列原圖＋6 法防禦圖，下列 blur σ=2 淨化後編輯；PhotoGuard、Mist、DIA-PT、DCT-Shield、BlurGuard、CAPGuard（2.5 倍），方法名附作者年份，不標 LPIPS。目視：CAPGuard 淨化後仍加上頭盔，差異主要是色偏 |
| Table 1 | 上段 ip2p 306 張全部方法：防禦圖 LPIPS、LAION、MUSIQ；編輯 LPIPS（None、Purified*、VAE-Trans）、SigLIP、ArcFace（Purified*、VAE-Trans），附 95% 區間小字（Purified* 跨七道、其他跨影像）；下段 UltraEdit（50 張）、FLUX 512（34 張）只列 PhotoGuard、DCT-Shield、DCT-Shield-Y、Mist、CAPGuard。保留率、CLIP-S、CLIP-I、FID、SSIM、PSNR 移補充。CAPGuard 列為 1.0 倍、紅字 |
| Fig. 3 失真—位移 | 實心星＝1.0 倍、空心星＝2.5 倍（82 張，0.302／0.367） |
| Fig. 4 淨化分析 | (a) 分組直方圖（DCT-Shield-Y 原生預算單獨著色）；(b–d) 8 張強度掃描，最左為未淨化點，DCT-Shield-Y 原生 |
| Table 3 消融 | 4 欄（防禦圖 LPIPS、LAION、MUSIQ、淨化前編輯 LPIPS）；A 為 306 張（MUSIQ 缺），D 以單角 6° 8 張暫代，B／C／C＋D／E 未跑 |
| 摘要與貢獻 | 倍數「1.37 to 2.60」為紅字，待重算 |
| 補充材料 | 協定細節、防禦圖 PSNR／SSIM／NIMA／NIQE 表、逐道淨化表；Pexels 選取規則與授權為 `\TODO` |

## 5. 2.5 倍與 BlurGuard 306 張到齊後要改的項目

color 會送 `color/trials/lut3d_defense_and_naturalness/results/color3d_scale2p5_pexels/per_image/main_table.csv`（306 張），baseline 會送 BlurGuard 306 張與以 2.5 倍為基準的消融、跨編輯器讀數。

1. Table 1：上段 CAPGuard 2.5 倍、BlurGuard 306 張；下段跨編輯器改為 VIBE-DistilledCFG（306 張，`--vibe baseline/results/pexels_portraits/vibe_distilled/summary.csv`）與 FLUX.2 [klein] 4B（`--klein`，未給時紅字佔位），方法為 Mist、PhotoGuard、DCT-Shield、BlurGuard、CAPGuard。UltraEdit／FLUX.1 Kontext 已停用。待辦：klein 讀數到齊後以 `--klein` 重產，並補 4.5 與 Conclusion 的 klein 句。
2. （已完成）Fig. 3：CAPGuard 只畫 2.5 倍一點，BlurGuard 只有等失真臂一點；遠端輸入 `in_pexels/points_blurguard.csv`、`condition_summary_blurguard.csv`、`color_summary_scale2p5.csv`、`perturbation_survival_blurguard.csv`（BlurGuard 列為 baseline `blurguard_306/per_purifier.csv` 的逐道平均），輸出 `out_pexels_blurguard/`。擬合 19 點，斜率 1.037、截距 0.049、r＝0.968；CAPGuard 在線上方 0.006。
3. Fig. 4：(a) 的 CAPGuard 仍為 1.0 倍（已加 `\prelimnote`；2.5 倍存續比七道 1.057、VAE-Trans 0.827 已寫入 4.4 正文）；(b–d) 強度掃描為 8 張、1.0 倍，需 306 張與 2.5 倍讀數。
4. 4.2–4.4 與 Conclusion 的 CAPGuard 數字已換 2.5 倍（0.353、0.364、1.058、0.288、7%、LAION 5.343、MUSIQ 5.308、FID 44.7／38.0、存續 1.057／83%）。待辦：BlurGuard 306 張、4.4 強度掃描、4.5 跨編輯器。
5. （已完成）摘要與貢獻第二條倍數為 1.32 to 3.23（CAPGuard Purified* 編輯 LPIPS 0.36383 ÷ 各加性方法 Purified*；下端 BlurGuard 0.27459、上端 DCT-Shield 0.1125），黑字。
6. （已完成）Method 3.3 上界為極角 < 25°、色相 < 7.5°（color 核對 `generate_color3d_budget_defenses.py:13–16`），求解時間 9.5 分鐘（中位，231 次求解）；Fig. 2 改用 man_60、`color3d_scale2p5_pexels` 的 LUT，像素對由 `generate_method_overview_panels.py` 自動挑選。
7. Table 2 消融：完整方法列已換 2.5 倍；A（隨機 Θ）為 1.0 倍上界、已改紅字；其餘待以 2.5 倍為基準的新讀數。
8. Limitations 的「19 of 32」計數（8 張）若有新數字則換。

## 6. 交付物

- 桌面 `paper.pdf`：`run_paper_preview.py` 成功後自動覆寫。
- 簡報 PNG（executor 取用）：前一 session 的 scratchpad `deck_figures/`，可能隨 session 清除；重出時以遠端 `~/paper_figures/out_pexels/`、`out_teaser/` 為準，表格 PNG 以 Tectonic 單獨編譯表格本體再於遠端 `mutool draw -r 300` 轉出。

## 7. 讀數來源

協定（未另註者）：InstructPix2Pix、FaceLock 配件四句、主種子 20260812；七道淨化＝JPEG 80／50／30、blur σ1／2、crop_resize0.1、rotate15，VAE-Trans 另列。

| 內容 | 來源 |
|---|---|
| Pexels 306 張主表（含 `_ci95`） | baseline 遠端 `runtime/pexels_portraits/summary/main_table.csv`、`main_table_supplementary.csv`、`per_purifier.csv` |
| BlurGuard（100 張） | `baseline/results/pexels_portraits/blurguard/summary.csv`（commit e84f78c） |
| UltraEdit 50 張 | baseline 遠端 `runtime/pexels_portraits/ultraedit_readouts/summary.csv`；CAPGuard：color `results/cross_model_pexels/sd3/transfer50/summary.txt` |
| FLUX 512、34 張 | `baseline/results/pexels_portraits/flux_512/summary.csv`（commit ccb4ad7） |
| 原生 DCT-Shield-Y 存續比與強度掃描 | baseline 遠端 `runtime/pexels_portraits/summary/native_dct_shield_y/`、`runtime/purifier_sweep_native_dct_shield_y/` |
| 2.5 倍（82 張，Fig. 3 空心星） | color `results/color3d_scale2p5_pexels/per_image/main_table.csv` |
| Fig. 2 的 LUT 與 x、x̃ | color `artifacts_pexels/defense_shards/color3d_scale2p5_pexels/man_60/` |
| teaser 影像 | 加性與 BlurGuard：`baseline/artifacts/pexels_portraits/<根>/`；CAPGuard 2.5 倍：color `artifacts_pexels/defense_shards/color3d_scale2p5_pexels/`、`edits_by_image/color3d_scale2p5_pexels/` |

## 8. 寫作規則

- 正文英文，學術、精確；避免冒號式揭示、問答式鋪陳、「not X but Y」、Moreover／Notably 類連接詞、比喻、口語、無出處的主張；不寫時間相依字眼；數字附協定。
- 摘要寫背景、痛點、本方法特色與預期效果，只保留一個淨化後位移倍數，不寫 baseline 名單或方法細節。
- 失真—位移直線是對所有方法的觀察（含本方法），寫成「等失真下可比」，不寫成對立。
- 方法名一律 `\capg{}`；首次出現寫全名。
- 表格：固定字級、不用 `\resizebox`；表頭依「對象 → 指標 → 淨化條件」分層；指標標 ↑／↓，各區塊最佳平均粗體，CAPGuard 列底色 `oursrow`；七道淨化平均寫 Purified$^{*}$ 並在 caption 註明七道；比較條件寫進 caption。
- 暫代數字：未達全規模的數字寫 `\prelim{值}{註解}`（紅字；註解非空時加上標並列入 PDF 末頁）；表格或圖的整體說明寫 `\prelimnote{註解}`，放在 `\caption` 之外。註解寫張數、協定、路徑與待補內容；無讀數寫 `\prelim{--}{}`。投稿前在 `main.tex` 設 `\prelimfinaltrue`，殘留者編譯失敗。
- 圖：與論文同字體（matplotlib 用 STIXGeneral＋stix，TikZ 用 texgyretermes），印刷 ≥ 7 pt；配色：加性方法紅 `#A94438`、CAPGuard teal `#1F6F7C`、飽和度藍 `#2F5DA8`、色相橘 `#C9781A`、參考灰。
- 不下「方法成立／不成立」的判定；不做色調／白平衡類淨化；不提「加性擾動偏高頻」。
- 正文上限 8 頁（參考文獻與 LLM Usage Statement 不計）；全文不放 URL。
- commit message 用英文；內容、清理、文件分開 commit；push 由 executor 執行；在 main 上 commit。

## 9. 工作方式

- 改論文：改 `docs/paper/` 後 `python docs/paper_tools/run_paper_preview.py`，檢查 `build/main.log` 的 Overfull 與未定義引用；看排版時把 `build/main.pdf` scp 到遠端 `/tmp`，`mutool draw -q -r 80 -o p.png main.pdf <頁>` 抓回看。
- 改 tex 或腳本：用 Edit 工具，或用 Write 寫成 scratchpad 內的 `.py` 再執行（以 `assert s.count(old) == 1` 確認唯一替換）。不要用 bash heredoc 寫含反斜線的 Python：此環境的 heredoc 會把 `\\` 變成 `\`，`\t` 會變成 tab，曾造成表格換行符號遺失與 `\times` 變成 tab。
- Fig. 3／4 重出（遠端）：`cd ~/paper_figures && ~/venvs/wacv/bin/python in/generate_paper_figures.py --points in_pexels/points.csv --aligned in_pexels/condition_summary.csv --color-summary in_pexels/color_summary.csv --survival in_pexels/perturbation_survival_native_dct_shield_y.csv --sweep in/retention_native_dct_shield_y.csv --ours-extra 0.302 0.367 "CAPGuard, bounds ×2.5" --out-dir out_pexels`，再 scp `out_pexels/*.pdf` 到 `docs/paper/fig/`。
- teaser 重出（遠端）：`~/venvs/wacv/bin/python in/generate_pexels_teaser_figure.py --image man_60 --prompt 2 --layout protected-purified --no-lpips --cite --scale2p5 --arms photoguard_encoder,mist,dia_pt,dct_shield,blurguard --out-dir out_teaser --name teaser`（每次約 1–2 分鐘，在搜尋影像檔）。

## 10. 近期決定（使用者，經 executor 轉達）

- 11/5：方法命名 CAPGuard（全名 Color-based Anti-Purification Image Immunization）；標題見第 3 節。
- 11/5：主表採草稿 G＋ArcFace（單表、ip2p 上段＋UltraEdit／FLUX 下段），原跨編輯器表併入；保留率移補充。
- 11/5：消融只量防禦圖與淨化前編輯（Table 3 四欄）；移除逐項懲罰與可行集合投影。
- 11/5：Fig. 4 (a) 改分組直方圖；(b–d) 最左加未淨化點、標出全部設定；DCT-Shield-Y 改原生預算。
- 11/6：teaser 用 man_60（helmet），放回正文為 Fig. 1；論文只保留 2.5 倍（第 5 節清單）。
- 摘要「at matched distortion」保留；倍數待 2.5 倍與 BlurGuard 306 張後重算。
