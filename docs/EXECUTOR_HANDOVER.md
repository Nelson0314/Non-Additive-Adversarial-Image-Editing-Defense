# executor 交接

接手 executor（專案管理、文獻管理、派工與轉達）的冷啟動文件。狀態以 2026-11-03 為準；各實驗的即時狀態以 `baseline/STATUS.md`、`color/STATUS.md`、color trial `HANDOVER.md` 與 `docs/PAPER_STRUCTURE.md` 為準。

## 1. 角色、分工與 pane

| 角色 | 職責 | 寫入範圍 |
|---|---|---|
| executor（本角色） | 進度管理、文獻管理（`docs/paper/main.bib` 歸 executor 核實）、派工與轉達、整理成果給使用者、代使用者 push | `docs/` 下的管理文件、`main.bib` 的核實修改 |
| baseline | 加性方法、等失真臂、Pexels 主表佇列與後續量測、跨編輯器 baseline 列 | `baseline/` |
| color | 本方法 color3d、Pexels 上的 color3d 與對照、跨編輯器 Ours、消融 | `color/` |
| paper | 論文正文 `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、圖表素材 | `docs/paper/`、`docs/paper_tools/` |

- pane 編號會隨關閉與新開而變動，每次派工前先 `list_panes`，依標題（executor、color、baseline、paper）確認。
- 同一時間一個目錄只有一個 session 有寫入權；換手時舊 session 先改為唯讀，確認本機沒有背景工作與 CronCreate 後才關。
- 回報給使用者：先講結果，數字附協定與張數，不下「成立／不成立」判定，外觀由使用者看圖決定；進度用「完成數／總數」，不用狀態文字。
- 看圖：遠端產物用 scp 抓到自己的 scratchpad，再用 `Start-Process` 開給使用者。本機不跑任何計算（Tectonic 編譯除外）。

## 2. 專案背景與方法

- **問題**：影像免疫。發布前處理人像，使攻擊方以擴散編輯器依指令編輯時，結果偏離未防禦的編輯。白盒、外掛；防禦方不知道指令，訓練只用空字串或類別名。
- **既有方法**：PGD 逐像素加性擾動，會被淨化（JPEG、模糊、縮放、VAE 重建）移除。
- **本方法 color3d**：每張影像一個 33³ RGB 3D LUT。自由參數 Θ 經色空間高斯平滑（σ = 4 格）成旋轉場 V；以黑點為原點、灰軸為極軸，去掉徑向分量後拆成子午與方位分量，分別以 tanh 限制極角 < 10°、方位角 < 3°（截斷到色域前；上界只在格點上保證，像素經三線性內插後可能超出）；‖c‖ 不變。目標為 ip2p VAE 編碼器後驗平均位移；損失只有目標，Adam 1800 步。公式見論文 3.3 節（`docs/paper/sec/3_method.tex`）。
- **淨化協定**：七道（jpeg80／50／30、blur σ1／2、crop_resize0.1、rotate15）＋ VAE-Trans（arXiv 2603.13028）；不做色調／白平衡淨化。
- **論文的 baseline**：9 臂（DAYN、DIA-R、DIA-PT、SIFM、DANP、Mist、PhotoGuard encoder、DCT-Shield、DCT-Shield-Y）；不列 photoguard_c、photoguard_linf、DiffVax。

## 3. 現有結果

### Pexels 主表（306 張，ip2p，4 句，主種子 20260812，不含 VAE-Trans）

完成於 11/3 06:50。檔案：遠端 `baseline/runtime/pexels_portraits/summary/main_table.csv`、`distribution_groups.csv`（逐組 FID／KID／CLIP-IQA，32/90 組 KID 為負）。已填入論文 Table 1 與補充材料。

| 條件 | 防禦圖 LPIPS | 淨化前 | 七道平均 | 最弱一道 | 保留率 | SigLIP 七道 ↓ | ArcFace 七道 ↓ |
|---|---|---|---|---|---|---|---|
| 9 臂 | 0.251–0.318 | 0.262–0.415 | 0.112–0.212 | 0.028–0.106 | 0.366–0.609 | 0.948–0.976 | 0.870–0.928 |
| color3d | 0.211 | 0.277 | 0.291 | 0.251 | 1.091 | 0.913 | 0.826 |

- 加性方法沿用 portraits 校準的等失真倍率，Pexels 上失真比 color3d 高（使用者決定暫不處理，caption 不宣稱 matched distortion）。
- FID 七道淨化後：color3d 35.0，9 臂 17.4–26.1；防禦圖對原圖 FID：color3d 39.1（最高）。CLIP-IQA：color3d 無一欄最佳。

### 同類隨機對照

- **同失真隨機**（`color3d_random_matched_pexels`，Pexels 270 張，36 張對不到 color3d 失真不列入，只有淨化前）：位移與 color3d 相同（編輯結果 LPIPS 0.265 vs 0.267，SigLIP 0.914 vs 0.914，ArcFace 0.821 vs 0.835）；隨機版 SSIM、VIF-p、L∞ 與 MUSIQ、NIMA、TOPIQ、CLIP-IQA 平均較差，LAION 相同。8 張 portraits 的同失真隨機見論文 Table 3 最後一列。
- **純隨機 α＝1**（`color3d_random_pexels`，306 張）：防禦圖 LPIPS 0.037，幾乎接近恆等。

### 其他已完成

- portraits 8 張：跨編輯器（論文 Table 2，含 Ours 與 Ours (transfer)）、消融（Table 3）、淨化強度掃描（Fig. 3 b–d）。
- 提亮再防禦試驗（3 張）：提亮後失真與位移上升，提亮本身失真 0.407。
- color3d 防禦圖 LPIPS 在 Pexels 較低的原因：與中性亮區比例（ρ 0.763）、平均明度（0.719）相關，見 color trial README。

## 4. 執行中、延後與暫停中的工作

- 執行中：baseline 佇列的 Pexels 主表 VAE-Trans（114 項）；其後由閘門 `vae_trans_gpu_done` 接遷移性實驗（`baseline/scripts/evaluate_pexels_transfer.sh`，`TRANSFER_DEADLINE` 2026-11-04 20:00）與主表各影像組的 CLIP-S（`baseline/scripts/measure_clip_score.py`），全局 6 卡。

| 負責 | 工作 | 位置 | 狀態 |
|---|---|---|---|
| baseline | 淨化強度掃描 P2、跨編輯器 baseline 列 P3（252 項） | `queue/jobs_deferred_p2_p3.txt` | 延後；第一步未驗證，恢復時先看 |
| color | Pexels 跨編輯器：FLUX transferred＋未防禦分母（50 張子集） | color trial `runtime/cross_model/` | 淨化前 200/200、七道 1,400/1,400、VAE-Trans 160/200 |
| color | Pexels 跨編輯器：UltraEdit transferred＋未防禦分母（306 張） | 同上 | 7/23 組 |
| color | Pexels 跨編輯器 Ours（以 FLUX／SD3 編碼器求解） | 同上 | 0 |
| color | Pexels 消融 | 子集 `pexels_subsets/ablation.txt`（97 張） | 只有隨機對照完成 |

- 子集：FLUX 50 張（`pexels_subsets/flux.txt`，依性別與 LPIPS 四分位分層抽樣）、消融 97 張（第一個影像組）。
- 遠端 Pexels 佇列由 crontab 每 5 分鐘的 `baseline/scripts/run_pexels_supervisor.sh` 推進；可用卡數＝6 − max(他人租約, `queue/color_reserve`)；第一項連續卡住會寫 `[WARN]`。color 用卡時設 `color_reserve`，用完歸零（color 曾兩次忘記歸零，派 color 用卡時要求它設自動歸零）。

## 5. 使用者 11/3 會議後的待辦（使用者已選定，依序）

1. **遷移性（最優先，週四 11/5 10:00 開會前）**：ip2p 求出的防禦圖直接送 UltraEdit 與 FLUX，Ours (transfer) 對 4 個 baseline（等失真臂的 photoguard_encoder、dct_shield、dct_shield_y、mist），全協定（淨化前、七道、VAE-Trans）。最終要跑完 306 張；先做 50 張。executor 的估計（未經 baseline／color 核實，估算）：
   - UltraEdit 50 張：每格約 6.6 秒，每條件每張 36 格；4 baseline＋Ours＋未防禦分母約 20 GPU 小時（Ours 與分母若已在已完成的影像組內則更少），basic-1 3 卡約 6–7 小時。
   - FLUX 50 張：每格約 90 秒；4 baseline 約 180 GPU 小時，Ours 只剩 VAE-Trans 40 格，分母已齊；只能在 basic-2，3–5 卡約 36–60 小時，週四前做不完全協定。
   - 核實後（baseline 實測每格 UltraEdit 6.6 秒、FLUX 83 秒；4 臂 50 張 UltraEdit 13.2、FLUX 166 GPU 小時；306 張 81、1,017 GPU 小時）。color 的 Ours (transfer) 與未防禦分母：50 張只缺 FLUX VAE-Trans 40 格；UltraEdit 306 張剩 149 張約 20 GPU 小時。
   - 使用者決定：VAE-Trans 跑完後接上，全局 6 卡跑到 2026-11-04 20:00，以「張」為單位（一張 4 臂 × 36 格做完才算），20:00 後不派新工作，回報完成張數，續跑與否由使用者決定。FLUX 在 basic-2、UltraEdit 在 basic-1（306 張，前 50 為 `flux.txt`）。
2. **表格的波動性**：Purified 與 Weakest 合成一欄，以 ± 或標準差等方式呈現各方法的影像間波動。
3. **主表美學**：只留 LAION、MUSIQ，其餘移到後面或補充材料。
4. **主表加指標**：在 LPIPS、SigLIP、ArcFace 之外，加入 SSIM、PSNR（編輯端或防禦圖端未定）、CLIP-S、CLIP-I 與編輯端 FID；不用 CLIP-IQA。定義（主表與遷移表共用）：CLIP-S＝cos(E_img(編輯), E_txt(指令))，文字端為實際送進該編輯器的指令（ip2p 與 FLUX 用 `edits.ip2p[k]`，UltraEdit 用 `ultraedit_templates.json` 的 add 句）；CLIP-I＝baseline 的 `clip_pair`（clip-vit-base-patch32，防禦後編輯對未防禦編輯，同一淨化）；FID 只算編輯端（防禦後編輯分布對未防禦編輯分布，淨化前／七道／VAE-Trans），不報防禦圖對原圖的 FID。論文已改版：Table 1 加三欄、美學移到 Table 2、遷移表為 Table 3、消融為 Table 4。
5. **本方法的多個預算點**：color3d 失真 0.21 與 baseline 0.3 脫節；需要 color3d 的多個失真預算點，證明本方法也能沿失真—位移直線移動。做不到時，退而把 300 多張逐張的失真—位移點全部畫出。
6. **BlurGuard**：先確定等失真的預算（旋鈕），再在 Pexels 跑全協定。
7. **淨化分析**：Fig. 3 的強度比較把 DCT-Shield 換成 DCT-Shield-Y；論文先分開呈現已有的淨化分類討論，中間缺的掃描先放著。
8. **消融改為「每次拿掉 pipeline 的一部分」，不做 LPIPS 對齊**（使用者理由：失真差異本身是最佳化的結果）：
   - A 拿掉最佳化（隨機 Θ）；
   - B 拿掉高斯平滑（σ＝0）；
   - C／D 合併：拿掉角度上界、拿掉極角與色相分開限制，分成一次拿掉一種，以及兩種都拿掉；
   - E 拿掉所有和亮度有關的限制（例如 ‖c‖ 不變）。
   規模、張數未定，需先估算後交使用者決定（每項 8 張約 4、97 張約 45、306 張約 140 GPU 小時，均為估算）。

## 6. 論文

- 投稿 CVPR 2027：摘要 11/10、全文 11/16、補充材料 11/23（AOE）。正文 8 頁；LLM Usage Statement 必備；不放 URL；匿名；2026-09-15 後的同期工作要引用（已知 MCIQA-2K、NullEdit，調查未完成）。
- 現況：9 頁，正文到第 8 頁。Fig. 1 方法總覽（BlurGuard 式版面、球面色彩旋轉）、Fig. 2 失真—位移（Pexels）、Fig. 3 淨化分析（(a) Pexels、(b–d) 8 張）；Table 1 Pexels 主表（VAE-Trans 欄 \TBD）、Table 2 跨編輯器（8 張）、Table 3 消融（8 張）；補充 CLIP-IQA、FID 已改 Pexels 306。
- 還在等使用者決定：摘要倍數的定義、是否保留「at matched distortion」、論文重心是否轉向參數化、|Δφ| < 3° 只在格點保證是否在文中註明、兩支舊產圖腳本（`generate_lut_overview_figure.py`、`generate_method_overview_figure.py`）是否刪除、LLM Usage 工具清單、Pexels 選取規則與授權。
- `main.bib`：46 條，全部被正文引用且已核實，無 TODO、無 URL；未引用的條目已移除（commit 歷史可查）。來源較弱的兩條：`doprotect2026`（作者取自第三方摘要頁）、`zeng2020lut`（TPAMI 卷期頁未在 IEEE 確認）。新增條目由 paper 通知後核實。

## 7. 簡報與其他成品

- 成果簡報（Slides，私人）：https://claude.ai/artifact/97hLkqveZu8We3he7abNSR 。使用者會在頁面上直接修改，發布前一定先 `read` 最新的 `project/deck.json` 與要改的投影片。目前頁序：method、main、plane、purif、clipiqa、fid、cross、ablation（已改為規劃中的條列）、random、random_median、random_distinct。圖表素材由 paper 從論文輸出到其 scratchpad 的 `deck_assets/`。
- 實驗總覽 PDF：桌面 `experiment_overview.pdf`（原始檔 `docs/paper_tools/experiment_overview/`）。

## 8. 規範（摘要；全文見根目錄 `CLAUDE.md`、`AGENTS.md` 與專案記憶 `MEMORY.md`）

- 回覆用繁體中文；寫進檔案的文件用客觀、學術語氣，不寫時間相依字眼；報數字用描述性名稱並附協定。
- 不設判準；刪除、覆寫、部署、對外發布、花錢前先確認；`archive/` 只保存。
- 主種子為標準，不跑多種子；訓練不可用任何編輯指令。
- **git push 由 executor 代使用者執行**（使用者 10/29 授權）：baseline／color 回報 commit 號後，確認 fast-forward、.sh 為 LF 再 push，並提醒對方遠端 pull 時不可覆寫執行中的腳本（這台 NFS 上 git 會就地改檔；先建 `queue/PAUSED` 或寫新檔）。
- GPU：卡數依使用者口頭指定，未指定時全局 ≤ 6；color 優先；監督一律放遠端，不依賴本機排程。
- 沒有使用者要求不派 subagent；工具呼叫超過約 50 次時主動提議換手。
- 硬體：`nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡），NFS 共用，repo `~/image-immunization`；FLUX 權重只在 basic-2，UltraEdit／SD3 只在 basic-1。
