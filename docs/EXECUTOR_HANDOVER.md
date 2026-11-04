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

## 4. 執行中的工作（2026-11-04 19:50 狀態）

全局 6 卡。baseline 的 supervisor 另讀 `baseline/runtime/pexels_portraits/queue/baseline_cap`（現為 4）：新開 worker 上限＝min(baseline_cap, 6 − max(他人租約, color_reserve))。

| 負責 | 工作 | 協定 | 進度與閘門 |
|---|---|---|---|
| baseline | FLUX 512：50 張（`pexels_subsets/flux.txt`）× 6 條件（等失真 photoguard_encoder、dct_shield、dct_shield_y、mist＋color3d_pexels＋未防禦），ip2p 防禦圖直接送入 | FLUX.1 Kontext 512×512、guidance 3.5、28 步、NF4、單次載入入口；淨化前、七道、VAE-Trans；資料版本 `baseline/results/pexels_portraits/flux_512/README.md` | 34/50（19:45）；`flux_512_readouts` 於 50 張完成後自動量讀數並彙整，完成後送 paper |
| baseline | color3d_scale2p5 的主表讀數（VAE-Trans 與主表全部欄位，121 項） | 同主表；程式已參數化（a6c4c7d） | 閘門為 color 的 `finish_color3d_scale2p5_pexels_full.log` 出現 `[FINISH]`；color 有 failed 時 followup 會因張數不齊失敗 |
| color | color3d_scale2p5_pexels（角度上界 2.5 倍：極角 25°、色相 7.5°）306 張，淨化前＋七道 | ip2p、4 句、主種子 | 2 線（FLUX 50 張完成前）；FLUX 完成後 color_reserve＝5、color 最多 5 線，baseline 讀數完成後改 6；預估 11/5 18:00 後完成 |
| color | 2.5 倍在 `pexels_subsets/budget_sweep.txt` 20 張的 VAE-Trans 與整組讀數，對照同 20 張的 color3d_pexels 與 9 臂 | 主表定義 | 降回 2 線後第一張卡優先跑；預估 11/4 22:00 前後 |

- UltraEdit 50 張（4 臂＋Ours＋未防禦）全協定已完成，讀數已填入論文 Table 2（遠端 `baseline/runtime/pexels_portraits/ultraedit_readouts/summary.csv`）。
- 延後（使用者另行安排）：UltraEdit／FLUX 其餘 256 張（`queue/jobs_deferred_transfer_rest.txt`；FLUX 1024 版已由 512 取代，`superseded_jobs_flux_1024.txt`）、P2 淨化強度掃描（dct_shield 已換 dct_shield_y，1a5f0be）、BlurGuard 等失真版（旋鈕待使用者定）、消融 A–E 與下半兩列（306 張）。
- 取消：以 SD3／FLUX VAE 重新求解的跨模型列（P3 移到 `queue/superseded_jobs_cross_model_resolve.txt`）；跨編輯器表不再區分 transfer。
- 預算掃描（20 張、只求解＋淨化前）：倍率 0.5／1.0／1.5／2.0／2.5 的防禦圖 LPIPS 0.126／0.213／0.258／0.279／0.288，淨化前編輯結果 LPIPS 0.182／0.278／0.326／0.347／0.357；20 張依性別×防禦圖 LPIPS 分層、20,000 組取 max KS 最小（0.085）。

## 5. 使用者 11/3–11/4 的決定（已實施）

- 主表：每方法一列，數字旁小字為 95% 信賴區間半寬（Purified 欄跨七道 t(0.975,6)·s/√7，其餘欄 306 張 1.96·s/√306）；不放 Weakest（移補充）；SSIM、PSNR 放位移端（未防禦編輯對防禦後編輯），防禦圖端移補充；美學主表只留 LAION、MUSIQ；加 CLIP-S、CLIP-I、編輯端 FID；不用 CLIP-IQA。
- 指標定義：CLIP-S＝cos(E_img(編輯), E_txt(實際送入該編輯器的指令))；CLIP-I＝clip-vit-base-patch32 的防禦後編輯對未防禦編輯（同一淨化）；FID 只算編輯端。
- 所有實驗以全規模（306 張）寫進論文；未跑完的數字以 `\prelim{值}{註解}` 紅字並附內部註解（`\prelimfinaltrue` 時殘留即報錯）。
- 消融：A 拿掉最佳化、B σ＝0、C 拿掉角度上界、D 單一 10° 夾角、C＋D、E 不正規化回 ‖c‖；不對齊失真；規模與排程由使用者安排。

## 6. 論文

- 投稿 CVPR 2027：摘要 11/10、全文 11/16、補充材料 11/23（AOE）。正文 8 頁；LLM Usage Statement 已寫；不放 URL；匿名。
- 現況：正文止於第 8 頁。Table 1 主表（306 張，紅字只剩 BlurGuard 列）、Table 2 跨編輯器（UltraEdit 50 張已填；FLUX 待 512 讀數）、Table 3 消融（A–E＋逐項懲罰、可行集合投影，紅字）；Fig. 2 含 306 張逐張點；Fig. 3a 標 DCT-Shield-Y，b–d 為 8 張 DCT-Shield 紅字；結論、社會影響、補充協定已寫。
- 待使用者：Pexels 選取規則與授權（4.1、補充）；限制節「配件仍出現 19/32」改 306 張需 VQA／偵測（GPU）；補充的 DISTS 與色彩對齊後失真換 306 張；未評估反推全域色彩映射的攻擊（使用者已決定不做色調／白平衡淨化）。
- 同期工作（2026-09-15 後）：spin2026（2610.06334）、srim2026（2610.07464）、cad2026（2610.35566）已入 bib 並寫入 Related Work 與引言。
- `main.bib`：49 條，皆被引用並核實；來源較弱：`doprotect2026`、`zeng2020lut`。

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
