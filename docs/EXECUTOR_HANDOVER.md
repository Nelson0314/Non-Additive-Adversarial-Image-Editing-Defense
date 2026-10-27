# executor 交接

接手 executor（專案管理、文獻管理、派工）的冷啟動文件。狀態以 2026-10-27 為準；各實驗的即時狀態以 `baseline/STATUS.md`、`color/STATUS.md` 與 color trial `HANDOVER.md` 為準。

## 1. 角色與分工

| pane | 角色 | 寫入範圍 |
|---|---|---|
| executor（本角色） | 進度管理、文獻管理、派工與轉達、整理成果給使用者；不做實驗、不寫論文正文 | `docs/` 下的管理文件（本檔、`PAPER_HANDOVER.md` 第 4 節的方法現況） |
| baseline | 11 個加性方法、等失真臂、主表、跨模型 baseline、Pexels baseline 長跑、指標量測 | `baseline/` |
| color | 本方法 color3d 的實驗與設計、簡報、Pexels 上的 color3d、跨模型 color3d | `color/` |
| paper | 論文正文（`docs/paper/`）與 `docs/PAPER_STRUCTURE.md` | `docs/paper/`、`docs/paper_tools/` |

pane 編號會變動，派工前先以 `list_panes` 確認。各 session 只改自己的目錄；同一時間一個目錄只能有一個 session 有寫入權（color 交接時備援 session 先唯讀）。

工作方式：
- 使用者以中文下指令，executor 轉成具體工作送到對應 pane。訊息要寫明目標、檔案、協定與回報格式（行數上限）。
- 方法設計與實驗方向由使用者直接與 color、baseline 討論；executor 只轉達。
- 回報給使用者：先講結果，表格附協定，不下「成立／不成立」判定。外觀由使用者看圖決定。
- 要給使用者看圖時，用 scp 抓到 scratchpad，再用 `start` 開啟或 SendUserFile。

## 2. 專案背景

- **問題**：影像免疫。發布前處理人像，使攻擊方以公開權重的擴散編輯器依指令編輯時，結果偏離未防禦的編輯。白盒、外掛（不改攻擊方權重）；防禦方不知道指令，訓練只用空字串或類別名。
- **既有方法**：PGD 逐像素加性擾動，會被淨化移除。
- **本方法 color3d**（2026-10-27 定案；原條件名 `color3d_smooth_polar10_hue3`）：
  - 每張影像一個 33³ RGB 3D LUT；自由參數經色空間 3D 高斯平滑（σ = 4 格點）成為旋轉場；
  - 以黑點為原點、灰軸為極軸旋轉色向量，‖c‖ 不變；極角變化 < 10°、方位角變化 < 3°（截斷到 RGB 色域前；截斷後約 3% 格點方位角超過 3°）；
  - 目標為 ip2p VAE 編碼器潛向量位移；損失只有目標，沒有投影、沒有 LPIPS 上下界。
- **消融條件**：
  - `color3d_phr`：δ = 30.6 三條上界，PHR 求解；
  - `color3d_orig`：最早的逐項懲罰版，用來說明懲罰式訓練不穩定（1/180 個檢查點可行）；
  - `color3d_smooth_angle6`：黑點 6° 單角度版。
- **主要讀數**（8 張 portraits、ip2p EulerAncestral 50 步、s_T 7.5、s_I 1.8、主種子 20260812、FaceLock 配件四句、32 格）：

| 條件 | 防禦圖 LPIPS | 淨化前 | 七道平均 | VAE-Trans 後 | 保留率 |
|---|---|---|---|---|---|
| color3d | 0.296 | 0.374 | 0.367 | 0.301 | 1.020 |
| 等失真 11 臂 | 0.30–0.37 | 0.24–0.39 | 0.12–0.24 | 0.02–0.07 | — |

  - color3d 的其他讀數：SSIM 0.942、ΔE00 12.9、LAION 5.48（原圖 5.48）、BRISQUE 14.1。
  - 淨化：jpeg80／50／30、blur σ1／2、crop_resize0.1、rotate15，以及 VAE-Trans（arXiv 2603.13028）。色調／白平衡類淨化依使用者決定不做。
- **其他已有結果**（多數尚未以最終 color3d 重量）：
  - 失真—位移直線：38 點，r = 0.955（`baseline/results/distortion_displacement/`）；
  - 擾動存續比、淨化強度掃描（color3d_orig）；
  - inpainting（color3d_orig）；
  - 感知指標與調色濾鏡比較；
  - 原圖對防禦圖頻譜（`color/trials/.../results/image_spectrum/`，使用者指示先不入論文）；
  - SD 系（ip2p、SD2.1-base、SD inpainting）共用同一個 VAE 編碼器；FLUX、SD3 不同。

## 3. 進行中的工作（2026-10-27 派出）

| 負責 | 工作 | 狀態 |
|---|---|---|
| color | color3d 在 Pexels 306 張上跑 baseline 協定（只 ip2p） | 10/27 13:17 開跑，預計 10/28 傍晚完成求解與編輯，之後約半天讀數 |
| color | color3d 跨模型：對 FLUX.1 Kontext（basic-2）與 SD3／UltraEdit（basic-1）的 VAE 求解、全協定，另做 ip2p 防禦圖的交叉遷移 | 已派，時程待回報 |
| color | 成果簡報改版為 color3d（https://claude.ai/artifact/VfAGamPdpErLFqNV9iVxdE ） | 已派 |
| color | 本地與遠端檔案整理、文件更新到最新、commit 未提交的 color 文件 | 已派；刪除前會列清單給使用者 |
| baseline | CLIP-IQA、FID（附 KID）、MCIQA（arXiv 2609.14495）的失真端與位移端量測，對象為 11 臂與 color3d | 已派 |
| baseline | 跨模型 encoder 類 baseline（photoguard_encoder、dct_shield、dct_shield_y、mist_textural）在 FLUX／SD3 | FLUX 未淨化已完成；淨化後與 SD3 狀態以 STATUS 為準 |
| baseline | 補 Pexels images_0307_0308 的三道淨化分母給 color | 已完成；color 已匯入，Pexels 分母 306 張齊全 |
| baseline | 本地與遠端檔案整理、文件更新 | 已派 |
| paper | 完成第 2–4 節（見第 5 節的指示） | 第 2–4 節已改寫；第 1 頁 teaser 為佔位，草案在 color 遠端 `results/paper_figures/` |
| paper | 兩張散布圖（淨化後等失真主表、淨化前失真—位移平面），在遠端畫成 PDF 放進 `docs/paper/fig/` | 已完成：`fig/plane.pdf`、`fig/purified_plane.pdf`，程式 `docs/paper_tools/generate_paper_figures.py` |
| baseline | Ours（color3d）的擾動存續比（七道＋VAE-Trans）與淨化強度掃描（13 個設定），協定同 color3d_orig | 已派 |
| 方法總覽圖的 subagent | 重做精緻版 teaser | 因帳號週額度用完中斷（10/30 22:00 重置）；使用者已改為：方法總覽圖移到 Method 前，第 1 頁另放簡單 teaser（交 paper） |

## 4. 論文

- 投稿 CVPR 2027：摘要（enrollment）11/10、全文 11/16、補充材料 11/23（皆 AOE）。
- 硬性規定：
  - 正文 8 頁；
  - 必須有 LLM Usage Statement；
  - 全文不放 URL（含參考文獻）；
  - 匿名；
  - 引用不存在的文獻會被拒；
  - 2026-09-15 後公開的同期工作要引用。
- 行政：作者名單在 11/10 前確定；所有作者的 OpenReview profile 要匯入論文；若沒有作者在指定 venue 被接受過論文，只能投一篇。
- 檔案：
  - `docs/paper/`：CVPR 2027 author kit，唯一正本；
  - `docs/PAPER_STRUCTURE.md`：節、內容、狀態、頁數，由 paper 維護；
  - `docs/PAPER_HANDOVER.md`：paper 的交接檔；
  - 預覽工具 `docs/paper_tools/run_paper_preview.py`（Tectonic）自動把 PDF 複製到桌面 `paper.pdf`；
  - 桌面 `paper_outline.html`：給指導者看的架構表（使用者已自行修改過，不要覆寫）。
- `docs/` 目前不在版控內；是否 commit 由使用者決定。
- 寫作規則（使用者定）：
  - 摘要當產品介紹寫，原則上不放數字（現有一個 `\NUM` 由使用者決定）；
  - 引言從領域研究現況開場，不寫日常情境；
  - 失真—位移直線是對所有方法的觀察；
  - 頻譜分析暫不入論文。
- 待辦（executor 負責）：`main.bib` 40 條中 28 處標 TODO，須逐條人工核實並改為出版社或 DBLP 條目，去除 URL。另需調查 2026-09-15 後的同期工作（已知候選：MCIQA-2K，arXiv 2609.14495）。

## 5. 使用者 2026-10-27 的論文指示（已轉給 paper）

1. Related Work 每段結尾回收到該方法或領域的限制，或本文為何選擇該方法。
2. 主表以淨化後結果為主（完整數字表＋散布圖），原生預算放後面。
3. 方法總覽圖移到 Method 前；第 1 頁 teaser 改放簡單、能呈現本文在做什麼的圖。
4. 主表失真欄只用 LPIPS 與有利或持平的美學指標；位移欄只用 LPIPS 與語義類。DISTS、PSNR、SSIM 保留數據，另找地方說明它們對色彩改變的偏差。
5. CLIP-IQA、FID、MCIQA 交 baseline 量測，報有利部分。
6. paper 先完成第 2–4 節，數據有多少放多少。

## 6. 硬體與環境

- **本機**：Windows 11，PowerShell 與 Git Bash。Python 指令為 `python`。本機不跑任何計算（含 CPU 分析與畫圖），只編輯、ssh、看圖；Tectonic 編譯論文除外。
- **遠端**：
  - `nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`（basic-1，8 卡）、`-p 10102`（basic-2，7 卡），RTX 3090 24 GB，NFS 共用；
  - repo 在 `~/image-immunization`，以 `git pull` 同步；venv 為 `~/venvs/wacv`。
  - HF 權重各機不同：FLUX 只在 basic-2，UltraEdit／SD3 只在 basic-1；ip2p 與 SD inpainting 兩台都有。
  - 卡常被實驗室其他人佔用。
- **GPU 規則**：
  - 卡數由使用者口頭指定；未指定時全局合計 ≤ 6 張（所有主機、session 加總）；color 優先；
  - 租約目錄 `~/gpu_leases`，工具 `run_with_gpu_lease.sh`；不要給 `--limit`；取卡失敗時迴圈重試，重試要在有卡的那台主機上開。
- **常見陷阱**（詳見專案記憶）：
  - 取卡函式的紀錄要寫 stderr；
  - 殺行程後用 ps 與租約目錄回頭確認；
  - 遠端腳本不可就地覆寫；
  - Python 寫 .sh 會變成 CRLF；
  - 只做編輯的條件不可用符號連結；
  - HF 離線模式會讓 SD v1-4 載不起來。

## 7. 規範（摘要；全文見根目錄 `CLAUDE.md`、`AGENTS.md` 與專案記憶 `MEMORY.md`）

- 回覆用繁體中文；寫進檔案的文件用客觀、學術語氣，不寫時間相依字眼；報數字用描述性名稱並附協定。
- 不設判準：數據完整回報，成立與否由使用者判斷；這條也適用於指標。
- 刪除、覆寫、部署、對外發布、花錢之前先確認；`archive/` 只保存，不刪除。
- 主種子為標準，不跑多種子；訓練不可用任何編輯指令。
- push 由使用者執行；commit message 用英文。
- 沒有使用者要求就不派 subagent；派時一次一個、限制輸出長度。
- 封存：舊的規劃文件（`PAPER.md`、`pseudo_paper.html`、`research_tree.html`）已移到 `archive/paper_planning/`，不再維護。
