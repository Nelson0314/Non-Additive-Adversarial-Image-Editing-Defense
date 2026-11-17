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
- pane 編號會變（session 重開或切帳號後常變），每次派工前 `list_panes`，依標題辨認。對方 session 重啟後可能沒收到訊息，派工後以 `read_pane` 確認對方確實開始。
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

## 4. 進行中（2026-11-17 晚）

- **inpainting（baseline）**：306 張中 263 完成，19:55 恢復（2 卡），預計 11/18 01:00 前後完成編輯，之後出讀數；`__raw.png` 讀數完成前不可刪。
- **消融擴充到 306 張（color）**：19:40 開跑，新增 106 張 × 8 臂＝848 項，約 190 GPU 小時（color 4 卡），預計 11/19 19:00 完成，之後出 306 張彙整與成對差，再換入正文消融段數字。
- **專案整理**：使用者已核可三個 session 的整理方案（2026-11-17）。不影響執行中工作的部分先做；color trial 升格入庫（ImmuLUT 程式、論文用結果 CSV、出圖工具，只收 CSV 不收 `.pt`）與 baseline 的 `flux_512` 共用程式改名，等消融與 inpainting 跑完。tex 中舊文註解等使用者問過 Lynn 再決定。

## 5. 待辦（補充材料階段）

- 補充材料內容：inpainting、消融（306 張）、user study（目前只有 2 人回答）、更多質性例子（man_07 p2 helmet 已出圖：baseline `artifacts/pexels_portraits/purification_example/man_07_p2.pdf`）。
- 兩份獨立審查的低優先項：參考文獻頁碼與卷號補齊、arXiv 條目改引正式版本、Fig. 3／Fig. 5 圖內字級 ≥ 7 pt 與 Fig. 5 灰階可辨、表 2／表 3 比較方法的選擇理由、行文小瑕疵。
- OpenReview 的 LLM Usage Declaration 全選 Generate；程式、實驗執行、資料標註實際接近 Agent 等級，使用者未決定是否修改。

## 6. 文獻

- `main.bib` 歸 executor 核實；期刊名縮寫統一為 IEEE TPAMI／TIFS／TIP。正文引用 CVPR 主會論文 7 篇＋1 篇 workshop。
- 不再被引用的條目保留未刪；來源較弱者為 `doprotect2026`、`zeng2020lut`。
