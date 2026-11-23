# executor 交接

接手 executor（專案管理、文獻管理、派工與轉達、代為 push）的冷啟動文件。冷啟動順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔 → 專案記憶 `MEMORY.md`（至少 executor-is-paper-pm、executor-owns-main-bib、no-handover-stops-before-85-percent、color-line-current、color-has-gpu-priority、no-local-compute、never-overwrite-running-scripts-on-nfs、paper-targets-cvpr-2027、advisor-text-is-final、keep-gpus-busy-on-advisor-direction、confirm-dispatch-was-received、paper-authors）。各專案即時狀態以 `baseline/STATUS.md`、`baseline/docs/HANDOVER.md`、`color/STATUS.md`、color trial `HANDOVER.md`〈現行狀態〉、`docs/PAPER_HANDOVER.md` 為準。本檔的歷史版本見 git log。

## 1. 角色與規則

| 角色 | 職責 | 寫入範圍 |
|---|---|---|
| executor | 進度管理、文獻管理（`docs/paper/main.bib` 核實）、派工與轉達、整理成果給使用者、代各 session push、ISR 評審 | `docs/` 的管理文件、`docs/isr_pilot/`、`main.bib` |
| baseline | 加性方法、BlurGuard、跨編輯器（VIBE、FLUX.2 klein、SD inpainting）、主表讀數、ISR 評審素材、論文圖 | `baseline/`、`core/` |
| color | ImmuLUT（條件名 color3d_scale2p5 系列）求解與編輯、讀數、消融、多色調搜尋 | `color/` |
| paper | 論文正文與補充材料 `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_HANDOVER.md` | 同左 |

- 使用者授權 executor 直接對各 session 下令；例外是 Overleaf 的任何操作（同步、刪檔），須使用者在 paper pane 親自下令並核准。
- pane 編號常變，每次派工前 `list_panes` 依標題辨認。對方 session 重啟或用量耗盡時訊息會遺失，派工後以 `wait_for_reply` 或 `read_pane` 確認對方回「收到」；沒有就重送全文。
- 排程由使用者決定；使用者未回應時依其最後指示自行排下一批，刪檔、改協定仍需確認。
- push：各 session 回報 commit 後，確認 `.sh` 無 CR 再 push；basic-1 repo 以 `git pull --ff-only` 跟進，跟進前確認 `run_pexels_supervisor.sh` 沒有在跑（`ps -eo args | grep "^bash.*[r]un_pexels_supervisor.sh"`，不要用 `pgrep -f`，會匹配 ssh 本身）。
- 回報使用者：繁體中文，先講結果，數字附協定，不下判定；看圖時把圖抓到 scratchpad、自己看過原尺寸再開給使用者（PowerShell `Start-Process <png>`）。
- 本機不跑計算（CPU 分析、計分、PDF 渲染都在 basic-1），本機只編輯、ssh、看圖、合併文字檔。
- GPU：全局 ≤ 6 張（basic-1／basic-2，NFS 共用 `~`），只取完全空的卡，color 優先。遠端 `nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`／`-p 10102`；遠端 Python 為 `python3` 或 `~/venvs/wacv/bin/python`。

## 2. 方法與協定

- **ImmuLUT**（論文巨集 `\capg`）：每張影像一個 33³ RGB 3D LUT；自由參數經色空間高斯平滑（σ＝4 格）成旋轉場；以灰軸為極軸，極角 < 25°、色相 < 7.5°（條件名 `color3d_scale2p5_pexels`）；損失只有 ip2p VAE 編碼器後驗平均差；Adam 1800 步。
- **資料與攻擊**：Pexels 人像 306 張；InstructPix2Pix（text 7.5、image 1.8、50 步），4 句 FaceLock 配件指令，主種子 20260812；最佳化不用任何指令。
- **淨化**：七道（JPEG 80／50／30、blur σ1／2、crop-resize 0.1、rotate15）＋ VAE-Trans。
- **加性對照**：10 個等失真臂。等失真縮放係數在 8 張校準集（data/portraits）上使平均防禦圖 LPIPS 等於 0.3344（早期顏色方法的值），原封套用到 306 張；306 張上各臂 0.251–0.318，ImmuLUT 0.287。
- **ISR（ip2p）**：Claude Sonnet 評審＋人工複核（1,038 格），`docs/isr_pilot/`（README、`audit/`、`audit206/`）。
- **ISR（SD inpainting）**：Claude Sonnet 評審 1,224 張拼圖 × 28 格＝34,272 格，`docs/isr_pilot/inpaint/`（README、rubric、judgments、計分程式、summary）。條件式 ISR（四淨化合計）：ImmuLUT 25.3%［23.1, 28.0］；BlurGuard 9.4、PhotoGuard 6.7、SIFM 6.6、DIA-PT 6.5、DCT-Shield 5.1。淨化後三道合計：ImmuLUT 25.8%、BlurGuard 8.3%。Undefended 成功率 65.8–72.3%（依淨化），hospital 52.8%、dog 77.9%（依指令）。未做人工複核；ImmuLUT 列的色調使盲化不完全。

## 3. 論文現況

- CVPR 2027 Submission 14011，正文 2026-11-17 上傳 OpenReview。作者順序：翁鈺翔、羅婈、帥宏翰（指導教授），通訊作者未定。
- Overleaf 最後同步點見 `docs/PAPER_HANDOVER.md`（commit `04ab6494` 起：正文 4.5 消融段為 306 張版；supp 接在正文後，只有 Ablation Study 與 Cross-Editor Evaluation on SD Inpainting 兩節；其餘六節存於 `sec/X_suppl_parked.tex`，不 `\input`）。
- 補充材料截止 2026-11-23 19:59（台灣時間）。是否另產單獨 supp PDF 上傳 OpenReview 由使用者決定，executor 未經手上傳，狀態需向使用者確認。
- Overleaf 根目錄的 `build/main.pdf`（編譯產物）使用者要求刪除，被 Claude Code 權限檢查擋下，需使用者在 paper pane 核准或自行刪除。
- SD inpainting ISR 數字尚未寫入 supp；寫入與否、寫在哪一節由使用者決定，交 paper 執行。

## 4. 多色調 ImmuLUT 搜尋（2026-11-20 至 11-23）

使用者目標：防禦圖像真人會套用的濾鏡（淡淡的、朦朧、不同色調皆可），且未淨化就擋下編輯；指令以 helmet、police suit、血腥句（Fig. 1 的 "give this person rotting zombie skin"）為主，不做 sunglasses、bowtie。完整結論、路徑與程式在 color trial `HANDOVER.md`〈現行狀態〉與 `color/STATUS.md`，摘要如下。

- 試過：色調初值與牽引、預算放寬（受步長限制，max|Θ|≈0.24）、濾鏡起點、參數式風格族（16＋8＋4 種色調）、ip2p 端到端項、低頻光影、粗糙 LUT、從使用者桌面參考圖擬合的 warm／cool／vintage／pink 濾鏡、去黑位與暗部保護的 clean 版、旋鈕中間設定。合計約 400 格。
- 核心發現：未淨化擋下 helmet 依賴旋轉場對暗部（頭髮）的大位移，正是霧與雜訊感的來源；去掉後外觀乾淨但 helmet 全部失守。police suit 在所有自然設定下都停在「保留原衣物但多肩章／臂章」，血腥句從未擋下。woman_111、woman_152、man_32、woman_59 結論一致，helmet 結果因影像而異。
- 外觀最自然的設定：參數式 golden／Kodachrome／cool_morning，暗部門檻 L*＝15、黑位 0、R1.5、σ4（`mid_d15_b0`）。
- 使用者尚未決定下一步（接受取捨、換方法類型、或停止）。GPU 自 2026-11-23 20:45 起空著。

## 5. 待辦

- 使用者待決：多色調方向；supp 單獨 PDF 與 OpenReview 上傳；Overleaf `build/` 刪除；inpainting ISR 是否入 supp 與是否做人工複核。
- color trial 升格入庫（求解鏈、論文用 CSV）與否決部分寫 `TRIALS.md`，見 color 交接檔。
- 兩份獨立審查的低優先項：參考文獻頁碼與卷號、arXiv 條目改正式版本（2026-11-20 查過 9 筆皆尚無正式版）、圖內字級、比較方法選擇理由、行文小瑕疵。

## 6. 工作方式與經驗

- **鬧鐘**：不經模型的外部喚醒以 Windows 工作排程器執行 `%LOCALAPPDATA%\executor_alarm\wake_executor.ps1`：讀 `%APPDATA%\multi-agent-cli\bridge\endpoint.json` 的 url／token，從 `bridge\*.mcp.json` 取一個標題不是 executor 的 pane 身分（橋接拒絕「送給自己」），對 executor pane 呼叫 `send_to_pane`；紀錄寫 `wake.log`。新增鬧鐘：`schtasks /Create /TN <名> /SC ONCE /SD yyyy/MM/dd /ST HH:mm /TR "powershell -NoProfile -ExecutionPolicy Bypass -File ...\wake_executor.ps1" /F`。
- **背景監看**：本機記憶體不足時，Claude Code 會停掉閒置的背景指令；長等待交給遠端腳本串接並寫事件到 log，用鬧鐘喚醒，不依賴本機監看。
- **Sonnet 評審**：brief 與 rubric 放 scratchpad，每個 agent 10 張拼圖（約 9 萬 token、2 分鐘），同時 4 個；碰到 5 小時用量上限就停，重置後以 `SendMessage` 讓中斷的 agent 從已寫入的 CSV 接續。123 批約用掉兩個多 5 小時週期。
- **磁碟**：`.git/objects/pack/tmp_pack_*` 是中斷的 git 傳輸殘留，2026-11-22 清掉 42.9 GB；C 槽再滿時先查這裡。
- **頁數**：以 Overleaf 或 basic-1 `pdflatex` 為準，本機 Tectonic 斷行不同。
- **刪除**：遠端刪除先請負責 session 列清單，經使用者確認後刪；trial 刪除前先寫 `TRIALS.md`。

## 7. 文獻

- `main.bib` 歸 executor 核實；期刊名縮寫統一為 IEEE TPAMI／TIFS／TIP。不再被引用的條目保留未刪；來源較弱者為 `doprotect2026`、`zeng2020lut`。
