# baseline：現況與待決

冷啟動先讀 `docs/HANDOVER.md`（範圍與分工、工作規範、結果正本、等失真設定、論文圖表出處、inpainting、遠端執行狀態、工具索引、已知問題、
待使用者決定），再讀本檔。資料組、協定、檔案位置與程式清單在 `README.md`；Pexels 資料組的選取在 `docs/PEXELS_PORTRAITS.md`；
等失真臂在 `results/aligned/README.md`；已刪除的 trial 在 `docs/TRIALS.md`。逐段工作紀錄以 `git log -- baseline/` 查閱。

## 現況

- 論文的圖表取自 Pexels 306 張：主表、ImmuLUT 2.5 倍、BlurGuard、VIBE-DistilledCFG、FLUX.2 klein、SD inpainting 的讀數皆已入版控
  （`docs/HANDOVER.md` 3.1），各圖表的來源依 LaTeX 標籤列於 3.3。
- ISR：ip2p 三批共 306 張的拼圖已產生，crosseditor100 已評審（根目錄 `docs/isr_pilot/`）；SD inpainting 1,224 張拼圖已產生，
  executor 已以 Claude Sonnet 全部評審（根目錄 `docs/isr_pilot/inpaint/`）。
- 質性圖：Fig. 4、補充資料的 ip2p 質性圖 2 張與 inpainting 質性圖 10 張皆已交 paper（`docs/HANDOVER.md` 3.3）。
- 遠端：baseline 沒有在跑的工作與租約；佇列無待執行工作，`baseline_cap` 為 2（`docs/HANDOVER.md` 3.6）。

## 待使用者決定

無。使用者決定（2026-11-24，`docs/HANDOVER.md` 6）：

1. BlurGuard 不在 Pexels 上重新校準，維持倍率 0.6071（306 張的防禦圖 LPIPS 0.2745，錨點 0.3344）。
2. deferred 的 UltraEdit／FLUX.1 Kontext 工作不執行。
3. 遠端暫存（inpainting ISR 100 張試評檔、`~/tmp_figs`、`~/tmp_inpaint`、8 張未入論文的 inpainting 候選、`inpaint_native_man_26`）已移到
   `~/trash/baseline/2026-11-24/`；`inpaint/**/__raw.png` 保留。

## 規矩

- 引用外部方法的數字依 `docs/reference/BASELINE_PROVENANCE.md` 規則 1（連同該方法的協定與來源版本）。
- 遠端停工作、crontab、刪除的程序見 `docs/HANDOVER.md` 2；共通規則見根目錄 `CLAUDE.md`。
