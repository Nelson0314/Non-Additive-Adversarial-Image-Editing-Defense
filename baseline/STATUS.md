# baseline：現況與待決

冷啟動先讀 `docs/HANDOVER.md`（範圍與分工、工作規範、結果正本、等失真設定、論文數字出處、inpainting、佇列、工具索引、已知問題），
再讀本檔。資料組、協定、檔案位置與程式清單在 `README.md`；Pexels 資料組的選取在 `docs/PEXELS_PORTRAITS.md`；等失真臂在
`results/aligned/README.md`；已刪除的 trial 在 `docs/TRIALS.md`。逐段工作紀錄保留在本檔 commit `05f785f0` 的版本。

## 現況

- 論文的圖表取自 Pexels 306 張：主表、ImmuLUT 2.5 倍、BlurGuard、VIBE-DistilledCFG、FLUX.2 klein 的讀數皆已入版控（`docs/HANDOVER.md` 3.1）。
- ISR：crosseditor100 已評審（根目錄 `docs/isr_pilot/`，論文 Table 2）；第二批 100 張與第三批 106 張的拼圖已產生，評審由 executor 派工（`docs/HANDOVER.md` 3.3）。
- 質性圖（Fig. 4、補充資料）與 Fig. 1 teaser 由 `scripts/generate_purification_example_figure.py`、`scripts/generate_teaser_flow_figure.py` 產生（`docs/HANDOVER.md` 3.3）。
- Inpainting 跨編輯器（補充資料，306 張 × 7 條件 × 4 淨化）執行中；完成後以 `scripts/evaluate_pexels_inpaint_readouts.sh` 出讀數，
  結果入 `results/pexels_portraits/inpaint/`（`docs/HANDOVER.md` 3.4）。

## 待使用者決定

1. BlurGuard 在 Pexels 上是否重新校準等失真倍率（現為 0.2745，錨點 0.3344）。
2. deferred 的 UltraEdit／FLUX.1 Kontext 工作是否執行。

## 規矩

- 引用外部方法的數字依 `docs/reference/BASELINE_PROVENANCE.md` 規則 1（連同該方法的協定與來源版本）。
- 遠端停工作、crontab、刪除的程序見 `docs/HANDOVER.md` 2；共通規則見根目錄 `CLAUDE.md`。
