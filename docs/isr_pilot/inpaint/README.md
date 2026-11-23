# ISR：SD Inpainting（306 張 Pexels 人像 × 4 句）

條件式 Immunization Success Rate，評審 SD Inpainting（`runwayml/stable-diffusion-inpainting`，CLIPSeg 主體遮罩補集，主種子 20260812）的既有編輯，不含新的 GPU 編輯。

## 協定

- 評審素材：baseline 遠端 `artifacts/pexels_portraits/inpaint_isr_sheets/`（1,224 張拼圖，由 `baseline/scripts/generate_inpaint_isr_sheets.py` 產生）。每張為一張影像 × 一句指令；列 R1–R7 為 7 個條件，列序以 `default_rng(20260812)` 逐張打亂，列標不含條件名；欄為 Input 與 none、JPEG50、blur σ2、VAE-Trans。解盲對照為同目錄 `key.csv`。
- 指令：p0 hospital、p1 gym、p2 snowy street、p3 dog（"A {man/woman} in a hospital" 等）。
- 評審：Claude Sonnet，每次 10 張拼圖、共 123 批，規則見 `rubric.md`。每格三個 0/1 標籤：content、mismatch、degraded。
- 規則的兩項取捨：snowy street 只要求戶外雪景，不要求街景；遮罩膨脹在人物輪廓外留下的原背景帶狀區不計入 mismatch 與 degraded。
- 條件式 ISR：只計同一影像、指令、淨化下 Undefended 判為成功（content=1、mismatch=0、degraded=0）的格子；防禦後任一標籤失敗即計為防禦成功。區間為以影像為單位的 bootstrap 95% CI（1000 次，種子 20260812）。
- 盲化限制：ImmuLUT 那一列的 Input 與編輯結果有明顯色調，評審可由外觀辨認該列。

## 檔案

| 檔案 | 內容 |
|---|---|
| `judgments.csv` | 34,272 格判定（1,224 張 × 28 格） |
| `rubric.md` | 評審規則全文 |
| `measure_inpaint_isr_summary.py` | 計分程式；參數為 `key.csv` 路徑，於遠端以 `python3` 執行 |
| `summary.txt` | 計分輸出：Undefended 成功率、各方法 × 淨化的 ISR 與 CI、分指令 ISR |
