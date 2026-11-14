# ISR pilot（InstructPix2Pix，crosseditor100）

指導教授回饋第 1 點的試驗：在 InstructPix2Pix 的既有編輯上，以視覺評審判定配件是否出現，計算條件式 Immunization Success Rate（ISR）。不含新的 GPU 編輯。

## 協定

- 影像：baseline `runtime/pexels_portraits/crosseditor100/images.txt` 的 100 張 Pexels 人像；4 句 FaceLock 配件指令（p0 sunglasses、p1 police suit、p2 helmet、p3 bowtie）；主種子 20260812；ip2p text 7.5／image 1.8／50 步。
- 條件 7 個：Undefended、PhotoGuard（等失真 `photoguard_encoder`）、DCT-Shield（等失真）、BlurGuard（306 張版）、SIFM（等失真）、Mist（等失真）、CAPGuard 2.5 倍（`color3d_scale2p5_pexels`）。
- 淨化 4 種：none、jpeg50、blur2（Gaussian σ=2）、vae_trans。
- 評審素材：baseline 遠端 `artifacts/pexels_portraits/ip2p_isr100_sheets/`（400 張拼圖，格號 C<欄>-R<列>，`cells.csv` 為逐格路徑）。
- 評審：Claude Sonnet（effort medium），每次 10 張拼圖、共 40 批，規則見 `rubric.md`。每格三個 0/1 標籤：accessory、mismatch、degraded。
- 條件式 ISR：只計同一影像、指令、淨化下 Undefended 判為成功（accessory=1、mismatch=0、degraded=0）的格子；防禦後任一標籤失敗即計為防禦成功。區間為以影像為單位的 bootstrap 95% CI（1000 次，種子 20260812）；方法間以 McNemar 檢定比較。

## 檔案

| 檔案 | 內容 |
|---|---|
| `judgments.csv` | 11,200 格判定（400 張 × 28 格） |
| `rubric.md` | 評審規則全文 |
| `summary.txt` | `measure_isr_summary.py` 的輸出：各條件 × 淨化的 ISR 與 CI、分指令 ISR、失敗類型、McNemar |
| `isr_by_prompt.png` | `generate_isr_by_prompt_figure.py` 產生的分指令表 |

兩支程式預設讀同目錄的 `judgments.csv`，於遠端以 `python3`（圖需 `~/venvs/wacv` 的 matplotlib）執行。

## 配對差值與人工複核

- `measure_isr_paired_difference.py` → `paired_difference.txt`：CAPGuard 對各方法的 ISR 差值，以影像為單位的 paired bootstrap 95% CI（2000 次，種子 20260812）。
- `generate_isr_audit_cases.py` 挑出人工複核的 606 格：CAPGuard 與 BlurGuard 判定不同的 233 對（466 格，四種淨化）與 7 條件 × 4 淨化每層 5 格的分層隨機 140 格；盲化面板為原圖、同淨化的未防禦編輯、受檢編輯，不顯示條件與評審標籤。使用者一人標註。
- `measure_isr_audit_agreement.py` → `audit/`：`human_labels.csv`（人工與評審標籤並列）、`judgments_human.csv`（複核格以人工標籤取代）、`summary_human.txt`、`paired_difference_human.txt`；`audit_key.csv` 為案例對照表。
- 人工標註的不符指令較評審嚴格：編輯額外加上未要求的物件（口罩、胸前另一個領結等）計為不符指令。

## 人工標註更正版

`audit/judgments_human.csv` 中有 8 格 CAPGuard（ImmuLUT）的標籤（7 格人工、1 格評審）記為編輯失敗，但拼圖上配件已穿戴在人物身上。
`generate_isr_corrected_judgments.py` 讀該檔、只改這 8 格為 accessory=1、mismatch=0、degraded=0，另存
`audit/judgments_human_corrected.csv`（多一欄 `correction_note`，記原標籤與依據；其餘 11,192 格逐字相同）。原檔不變。

| 拼圖 | 格 | 淨化 | 原標籤（accessory／mismatch／degraded） | 拼圖所見 |
|---|---|---|---|---|
| `man_37__p2` | C2-R7 | none | 0／1／0 | 頭上有金屬頭盔，與未防禦編輯相同 |
| `man_37__p2` | C3-R7 | jpeg50 | 0／1／0 | 同上 |
| `man_37__p2` | C4-R7 | blur2 | 0／1／0 | 頭上有深色頭盔，與未防禦編輯相同 |
| `man_37__p2` | C5-R7 | vae_trans | 0／0／0（評審標籤，note「faint」；不在人工複核的 606 格內） | 頭上有金屬頭盔，與未防禦編輯相同 |
| `woman_50__p1` | C5-R7 | vae_trans | 0／1／0 | 警察制服，含臂章與徽章 |
| `woman_22__p3` | C5-R7 | vae_trans | 0／0／0 | 領口有紅色領結 |
| `woman_97__p2` | C2-R7 | none | 0／0／0 | 頭上有金屬頭盔 |
| `woman_97__p2` | C3-R7 | jpeg50 | 0／1／0 | 同上 |

`audit/summary_corrected.txt`、`audit/paired_difference_corrected.txt` 為兩支量測程式讀更正檔的輸出。與 `*_human.txt` 的差異只在 CAPGuard 列
與含 CAPGuard 的配對差值（協定同上：crosseditor100 100 張 × 4 句，條件式 ISR，影像層 bootstrap）：

| 讀數 | none | jpeg50 | blur2 | vae_trans | 三道合計 |
|---|---|---|---|---|---|
| CAPGuard 條件式 ISR，`judgments_human.csv` | 17.0%（55/324） | 18.4%（58/316） | 16.9%（52/307） | 14.0%（41/293） | 16.5%（151/916） |
| CAPGuard 條件式 ISR，更正版 | 16.4%（53/324） | 17.7%（56/316） | 16.6%（51/307） | 13.0%（38/293） | 15.8%（145/916） |
| CAPGuard − BlurGuard，`judgments_human.csv` | +2.2 [−1.5, +6.0] | +1.3 [−1.9, +4.4] | +5.5 [+1.7, +9.9] | +6.1 [+2.8, +9.6] | +4.3 [+2.0, +6.9] |
| CAPGuard − BlurGuard，更正版 | +1.5 [−2.2, +5.5] | +0.6 [−2.8, +3.9] | +5.2 [+1.3, +9.5] | +5.1 [+1.7, +8.7] | +3.6 [+1.1, +6.3] |

## 已知限制

- 人工複核只涵蓋 606 格；其餘 10,594 格仍為評審標籤。
- 評審標準在部分批次較寬，尚未重判：batch 12（woman_20–22 前兩句，淡色配件計為出現）、batch 15（woman_28__p3 手腕上的領結計為出現）、batch 34（woman_84__p2 部分淡色頭盔計為出現）、batch 37（雙領結計為出現）。
- 只有 ip2p；VIBE 與 klein 的同版面拼圖未評。
