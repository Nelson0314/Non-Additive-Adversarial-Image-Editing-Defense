# Inpainting ISR 評審規則（草稿）

適用於 `artifacts/pexels_portraits/inpaint_isr_sheets/`（306 張 Pexels 人像 × 4 句 inpaint 指令，1,224 張拼圖），以
`scripts/generate_inpaint_isr_sheets.py` 產生。編輯器為 `runwayml/stable-diffusion-inpainting`：只重繪主體以外的背景（CLIPSeg 主體遮罩的
補集），人物本身保留。判定欄位與 ip2p 版（根目錄 `docs/isr_pilot/rubric.md`）相同，`accessory` 改為 `content`。

## 拼圖

- 標題：影像 ID 與送入編輯器的指令全文，四句為 "A {man/woman} in a hospital"、"A {man/woman} in the gym"、
  "A {man/woman} in a snowy street"、"A {man/woman} sitting next to a dog"。
- 列 R1–R7：7 個條件，列序逐張打亂，列標不含條件名（解盲對照在 `key.csv`）。
- 欄：C1 Input（送入編輯器的影像）、C2 No purification、C3 JPEG 50、C4 Gaussian blur σ2、C5 VAE-Trans。格號印於每格左上。
- 判定 C2–C5 共 28 格。C1 只作參照，不判定。輸入影像本身可能有明顯色偏，色偏本身不算失敗。

## 欄位（每格三個獨立的 0／1 標籤）

- `content`：指令要求的內容清楚出現在重繪的背景中。
  - hospital：可辨認的醫療環境或設備，例如病床、點滴架、監視器、壁掛醫療設備、醫院走廊、診間。只有白牆或一般室內 = 0。
  - gym：可辨認的健身器材或健身房，例如啞鈴、槓鈴、重訓架、跑步機。只有一般室內或一般牆面 = 0。
  - snowy street：可辨認的雪景街道。積雪或飄雪，且有街道、建築或車輛等戶外街景要素，兩者皆需具備。只有雪沒有街景、只有街景沒有雪 = 0。
  - dog：畫面中可辨認的一隻狗（頭部或身體清楚）。只有模糊色塊、殘缺到無法辨認、被裁到只剩極小局部 = 0。
  - 內容淡、只畫一半、需仔細辨認才能看出者 = 0。
- `mismatch`：編輯做了指令以外的事，例如：
  - 人物被換成另一人，或主體區域被改動。遮罩邊緣的細小接縫不算。
  - 多出的人物；指令為 dog 時畫出人或其他動物而非狗。
  - 背景被換成與指令無關的場景。
- `degraded`：嚴重的結構或品質崩壞，例如大面積雜訊、背景成為無法辨識的紋理或塗鴉、主體與背景交界大片破碎。整體色偏、輕微模糊、輕微霧化不算。
- 交界判斷不確定的格可加簡短 note（≤ 8 字）。

## 輸出

CSV 表頭（只寫一次）：`sheet,cell,purification,content,mismatch,degraded,note`。`purification` 取 none、jpeg50、blur2、vae_trans。
每張 28 列，每完成一張即寫入。條件由 `key.csv` 依 sheet 與列號對回。

## 條件式 ISR（與 ip2p 版相同的定義）

只計同一影像、指令、淨化下，未防禦判為成功的格；未防禦成功的條件為 `content`=1、`mismatch`=0、`degraded`=0。在這些格中，防禦後只要任一標籤失敗即計為防禦成功。

## 待審事項

1. inpainting 的指令描述的是場景而非配件。頭肩構圖下重繪區只是人物周圍一圈，hospital 與 snowy street 常只畫出牆面紋理或局部雪景，
   未防禦本身的成功率可能偏低（`data/pexels_portraits/prompts.yaml` 註解中有對應實測），條件式 ISR 的分母會因此變小。
2. snowy street 是否只要求「雪」即可（目前要求雪與街景兩者兼具）。
3. 遮罩比人物輪廓寬一圈，輪廓外留有原背景色的帶狀區，所有條件皆有；建議明定此帶狀區不計入 mismatch 或 degraded。
