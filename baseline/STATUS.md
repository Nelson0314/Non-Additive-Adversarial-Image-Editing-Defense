# baseline：現況與接續指引

接手先讀這一份；協定、檔案位置、程式清單在 `README.md`；兩組進階量測各有說明：
`results/aligned/README.md`（等失真臂）、`results/ADDITIVE_TRANSFER.md`（加性穿透拆解）。

## 資料組

| 資料 | 回答什麼 | 位置 |
|---|---|---|
| 主表 | 12 個方法在各自原生預算下，把 ip2p／inpaint 的編輯結果推開多少 | `results/*.csv` |
| 等失真臂 | 10 個方法縮到同一個防禦圖 LPIPS 之後，還推開多少 | `results/aligned/` |
| 穿透拆解 | 位移中有多少是防禦端的改動原樣穿過編輯器（ip2p） | `results/additive_transfer.csv`、`results/ADDITIVE_TRANSFER.md` |
| FLUX 全表 | 同一批防禦圖換成 FLUX.1-Kontext 編輯 | `results/flux/` |
| UltraEdit 全表 | 同一批防禦圖（含淨化後）換成 UltraEdit（SD3）編輯 | `results/ultraedit/` |

主讀數是 `displacement.csv` 的 `disp_lpips_full`（編輯結果 LPIPS：未防禦編輯 vs 防禦後編輯，
另有主體／背景分區）。主表的位移排名與防禦圖失真排名幾乎同序（原生防禦圖 LPIPS 0.043–0.666），
等失真臂與穿透拆解是為了把「防禦有效」與「擾動較大」分開。各組協定不同，引用數字時連協定一起引用。

## 現況

全部資料組都已跑完，沒有進行中的遠端工作，沒有指定中的待辦。

### 顏色列是 `color`
主表的顏色那一列是 color 專案的方法 `color`（預設參數）。防禦圖由 color 專案產出，經
`import_defense_artifacts --variant color` 整理成主表版面並重算保真欄；ip2p／inpaint 編輯、7 道淨化與
淨化後重編、FLUX、UltraEdit 都重跑過（`scripts/evaluate_color_condition.sh`），讀數以同一程式只算 `color`
後替換進各聚合 CSV，`results/additional_metrics/` 整張重算。
舊顏色列 `color_curve`（CSV 識別值原為 `colour_curve_ours`）的數值留在 commit `0dd243b`。

`color` 的防禦圖 LPIPS 是 0.2317（8 張平均），舊顏色列是 0.3344，量法相同（`piq.LPIPS`）。差距來自上限的
組成：舊顏色列只有整圖平均 ΔE00 ≤ 16（8 張皆頂到）；`color` 另有逐像素 Lab 位移上限（a*＋ ≤ 4、a*－ ≤ 15、
b*＋ ≤ 4、b*－ ≤ 25、|ΔL*| ≤ 15）、臉框與膚色 ΔE00 ≤ 16、彩度 p95、對原圖 LPIPS 與位移場半徑 80，
8 張中 a*－ 與位移場半徑各有 6 張頂到，整圖平均 ΔE00 為 9.0–13.8。

**未跟著換的兩處**：等失真臂的錨點仍是舊顏色列的 0.3344（使用者指示不重新對齊）；`results/ADDITIVE_TRANSFER.md`
的文字與表格是舊顏色列的數字，`additive_transfer.csv` 的顏色列已是 `color`。

### 跨編輯器
| 編輯器 | 狀態 | 協定 |
|---|---|---|
| FLUX.1-Kontext-dev | 全表 13 arm × 32 格 | guidance 3.5、28 步；管線強制 1024×1024；4-bit NF4 量化，單卡 3090 |
| UltraEdit（`BleachNick/SD3_UltraEdit_freeform`） | 全表 13 arm ×（未淨化＋7 道淨化）× 32 格 | 指令 `Add … to the person`、guidance 2.5、image guidance 1.5、512²、50 步 |
| SDEdit（SD 1.5、SD 2.1-base） | 放棄 | strength 0.2–0.6、guidance 3–10 掃過，身分保留與指令執行沒有交集 |
| SDXL-InstructPix2Pix（`diffusers/sdxl-instructpix2pix-768`） | 只掃參數 | 空指令也把背景換灰、加亮暈（背景 ΔE00 21–26，主表 ip2p 為 2.9），guidance 壓不掉 |
| CosXL Edit | 未跑 | gated 權重，遠端沒有 HF token |

各方法的編輯結果 LPIPS 排名，與 ip2p 的 Spearman 相關：inpaint 0.979、FLUX 0.951、UltraEdit 0.951。

**UltraEdit 的參數選法**（`results/sweeps/ultraedit/` 與對應的 `*_off_target.csv`，
2–4 張未防禦影像）：作者 README 的範例值 g7.5／ig1.5 會換臉、改動指令以外的區域；guidance 越低指令以外的
改動越少，image guidance 2.5 反而提高對比；g2.5／ig1.5 下六種句型（`configs/prompts/ultraedit_templates.json`）的
背景 ΔE00 為 2.21–2.34，`add` 句型三項改動讀數最低；換名詞與 g4（`configs/prompts/ultraedit_noun_placement_variants.json`）
沒有讓 `police suit` 被執行，且加重墨鏡那格的換臉。管線不在官方 diffusers，`third_party/ultraedit/pipeline.py`
移植自作者的 fork，只改 import。

### 已知限制
- **幾何淨化的分區欄已更正**：`retention.csv` 中 `color` 以外 11 個條件的 `crop_resize0.1`、`rotate15`（1,408 列）的
  `disp_purified_subject`／`disp_purified_background` 原以未變換的遮罩算出，已用 `purified_mask()` 重算並寫回
  （2,814 個值改變，最大絕對變化 0.10288）。重算時其餘欄與非幾何算子的分區欄逐值與原表相同。更正前的數值見
  commit `228c59b` 的 `baseline/results/retention.csv`。
- **等失真臂的幾何淨化分區欄已查證**：`results/aligned/retention.csv` 全部 4,480 列以 `purified_mask()` 重算，
  包括 `crop_resize0.1`、`rotate15` 的 1,280 列分區欄在內逐值與原表相同，原表即以變換後的遮罩算出，數值未改。
- **SD 2.x 只能用 epsilon-prediction 權重**：`sd2-community/stable-diffusion-2-1` 是 v-prediction，`SDWrapper`
  的 DDIM 遞迴假設 ε-prediction，要用 `sd2-community/stable-diffusion-2-1-base`。

## 遠端

- 遠端 `~/image-immunization/baseline` 為本專案（GitHub `main` 的 clone，以 `git pull` 同步）；影像產物位於 `artifacts/`
  （見 `README.md`「影像產物」）。舊位置與新目錄的對照記錄於 `archive/migration/RESTRUCTURE_LOG.md` 第 10 項。
- `requirements.lock` 未入庫；由遠端執行環境以 `vendor/scripts/freeze_env.py` 產生。
- HF 權重在各機的 `/var/cache/huggingface`，兩台不同：FLUX 只在 basic-2，UltraEdit 只在 basic-1，ip2p 與
  SD-inpainting 兩台都有。機器相關設定以 `ENV_FILE` 交給 `scripts/env.sh`。
- `measure_additional_metrics` 讀 `artifacts/` 的版面與 CSV 中的影像路徑；無 GPU 亦可執行（固定使用 CPU）。

## 規矩

- **GPU 一律送遠端**。全局可用卡數由使用者逐次授權；未說明或說明不清時預設 6 張，所有 session、主機與排程合計。
  取卡一律經 `vendor/scripts/`（`run_with_gpu_lease.sh`、`gpu_lease.sh`），租約目錄 `~/gpu_leases/`
  （`<主機> <pid> <名稱> <擁有者 token>`，兩台共用）；`free_cards.sh` 只擋別人佔用超過 512 MiB 的卡
  （別人單一行程在每張卡上各留約 256 MiB 的 context 可放行，使用者裁定）。
- **不設判準**：數據與圖擺出來為止，不下「成立／不成立」「值得／不值得」的結論，指標本身也一樣。
- 報數字寫描述性名稱（編輯結果 LPIPS、防禦圖 LPIPS），不用 D、D_T 這類代號；引用數字連協定一起引用
  （`docs/reference/BASELINE_PROVENANCE.md` 規則 1）。
- 本專案範圍只到 `baseline/`；`vendor/` 不就地修改，修正回到 `core/` 後重新匯出。
- 共通規則（命名、書面用語、commit、LF）見根目錄 `CLAUDE.md`。
