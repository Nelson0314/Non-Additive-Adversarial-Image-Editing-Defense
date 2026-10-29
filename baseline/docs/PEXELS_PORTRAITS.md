# Pexels 人像資料組

`data/pexels_portraits/`：男 108 張、女 198 張，共 306 張，512² RGB PNG，構圖與 `data/portraits/` 同型（單人、近正面、半身或全身）。
用於在較大的影像集上重跑外部條件的原生預算評測（ip2p 場景），入口為 `scripts/evaluate_pexels_portraits.sh`。

## 來源與授權

- 中繼資料：Hugging Face `gaunernst/pexels-portrait`（`pexels_portrait_part1.csv.gz`，9,853 列，Pexels API 欄位，已以 YOLOv8-Face 去除無臉影像）。
- 影像：`images.pexels.com` 的原圖，參數 `auto=compress&cs=tinysrgb&w=1200`，依 Pexels License 使用。
- `provenance.json` 逐張記錄 Pexels ID、頁面與影像網址、攝影師、alt 文字、原尺寸、臉框、裁切框與輸出 PNG 的 SHA-256。

## 選取

1. alt 文字只含男性或女性其中一類詞，且不含兒童、群體、黑白、遮擋、藝術化描述等排除詞：5,763 張。
2. MTCNN（`facenet_pytorch`，與 `immunization_core.metrics.identity` 相同偵測器）：機率 ≥ 0.90 的臉恰一張且其機率 ≥ 0.98；
   臉框高度 / 短邊介於 0.07 與 0.50；鼻尖對兩眼中點的水平偏移 ≤ 0.25 倍眼距；Lab 平均 |a*| + |b*| ≥ 6；
   臉框完整落在裁切內：2,486 張。
3. CLIP ViT-L/14 零樣本 `a photo of a man` / `a photo of a woman` 的機率 ≥ 0.8 且與 alt 的類別一致，且 FaceNet 取得到身分向量：
   2,459 張（男 407、女 2,052）。
4. 依裁切後影像與 `data/portraits` 八張的 CLIP 影像餘弦平均值排序，同一攝影師至多兩張；
   與同類別已選影像的 FaceNet 身分餘弦 ≥ 0.65 者略過（同一人或同一場拍攝）。
5. 目視剔除：臉被遮擋、低頭或側臉、閉眼、人物過小、臉部彩繪或戲服、兒童、赤裸上身、過暗。
6. 主體遮罩面積落在 [0.01, 0.80] 之外的影像不納入（3 張）。
7. 主體遮罩未涵蓋人物的影像不納入（3 張：`woman_115` 只標到畫面下緣兩小塊，`man_70`、`man_75` 只標到頭部）；
   檔名保留空號，其餘影像的編號不變。

影像序 0–96（`man_00`–`man_47`、`woman_00`–`woman_48`）為第一批，由各類別前 80 名候選目視選出。
影像序 97 起為擴充：男性候選 161 張（通過步驟 4 的全部男性）目視通過 62 張，女性前 430 名候選目視通過 153 張、其中 3 張因步驟 6 排除，
依 CLIP 相似度由高到低續編號。男性在來源中通過步驟 3 的只有 407 張，兩類別的張數因此不相等。

裁切沿用 `archive/anti-purification/scripts/crop_portraits.py`：邊長 min(H, W)，中心為臉框中心往下 0.18 倍邊長，
超界夾回，LANCZOS 縮至 512²。

## 遮罩

`masks/` 由 `immunization_baseline.cli.generate_subject_masks` 產生（CLIPSeg `CIDAS/clipseg-rd64-refined`，
文字為類別的 `content`，門檻 0.30，膨脹 4 px，白 = 重繪）。同一程式在 `data/portraits` 上的輸出與已入庫的八張遮罩逐位元相同。
306 張的主體面積為 0.168–0.792。`masks/overview/`（不入版控，由同一 CLI 重新產生）中，第一批可見的分割誤差：`man_00`（窗框併入主體）、`man_36`（軀幹有缺口）、
`woman_03`（帽頂未含）、`woman_10` 與 `woman_46`（頭髮上緣未含）。遮罩只影響主體／背景分區欄與 `diffvax`。

## 評測協定

與主表相同：`timbrooks/instruct-pix2pix`，`s_t` 7.5、`s_i` 1.8，種子 20260812，50 步，512²，四條 FaceLock 指令；
七道淨化同 `immunization_core.purifiers.protocol`；LPIPS 為 `piq.LPIPS()`。只跑 ip2p 場景，不含 `color` 條件。

條件（`scripts/evaluate_pexels_portraits.sh` 的預設）：`diffvax`、`photoguard_encoder`、`mist`、`dia_pt`、`sifm`、`dia_r`、
`dct_shield_y`、`dct_shield`、`dayn`、`danp`。`photoguard_encoder` 為 PhotoGuard 的 encoder attack
（官方 img2img notebook：L∞ 0.06／[-1,1]、1,000 步、步長由 0.02 線性衰減、隨機起點）；
主表的 `photoguard_c`、`photoguard_linf` 為 diffusion attack，每張求解約 6,400 秒，不在預設內。

## 影像組與產物

`image_sets.json` 把影像序切成影像組：`images_0000_0096` 為第一批 97 張，其後每 10 張一組（`images_0097_0106`…）。
每個影像組完成全部條件的防禦生成、編輯、七道淨化、量測與補充指標（FSIM、ΔE00、MSE 與七項無參考美學指標）後才進行下一組。

| 影像組 | 產物 | 狀態與讀數 |
|---|---|---|
| `images_0000_0096` | `artifacts/pexels_portraits/` | `runtime/pexels_portraits/*.done`、`runtime/pexels_portraits/results/` |
| 其餘 | `artifacts/pexels_portraits/sets/<影像組>/` | `runtime/pexels_portraits/sets/<影像組>/`（`*.done` 與 `results/`） |

各影像組的資料子集為 `runtime/pexels_portraits/data/<影像組>/`（指向本資料組的符號連結）。
入版控的讀數在 `results/pexels_portraits/`，見「讀數」。

執行：`ENV_FILE=~/env.sh bash scripts/evaluate_pexels_portraits.sh <卡號> <影像組> [<影像組>...]`；
中斷後以同一指令接續（完成標記與 `results_*.csv` 的既有列）。

## 讀數

`results/pexels_portraits/` 為已完整完成的影像組的合併表，由 `scripts/generate_pexels_result_tables.py` 自各影像組的 `results/` 產生；
納入的影像組列於同目錄的 `image_sets.txt`。現有 186 張（`images_0000_0096` 至 `images_0177_0186`）、十個條件：
`defense_<條件>.csv`（186 列）、`displacement_<條件>.csv`（744 列）、`retention_<條件>.csv`（5,208 列）、
`fidelity.csv`（FSIM、ΔE00、MSE）與 `aesthetic.csv`（七項無參考美學指標，含 `condition=original` 參照列）。

186 張的平均（ip2p；編輯結果每條件 744 格；淨化後四欄為淨化後的編輯結果 LPIPS）：

| 條件 | 防禦圖 LPIPS | 防禦圖 PSNR | 編輯結果 LPIPS | 擋下 | `jpeg30` | `blur2` | `crop_resize0.1` | `rotate15` | 每張求解秒數 |
|---|---|---|---|---|---|---|---|---|---|
| `diffvax` | 0.044 | 36.05 | 0.081 | 4/744 | 0.077 | 0.058 | 0.087 | 0.088 | 0 |
| `dia_r` | 0.282 | 40.35 | 0.335 | 75/744 | 0.181 | 0.093 | 0.283 | 0.235 | 145 |
| `dia_pt` | 0.386 | 36.64 | 0.449 | 286/744 | 0.239 | 0.150 | 0.391 | 0.317 | 69 |
| `dayn` | 0.447 | 33.47 | 0.496 | 405/744 | 0.328 | 0.226 | 0.455 | 0.381 | 478 |
| `sifm` | 0.471 | 33.03 | 0.519 | 470/744 | 0.331 | 0.238 | 0.474 | 0.395 | 122 |
| `danp` | 0.488 | 32.97 | 0.543 | 491/744 | 0.350 | 0.253 | 0.498 | 0.417 | 563 |
| `dct_shield` | 0.549 | 29.66 | 0.681 | 719/744 | 0.138 | 0.115 | 0.487 | 0.383 | 154 |
| `photoguard_encoder` | 0.558 | 31.72 | 0.703 | 734/744 | 0.297 | 0.206 | 0.505 | 0.411 | 157 |
| `mist` | 0.623 | 26.69 | 0.636 | 738/744 | 0.436 | 0.360 | 0.607 | 0.515 | 78 |
| `dct_shield_y` | 0.633 | 24.38 | 0.698 | 733/744 | 0.218 | 0.111 | 0.519 | 0.446 | 154 |

「擋下」為兩張編輯結果的 SigLIP 影像餘弦低於 0.837 的格數。
`artifacts/pexels_portraits/report/index.html`（不入版控）為第一批 97 張、七個條件時產生的報告頁。

## 操作

遠端專案根為 `~/image-immunization/baseline`；狀態目錄 `R=runtime/pexels_portraits`。

| 項目 | 位置或指令 |
|---|---|
| 兩個 worker 的影像組清單 | `$R/worker_a.sets`、`$R/worker_b.sets`（已完成的影像組由完成標記跳過） |
| worker 的主機、卡與結束碼 | `$R/worker_<名>.lock/owner` |
| log | `$R/worker_<名>.log`（`[STEP]` 行為步驟邊界） |
| 啟動或恢復一個 worker | 把 `$R/worker_<名>.lock` 移到 `$R/superseded/`，再於兩台主機各執行 `setsid nohup bash scripts/run_worker_when_card_free.sh <名> $(cat $R/worker_<名>.sets) > /dev/null 2>&1 < /dev/null &` |
| 等失真臂的 worker | 兩台各一個 `setsid nohup bash scripts/run_pexels_aligned_worker.sh <a|b>`；取卡、影像組鎖 `$R/aligned/locks/`、紀錄 `$R/aligned/run.log` 與 `worker_<名>.log` |
| 為 color 讓卡並自動恢復 | 兩台各執行 `setsid nohup bash scripts/run_pause_for_color_queue.sh <標籤> <color log>[=<標記>] ...`：停止本機的等待行程與等失真取卡迴圈，worker 在步驟邊界暫停（`IMMEDIATE=1` 立即終止），color log 新增標記（預設 `[ALL-DONE]`，`@<檔名>` 表示檔案出現）後恢復；`NO_PAUSE=1` 只負責恢復，`NATIVE_WORKERS`、`EXTRA_ALIGNED_LOOPS` 指定要恢復的 worker；紀錄於 `$R/pause_for_color_<標籤>.log` |
| 佇列與監督（現行） | 工作清單 `$R/queue/jobs.txt`（每行「<工作 ID> [AFTER=<工作>[,…]] [CPU] <指令>」；`AFTER=` 所列工作完成後才可認領，`CPU` 為不取卡的 CPU 工作，每台至多 2 個，不計入卡數）；兩台的 crontab 每 30 分鐘執行 `scripts/run_pexels_supervisor.sh`，依可用卡數以 `scripts/run_pexels_job.sh` 開 worker（每個 worker 認領一項工作、取一張卡、做完即退出）。紀錄：`$R/queue/supervisor.log`、`worker.log`、`logs/<工作>.log`；完成標記 `$R/queue/done/`，失敗 3 次記於 `$R/queue/failed/` |
| baseline 可用卡數 | 6 − max(`~/gpu_leases` 中名稱不以 `baseline_` 開頭的租約數, `$R/queue/color_reserve` 的整數)；baseline 的租約名皆為 `baseline_pexels_<工作>` |
| color 取卡 | 在 `$R/queue/color_reserve` 寫入 color 要的張數（0–6），再於兩台各執行一次 `bash scripts/run_pexels_supervisor.sh`：已認領工作多於可用卡數時，監督停止本機最新認領的 worker 並交還其卡，之後不再開回；中斷的步驟由下一個 worker 依完成標記重做。color 用完後把該檔改回 0（或刪除），下一個週期 baseline 自動擴回 |
| 後續量測 | 每個影像組五項工作：`followup_<影像組>_<undefended|native|aligned|color3d>`（`scripts/evaluate_pexels_followup.sh`，GPU：VAE-Trans 淨化與編輯、VAE-Trans 位移保留、語意與身分）與 `cpu_<影像組>`（`scripts/measure_pexels_followup_cpu.sh`：編輯結果美學、color3d_pexels 防禦圖失真與美學）；產物 `artifacts/pexels_portraits/vae_trans/`，狀態與讀數 `$R/followup/<根>/`（`done/`、`results/<分組>/`）；前置條件不成立時結束碼 4，不計失敗 |
| 暫停整個佇列 | 建立 `$R/queue/PAUSED`（監督不再開 worker，已在跑的做完即退出）；刪除即恢復 |
| 合併讀數 | `python scripts/generate_pexels_result_tables.py <輸出目錄>`（遠端專案根），再把輸出複製到本機 `results/pexels_portraits/` |

- `run_worker_when_card_free.sh` 每 5 分鐘尋找沒有 compute app 且沒有租約的卡；同名 worker 以 `worker_<名>.lock` 保證只啟動一次，
  worker 結束後不重新啟動（租約被拒的結束碼 3 除外）。
- worker 的 PID 以 `pgrep -f "^bash scripts/evaluate_pexels_portraits.sh <卡號> "` 取得；樣式不加 `^` 時會符合執行該指令的 ssh 殼。
  `pkill -f` 的樣式出現在自身指令列時會終止該連線。
- 暫停後 worker 以結束碼 143 結束並釋放租約；恢復時只重跑被中斷的那一步。
- 以 `run_pause_for_color_queue.sh` 在 basic-1 終止 worker 時，監看本身曾兩次在終止後結束、未完成恢復（原因未查明）；
  較穩的作法是在 ssh 指令中以 PID 直接停止 worker、移除被停 driver 的影像組鎖，再以 `NO_PAUSE=1` 啟動只負責恢復的監看。
- 不要以 `HF_HUB_OFFLINE=1` 啟動 worker：遠端快取的 SD v1.4 在離線模式下無法解析而立即失敗。
- 已取得租約的卡仍可能被其他使用者的行程佔用顯存，worker 會以 CUDA out of memory 結束（結束碼 1），需依上表恢復。
- 主體遮罩未涵蓋人物的影像會使幾何淨化後的主體區為空，量測步驟即中止；處理方式為依「選取」第 7 點自資料組移除該影像。

## GPU 時間

每張影像的防禦生成時間（秒；`defense_<條件>.csv` 的 `total_seconds` 平均）：`diffvax` 0、`dia_pt` 69、`mist` 77、`sifm` 123、
`dia_r` 147、`dct_shield` 155、`dct_shield_y` 155、`dayn` 483（73 張），以上為第一批實測；`danp` 569、`photoguard_c` 6,414、
`photoguard_linf` 6,481 為主表 8 張的實測。ip2p 編輯為 64 個 arm（每個 388 格）的平均，每格 11.6 秒。

| 配置 | 每張 GPU 小時 | 100 張 | 200 張 | 500 張 | 2,000 張 |
|---|---|---|---|---|---|
| 11 條件（含兩個 photoguard diffusion attack），含七道淨化（每張 384 格） | 5.31 | 531 | 1,063 | 2,657 | 10,629 |
| 同上，只跑未淨化（每張 48 格） | 4.23 | 423 | 846 | 2,115 | 8,461 |
| 去掉兩個 photoguard diffusion attack，含七道淨化（每張 320 格） | 1.53 | 153 | 305 | 763 | 3,052 |
