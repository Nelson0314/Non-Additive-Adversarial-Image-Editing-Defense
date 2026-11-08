# baseline 交接

冷啟動順序：根目錄 `CLAUDE.md` 與 `AGENTS.md` → 本檔 → `STATUS.md` → 專案記憶 `MEMORY.md`（至少 color-has-gpu-priority、no-local-compute、
never-overwrite-running-scripts-on-nfs、lease-limit-counts-global-leases、no-agents-unless-asked、no-handover-stops-before-85-percent）。

## 職責範圍

baseline 專案負責外部免疫方法（baseline 臂）的移植、求解與評測，並以同一套讀數程式量測 color 專案的方法 CAPGuard（條件名
`color3d_scale2p5_pexels`，防禦圖由 color trial `trials/lut3d_defense_and_naturalness/` 產生）。論文的表與圖由 paper session 取用；
executor 負責派工、push 與轉知 paper。baseline session 只寫 `baseline/` 與遠端 `~/image-immunization/baseline/`。

## 結果正本

| 項目 | 版控正本 | 遠端工作區 | 產生指令 | commit |
|---|---|---|---|---|
| Pexels 306 張主表（等失真 9 臂＋`color3d_pexels`） | `results/pexels_portraits/main_table/` | `runtime/pexels_portraits/summary/` | `scripts/measure_pexels_summary_inputs.sh summary` | 0fbc5e11 |
| CAPGuard 2.5 倍主表列（306 張） | `results/pexels_portraits/color3d_scale2p5/` | `runtime/pexels_portraits/summary/color3d_scale2p5/` | `COLOR3D_GROUP=color3d_scale2p5 bash scripts/measure_pexels_summary_inputs.sh summary` | 0fbc5e11（程式 2c33b25） |
| BlurGuard 主表列（306 張） | `results/pexels_portraits/blurguard/` | `runtime/pexels_portraits/blurguard_306/`（兩批：`blurguard/`、`blurguard_rest/`） | `BLURGUARD_ROOTS=… bash scripts/evaluate_pexels_blurguard_summary.sh` | 0fbc5e11（100 張版 e84f78c3） |
| UltraEdit 跨編輯器 50 張（等失真 4 臂） | `results/pexels_portraits/ultraedit/` | `runtime/pexels_portraits/ultraedit_readouts/` | `scripts/evaluate_pexels_ultraedit_readouts.sh` | 0fbc5e11 |
| FLUX.1 Kontext 512² 跨編輯器 34 張 | `results/pexels_portraits/flux_512/` | `runtime/pexels_portraits/flux_512/readouts/` | `IMAGES_FILE=… bash scripts/evaluate_pexels_flux_512_readouts.sh` | ccb4ad7 |
| VIBE-DistilledCFG 跨編輯器 100／306 張 | `results/pexels_portraits/vibe_distilled/`（100 張在 `crosseditor100/`） | `runtime/pexels_portraits/vibe_distilled/readouts_{100,306}/` | `EDITOR_NAME=vibe_distilled … scripts/evaluate_pexels_flux_512_readouts.sh`；306 張的語意步驟以 `runtime/pexels_portraits/vibe_distilled/readouts_parts.sh` 分 4 片 | b78e9daf |
| FLUX.2 klein 4B 蒸餾跨編輯器 100／306 張 | `results/pexels_portraits/klein_distilled/`（100 張在 `crosseditor100/`） | 編輯 `runtime/pexels_portraits/klein/`，讀數 `runtime/pexels_portraits/klein_distilled/readouts_{100,306}/` | 同上，`EDITOR_NAME=klein` | b78e9daf |
| 原生預算 dct_shield_y（Fig. 3） | 尚未入版控 | 存續比 `runtime/pexels_portraits/summary/native_dct_shield_y/survival_{seven,vae_trans}.csv`（306 張）；淨化強度掃描 `runtime/purifier_sweep_native_dct_shield_y/retention_dct_shield_y.csv`（portraits 8 張） | `runtime/pexels_portraits/summary/run_native_*.sh` | — |

data/portraits 8 張的主表、等失真臂、FLUX／UltraEdit 全表見 `STATUS.md` 與 `README.md`「結果表」。

## 方法與各臂設定

- 等失真臂：各方法的原生 eps 乘 `eps_scale`，使 data/portraits 8 張的平均防禦圖 LPIPS（`piq.LPIPS`，交付 PNG）等於錨點 0.3344；倍率在
  `results/aligned/defense_<條件>.csv`（例如 dct_shield_y 0.073、dct_shield 0.211、mist 0.125），`photoguard_encoder` 為 eps_scale 0.2614（2/255）、1000 步。
  全部列標 `modified_from_paper=True`。論文主表的 baseline 為 9 臂（`mist`、`dia_pt`、`sifm`、`dia_r`、`dct_shield_y`、`dct_shield`、`dayn`、`danp`、`photoguard_encoder`）。
- BlurGuard：官方 `pgd_freq`，ε 16/255、150 步、sigma_weighting 0、learning_rate 0.06，求解後擾動乘 0.6071（data/portraits 8 張校準到 0.3344；
  sigma_weighting 10／5／2／1／0.5／0.25／0.1 於 lr 0.06 皆約 0.232，0 為 0.419；紀錄 `runtime/blurguard/calibration/run.log`）。
  標 `modified_from_paper`。同一倍率下 Pexels 306 張的防禦圖 LPIPS 為 0.2745（100 張 0.2711）。程式 `src/…/generate_blurguard_defenses.py`（d11247a）、
  `scripts/apply_blurguard_scale.py`、`scripts/evaluate_pexels_blurguard.sh`、`scripts/generate_pexels_blurguard_subset.py`。
- DCT-Shield-Y：主表為等失真版；Fig. 3 依使用者指示改用原生預算（eps 1、Q_alg 0.85），讀數見上表。等失真倍率 0.073 下擾動不足一個 JPEG 量化級，
  論文 §4.2 的抗 JPEG 條件不成立（`docs/reference/AUDIT_DCT_SHIELD.md`）。
- CAPGuard：`color3d_scale2p5_pexels`（color 專案；極角 < 25°、色相 < 7.5°）。1.0 倍的 `color3d_pexels` 只保留既有數據。

## 資料

- Pexels 人像 306 張（男 108、女 198），512² PNG，`data/pexels_portraits/`；選取規則與授權見 `docs/PEXELS_PORTRAITS.md`，逐張出處 `provenance.json`。
- 影像組：`data/pexels_portraits/image_sets.json`（第一組 `images_0000_0096` 97 張，其餘每組 10 張）。
- 跨編輯器子集 `runtime/pexels_portraits/crosseditor100/images.txt`：color trial `pexels_subsets/flux.txt` 的 50 張（性別 × color3d_pexels 防禦圖 LPIPS 四分位分層）
  ＋影像組 `images_0097_0106` 至 `images_0137_0146` 的 50 張（color 已有這幾組的 UltraEdit 未防禦分母）。306 張順序為這 100 張再接其餘 206 張
  （`runtime/pexels_portraits/<編輯器>/images_306.txt`）。
- BlurGuard 第一批 100 張：`results/pexels_portraits/blurguard/images.txt`（分層抽樣，含 color 的 budget_sweep 20 張，KS D＝0.029）；第二批 206 張分 29 個單位
  （`runtime/pexels_portraits/blurguard_rest/units/`）。

## 協定

- ip2p：`timbrooks/instruct-pix2pix`，text guidance 7.5、image guidance 1.8、50 步、512²、種子 20260812（主種子，不跑多種子）；四句指令
  （FaceLock；UltraEdit 與非 SD 系編輯器用 `configs/prompts/ultraedit_templates.json` 的 add 句型）。
- 淨化：七道 `blur1`、`blur2`、`crop_resize0.1`、`jpeg30`、`jpeg50`、`jpeg80`、`rotate15`（`immunization_core.purifiers.protocol`）與 VAE-Trans
  （color trial `artifacts/purification/vae_trans/model/encoder.pt`）。
- 讀數定義與區間：`scripts/generate_pexels_main_table_summary.py` 開頭。主讀數為編輯結果 LPIPS（未防禦編輯對防禦後編輯，兩者經同一淨化）；
  七道欄為 7 個逐道平均的平均、區間 t(0.975, 6)·s／√7；其餘欄為影像間 1.96·s／√n。另有 SSIM／PSNR、保留率、SigLIP、CLIP-I、CLIP-S、
  ArcFace（原圖對編輯、E_u 對 E_d）、存續比、FID／KID、防禦圖 LPIPS 與美學。不設判準，數字連同協定回報。
- FID、KID、CLIP-IQA 的特徵以 GPU 計算（`measure_distribution_metrics.py --device cuda`）；主表、CAPGuard 2.5 倍、BlurGuard、VIBE-DistilledCFG 已由
  CPU 版取代，比對與 CPU 版副本位置見 `results/pexels_portraits/distribution_device/README.md`。`evaluate_pexels_flux_512_readouts.sh` 在有卡時以 GPU 計算。
- 跨編輯器：防禦圖與淨化後輸入一律取 ip2p 流程的產物，不以替代編輯器重新求解。

## 跨編輯器現況

- 現行：VIBE-DistilledCFG（`iitolstykh/VIBE-Image-Edit-DistilledCFG`，20 步）與 FLUX.2 [klein] 4B 蒸餾（`black-forest-labs/FLUX.2-klein-4B`，4 步、guidance 1.0），
  512²。條件：undefended、photoguard_encoder、dct_shield、mist、blurguard、color3d_scale2p5_pexels；全協定 216 次編輯／張。
  入口 `scripts/evaluate_pexels_nonsd_editor.sh <vibe_distilled|klein> <影像>`（版面同 FLUX 512），編輯 `scripts/run_nonsd_edits.py`。
  實測（RTX 3090、含 VAE 編解碼）：VIBE-DistilledCFG 每次 0.97 s、峰值 8.5 GB；klein 蒸餾 1.55 s、15.5 GB（`scripts/measure_nonsd_editor_speed.py`，結果 `results/pexels_portraits/nonsd_editor_speed/speed_<模型>.json`）。
  授權（2026-11-07 查證，未判定）：FLUX.2 klein 4B 模型卡為 apache-2.0。VIBE-Image-Edit(-DistilledCFG) 模型卡沒有 license 標籤，README 寫
  「built upon SANA, refer to the original SANA license」，所附連結（SANA1.5_4.8B…/LICENSE.txt）回傳 404；SANA 程式庫 NVlabs/Sana 為 Apache-2.0；
  SANA1.5_1.6B_1024px_diffusers 模型卡標 apache-2.0，Direct Use 段寫「intended for research purposes only」；VLM Qwen3-VL-2B 為 Apache-2.0。
  trial `non_sd_editor_speed` 已 promote（`docs/TRIALS.md`）；範例對比圖在遠端 `artifacts/pexels_portraits/vibe_distilled/examples/`。
- 已停：UltraEdit 與 FLUX.1 Kontext 的 2.5 倍與補到 306 張（UltraEdit 2.5 倍完成 21／100）。未執行的工作行在遠端 `runtime/pexels_portraits/queue/jobs_deferred_manual.txt`。
- 讀數：`scripts/evaluate_pexels_flux_512_readouts.sh`（`EDITOR_NAME`、`IMAGES_FILE`、`CONDS`、`READOUTS_DIR`），彙整 `generate_pexels_flux_512_summary.py`。

## 進行中的工作（佇列自動執行）

- VIBE-DistilledCFG：100／306 張讀數完成並入版控。306 張的 `vibe_distilled_r306_finish` 已自佇列移除（改前清單 `queue/superseded_jobs_before_fid_gpu.txt`），
  其 ArcFace、SSIM／PSNR 以手動 CPU 行程、FID 以 `runtime/pexels_portraits/fid_cuda/run_fid_cuda.sh` 完成；前 100 張的逐格讀數與 100 張版逐值相同。
- FLUX.2 klein 4B 蒸餾：100／306 張讀數完成並入版控（306 張以 `scripts/run_pexels_readout_parts.sh` 分段，FID 走 GPU）；前 100 張的逐格讀數與 100 張版逐值相同。
- 完成後：summary 複製到 `results/pexels_portraits/<編輯器>/` 並提交，路徑與主要數字回報 executor，由 executor 轉 paper。

## 佇列與 GPU

- 佇列：遠端 `runtime/pexels_portraits/queue/jobs.txt`，每行「<ID> [AFTER=…] [HOST=…] [CPU] <指令>」；兩台 crontab 每 5 分鐘執行
  `scripts/run_pexels_supervisor.sh`，以 `scripts/run_pexels_queue_worker.sh` 認領（取鎖後重驗完成狀態），完成標記 `done/`、失敗 3 次 `failed/`、紀錄 `logs/`。
  暫停新認領：`touch queue/PAUSED`。清單改動寫 `jobs.txt.next` 再 `mv`，改前清單另存，舊的改前清單在 `~/trash/baseline/2026-11-07/queue/`。
- 卡數：全局 ≤ 6（所有 session 合計），color 優先。baseline 可用卡數 ＝ min(`queue/baseline_cap`, 6 − max(非 baseline 租約數, `queue/color_reserve`))。
  只取完全空的卡，不與他人共用；租約工具 `vendor/scripts/run_with_gpu_lease.sh`（不給 `--limit`）。線數以 AFTER 串鏈控制。
- 排程只依使用者經 executor 下達的指令；不自行接續下一階段。

## 遠端環境

- basic-1 `ssh -p 10101`、basic-2 `-p 10102`（`nelson0314@server.basiclab.lab.nycu.edu.tw`），NFS 共用 `~`；repo `~/image-immunization`。
  basic-1 上無法 ssh 到 basic-2，取卡重試要在有卡的主機開。
- Python：`export ENV_FILE=~/env.sh; source scripts/env.sh` 後的 `$PY`（`~/venvs/wacv`）；VIBE 用 `~/venvs/vibe`（torch 2.6.0、diffusers 0.33.1、transformers 4.57.1、vibe）。
- HF 快取：各機 `/var/cache/huggingface`（`HF_HOME`，不設 `HF_HUB_OFFLINE`）。FLUX.1-Kontext 兩台都有；FLUX.2 klein 4B 與 VIBE 兩台都有；UltraEdit 只在 basic-1。

## 已知的坑

- 遠端 `git pull` 會就地改寫檔案；pull 前確認改到的 `.sh` 沒有行程在執行（必要時以 `/proc/<pid>/fd/255` 對 inode）。遠端腳本更新一律寫暫存檔再 `mv`。
- 租約取得後他人可能隨即佔用同一張卡，造成 OOM（UltraEdit woman_04、klein 測速各一次）；worker 會重試，連續失敗才進 `failed/`。
- 語意與身分讀數（`measure_semantic_identity`）單一行程約每秒 3 格；大量格數以 `--shard/--shards` 分成多個 GPU 工作再合併。
- Python 寫 `.sh` 時保持 LF；ssh 單引號內嵌 Python 會吃掉引號，較長的遠端程式先寫檔再 scp。

## 規範

- `CLAUDE.md`、`AGENTS.md`：只改 `baseline/`；`archive/` 不動；不設判準；不改科學協定；必要輸入失敗即失敗；commit 依機械搬檔／行為修正／文件更新分開，
  訊息用英文；push 由 executor 代為執行。命名規則由 `core/tests/test_repository_naming.py` 檢查。
- 刪除改為移到遠端 `~/trash/baseline/<日期>/`；不確定是否被引用的檔先列給 executor。

## 待使用者決定

- BlurGuard 倍率 0.6071 在 Pexels 上的防禦圖 LPIPS 為 0.2745，未達 0.3344；是否在 Pexels 上重新校準。
- deferred 的 UltraEdit／FLUX.1 Kontext 工作（2.5 倍與補到 306 張）是否恢復。
- Pexels 306 張的淨化強度掃描（約 100 GPU 小時）是否執行。
