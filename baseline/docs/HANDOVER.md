# baseline 交接

冷啟動順序：根目錄 `CLAUDE.md` 與 `AGENTS.md` → 本檔 → `STATUS.md` → 專案記憶 `MEMORY.md`（至少 color-has-gpu-priority、no-local-compute、
never-overwrite-running-scripts-on-nfs、lease-limit-counts-global-leases、no-agents-unless-asked、dont-touch-others-work、
no-handover-stops-before-85-percent）。協定細節、檔案位置與程式清單在 `README.md`；等失真臂的說明在 `results/aligned/README.md`。

## 1. 範圍與分工

- baseline 負責外部免疫方法（加性擾動）的移植、求解與評測，並以同一套讀數程式量測 color 專案的方法 ImmuLUT（原名 CAPGuard；條件名
  `color3d_scale2p5_pexels`，防禦圖由 color trial `trials/lut3d_defense_and_naturalness/` 產生）。
- 方法：PhotoGuard（encoder 攻擊 `photoguard_encoder`；diffusion 攻擊 `photoguard_c`、`photoguard_linf` 只保留 data/portraits 的既有結果）、
  Mist、DIA-PT、DIA-R、DCT-Shield、DCT-Shield-Y、SIFM、DAYN、DANP、BlurGuard；DiffVax 只保留既有結果。
- 論文定調：加性擾動在淨化後大幅減弱，ImmuLUT 的全域色彩變換在淨化後仍保留；主讀數為編輯結果 LPIPS。
- 分工：color 專案負責 ImmuLUT 的求解；paper session 負責 `docs/paper/`、`docs/paper_tools/` 與論文圖表；executor 負責派工、push、文獻與轉知。
  baseline session 只寫 `baseline/` 與遠端 `~/image-immunization/baseline/`，不改論文檔與其他專案的檔案（需要時把事實與數字交給 executor）。

## 2. 工作規範

- 不設判準：數據與影像完整回報，「成立／值得再跑」由使用者判斷。只用主種子 20260812（防禦求解為種子 0，見各 `results/defense_*.csv`）。
- 不改科學協定（LPIPS 後端 `piq.LPIPS`、resize、插值、遮罩、seed、precision、量化、樣本排除）；需要修正時另立資料版本並記錄。
- 移植外部方法時使用其原損失與設定；與論文不同處在結果列標 `modified_from_paper` 並寫 `modification_note`。
- GPU：一律遠端（basic-1 `ssh -p 10101`、basic-2 `-p 10102`，`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 共用 `~`）。全局合計 ≤ 6 張，
  color 優先；只取完全空的卡；租約工具 `vendor/scripts/run_with_gpu_lease.sh`（不給 `--limit`）；取卡重試在有卡的主機上開（basic-1 無法 ssh 到 basic-2）。
  baseline 佇列的卡數上限為遠端 `runtime/pexels_portraits/queue/baseline_cap`；color 要卡時以 `scripts/run_pause_for_color_queue.sh` 停 baseline worker。
- 本機不跑計算（CPU 讀數、畫圖亦送遠端）；本機只編輯、git、ssh、看圖。
- 遠端執行前 `export ENV_FILE=~/env.sh; source scripts/env.sh`，之後以 `$PY`（`~/venvs/wacv`）執行；VIBE 系用 `~/venvs/vibe`。
- 遠端腳本不可就地覆寫：先寫暫存檔再 `mv`；遠端 `git pull` 前確認改到的 `.sh` 沒有行程在執行。佇列清單改動寫 `jobs.txt.next` 再 `mv`，改前清單另存。
- 刪除一律改為移到遠端 `~/trash/baseline/<日期>/`，事先列清單給使用者確認。別人做的東西只呼叫不改。commit 由 executor push。

## 3. 現況

### 3.1 結果正本（Pexels 人像 306 張；ip2p：text 7.5、image 1.8、50 步、種子 20260812，FaceLock 四句）

| 項目 | 版控正本 | 遠端工作區 |
|---|---|---|
| 主表（等失真 9 臂＋`color3d_pexels`） | `results/pexels_portraits/main_table/` | `runtime/pexels_portraits/summary/` |
| ImmuLUT 2.5 倍主表列 | `results/pexels_portraits/color3d_scale2p5/` | `runtime/pexels_portraits/summary/color3d_scale2p5/` |
| BlurGuard 主表列 | `results/pexels_portraits/blurguard/` | `runtime/pexels_portraits/blurguard_306/` |
| VIBE-DistilledCFG 跨編輯器 100／306 張 | `results/pexels_portraits/vibe_distilled/` | `runtime/pexels_portraits/vibe_distilled/readouts_{100,306}/` |
| FLUX.2 klein 4B 蒸餾跨編輯器 100／306 張 | `results/pexels_portraits/klein_distilled/` | `runtime/pexels_portraits/klein_distilled/readouts_{100,306}/` |
| FLUX.1 Kontext 512² 34 張、UltraEdit 50 張 | `results/pexels_portraits/{flux_512,ultraedit}/` | `runtime/pexels_portraits/{flux_512,ultraedit_readouts}/` |
| 分布指標 CPU／GPU 特徵比對 | `results/pexels_portraits/distribution_device/` | `runtime/pexels_portraits/fid_cuda/` |
| 非 SD 編輯器測速 | `results/pexels_portraits/nonsd_editor_speed/` | — |

- 讀數定義與區間：`scripts/generate_pexels_main_table_summary.py` 開頭。主讀數為編輯結果 LPIPS（未防禦編輯對防禦後編輯，同淨化）；
  七道欄為 7 個逐道平均的平均；另有 SSIM／PSNR、保留率、SigLIP、CLIP-I、CLIP-S、ArcFace、存續比、FID／KID、防禦圖 LPIPS 與美學。
- 淨化：七道 `blur1`、`blur2`、`crop_resize0.1`、`jpeg30`、`jpeg50`、`jpeg80`、`rotate15`（`immunization_core.purifiers.protocol`）與 VAE-Trans
  （color trial `artifacts/purification/vae_trans/model/encoder.pt`）。
- FID／KID／CLIP-IQA 的特徵以 GPU 計算（`measure_distribution_metrics.py --device cuda`），與 CPU 版的最大 FID 差 0.001，排名不變。
- 跨編輯器的防禦圖與淨化後輸入一律取 ip2p 流程的產物，不以替代編輯器重新求解。
- data/portraits 8 張的主表、等失真臂、FLUX／UltraEdit 全表與跨模型移植見 `README.md`「結果表」與 `results/` 各子目錄的 README。

### 3.2 等失真設定

各方法的原生 ε 乘 `eps_scale`，使 data/portraits 8 張的平均防禦圖 LPIPS 等於錨點 0.3344（`piq.LPIPS`，交付 PNG）；倍率在 `results/aligned/defense_<條件>.csv`：

| 條件 | eps_scale | 步數 |
|---|---|---|
| `photoguard_encoder` | 0.2614（2/255） | 1000 |
| `mist` | 0.125 | 100 |
| `dia_pt` | 0.418 | 20 |
| `dia_r` | 0.969 | 20 |
| `dct_shield` | 0.211 | 1000 |
| `dct_shield_y` | 0.073 | 1000 |
| `danp` | 0.323 | 100 |
| `dayn` | 0.43 | 100 |
| `sifm` | 0.289 | 100 |

BlurGuard（官方 `pgd_freq`，github.com/jsu-kim/BlurGuard commit e1b2d5e）的擾動幅度由各區塊學到的高斯模糊 σ 決定，ε 只是上限；等失真版本設
`sigma_weighting` 0（σ 不更新、固定為 1）並於求解後把擾動乘 0.6071（`scripts/apply_blurguard_scale.py`；校準紀錄 `runtime/blurguard/calibration/run.log`）。
同倍率下 Pexels 306 張的防禦圖 LPIPS 為 0.2745。官方設定（`sigma_weighting` 10）在 data/portraits 8 張的防禦圖 LPIPS 為 0.231，論文圖表未使用。

### 3.3 論文引用數字的出處

- Table 1 與 ISR 表：3.1 各 `results/pexels_portraits/*` 的 summary（paper session 以 `docs/paper_tools/generate_main_table.py` 讀取）。
- Fig. 4（失真—位移平面）：遠端 `~/paper_figures/in_pexels/points_blurguard.csv`、`condition_summary_blurguard.csv`、`color_summary_scale2p5.csv`
  （paper session 以 `docs/paper_tools/import_pexels_figure_inputs.py` 與 `generate_paper_figures.py` 產生）。方形為 9 個方法的原生預算
  （遠端 `runtime/pexels_portraits/interim/fig3_points.csv`），圓點為 10 個等失真臂（含 BlurGuard）。
- ISR 評審（論文 Table 2 與正文的條件式 ISR）：素材為 `artifacts/pexels_portraits/ip2p_isr100_sheets/`（crosseditor100 的 100 張 × 4 句，
  7 條件 × 4 淨化，`cells.csv` 11,200 列）；已完成（Claude Sonnet 判定＋人工複核 606 格），結果見根目錄 `docs/isr_pilot/`
  （`judgments.csv`、`audit/judgments_human.csv`、`summary.txt`、`paired_difference.txt`）。
- isr50 的 `ip2p_isr_sheets/` 與 `vibe_distilled/isr_sheets/`：未排程，使用者指示才評審。
- 拼圖產生程式在遠端 `runtime/pexels_portraits/{crosseditor100,vibe_distilled}/generate_*.py`。

### 3.4 暫時性嘗試（`trials/`，不入版控，本機與遠端各一份）

| trial | 內容 | 狀態 |
|---|---|---|
| `blurguard_paper_protocol` | BlurGuard 依論文協定（ImageNet-Edit 80 張、SD v1.4 img2img、未淨化）重跑 | 完成；遠端 `trials/blurguard_paper_protocol/results.json`，見 5. |
| `isr_guidance_and_budget` | isr50 前 10 張：ip2p image guidance 1.8／1.2、VIBE、原生預算 PhotoGuard 的拼圖與編輯結果 LPIPS | 完成；`outputs/sheets/`、`outputs/lpips_*.csv` |
| `isr_mllm_judge` | Qwen2.5-VL-7B 與 Qwen3-VL-8B 試判 50 格 | 完成；兩模型逐欄一致 21–41／50 |
| `unsplash_portrait_pool` | Pexels 資料組的選取程式與候選快取 | 保留（資料組來源） |

前三者待使用者決定 promote 或 drop（drop 前在 `docs/TRIALS.md` 寫一列並提交）。

### 3.5 佇列

遠端 `runtime/pexels_portraits/queue/jobs.txt`，兩台 crontab 每 5 分鐘執行 `scripts/run_pexels_supervisor.sh`，以 `scripts/run_pexels_queue_worker.sh`
認領；完成標記 `done/`、失敗 3 次 `failed/`、紀錄 `logs/`；暫停新認領 `touch queue/PAUSED`。目前無待執行工作。未執行的 UltraEdit／FLUX.1 Kontext
工作在 `queue/jobs_deferred_manual.txt`、`jobs_deferred_transfer_rest.txt`、`jobs_deferred_p2_p3.txt`。

## 4. 工具與路徑索引

- 遠端版面：`artifacts/pexels_portraits/{aligned,sets}/`（等失真臂防禦圖與編輯；`aligned/` 為第一個影像組，`sets/<影像組>/` 為其餘）、
  `defenses/`、`defended_edits/`、`purified_edits/`、`vae_trans/`、`vibe_distilled/`、`klein/`；`runtime/pexels_portraits/{summary,followup,queue,
  blurguard,blurguard_rest,blurguard_306,vibe_distilled,klein_distilled,crosseditor100}/`。影像組定義 `data/pexels_portraits/image_sets.json`。
- 求解：`immunization_baseline.cli.generate_defenses`（`--eps-scale`、`--vae-model`）、`immunization_baseline.cli.generate_blurguard_defenses`、
  `scripts/apply_blurguard_scale.py`。
- 主表管線：`scripts/evaluate_pexels_aligned*.sh`、`evaluate_pexels_followup*.sh`、`measure_pexels_summary_inputs.sh`、`generate_pexels_main_table_summary.py`。
- 跨編輯器：`scripts/evaluate_pexels_nonsd_editor.sh`（`run_nonsd_edits.py`）、`evaluate_pexels_flux_512_readouts.sh`、`run_pexels_readout_parts.sh`
  （語意分片＋finish_gpu／finish_cpu）、`generate_pexels_flux_512_summary.py`。
- 分布指標：`scripts/measure_distribution_metrics.py`（`--device cpu|cuda`）。
- teaser：`scripts/generate_synthetic_portraits.py`（FLUX.2 klein 4B 文生圖，輸出 `artifacts/synthetic_portraits/<名稱>/`）、
  `scripts/run_teaser_edits.py`（防禦、淨化、ip2p 編輯；ImmuLUT 從 `--immulut-dir` 讀）、`scripts/generate_teaser_overview.py`。teaser 已定案（Fig. 1 殭屍版，由 color 產生），baseline 入口不需執行。
- BlurGuard 對外套件：使用者桌面 `BlurGuard_package.zip`（官方 e1b2d5e 副本＋以 diffusers 載入 VAE 的替代入口；VAE 權重與官方 `sd-v1-4.ckpt` 逐元素相同）。

## 5. 已知問題與待決

- BlurGuard 依論文協定重跑（ImageNet-Edit 80 張、SD v1.4 img2img、官方設定）未對上論文：未淨化編輯結果 LPIPS（AlexNet）0.228 對論文 0.36，
  SSIM 0.757 對 0.65，PSNR 29.65 對 28.5，IA 0.932 對 0.89；防禦圖對原圖 LPIPS 0.086 對 0.21。重產的未防禦編輯與官方隨附者幾乎相同
  （LPIPS 0.005），差距在防禦圖：學到的 σ 中位 3.21（0.71–7.43），擾動被重度模糊。原因未查明；可用官方 `blurguard.py`＋原始 checkpoint 對照。
- 論文正文與 BlurGuard 的實際做法不一致（`docs/paper/sec/4_experiments.tex` 第 14 行寫「each method is run at its native budget」與「budget rescaled」）；
  實際上 BlurGuard 無原生預算點，等失真版本為 `sigma_weighting` 0＋求解後乘 0.6071。Fig. 4 的 DIA-R 方形與圓點重疊（eps_scale 0.969）。論文修改屬 paper session。
- BlurGuard 倍率 0.6071 在 Pexels 上的防禦圖 LPIPS 為 0.2745，未達 0.3344；是否在 Pexels 上重新校準待使用者決定。
- deferred 的 UltraEdit／FLUX.1 Kontext 工作與 Pexels 淨化強度掃描（約 100 GPU 小時）是否執行待使用者決定。
- 遠端可清理的中間產物清單已交 executor，待使用者確認後才移到 `~/trash/baseline/`。
