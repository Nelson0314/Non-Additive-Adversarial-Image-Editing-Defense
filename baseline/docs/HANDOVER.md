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
- 遠端 `~/image-immunization/baseline` 以 `git pull` 同步（push 由 executor 執行）；影像產物在 `artifacts/`，執行狀態在 `runtime/`。
  停工作：以 `queue/claims/<工作>/owner` 的 PID 執行 `kill -TERM -- -<PID>`，再以 `pgrep -g <PID>` 確認 0 個行程、租約已釋放。
  crontab 的修改需使用者同意（備份在兩台 `~/crontab.backup.*`）。
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
| SD inpainting（7 條件 × 4 淨化，306 張） | `results/pexels_portraits/inpaint/` | `runtime/pexels_portraits/inpaint/readouts/` |
| BlurGuard 防禦圖 PSNR、SSIM、NIMA、NIQE | `results/pexels_portraits/blurguard/defense_quality.csv` | — |
| 非 SD 編輯器測速 | `results/pexels_portraits/nonsd_editor_speed/` | — |

- 讀數定義與區間：`scripts/generate_pexels_main_table_summary.py` 開頭。主讀數為編輯結果 LPIPS（未防禦編輯對防禦後編輯，同淨化）；
  七道欄為 7 個逐道平均的平均；另有 SSIM／PSNR、保留率、SigLIP、CLIP-I、CLIP-S、ArcFace、存續比、FID／KID、防禦圖 LPIPS 與美學。
- 淨化：七道 `blur1`、`blur2`、`crop_resize0.1`、`jpeg30`、`jpeg50`、`jpeg80`、`rotate15`（`immunization_core.purifiers.protocol`）與 VAE-Trans
  （color trial `artifacts/purification/vae_trans/model/encoder.pt`）。
- FID／KID／CLIP-IQA 的特徵以 GPU 計算（`measure_distribution_metrics.py --device cuda`），與 CPU 版的最大 FID 差 0.001，排名不變。
- 跨編輯器的防禦圖與淨化後輸入一律取 ip2p 流程的產物，不以替代編輯器重新求解。
- data/portraits 8 張的主表、等失真臂、FLUX／UltraEdit 全表與跨模型移植見 `README.md`「結果表」與 `results/` 各子目錄的 README。

### 3.2 等失真設定

各方法的原生 ε 乘 `eps_scale`，使 data/portraits 8 張的平均防禦圖 LPIPS 等於錨點 0.3344（`piq.LPIPS`，交付 PNG）。錨點是 color 早期顏色方法（舊顏色列）
在這 8 張的值，不是 ImmuLUT；倍率原封套用到 Pexels 306 張，306 張上各臂的防禦圖 LPIPS 為 0.251–0.318（ImmuLUT 0.287）。倍率在 `results/aligned/defense_<條件>.csv`：

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

### 3.3 論文圖表的出處（依 LaTeX 標籤；編號隨版面變動，以標籤為準）

| 標籤（檔案） | 內容 | 數據或圖的來源 |
|---|---|---|
| `tab:main`（`sec/3_method.tex`） | 306 張 ip2p 主表 | `results/pexels_portraits/{main_table,color3d_scale2p5,blurguard}/`；paper 以 `docs/paper_tools/generate_main_table.py` 讀取 |
| `fig:plane`（`fig/distortion_displacement.pdf`） | 失真—位移平面 | 遠端 `~/paper_figures/in_pexels/`（`points_blurguard.csv`、`condition_summary_blurguard.csv`、`color_summary_scale2p5.csv`）；方形為原生預算（`runtime/pexels_portraits/interim/fig3_points.csv`），圓點為 10 個等失真臂；paper 以 `docs/paper_tools/generate_paper_figures.py` 產生 |
| `fig:purification_example`（`fig/purification_example.pdf`） | woman_111 p1 police suit，blur σ2 | `scripts/generate_purification_example_figure.py`，遠端 `artifacts/pexels_portraits/purification_example/woman_111_p1.pdf` |
| `fig:purification`（`fig/purification.pdf`） | 逐淨化分析圖 | `docs/paper_tools/generate_paper_figures.py`（paper 維護） |
| `tab:isr` | ip2p 條件式 ISR | 根目錄 `docs/isr_pilot/`（Claude Sonnet 判定＋人工複核 606 格；素材 `artifacts/pexels_portraits/ip2p_isr100_sheets/`） |
| `tab:transfer` | VIBE-DistilledCFG、FLUX.2 klein 4B 跨編輯器 | `results/pexels_portraits/{vibe_distilled,klein_distilled}/` |
| `tab:supp_inpainting`（`sec/X_suppl.tex`） | SD inpainting 讀數 | `results/pexels_portraits/inpaint/summary.csv`（見 3.4） |
| `tab:supp_inpaint_isr` | SD inpainting ISR（1,224 張，Sonnet 判定、無人工複核） | 根目錄 `docs/isr_pilot/inpaint/`（executor 產生）；素材與解盲表見 3.4 |
| `tab:supp_aesthetics` 的 BlurGuard 列 | 防禦圖 PSNR、SSIM、NIMA、NIQE | `results/pexels_portraits/blurguard/defense_quality.csv` |
| 補充資料 ip2p 質性圖 | woman_23 helmet（編輯種子 3）、man_07 helmet（主種子） | `artifacts/pexels_portraits/purification_example/{woman_23_p2_seed3,man_07_p2}.pdf`；woman_23 的種子 3 編輯在 `artifacts/teaser/woman_23_helmet_seed3_relabeled/` |
| 補充資料 inpainting 質性圖 10 張（`fig/inpaint_<影像>_<指令>.pdf`） | man_26 dog、woman_14 hospital、woman_138／man_51 hospital、woman_39／man_07 gym、woman_146／woman_87 snowy street、woman_27／man_81 dog | `scripts/generate_inpaint_example_figure.py`；遠端 `artifacts/pexels_portraits/inpaint_example/`（man_26、woman_14）與 `inpaint_example_candidates/`（其餘 8 張，同名 `.csv` 列出所用檔案） |
| Fig. 1 teaser（`fig/teaser.pdf`） | man_x 殭屍，JPEG 30 | `scripts/generate_teaser_flow_figure.py`（呼叫 color trial 的版面），run-dir 遠端 `artifacts/teaser/teaser_man_x_zombieskin_tg7p5_ig1p8`，輸出 `artifacts/teaser_paper_format/` |

- 各方法原生預算、等失真倍率與實作來源：`docs/reference/BASELINE_PROVENANCE.md` 與 `results/aligned/defense_<條件>.csv` 的 `eps_native`、`eps_scale`、`spec_source` 欄。
- 圖內用詞為 Undefended、Immunized（與正文一致），由 `immunization_baseline.figure_text.relabel` 或本專案的繪圖函式產生；color 的程式未修改。

### 3.4 SD inpainting（補充資料）

- 協定：`runwayml/stable-diffusion-inpainting`（SD 1.5 inpainting），fp32、strength 1.0、50 步、guidance 7.5、種子 20260812、512²；遮罩
  `data/pexels_portraits/masks/`（CLIPSeg `CIDAS/clipseg-rd64-refined` 主體分割，白＝重繪＝主體的補集），輸出合成回遮罩外原輸入；
  指令 `prompts.yaml` 的 `edits.inpaint` 四句（hospital、gym、snowy street、dog）。
- 防禦圖與淨化後輸入沿用 ip2p 流程，不重新求解：SD1.4（各 baseline 的求解目標）、SD1.5 inpainting 與 ip2p 的 VAE 逐張量相同（最大差 0）。
- 306 張 × 7 條件（未防禦、等失真 PhotoGuard、DIA-PT、DCT-Shield、SIFM、BlurGuard、ImmuLUT）× none／jpeg50／blur2／VAE-Trans × 4 句，全部完成。
- 結果正本：`results/pexels_portraits/inpaint/`（`summary.csv`、`images.txt`、`README.md`）；逐格讀數在遠端 `runtime/pexels_portraits/inpaint/readouts/`。
- ISR 素材：遠端 `artifacts/pexels_portraits/inpaint_isr_sheets/`（1,224 張拼圖，列序以 `default_rng(20260812)` 逐張打亂；`cells.csv`、
  解盲表 `key.csv`），由 `scripts/generate_inpaint_isr_sheets.py` 產生；評審規則草稿 `docs/INPAINT_ISR_RUBRIC.md`。
- 原生預算對照（僅 man_26 p3，論文未用）：遠端 `artifacts/pexels_portraits/inpaint_native_man_26/readouts.csv`。

### 3.5 暫時性嘗試（`trials/`，不入版控）

| trial | 內容 | 狀態 |
|---|---|---|
| `unsplash_portrait_pool` | Pexels 資料組的選取程式與候選快取 | 保留（資料組來源） |

已刪除的 trial 與其數字見 `docs/TRIALS.md`。

### 3.6 遠端執行狀態

- 佇列 `runtime/pexels_portraits/queue/jobs.txt`：無待執行工作，`failed/` 為空；`baseline_cap` 為 2（color 的要求），`PAUSED` 不存在。
  兩台 crontab 各有一行，每 5 分鐘執行 `scripts/run_pexels_supervisor.sh`，無工作時不取卡。暫停新認領：`touch queue/PAUSED`。
- 常駐行程：baseline 沒有在跑的工作，也沒有租約（兩台 `pgrep` 只見 color 的行程）。
- 未執行的 UltraEdit／FLUX.1 Kontext 工作：`queue/jobs_deferred_manual.txt`、`jobs_deferred_transfer_rest.txt`、`jobs_deferred_flux_512_rest.txt`
  （使用已刪除腳本的工作項已移除，改前清單 `*.before_cleanup`）。
- 卡數規則：全局 ≤ 6，扣除其他 session 的租約，color 優先；color 要卡時設 `PAUSED` 並把 `baseline_cap` 改為 0，讓在跑的影像跑完。

## 4. 工具與路徑索引

- 遠端版面：`artifacts/pexels_portraits/{aligned,sets}/`（等失真臂防禦圖與編輯；`aligned/` 為第一個影像組，`sets/<影像組>/` 為其餘）、
  `defenses/`、`defended_edits/`、`purified_edits/`（原生預算）、`vae_trans/`、`vibe_distilled/`、`klein/`、`inpaint/`；`runtime/pexels_portraits/{summary,
  followup,queue,blurguard,blurguard_rest,blurguard_306,vibe_distilled,klein_distilled,crosseditor100,isr_second100,isr_third106,inpaint}/`。
  影像組定義 `data/pexels_portraits/image_sets.json`；BlurGuard 第二批的產物在 `runtime/pexels_portraits/blurguard_rest/`。
- 求解：`immunization_baseline.cli.generate_defenses`（`--eps-scale`、`--vae-model`）、`immunization_baseline.cli.generate_blurguard_defenses`、
  `scripts/apply_blurguard_scale.py`。
- 主表管線：`scripts/evaluate_pexels_aligned*.sh`、`evaluate_pexels_followup.sh`、`measure_pexels_summary_inputs.sh`、`generate_pexels_main_table_summary.py`。
- 跨編輯器：`scripts/evaluate_pexels_nonsd_editor.sh`（`run_nonsd_edits.py`）、`evaluate_pexels_flux_512_readouts.sh`、`run_pexels_readout_parts.sh`、
  `generate_pexels_editor_{cell_tables,summary}.py`；inpainting：`evaluate_pexels_inpaint_editor.sh`（`run_inpaint_edits.py`）、
  `evaluate_pexels_inpaint_readouts.sh`、`generate_pexels_inpaint_summary.py`、`generate_inpaint_isr_sheets.py`。
- 圖：`generate_purification_example_figure.py`（ip2p 質性圖；`--purification jpeg50|blur2|vae_trans`、`--extra` ImmuLUT 變體欄）、
  `generate_inpaint_example_figure.py`、`generate_teaser_flow_figure.py`；字級以 0.62 倍正文寬度置入時為 7 pt。
- ISR 拼圖：遠端 `runtime/pexels_portraits/{crosseditor100,isr_second100,isr_third106}/generate_*.py`（ip2p 三批共 306 張）、`scripts/generate_inpaint_isr_sheets.py`。
- 分布指標：`scripts/measure_distribution_metrics.py`（`--device cpu|cuda`）。
- BlurGuard 對外套件：使用者桌面 `BlurGuard_package.zip`（官方 e1b2d5e 副本＋以 diffusers 載入 VAE 的替代入口）。

## 5. 已知問題

- BlurGuard 依論文協定重跑（ImageNet-Edit 80 張、SD v1.4 img2img、官方設定）未對上論文數字；數字見 `docs/TRIALS.md` 的
  `blurguard_paper_protocol` 列，原因未查明。
- BlurGuard 倍率 0.6071 在 Pexels 上的防禦圖 LPIPS 為 0.2745，未達錨點 0.3344。
- 幾何淨化的分區欄：`results/retention.csv` 中 11 個條件的 `crop_resize0.1`、`rotate15` 分區欄已以 `purified_mask()` 重算（更正前見 commit `228c59b`）。
- SD 2.x 只能用 epsilon-prediction 權重（`sd2-community/stable-diffusion-2-1-base`）。
- `results/distortion_displacement/points.csv` 的產生程式已刪除（引用的 color 條件產物已不存在）；檔案保留為紀錄。
- 遠端 `runtime/pexels_portraits/vibe_distilled/readouts_parts.sh` 與 `fid_cuda/replace_with_cuda.sh` 仍引用改名前的 `generate_pexels_flux_512_*`；重跑前需改名。
- ImmuLUT 的防禦圖與編輯有明顯色調，ISR 拼圖即使打亂列序也無法完全盲化。
- 遠端跑 `core/tests` 前需 `export ENV_FILE=~/env.sh; source baseline/scripts/env.sh`，否則 `test_generate_requirements_lock.py` 的 round-trip 測試找不到 uv。

## 6. 待使用者決定

1. BlurGuard 是否在 Pexels 上重新校準等失真倍率。
2. deferred 的 UltraEdit／FLUX.1 Kontext 工作是否執行。
3. 遠端可清理的暫存（刪除需使用者確認，確認後移到 `~/trash/baseline/`）：
   - `artifacts/pexels_portraits/inpaint_isr_sheets/` 中 inpainting ISR 100 張試評的 `pilot100.txt`、`pilot100_codes_partial.txt`、`score.py`
     （試評停在 68 張，已由 executor 的 1,224 張全評取代）；
   - `~/tmp_figs/`（83 MB，腳本副本、拼圖與測試紀錄）、`~/tmp_inpaint/`（28 KB）；
   - `artifacts/pexels_portraits/inpaint_example_candidates/` 中未入論文的 8 張候選，以及 `inpaint_native_man_26/`。
4. inpainting 的 pipeline 原始輸出（`artifacts/pexels_portraits/inpaint/**/__raw.png`，約 11 GB）依使用者指示保留。

逐段工作紀錄：`git log -- baseline/`；STATUS 的舊版逐段紀錄見 commit `05f785f0`。
