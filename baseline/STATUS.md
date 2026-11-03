# baseline：現況與接續指引

接手先讀這一份；協定、檔案位置、程式清單在 `README.md`；等失真臂的說明在 `results/aligned/README.md`。

## 資料組

| 資料 | 回答什麼 | 位置 |
|---|---|---|
| 主表 | 12 個方法在各自原生預算下，把 ip2p／inpaint 的編輯結果推開多少 | `results/*.csv` |
| 等失真臂 | 10 個方法縮到同一個防禦圖 LPIPS 之後，還推開多少 | `results/aligned/` |
| FLUX 全表 | 同一批防禦圖換成 FLUX.1-Kontext 編輯 | `results/flux/` |
| UltraEdit 全表 | 同一批防禦圖（含淨化後）換成 UltraEdit（SD3）編輯 | `results/ultraedit/` |
| Pexels 人像 | 外部條件在 306 張 Pexels 人像上的原生預算讀數（只 ip2p） | `results/pexels_portraits/`、`docs/PEXELS_PORTRAITS.md` |
| 跨模型移植 | encoder 類條件改以 SD3／FLUX 的 VAE 求解後由該編輯器編輯 | `results/cross_model/` |
| CLIP-IQA、FID、KID | 等失真臂與 color3d 的分布距離與 CLIP-IQA（portraits、Pexels） | `results/distribution_metrics/` |

主讀數是 `displacement.csv` 的 `disp_lpips_full`（編輯結果 LPIPS：未防禦編輯 vs 防禦後編輯，
另有主體／背景分區）。主表的位移排名與防禦圖失真排名幾乎同序（原生防禦圖 LPIPS 0.043–0.666），
等失真臂是為了把「防禦有效」與「擾動較大」分開。各組協定不同，引用數字時連協定一起引用。

## 現況

上表除 Pexels 人像外皆已有完整結果表。論文的 baseline 只列 9 個條件（不含 `photoguard_c`、`photoguard_linf`、`diffvax`），
其餘條件的既有數據保留。BlurGuard 不併入主表（使用者決定），讀數留在遠端 `runtime/blurguard/`。

### Pexels 人像

資料組 306 張（男 108、女 198），十個條件（`diffvax`、`photoguard_encoder`、`mist`、`dia_pt`、`sifm`、`dia_r`、`dct_shield_y`、
`dct_shield`、`dayn`、`danp`），只跑 ip2p，分為 23 個影像組。原生評測的進度以遠端各影像組的 `measured_<條件>.done`、
`additional_metrics.done` 為準；`results/pexels_portraits/` 為 186 張（`images_0000_0096` 至 `images_0177_0186`）的合併表，
全部影像組完成後以 `scripts/generate_pexels_result_tables.py` 重新產生。
資料選取、讀數表與操作見 `docs/PEXELS_PORTRAITS.md`。

- 執行方式：遠端佇列。兩台主機的 crontab 每 5 分鐘（2026-11-03 起；原為每 30 分鐘，工作結束後空出的卡與 CPU 槽要等下一輪才補）執行 `scripts/run_pexels_supervisor.sh`，依 `runtime/pexels_portraits/queue/jobs.txt`
  開 worker（`scripts/run_pexels_job.sh`），不依賴本機。工作清單含等失真臂各影像組（`aligned_<影像組>`）與後續量測
  （`followup_<影像組>_<分組>`、`cpu_<影像組>`）；格式、狀態與 color 取卡方式見 `docs/PEXELS_PORTRAITS.md`「操作」。
- GPU：全局合計 ≤ 6，color 優先；baseline 可用卡數 = 6 − max(非 baseline 租約數, `queue/color_reserve`)，由 supervisor 自動調整。
- 後續量測（原生 10、等失真 8、`color3d_pexels`）：VAE-Trans（含未防禦分母）、語意與身分、編輯結果美學，以及 `color3d_pexels`
  的防禦圖失真與美學。原生與等失真臂的防禦圖失真（交付 PNG、CPU，`fidelity.csv`）與美學已在各影像組的 `results/` 中。
  `color3d_pexels` 的影像取自 color trial `artifacts_pexels/`（只讀），讀數寫在 baseline。
- CLIP-IQA／FID／KID：`results/distribution_metrics/pexels_*.csv` 的等失真臂只含合併當時已完成的影像（42–166 張，同條件內各淨化的影像集合不同），
  `color3d` 為 306 張；等失真臂補齊 306 張且有 VAE-Trans 組後重新合併。
- 資料組的目標張數為 500；現行選取規則下來源只得到 306 張，補足方式（放寬規則、增加來源或只增加女性影像）未定。
- 資料選取的程式與候選快取在 `trials/unsplash_portrait_pool/`（本機與遠端，不入版控），內容見該目錄的 `README.md`。
- 語意與身分讀數：`results/pexels_portraits/semantic_identity_<條件>.csv`（含 `undefended`）為 186 張；
  主表、等失真臂、UltraEdit、FLUX 見 `README.md`「結果表」。產生程式 `measure_semantic_identity`（CUDA；
  與 CPU 計算相比 CLIP／SigLIP 方向差 ≤ 1e-5、ArcFace 逐位元相同、FaceNet 最大差 0.028）。
- 等失真臂擴至 Pexels：`scripts/evaluate_pexels_aligned.sh`，八條件（`mist`、`dia_pt`、`sifm`、`dia_r`、`dct_shield_y`、
  `dct_shield`、`dayn`、`danp`）沿用 `results/aligned/defense_<條件>.csv` 的 `eps_scale`；`photoguard_encoder`（eps_scale 0.2614，
  即 2/255，1000 步）由 `scripts/evaluate_pexels_aligned_condition.sh` 另跑（逐條件的完成標記 `chain_<條件>.done`、鎖
  `locks/<條件>/`，防禦圖失真與美學在 `results/<條件>/`）；`photoguard_c`、`photoguard_linf` 與 `diffvax` 不在此臂。
  產物與狀態在 `artifacts/pexels_portraits/aligned/`、`runtime/pexels_portraits/aligned/`。八臂的 `fidelity.csv` 若在
  photoguard_encoder 防禦生成期間量測，可能含部分 photoguard_encoder 列，合併時依條件過濾。
- FLUX 七道淨化：`run_flux_edits --purifier`，13 arm × 7 算子 × 32 格，輸出 `results/flux/purified_edits/<算子>/edits_<arm>.csv`。
- 文獻的評估集規模見 `docs/reference/EVALUATION_SET_SIZES.md`；完整版三張表的規劃與 GPU 時間見 `docs/PEXELS_PORTRAITS.md`「GPU 時間」。

### Pexels 的論文工作（使用者 2026-10-29 定案：論文圖表全部改用 Pexels，以影像組推進）

遠端佇列 `runtime/pexels_portraits/queue/jobs.txt` 的工作分類與狀態（以 `queue/done/` 的完成標記為準）：

| 項目 | 內容 | 程式與佇列工作 | 狀態 |
|---|---|---|---|
| 主表（不含 VAE-Trans） | 等失真 9 臂（8 臂＋`photoguard_encoder`）＋`color3d_pexels`，306 張 | `nv_*`（`scripts/evaluate_pexels_followup_semantic.sh`）、`nv_distribution`、`nv_summary` | 已完成（2026-11-03）；產物 `runtime/pexels_portraits/summary/` |
| VAE-Trans 那一半 | VAE-Trans 淨化與編輯、位移保留、語意、存續比，以及含 VAE-Trans 的 distribution 與彙整 | `followup_<組>_{undefended,color3d,aligned}`（`scripts/evaluate_pexels_followup.sh`）、`survival_<組>_vae_trans`、`cpu_<組>`、`arcface_pair_<組>`、`distribution_pexels`、`summary_pexels`（`scripts/measure_pexels_summary_inputs.sh`） | 執行中；`summary_pexels` 完成後覆寫 `summary/main_table.csv`（VAE-Trans 欄填入），送 executor 與 paper |
| 閘門 | 69 項 VAE-Trans GPU 工作完成 | `vae_trans_gpu_done`（CPU，`true`） | color 等它出現後在 `queue/color_reserve` 寫 1，取 1 卡補 FLUX Ours 的 VAE-Trans 40 格 |
| 遷移性實驗 | 等失真 `photoguard_encoder`、`dct_shield`、`dct_shield_y`、`mist` 的 ip2p 防禦圖直接送 FLUX（basic-2）與 UltraEdit（basic-1），全協定（淨化前、七道、VAE-Trans），每張 4 臂 × 36 格；兩編輯器皆 306 張 | `scripts/evaluate_pexels_transfer.sh`（UltraEdit；FLUX 的舊路徑）、`scripts/evaluate_pexels_flux_transfer.sh`（FLUX，每張只載入一次模型）；`transfer_<編輯器>_<影像>`，順序見下 | 排在閘門後，第一步未驗證；產物 `artifacts/pexels_portraits/transfer/<編輯器>/`，逐格表與完成標記 `runtime/pexels_portraits/transfer/<編輯器>/{chunks,done}/`；讀數（位移、保留、語意、CLIP-S、FID）尚未寫 |
| color 的遷移列 | 未防禦分母與 Ours (transfer)：UltraEdit 其餘 16 組（`images_0157_0166` 起）、FLUX 其餘 256 張 | `scripts/run_color_transfer_job.sh <sd3\|flux> <工作>` 呼叫 color trial 的 `tools/run_pexels_{ultraedit,cross_model}_job.sh`；佇列工作 `color_<sd3\|flux>_<工作>`，清單取自 color trial `runtime/cross_model_pexels/<sd3\|flux>/baseline_queue.txt`；完成標記與 log 在 color 的 `jobs/`、`logs/` | 排在第三段（sd3）與第四段（flux）；flux 另需 `queue/done/flux_single_load_accepted` |
| CLIP-S | cos(E_img(編輯), E_txt(實際送進編輯器的指令))，CLIP 與 SigLIP | `scripts/measure_clip_score.py`；`clip_score_<組>`（GPU） | 排在各組 followup 後，第一步未驗證；彙整程式尚未加 CLIP-S 欄 |
| P2 淨化強度掃描 | 306 張，`color3d_pexels`＋`dct_shield`、`mist`、`dia_pt` | `scripts/sweep_pexels_purifier_strength.sh` | 延後；佇列行在 `queue/jobs_deferred_p2_p3.txt`，第一步未驗證 |
| P3 跨編輯器（替代 VAE 重新求解） | `photoguard_encoder`、`dct_shield`、`dct_shield_y`、`mist_textural` | `scripts/evaluate_pexels_cross_model.sh` | 延後；同上 |

- 卡數（使用者 2026-11-04 指示）：baseline 合計上限 4 張，留 2 張給 color；由 `queue/baseline_cap`（現值 4）限制 supervisor 新開的 worker，
  執行中的不受影響。
- FLUX 改用 512×512 協定（使用者 2026-11-04 定案，資料版本 `flux_512`，說明見 `results/pexels_portraits/flux_512/README.md`）：佇列工作
  `flux_512_<影像>`（flux.txt 50 張，6 條件 × 36 格，`scripts/evaluate_pexels_flux_512.sh`），等 UltraEdit 50 張與 `queue/done/flux_512_accepted`。
  實測每格 21.3 s、每張約 1.34 GPU 小時；1024 版的 FLUX 工作移到 `queue/superseded_jobs_flux_1024.txt`，已有產物與標記不動。
  讀數（位移、保留率、SigLIP、CLIP-S、CLIP-I、ArcFace、編輯端 SSIM／PSNR、FID）由 baseline 統一量，尚未寫。
- 遷移實驗目前只跑兩編輯器各 50 張（使用者 2026-11-04 指示）：第三、四段（UltraEdit 與 FLUX 其餘 256 張、color 的 91 項、閘門
  `transfer_ultraedit_rest_done`）移到遠端 `queue/jobs_deferred_transfer_rest.txt`，完成標記未動；續跑時移回 `jobs.txt`。
- 遷移實驗的順序（使用者 2026-11-03 定案，`TRANSFER_DEADLINE` 取消）：UltraEdit 的 FLUX 子集 50 張（color trial `pexels_subsets/flux.txt`）→ FLUX 同 50 張
  → UltraEdit 其餘 256 張與 color 的 sd3 工作 → FLUX 其餘 256 張與 color 的 flux 工作；其餘 256 張依 `runtime/pexels_portraits/transfer/ultraedit_order.txt`
  第 51–306 行（影像組順序）。段與段之間由 CPU 閘門 `transfer_ultraedit_flux_subset_done`、`transfer_flux_subset_done`、`transfer_ultraedit_rest_done`
  銜接：前一段每項皆有 `done/` 或 `failed/` 標記才放行（失敗的影像不擋後段，回報時列出）。佇列清單的產生程式在遠端 `queue/generate_jobs_transfer_sequence.py`、
  `queue/add_color_transfer_jobs.py`，改動前的清單為 `queue/superseded_jobs_before_{transfer_sequence,color_transfer}.txt`。
- FLUX 單次載入：`run_flux_edit_jobs --jobs-file` 讀多行 `run_flux_edits` 參數，共用一次載入依序執行，各行的協定、輸出與續跑規則不變。
  與逐次載入的等價性檢查在遠端 `runtime/pexels_portraits/transfer/single_load_check/`（data/portraits、`mist`、`man_00`，未淨化與 `jpeg50`，
  basic-2 一卡）：單次載入、同卡逐次載入與既有 `artifacts/flux/` 的 12 格輸出逐像素相同（max|Δ| = 0），`id_orig`、`arcface_orig` 相同。
  據此寫入 `queue/done/flux_single_load_accepted`，306 個 `transfer_flux_*` 行改用 `evaluate_pexels_flux_transfer.sh`。
- 遷移的實測：UltraEdit 每張約 24 分鐘（4 條件 × {淨化前 4 格、VAE-Trans 4 格、七道 28 格}，含 12 次載入）；FLUX 單格 83 s、載入 99 s。
- 主表彙整的欄定義見 `scripts/generate_pexels_main_table_summary.py` 開頭；`disp_weakest` 為各方法七道中平均編輯結果 LPIPS 最低的一道（論文 Weakest），`disp_worst` 為逐格取七道最小值後平均。論文不用 CLIP-IQA 與防禦圖對原圖的 FID；CLIP-I＝`clip_pair`，`clip_dir` 保留在表內。
- 共用指標的通用輸入（color 專案直接呼叫）：`measure_clip_score.py --edits-csv`、`measure_distribution_metrics.py --pairs-csv`、`measure_pexels_arcface_pair.py --semantic-csv --output-csv`。
- 佇列機制：worker 為 `scripts/run_pexels_queue_worker.sh`（`AFTER=`、`HOST=`、`CPU`；結束碼 4 略過後繼續認領）；supervisor 由兩台 crontab 每 5 分鐘執行，清除本機擁有、行程已消失的等失真鎖，佇列第一項連續 4 次 not ready 時在 `supervisor.log` 寫 `[WARN]`。舊 worker `scripts/run_pexels_job.sh` 已不再使用。
- `aligned_images_0000_0096`、`native_images_0307_0308` 的佇列完成標記為手動補上（log 有 `ALL SETS DONE`，worker 因 stale handle 在寫標記前結束），已核對無重複寫入。
- 遠端 `runtime/` 的刪除候選：`semantic_identity/{superseded_cpu,timing,merged,merged_cpu,merged_fidelity}`、`aligned_recalibration/discarded`、`superseded_scripts`、
  `cross_model/{superseded,*.patch,generate_defenses.py}`、`pexels_portraits/superseded`、各處的 `.py` 副本；處理方式為 mv 到 `~/trash/baseline/<原相對路徑>`。
  `runtime/blurguard/` 保留（使用者待辦）；`distribution_metrics/` 與 `summary/distribution_metrics/` 的特徵檔保留，供重新合併使用。
- 改到正在執行的遠端腳本時，先建立 `queue/PAUSED` 並等工作結束再 `git pull`，或另寫新檔：此 NFS 上 `git pull` 會就地改寫檔案（inode 不變）。
  pull 前以 `ps -eo args | grep -o "scripts/[a-z_]*\.sh"` 對照 `git diff --name-only HEAD origin/main`；`.py` 在啟動時讀入，可改。
- 已知的單次失敗：他人行程佔顯存造成 OOM、HF 取模型連線失敗；worker 自動重跑，同一工作失敗 3 次才進 `queue/failed/`。
- 實測每格秒數（不含載入模型）：ip2p 11.4 s、UltraEdit 6.6 s、FLUX 83 s。遷移 4 臂 × 50 張的估算：UltraEdit 13.2、FLUX 166 GPU 小時。
  後續量測每 10 張約 3.0 卡時（`undefended` 0.44、`color3d` 0.69、`aligned` 1.87）。

待辦（依優先序）：

1. 閘門 `vae_trans_gpu_done` 出現後，驗證 `transfer_flux_<影像>`（basic-2）、`transfer_ultraedit_<影像>`（basic-1）與 `clip_score_<組>` 的第一步：
   log 的 `[STEP] 遷移 …` 與逐格輸出、`artifacts/pexels_portraits/transfer/<編輯器>/` 的第一張輸出；`clip_score` 的
   `[ALLDONE] … clip_score.csv（N 格）`，10 張的組 N＝3,200＋400＋640＋80（9 臂與未防禦 × 4 指令 × 8 淨化設定、9 臂與未防禦 × 4、color3d 兩組）。異常即停並回報 executor。
2. 2026-11-04 20:00 向 executor 回報完成張數快照（FLUX、UltraEdit 各／50 與／306）與 GPU 小時，排程不停：
   張數取 `runtime/pexels_portraits/transfer/<編輯器>/done/<影像>.done`，GPU 小時取 `queue/worker.log` 的 claim 與 done 時刻相減。
   等價性檢查通過後寫 `queue/done/flux_single_load_accepted`，並把尚未開始的 `transfer_flux_*` 行改為 `evaluate_pexels_flux_transfer.sh`。
3. `summary_pexels` 完成後（`summary/main_table.csv` 的 `*_vae_trans` 欄有值）送 executor 與 paper，附路徑與 n；
   再於 `scripts/generate_pexels_main_table_summary.py` 加 CLIP-S 欄（讀各組 `followup/<根>/results/clip_score.csv`，淨化前、七道、VAE-Trans 各一）並重跑彙整。
4. 遷移表的讀數：位移 `measure_flux_displacement`／`measure_edit_displacement`、保留 `measure_flux_retention`／`measure_purified_displacement`、
   語意 `measure_semantic_identity --edits-csv`、CLIP-S `measure_clip_score.py --edits-csv`、FID `measure_distribution_metrics.py --pairs-csv`。
   未防禦分母與 Ours (transfer) 由 color 量（FLUX：color trial `artifacts_pexels/cross_model/flux/{edits,purified_edits}/undefended/`；
   UltraEdit：`artifacts_pexels/cross_model/sd3/edits/{undefended_edits,purified_edits/undefended}/`），程式共用。
5. baseline/ 清理（executor 派工，排在遷移實驗之後）：只動 baseline/、不改行為；被 color trial、`docs/paper_tools`、簡報、佇列引用的路徑不動
   （改前 grep 整個 repo 與遠端 color trial 的 `tools/`）；測試在遠端跑（含 `core/tests/test_repository_naming.py`）；遠端產物 mv 到 `~/trash/baseline/`。
   完成後以 ≤ 8 行回報類別與數量、commit、trash 總量、刻意保留的項目與理由。
6. 待使用者決定：P2、P3 是否開跑。

### 論文實驗

以下讀數皆為 data/portraits 8 張、四句、主種子；防禦圖失真讀交付的 PNG、CPU，逐格編輯讀數用 GPU。

| 項目 | 讀數 | 程式 |
|---|---|---|
| 失真—位移平面（ip2p） | `results/distortion_displacement/points.csv`（38 點）、`fits.csv`（LPIPS／DISTS／PieAPP 三種橫軸） | `scripts/generate_distortion_displacement_plane.py` |
| 擾動存續比 LPIPS(P(x̃),P(x)) ÷ LPIPS(x̃,x) | `results/aligned/perturbation_survival.csv`（等失真 11 臂＋color 兩條件，七道＋VAE-Trans） | `measure_perturbation_survival` |
| 淨化強度掃描（ip2p） | `results/sweeps/ip2p/purifier_strength/retention.csv`（13 個淨化設定 × 5 條件） | `apply_purifier_sweep`、`pipelines.retention` |
| color 條件的 inpaint 全協定 | `results/color_conditions/<條件>/` | 遠端 `runtime/color_inpaint/` 的流程 |

- 等失真臂的 `mist`、`photoguard_linf`、`photoguard_encoder` 取 2/255（PNG 上擾動為整數灰階，見 `results/aligned/README.md`）。
- BlurGuard：`generate_blurguard_defenses` 直接呼叫官方 `pgd_freq`（第一階段編碼器以 diffusers 的 SD v1.4 VAE 代替 ldm checkpoint）。原生 8 張防禦圖 LPIPS 0.231；ε 乘 1.0／1.5／2.0／2.5 時為 0.231／0.232／0.217／0.216，ε 不約束擾動幅度，等失真版本的旋鈕待使用者決定。主表原生全鏈的產物在 `artifacts/{defenses,defended_edits,purified,purified_edits}/blurguard`，讀數在遠端 `runtime/blurguard/native/results/`（完成後併入主表）。
- 跨模型移植：`generate_defenses --vae-model` 只接受損失只經過 VAE 編碼器的條件（`photoguard_encoder`、`dct_shield`、`dct_shield_y`，另有只保留 textural 項的 `mist_textural`）。SD3（UltraEdit）與 FLUX.1-Kontext 兩者的未淨化、七道與 VAE-Trans 讀數在 `results/cross_model/<sd3|flux>/`，協定與 `mist_textural` 和原方法的差異見該目錄的 `README.md`；產物 `artifacts/cross_model/`。
- `scripts/generate_distortion_displacement_plane.py` 引用的 color 條件 `color3d_adaptive_set`、`color3d_0d4` 的產物已由 color 刪除，重跑前須指定新的條件清單。

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

**未跟著換的一處**：等失真臂的錨點仍是舊顏色列的 0.3344（使用者指示不重新對齊）。

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
- `requirements.lock` 未入庫；由遠端執行環境以 `vendor/scripts/generate_requirements_lock.py` 產生。
- HF 權重在各機的 `/var/cache/huggingface`，兩台不同：FLUX 只在 basic-2，UltraEdit 只在 basic-1，ip2p 與
  SD-inpainting 兩台都有。機器相關設定以 `ENV_FILE` 交給 `scripts/env.sh`。
- `measure_additional_metrics` 讀 `artifacts/` 的版面與 CSV 中的影像路徑；無 GPU 亦可執行（固定使用 CPU）。
- 連線：basic-1 `ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`、basic-2 `-p 10102`，NFS 共用，repo 在 `~/image-immunization`；
  Python 為 `export ENV_FILE=~/env.sh; source scripts/env.sh` 後的 `$PY`。basic-2 的 ssh 逾時時從 basic-1 查詢。
- 停工作：以 `queue/claims/<工作>/owner` 的 PID 執行 `kill -TERM -- -<PID>`，再以 `pgrep -g <PID>` 確認 0 個行程、租約已釋放、完成標記未被覆寫。
- 遠端產物不 rm，mv 到 `~/trash/baseline/<原相對路徑>`，事先列清單給使用者確認。crontab 的修改需使用者同意（備份在兩台 `~/crontab.backup.*`）。
- 本 session 的 commit 由 executor push；push 後在遠端 pull。

## 規矩

- 引用外部方法的數字依 `docs/reference/BASELINE_PROVENANCE.md` 規則 1（連同該方法的協定與來源版本）。
- 共通規則見根目錄 `CLAUDE.md`。
