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

- 執行方式：遠端佇列。兩台主機的 crontab 每 30 分鐘執行 `scripts/run_pexels_supervisor.sh`，依 `runtime/pexels_portraits/queue/jobs.txt`
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

### Pexels 的論文工作（使用者 2026-10-29 定案：論文圖表全部改用 Pexels，以影像組推進，每完成一批交 paper，每天向 executor 回報完成數/總數）

使用者 2026-10-31 指示：全部 GPU 只跑 P0，P1（CPU）照跑；P2、P3 暫停，下一步待主表完成後由使用者決定。P2、P3 的佇列行移到遠端 `runtime/pexels_portraits/queue/jobs_deferred_p2_p3.txt`（恢復時併回 `jobs.txt`）。兩者的程式尚未實際驗證第一步，恢復時先確認；`sweep_images_0097_0106_undefended` 與 `cross_flux_mist_textural_0` 曾在起步階段被終止。

| 優先 | 內容 | 狀態與位置 |
|---|---|---|
| P0 | 主表：等失真 9 臂（8 臂＋`photoguard_encoder`）＋`color3d_pexels`；欄為防禦圖 LPIPS、防禦圖美學（LAION、NIMA、MUSIQ、NIQE）、編輯結果 LPIPS（淨化前、七道平均、最差一道、VAE-Trans、保留率）、SigLIP（七道、VAE-Trans）、ArcFace（七道） | 遠端佇列：`aligned_photoguard_encoder_<組>`、`aligned_<組>`、`followup_<組>_{undefended,color3d,aligned}`、`cpu_<組>`；原生 10 臂的 followup 不做。2 卡下預估 11/5–11/6 完成 |
| P1 | 擾動存續比 LPIPS(P(x̃),P(x))/LPIPS(x̃,x)，9 臂＋`color3d_pexels`，七道＋VAE-Trans（CPU） | `scripts/measure_pexels_perturbation_survival.sh`；佇列 `survival_color3d_seven`、`survival_<組>_{seven,vae_trans}`；輸出 `runtime/pexels_portraits/followup/<根>/results/<aligned｜color3d>/survival_*.csv`、`followup/results/color3d_all/` |
| P2 | 淨化強度掃描：306 張，`color3d_pexels`＋`dct_shield`、`mist`、`dia_pt`（等失真）＋未防禦分母；13 設定中 `blur1`、`blur2`、`jpeg30`、`jpeg50` 重用七道編輯，新增 9 個（`blur0.5`、`blur3`、`jpeg10`、`jpeg70`、`jpeg90`、`jpeg95`、`resample_roundtrip128/256/384`） | `scripts/sweep_pexels_purifier_strength.sh`；佇列 `sweep_<組>_<條件>`（115 項），排在 P0 之後；約 180 GPU 小時。portraits 版見 `results/sweeps/ip2p/purifier_strength/`、`apply_purifier_sweep` |
| P3 | 跨編輯器 baseline 列（`photoguard_encoder`、`dct_shield`、`dct_shield_y`、`mist_textural`）：UltraEdit 306 張全協定（未淨化、七道、VAE-Trans）；FLUX 只跑 color trial `pexels_subsets/flux.txt` 的影像（一行一張影像名，與 Ours 同一份名單） | `scripts/evaluate_pexels_cross_model.sh`；佇列 `cross_sd3_*`（`HOST=basic-1`）、`cross_flux_*`（`HOST=basic-2`），排在 P2 之後。協定見 `results/cross_model/README.md`。未防禦分母在 color trial `artifacts_pexels/cross_model/`：FLUX 由 color 完成；UltraEdit 的 7 組由 color 完成，其餘由佇列沿用 color 的工具與完成標記補齊 |
| — | Fig. 3 散布點：原生 10 臂與等失真臂的淨化前、七道平均（編輯結果 LPIPS 對防禦圖 LPIPS），在數據齊的影像組上整理給 paper | 不需 GPU；原生讀 `runtime/pexels_portraits/<根>/results/`，等失真讀 `runtime/pexels_portraits/aligned/<根>/results/` |

- CLIP-IQA／FID／KID 在 9 臂補齊 306 張且有 VAE-Trans 組後重新合併（需加入 `photoguard_encoder` 與 VAE-Trans 組），通知 paper 一次，附 `n_images`，KID 為負值的組在表中加欄標註。
- 306 張的原生合併表以 `scripts/generate_pexels_result_tables.py` 重新產生後入庫。
- `native_images_0307_0308` 的佇列完成標記為手動補上（log 於 2026-10-29 16:40:49 有 `ALL SETS DONE`，worker 因腳本檔 stale handle 在寫標記前結束，見 `queue/worker.log`）。
  `aligned_images_0000_0096` 的 worker 有同樣情形，結束後須核對 log 與完成標記、確認沒有重複寫入。
- 遠端 `runtime/` 的刪除候選（等使用者確認）：`semantic_identity/{superseded_cpu,timing,merged,merged_cpu,merged_fidelity}`、`aligned_recalibration/discarded`、
  `superseded_scripts`、`cross_model/{superseded,*.patch,generate_defenses.py}`、`pexels_portraits/superseded`、各處的 `.py` 副本、`blurguard/`（讀數保留與否由使用者決定）。
  `distribution_metrics/` 的特徵檔保留，供重新合併使用。
- 改到正在執行的遠端腳本時，先建立 `queue/PAUSED` 並等工作結束再 `git pull`，或另寫新檔：此 NFS 上 `git pull` 會就地改寫檔案（inode 不變）。

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

## 規矩

- 引用外部方法的數字依 `docs/reference/BASELINE_PROVENANCE.md` 規則 1（連同該方法的協定與來源版本）。
- 共通規則見根目錄 `CLAUDE.md`。
