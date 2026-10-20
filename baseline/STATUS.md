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

主讀數是 `displacement.csv` 的 `disp_lpips_full`（編輯結果 LPIPS：未防禦編輯 vs 防禦後編輯，
另有主體／背景分區）。主表的位移排名與防禦圖失真排名幾乎同序（原生防禦圖 LPIPS 0.043–0.666），
等失真臂是為了把「防禦有效」與「擾動較大」分開。各組協定不同，引用數字時連協定一起引用。

## 現況

上表除 Pexels 人像外皆已有完整結果表。

### Pexels 人像

資料組 306 張（男 108、女 198），十個條件（`diffvax`、`photoguard_encoder`、`mist`、`dia_pt`、`sifm`、`dia_r`、`dct_shield_y`、
`dct_shield`、`dayn`、`danp`），只跑 ip2p，以影像組為單位由兩個 worker 各佔一張卡執行。
`results/pexels_portraits/` 為 186 張（`images_0000_0096` 至 `images_0177_0186`）的合併表；其餘影像組的進度以遠端
`runtime/pexels_portraits/` 的完成標記與 `worker_<名>.lock/owner` 為準。資料選取、讀數表、暫停與恢復的操作見 `docs/PEXELS_PORTRAITS.md`。

- GPU：color 有佇列時優先取卡，baseline 用剩下的，全局合計 ≤ 6；讓卡與恢復用 `scripts/run_pause_for_color_queue.sh`（見 `docs/PEXELS_PORTRAITS.md`「操作」）。
- 資料組的目標張數為 500；現行選取規則下來源只得到 306 張，補足方式（放寬規則、增加來源或只增加女性影像）未定。
- 資料選取的程式與候選快取在 `trials/unsplash_portrait_pool/`（本機與遠端，不入版控），內容見該目錄的 `README.md`。
- 語意與身分讀數：`results/pexels_portraits/semantic_identity_<條件>.csv`（含 `undefended`）為 186 張；
  主表、等失真臂、UltraEdit、FLUX 見 `README.md`「結果表」。產生程式 `measure_semantic_identity`（CUDA；
  與 CPU 計算相比 CLIP／SigLIP 方向差 ≤ 1e-5、ArcFace 逐位元相同、FaceNet 最大差 0.028）。
- 等失真臂擴至 Pexels：`scripts/evaluate_pexels_aligned.sh`，八條件（`mist`、`dia_pt`、`sifm`、`dia_r`、`dct_shield_y`、
  `dct_shield`、`dayn`、`danp`）沿用 `results/aligned/defense_<條件>.csv` 的 `eps_scale`；`photoguard_c`、`photoguard_linf`、
  `photoguard_encoder` 與 `diffvax` 不在此臂。產物與狀態在 `artifacts/pexels_portraits/aligned/`、`runtime/pexels_portraits/aligned/`。
- FLUX 七道淨化：`run_flux_edits --purifier`，13 arm × 7 算子 × 32 格，輸出 `results/flux/purified_edits/<算子>/edits_<arm>.csv`。
- 文獻的評估集規模見 `docs/reference/EVALUATION_SET_SIZES.md`；完整版三張表的規劃與 GPU 時間見 `docs/PEXELS_PORTRAITS.md`「GPU 時間」。

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
- 跨模型移植：`generate_defenses --vae-model` 只接受損失只經過 VAE 編碼器的條件（`photoguard_encoder`、`dct_shield`、`dct_shield_y`）。以 SD3（UltraEdit）與 FLUX.1-Kontext 的 VAE 求解並做未淨化編輯，產物 `artifacts/cross_model/<sd3|flux>/`，狀態與讀數 `runtime/cross_model/`；淨化後編輯尚未排程。`mist` 只有 textural 項可移植，其餘條件需要去噪器或反演，不移植。
- Pexels 長跑與 Pexels 等失真臂在論文實驗期間暫停；恢復方式見 `docs/PEXELS_PORTRAITS.md`「操作」（等失真臂為 `scripts/run_pexels_aligned_worker.sh`）。

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
