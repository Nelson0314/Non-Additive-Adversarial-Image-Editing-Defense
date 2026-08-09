#!/usr/bin/env bash
# 逐影像分片：用多張卡把段 1–3 平行跑完，**不改任何 Python 程式**。
#
#   bash scripts/shard.sh calib   <影像1> <影像2> ...     # 段 0，一次，單卡
#   bash scripts/shard.sh fanout  <影像1> <影像2> ...     # 段 1–3，每張圖一卡
#       環境變數 `STAGES` 可只跑其中幾段，預設 "train rayscale eval"。
#   bash scripts/shard.sh watch                           # 看各分片進度
#   bash scripts/shard.sh merge   <影像1> <影像2> ...     # 合併分片並跑段 4
#
# 批次由環境變數 `BATCH` 指定，它同時選定模型設定（見下方 profile 區）：
#
#   BATCH=b3  → SDXL 1.0 base / 1024² / bf16      （預設，img2img）
#   BATCH=v14 → SD v1.4 / 512² / fp32              （img2img，strength 0.6）
#   BATCH=v14r→ 同上但 strength 0.4                （2026-08-08 的重做）
#   BATCH=ip* → SD inpainting / 512² / fp32        （**inpainting**，無 strength）
#   BATCH=s3a*→ 第三階段批次 A：site apa + Lo 式 (5)，τ_train **0.50**
#   BATCH=s3b*→ 第三階段批次 B：同上但 τ_train **0.20**（等使用者指示才跑）
#
# fanout 產生的 tmux session 名為 `wacv-<批次>-<影像>`。
#
# **本腳本以自己的位置推導 repo 根目錄**（`WACV_ROOT`），不再寫死
# `$HOME/WACV`——遠端只有一個工作目錄而兩個 session 共用它，見該變數的說明。
#
# ---------------------------------------------------------------------------
# 為什麼這樣可行（三個前提，改動任一個之前先確認它們還成立）
#
# 1. `ProgressWriter._acquire()` 的鎖是**逐批次目錄**的，不是全域的。
#    不同批次目錄＝不同寫入者，完全合法。該鎖存在的理由是「同一批資料
#    不可有兩個寫入者」，分片後每個寫入者動的是不相交的資料。
#
# 2. 校準表可跨分片共用：`Calibration.REQUIRED_CONTEXT` 是
#    (model, resolution, guidance, steps, gpu, precision)，**不含 image**。
#    `calibrate_lr` 本來就只探測 `next(iter(res.images.values()))` 一張圖，
#    也就是說學習率在設計上與影像無關。故段 0 只跑一次，把
#    `calib/calibration.json` 複製進每個分片目錄即可。
#
#    **這一點是整個作法的關鍵。** 若改成每個分片各跑自己的段 0，各分片會
#    得到不同的學習率，而那是一個無法歸因的變因——跨影像的比較就毀了。
#
# 3. 合併無衝突：產物在 `batch_dir/<條件>/<影像>/`，逐格紀錄的檔名是
#    `<段>__<條件>__<影像>[__tau…].json`，兩者都帶影像。`run_report` 只讀
#    `_cells/*.json` 且只寫相對路徑的 `<img src>`，故把各分片的檔案疊進
#    同一個目錄再跑段 4，得到的就是完整的 compare.html。
#
# ---------------------------------------------------------------------------
# 分片的粒度就是影像數：N 張圖用 N 張卡，牆鐘時間 ≈ 總時數 / N。
# 影像是唯一有 CLI 入口的軸，條件與 τ 都沒有，故不要試圖再細分。
set -euo pipefail

MODE=${1:?用法見檔頭}
shift || true

# 這個 repo 的根目錄，由本腳本自己的位置推導。
#
# 2026-08-09 由 `$HOME/WACV` 改為推導（三處 `cd`，含兩處 tmux 指令內部）。
# before 的症狀在第三階段才會出現，但它一出現就是最貴的一種：**遠端只有
# 一個工作目錄，而兩個 session 共用它**。第三階段要跑的是
# `claude/stage3-apa-attn`，ip3 那個 session 跑的是
# `claude/e20-fidelity-constraint`；在 `$HOME/WACV` 裡切分支會把對方正在
# 用的程式碼換掉，而已經載入記憶體的 python 行程不受影響、**下一個起來的
# 才會用到新碼**——那正是「跑起來很正常、數字也合理」的那一類失敗
# （`RESULTS_2026-08-08` §9.8 已經記錄過同一型的干擾一次）。
#
# 推導之後，第三階段在 `$HOME/WACV-s3`（git worktree）裡跑，對方的工作
# 目錄完全不動。`WACV_ROOT` 可覆寫供特殊情形使用。
WACV_ROOT=${WACV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# shellcheck disable=SC1090
source "$HOME/env.sh"

# **`env.sh` 的最後一行是 `cd $HOME/WACV`**，故 source 之後工作目錄會被換到
# 那裡——而在有第二個 worktree 的情況下，那是**另一個 session 的樹**。
#
# 2026-08-09 實測到的後果：`merge` 分支呼叫 `$PY scripts/run_stage.py` 時
# 沒有自己的 `cd`（tmux 那幾條有），於是跑到 `$HOME/WACV/scripts/run_stage.py`
# ——ip3 那條分支的版本，不認得本批的 `--attn-mask-tau`，段 4 以
# 「unrecognized arguments」中止。段 1–3 沒中招純粹是因為它們的 tmux 指令
# 各自明寫了 `cd $WACV_ROOT`。
#
# 危險的是它**只在旗標剛好不存在時才報錯**：若兩條分支的 CLI 介面相容，
# 就會安靜地用另一棵樹的程式碼跑完，而 `sys.path` 由 `run_stage.py` 自己
# 的位置決定（`sys.path.insert(0, parents[1])`），故連 `src` 都會跟著換。
# 那正是本專案記錄過的「跑起來很正常、數字也合理」那一型。
#
# 修在源頭而不是逐處補 `cd`：一行涵蓋全部分支，且新增分支不會漏。
cd "$WACV_ROOT"

BATCH=${BATCH:-b1}
# `fanout` 每個分片依序跑的段。預設值即原本寫死的三段，故既有命令列的行為
# 逐字不變。
#
# 2026-08-08 由寫死改為可覆寫。理由：`HANDOVER_2026-08-08` §8.3 要求 v14r 在
# **段 2 之後停下來**，先用人眼看 τ=0.20 的 `x_def` 再決定要不要跑段 3——
# 放寬 `tau_acut` 買到的是訓練自由度，代價可能是同一 LPIPS 下可見失真變差，
# 而段 3 要 7 小時。寫死三段時唯一的做法是在段 2 結束後去 kill session，
# 那會落在 `eval` 已經開始之後，且 log 尾端不會有 `[exit`，監控無法分辨
# 「人為中止」與「掛掉」。
#
#   STAGES="train rayscale" BATCH=v14r bash scripts/shard.sh fanout …   # 先停
#   STAGES="eval"           BATCH=v14r bash scripts/shard.sh fanout …   # 看完再續
#
# 續跑是安全的：`run_stage` 依逐格紀錄跳過已完成的格，段 1／2 的 `_cells`
# 已在，第二次只會跑 `eval`。
STAGES=${STAGES:-"train rayscale eval"}
GPU_TAG=RTX-3090
PRECISION=bf16
# 模型設定的預設值＝b1／b2／b3 那一組（SDXL 1.0 base，程式預設值），
# 故此處留空以維持原本的命令列逐字不變。v14 的覆寫在本檔末的 profile 區。
MODEL=""
# `--purify-mode rotate --attn-timesteps 2` 不是調味，是 24 GB 下的必要條件。
# N1（`optimize._build_attn_step`）不能開 UNet checkpoint——hook 在 backward
# 重算時已卸除，兩次存檔的張量數對不上——故每步要留住
# `attn_timesteps × 淨化算子數` 個完整的 SDXL UNet 計算圖，1024² 下每個約
# 4.5 GB。2026-08-06 於 RTX 3090（23.56 GB）逐一實測：
#
#   all   + t=4（預設，12 圖）  OOM
#   rotate + t=4（4 圖）        OOM
#   rotate + t=3（3 圖）        峰值 24124 MiB → OOM
#   rotate + t=2（2 圖）        峰值 22964 MiB → 通過
#
# 兩者都必須帶：rotate 下前向數等於 attn_timesteps，缺 rotate 就是 6 圖。
# 改用顯存更大的卡時可以調回，但 `gpu` 進 config_hash，那本來就是新批次。
MEM="--purify-mode rotate --attn-timesteps 2"

# `--warp-max-disp` 由段 0 的 `warp_reach` 決定，不是預設值。程式預設 1.5,
# 而 2026-08-06 於 RTX 3090、1024²、grid_size=32 實測的可達 LPIPS 為：
#
#   max_disp    bird_03   cat_02   dog_03   三者最小
#      1.5       0.1091   0.2182   0.1544    0.109   ← 程式預設，蓋不住
#      3.0       0.2170   0.3846   0.2905    0.217
#      5.0       0.3099   0.4946   0.3918    0.310
#      8.0       0.3914   0.5579   0.4641    0.391   ← 採用
#     12.0       0.4480   0.5928   0.5111    0.448
#     20.0       0.4845   0.6164   0.5446    0.485
#
# 判準是「`TAUS` 中最大的那個點必須可達」，即 0.35——**不是** `--tau-train`。
# 段 2 要把 φ 縮放到每一個 τ 並照實報告，任一點拋出主表就有空格。1.5 下三張
# 圖有兩張連 0.20 都到不了。取 8.0 是蓋得住且仍在曲線膝部的最小格點；
# 再往上收益遞減（12→20 只多 0.037）。
#
# 2026-08-06 修正這段說明：原文寫「--tau-train 是 0.35 故最小值必須 ≥ 0.35」，
# 而訓練點當日已改為 0.20（見 grid.TRAIN_TAU）。結論不變但理由變了——
# 8.0 是為了讓 τ=0.35 這一**報告**點可達，與訓練點無關。
#
# 附帶記錄：τ=0.20 與 0.35 量到的 disp_max 都是 8√2 = 11.31 px，即位移場
# 從主表那一點起就已頂死本上界。放寬它不會讓防禦白白變強（實際失真仍由
# 段 2 固定在 τ 上），但會改變飽和的程度，故它進 config_hash。
#
# 這個量測用的是 **R**（高斯隨機位移）——而 R 本身就是格點裡的一個條件，
# 故它必須自己達得到 τ=0.35，不能只看最佳化後的 N1／N2。先驗實測「同振幅
# 下最佳化解的可辨失真是隨機的 2–3 倍」，所以對 N1／N2 而言 8.0 是寬鬆的。
#
# `max_disp` 是硬上界（`site_warp.displacement` 直接 clamp），不是損失權重：
# 放寬它不會讓防禦白白變強，實際失真仍由段 2 的射線縮放固定在 τ 上。
REACH="--warp-max-disp 8.0"

# N1 的 decoy 位置。程式預設 0（BOS，忠於 PromptFlare 的 L_CA 原形式），
# 但那在 SDXL 上拿不到注意力：2026-08-06 實測 70 層 attn2 的平均 token 質量
# 為 7.2e-06（空 prompt）／5.6e-06（"a cat"），等於零，`1 − mass` 坐在最大值
# 上、梯度為零。原因是 BOS 的嵌入 L2 範數 1193 而其餘 token 只有 24–37——
# CLIP 的 massive-activation 現象，BOS 是暫存槽不是被 attend 的對象。
# 末位 PAD（第 76 格）的質量是 1.59e-02／4.84e-03，索引固定且不承載語意，
# 同樣 prompt-free。`optimize.MIN_SHARED_MASS` 會在第 0 步檢查並拋出。
ATTN="--shared-tokens 76"

# 走生成路徑的條件（N3／site apa）的反演設定。**這兩項不可省。**
#
# `generator.py` 的 t_max 註解寫明「此參數必須由呼叫端依 E0c 的量測結果
# 指定，不可沿用預設值」，而 `run_stage.py` 的預設是 None（走滿 [0,999]）、
# 段 0 的 `run_calibration` 又不產生它。b1 因此以 None + DDIM 執行，
# G(x;0) 的 LPIPS 是 0.7325／0.6736／0.6917——φ=0 時產出的已經是另一張圖，
# 保真預算在防禦起作用之前就被吃光，N3 的階段二第 24 步即被判為收斂。
#
# 2026-08-06 於 RTX 3090、SDXL、1024²、bf16、k_inv=10 實測（三張圖）：
#
#   DDIM   + t_max 走滿   0.7325 / 0.6736 / 0.6917  ← b1 用的
#   DDIM   + t_max 200    0.1395 / 0.2846 / 0.1400
#   BDIA   + t_max 走滿   0.1237 / 0.2270 / 0.1017
#   BDIA   + t_max 200    0.1092 / 0.2208 / 0.0976  ← 採用
#   （純 VAE 來回下限     0.0986 / 0.2130 / 0.0889）
#
# 取 BDIA + 200 使 G(x;0) 落在純 VAE 下限上方 0.008–0.010，即生成路徑本身
# 幾乎不再額外收費。k_inv 維持 10：開了 BDIA 之後 20 反而較差（0.1393）。
#
# 註：那 0.008–0.010 是 bf16 的累積誤差。fp32 下 BDIA 精確到與純 VAE 來回
# 逐位相同，`--t-max` 屆時是無關參數（SD v1.4/512²/fp32 實測，
# 見 RESULTS_2026-08-06 §8.1）。
INV="--exact-inversion --t-max 200"

# 產物寫在**版控範圍之外**，預設 `~/wacv_runs`。
#
# 2026-08-06 實測到的事故：`runs/b1*` 依「runs/ 是唯一證據來源」的規定提交
# 進版控之後，機器上的一次 `git pull` 把**實驗正在寫的目錄刪掉了**。機制是
# 機器的 sparse-checkout 為 `/*` 加 `!/runs/`，於是那些路徑一旦成為被追蹤的
# 檔案就會被標成 skip-worktree（`git ls-files -v` 顯示 `S`）並從工作區移除。
#
# 當時只有與已提交版本**逐位元不同**的檔案（該次新產生的 calibration.json、
# lr_probe.csv）因為 git 不敢刪而倖存，其餘全部消失。也就是說這個機制專門
# 刪掉「已經回收過的結果」，而那正是最容易被誤判為安全的一類。
#
# 把輸出移出 repo 之後，git 完全看不到實驗產物，pull 與實驗不再互相干擾。
# 回收流程不變：從 $RUNS 打包拉回本機的 `runs/` 再入版控。
RUNS=${WACV_RUNS:-$HOME/wacv_runs}

# 學習率探測的步數。程式預設 12，而 2026-08-06 實測 12 步對位移場**沒有
# 鑑別力**：N2 的五個候選末端損失全距 0.9%（0.7324–0.7414），小於「當步輪到
# 哪個淨化算子」單獨造成的 9% 落差，`_pick_best` 的 argmin 等於抽籤。
# 拉到 60 步後三個候選單調分開（0.7191 / 0.7037 / 0.6943），且 disp_max
# 由 0.068 px 分到 3.046 px。詳見 RESULTS_2026-08-06 §2。
PROBE="--probe-steps 60"

# ---------------------------------------------------------------------------
# 批次 profile
#
# `DESIGN_2026-08-05` §2.0a（使用者 2026-08-06 定案）並行兩組實驗，兩組的
# 模型／解析度／精度不同，而上面那些常數是為 b3（SDXL）寫的。此處以 BATCH
# 名稱選 profile，**不改上面任何一行**，使 b1／b2／b3 的命令列逐字不變。
#
# 為什麼必須逐字不變：`Calibration.REQUIRED_CONTEXT` 是
# (model, resolution, guidance, steps, gpu, precision)，段 1 拿段 0 的
# calibration.json 時會比對這六項。差一項就是 `CalibrationMismatch`，
# 而段 0 要兩小時。
#
# v14 的四項差異與其理由（`HANDOVER_2026-08-07` §4）：
#
#   --wrapper sd --resolution 512 --precision fp32
#       Mist 與 DIA 的原生模型，且為指導者協定所指定。
#   --purify-mode all（不帶 --attn-timesteps）
#       512² 下 N1 的計算圖只有 1024² 的四分之一，24 GB 沒有壓力，
#       故不必用 rotate + t=2 這個為 SDXL 省記憶體的設定。
#   --shared-tokens 0
#       BOS，PromptFlare 原作的選擇。SDXL 上 BOS 拿不到注意力質量
#       （實測 7.2e-06）才改用第 76 格的 PAD；SD v1.4 上 BOS 實測質量
#       93.9%，原形式成立。
#   不帶 --t-max
#       fp32 下 BDIA 反演與純 VAE 來回逐位相同，該參數無關
#       （RESULTS_2026-08-06 §8.1）。帶上去只會多一項 config 差異。
#
#   ip*：inpainting 威脅模型（使用者 2026-08-07 定案的下一批）。
#       權重取 `runwayml/stable-diffusion-inpainting`——PhotoGuard-c、
#       AdvPaint、PromptFlare 三篇原作共同指定的那一份，換過去之後它們
#       回到原生的 9 通道形態。
#       **不帶 `--strength`**：inpainting 沒有這個參數，pipeline 由純噪聲
#       起跑並跑滿自己的排程，`SDWrapper.edit` 收到它會拋出。五篇 baseline
#       的原始碼裡也都沒有這個數（`photoguard.py:124` 等三處各自拒絕預設值）。
#       `--mask-mode` 是切換威脅模型的旗標，同時決定 `base_config` 多不多
#       一個 `mask` 鍵；沒有它就是 img2img。
MASK=""
# 攻擊方的 strength。留空即沿用 `run_stage` 的預設 0.6，故 b1/b2/b3/v14 的
# 命令列逐字不變。
#
# `v14r*`（2026-08-08 的重做）取 **0.4**：strength 掃描顯示攻擊的區間在 0.5
# 就飽和而防禦效果在 0.4 之後崩掉，0.4 是唯一同時滿足「攻擊確實有效」與
# 「防禦仍有著力點」的點（`RESULTS_2026-08-07` §6c）。0.6 是段 0 的
# `calibrate_strength` 取「編輯效果最大」選出的，那由建構上就落在防禦最
# 無力的一端。
#
# **inpainting（ip*）不可帶**：那個威脅模型沒有 strength，`run_stage` 會在
# `--mask-mode` 下拒絕並中止。
STRENGTH=""
# 本批要跑哪些條件，以及失真預算軸。留空即 `run_stage` 的預設（全部條件、
# τ_train=0.20），故 b1/b2/b3/v14/v14r/ip3 的命令列逐字不變。
GRID=""
case "$BATCH" in
  s3a*|s3b*)
    # ---------------------------------------------------------------------
    # 第三階段（2026-08-09）。模型設定與 v14r **逐字相同**——換掉的只有
    # 條件、防禦目標與訓練點，威脅模型不動。
    #
    # 為什麼 strength 沿用 0.4 而不是回到 0.6：0.3／0.4／0.5／0.6 已全部量過，
    # 沒有任何一點產生可與雜訊區分的正效果（`RESULTS_2026-08-08` §7.3、§7.4），
    # 使用者裁決「下一輪不要再調 strength」。沿用 v14r 的 0.4 使本批與它在
    # 同一個工作點上，兩批的加性 baseline 因此可以直接對照。
    #
    # 五個條件：`N4`（apa + Lo 式 5）、`Ra`（apa 上的同失真隨機對照）、
    # 三個加性 baseline。位移場的 N1／N2／R 移出格點但原始碼保留——依據見
    # `docs/DECISION_stage3.md`：v14r 實測 N1 對 R 的 `edit_lpips` 比值 1.046、
    # 語意失敗 4/15 對 4/15，即訓練對位移場幾乎沒有貢獻。
    #
    # `--attn-mask-tau 0.5` 是式 (4) 的遮罩門檻，作用在峰值正規化後的尺度上。
    # 論文（Lo et al., CVPR 2024）未給值，本專案選定並記錄；它進
    # `config_hash`，且**只在給定時出現**，故 img2img 既有批次的雜湊逐位不變
    # （`tests/test_executors.py::test_img2img既有批次的config_hash逐位不變`）。
    # 0.5 沿用 `linf_attack` 那條文獻基準路徑已經在用的值，使兩條路徑的遮罩
    # 定義一致；覆蓋率兩端的警告由 `optimize._build_attn_step` 印出。
    #
    # `--purify-mode all` 沿用 v14r：512² 下 N4 的計算圖是 1024² 的四分之一。
    # **但 N4 與 N1 一樣不能開 UNet checkpoint**（hook 在 backward 重算時已
    # 卸除，兩次存檔的張量數對不上），且它走生成路徑、圖比 N1 更長，故這是
    # 本批記憶體最緊的一格。段 0 的乾跑若 OOM，先降 `--attn-timesteps`。
    PRECISION=fp32
    MODEL="--model CompVis/stable-diffusion-v1-4 --wrapper sd --resolution 512"
    # `--attn-timesteps 2`（不是預設的 4）是**記憶體實測的結果**，不是調味。
    #
    # N4 的注意力前向恆不能 checkpoint（hook 與 checkpoint 重算不相容），故每步
    # 要同時留住 `attn_timesteps × 淨化算子數` 份完整的 UNet 計算圖；
    # `--purify-mode all` 下淨化算子是 3 個，t=4 就是 12 份。
    # 2026-08-09 於 RTX 3090（24576 MiB）、SD v1.4、512²、fp32 實測段 0：
    #
    #   attn_timesteps=4   峰值 23924 MiB（97.3%，只剩 652 MiB）  跑得完但貼著上限
    #   attn_timesteps=2   峰值 15126 MiB（61.5%，餘裕 9450 MiB） 採用
    #
    # 兩者都跑得完，取 2 是因為 652 MiB 的餘裕撐不住段 1 的 250 步——碎片化
    # 累積之後在第 200 步 OOM 會賠掉數小時，而那正是本專案最貴的一種失敗。
    # 代價是注意力目標在 [0, t_edit] 上只取兩個 timestep 而非四個；t=2 在本
    # 專案有前例（SDXL 的 b3 因同一個記憶體理由用 `rotate + t=2`）。
    #
    # **它在 `loss_params` 因而進 `config_hash`**，故這是一個被記錄下來的批次
    # 選擇，不是隱形的預設值。要拿回 t=4 的正解是把 `_build_attn_step` 改成
    # 逐 pair 反傳再累加梯度（DAYN 自己的 Algorithm 1 就是那樣做的，
    # 見 `linf_attack.pgd_linf`），那會讓峰值與 t 無關；本輪不做該改動。
    MEM="--purify-mode all --attn-timesteps 2"
    ATTN="--shared-tokens 0 --attn-mask-tau 0.5"
    INV="--exact-inversion"
    STRENGTH="--strength 0.4"
    GRID="--conditions N4 Ra photoguard_c mist dia_r"
    ;;&
  s3a*)
    # 批次 A：τ_train = 0.50，加性方法普遍所在的量級。
    #
    # 依據是本專案重現 DAYN 時實測其 `pert_lpips ≈ 0.51`（L∞ = 0.06，n=480），
    # 可由 `git show fc23d2278:runs/lo_baseline/summary.csv` 取回。
    #
    # **四個 τ 常數一致地跟著批次走**（`grid.tau_plan_for`）：0.50 進 TAUS、
    # 成為 MAIN_TAU 與 TRAIN_TAU、並落在 FULL_PURIFY_TAUS 內。只改訓練點而
    # 不動其餘三個的話，`purifiers_for` 對不在 FULL_PURIFY_TAUS 的 τ 只回傳
    # identity，**訓練點上一個淨化格都不會有**——而抗淨化是主張一。
    #
    # 兩道 hinge 的門檻不寫在這裡：`run_stage` 由 `--tau-train` 依比例導出
    # （0.8×τ = 0.40、16×τ = 8.0）並印在 log 的 `[thresholds]` 行。
    GRID="$GRID --tau-train 0.50"
    ;;
  s3b*)
    # 批次 B：τ_train = 0.20，使用者判讀的人眼上限。**等使用者指示才跑。**
    #
    # 0.20 是 `grid.TRAIN_TAU` 的預設值，故 `tau_plan_for` 會回傳模組常數
    # 本身，完整淨化組落在 (0.20, 0.35) —— 與 v14／v14r 同一組報告點。
    GRID="$GRID --tau-train 0.20"
    ;;
  v14r*)
    # 重做：模型設定與 v14 完全相同，只改 strength。門檻不必在此指定——
    # `run_stage` 會由 `--tau-train` 依比例導出（0.8×τ、16×τ）並印在 log 上。
    PRECISION=fp32
    MODEL="--model CompVis/stable-diffusion-v1-4 --wrapper sd --resolution 512"
    MEM="--purify-mode all"
    ATTN="--shared-tokens 0"
    INV="--exact-inversion"
    STRENGTH="--strength 0.4"
    ;;
  v14*)
    PRECISION=fp32
    MODEL="--model CompVis/stable-diffusion-v1-4 --wrapper sd --resolution 512"
    MEM="--purify-mode all"
    ATTN="--shared-tokens 0"
    INV="--exact-inversion"
    ;;
  ip*)
    PRECISION=fp32
    MODEL="--model runwayml/stable-diffusion-inpainting --wrapper sd_inpaint --resolution 512"
    MEM="--purify-mode all"
    ATTN="--shared-tokens 0"
    INV="--exact-inversion"
    #       `--warp-mask-gate`（2026-08-08，使用者裁決的處置 B）：位移場在
    #       遮罩內歸零。遮罩內是攻擊方會整片覆寫的區域，我方在那裡的擾動
    #       **零防禦價值、全額保真成本**，而 PhotoGuard-c 與 PromptFlare 的
    #       原始碼本來就把梯度乘 (1 − mask)。不加閘等於先丟掉「涵蓋率」比例
    #       的預算才開始比較。它進 `module_params` 因而進 `config_hash`。
    #       套用範圍是 site warp 的三個條件（N1／N2／R）；N3 走生成路徑，
    #       其擾動經 VAE 解碼後本來就不是逐像素定域的，無法以同一方式加閘。
    #
    #       兩道 hinge 的門檻（`--tau-acut`／`--tau-chroma`）不寫在這裡：
    #       2026-08-08 起 `run_stage` 在未明給時由 `--tau-train` 依比例導出
    #       （0.8 × τ、16 × τ），τ_train=0.20 下為 0.16 與 3.2，並印在 log
    #       的 `[thresholds]` 行。明給只會多一項與規則不同步的風險。
    #
    #       `--mask-mode attention_box --mask-tau 0.5` 維持不變，**但影像必須
    #       另選**（2026-08-08）。整段查證如下，中途一度改成 `attention` 又改
    #       回來，兩次的證據都留著。
    #
    #       起因：ip1 段 0 實跑量到的涵蓋率是 bird_03 0.483／cat_02 0.314／
    #       **dog_03 0.875**，第三張超過 `HANDOVER` §3.2a 的 0.6 停止線。疊圖
    #       確認框的**對位正確**——問題不在框錯，而在那隻狗本身佔滿畫面，
    #       外接矩形因此逼近全圖。0.875 等於攻擊方重畫八分之七的畫面，防禦只
    #       剩八分之一的脈絡，那正是第一階段的失效模式換一扇門走回來。
    #
    #       中途的處置與它為什麼不行：改用 `attention`（輪廓，不取外接矩形）
    #       後三張圖降到 0.058／0.143／0.360，數字合格，**但疊圖不合格**——
    #       那不是物件剪影而是破碎斑塊，bird_03 只蓋到背與頭的一部分，dog_03
    #       還蓋到左上與右上的**背景**。攻擊方不會選那種區域，遮罩必須是一塊
    #       連通且含住物件的區域（`runs/ip2_maskprobe2/`）。
    #
    #       定案：維持矩形框，改以**規則**選圖——涵蓋率落在 [0.15, 0.45]，
    #       每個類別取一張。遮罩由一次加噪前向決定，故涵蓋率**依賴 seed**
    #       （dog_02 在 seed 0 與 7 下量到 0.4475 與 0.8750，差一倍），
    #       判準因此在批次實際使用的 `--seed 20260805` 上量
    #       （`runs/ip2_maskprobe3/coverage_seed20260805.csv`，全 24 張）：
    #
    #         在窗內：bird_02 0.401、cat_01 0.234、cat_02 0.314、
    #                 horse_00 0.205、horse_01 0.234、horse_02 0.275
    #         人像 8 張全部 0.92–1.00（物件佔滿畫面），整組不可用
    #
    #       每類取一張，並以疊圖判定框是否含住物件：bird 只有一張合格；
    #       cat 取 cat_02（第一階段用過同一張，可對照）；horse 取 horse_02
    #       ——horse_00 的框只蓋到畫面左半、馬在右緣外，horse_01 的框切掉馬頭。
    #
    #       故本批的影像是 **bird_02 cat_02 horse_02**，不是第一階段那三張。
    MASK="--mask-mode attention_box --mask-tau 0.5 --warp-mask-gate"
    ;;
esac

COMMON="--runs-root $RUNS --gpu-tag $GPU_TAG --precision $PRECISION $MODEL --mist-target data/targets/MIST.png $MEM $REACH $ATTN $INV $PROBE $MASK $STRENGTH $GRID"

shard_dir() { echo "$RUNS/${BATCH}_$1"; }

# 空閒的卡，記憶體用量由低到高。排除他人正在用的。
#
# `WACV_GPUS` 明給要用哪幾張（空白分隔的 index），給定時**完全跳過偵測**。
#
# 2026-08-09 新增。理由是「保留」與「起跑」之間的空窗：這台機器有其他使用者，
# 段 0 只吃一張卡而段 1–3 要三張，中間隔著一到兩小時。若在段 0 期間放著另外
# 兩張不管，等 fanout 要用時很可能已經被別人拿走，而 fanout 的失敗訊息是
# 「空閒卡 N 張，少於影像 3 張」——那時段 0 的機時已經花掉了。
#
# 佔卡的作法是在那些卡上放一個佔用行程，於是它們在 `nvidia-smi` 上就是忙的，
# **自動偵測會把自己佔的卡也排除掉**。故必須有一條明給的路徑。
#
# 明給時不做任何檢查：呼叫端既然指名了，就由呼叫端負責那些卡是可用的。
# 自動偵測那條路徑的行為逐字不變，既有批次的命令列不受影響。
free_gpus() {
  if [ -n "${WACV_GPUS:-}" ]; then
    printf '%s\n' $WACV_GPUS
    return
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1}'
}

case "$MODE" in

calib)
  # `--dry-run` 走前景、不開 tmux、不佔卡。2026-08-08 補入：
  # `HANDOVER_2026-08-08` §3.2 的第二步逐字寫了這個用法，而在此之前
  # `--dry-run` 會被當成一個影像 id 塞進 `--images`，argparse 於是把它解析
  # 成旗標、把真正的三個 id 當成多餘的位置參數而拒絕。錯誤訊息看不出原因。
  DRY=""
  if [ "${1:-}" = "--dry-run" ]; then DRY="--dry-run"; shift; fi
  IMAGES=("$@")
  [ ${#IMAGES[@]} -gt 0 ] || { echo "要給影像 id" >&2; exit 1; }
  if [ -n "$DRY" ]; then
    cd "$WACV_ROOT"
    PYTHONIOENCODING=utf-8 $PY scripts/run_stage.py calib --batch "$BATCH" \
      $COMMON --images "${IMAGES[@]}" --dry-run
    exit $?
  fi
  GPU=$(free_gpus | head -1)
  # 批次目錄必須先存在：tmux 指令裡的 `> …/calib.log` 在目錄不存在時
  # 直接失敗，而那個失敗發生在 tmux 內，外面只看得到「session 沒了」。
  mkdir -p "$RUNS/$BATCH"
  echo "段 0 在 GPU $GPU 上跑，影像：${IMAGES[*]}"
  # session 名帶批次。2026-08-08 改，before：固定的 `wacv-calib`。
  #
  # 那個名字使**兩個批次的段 0 不能並存**，而 fanout 早就是帶批次的
  # （`wacv-$BATCH-$IMG`，理由見該處註解），兩者不一致。2026-08-08 實測撞上：
  # v14r 的段 0 佔著這個名字，ip2 的 `tmux new-session` 以 `duplicate session`
  # 失敗。`set -e` 確實在該處中止（實測 rc=1，其後的 `echo` 沒有執行），故這
  # **不是**一個靜默失敗；危險的是那行訊息只說「名字重複」，看不出「是另一個
  # 批次佔著」，而呼叫端若把輸出接進 pipeline 就連 rc 都收不到。
  #
  # 若當時 `set -e` 沒有生效，接下來的 `tmux has-session` 會拿別的批次的
  # session 判定為成功——那才會是 §7 第 3 項那種型態。名字帶批次之後兩條路
  # 都不成立。
  SESS="wacv-$BATCH-calib"
  tmux new-session -d -s "$SESS" \
    "cd $WACV_ROOT && PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=$GPU \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/run_stage.py calib --batch $BATCH $COMMON --images ${IMAGES[*]} \
> $RUNS/$BATCH/calib.log 2>&1; echo \"[exit \$?]\" >> $RUNS/$BATCH/calib.log"
  # 起不來要當場知道。先前漏了 mkdir，症狀是 session 靜默消失、log 不存在，
  # 而輪詢的一方只會看到「還在跑」——那是最貴的一種失敗。
  sleep 5
  tmux has-session -t "$SESS" 2>/dev/null || {
    echo "$SESS 沒有起來。log：" >&2
    cat "$RUNS/$BATCH/calib.log" 2>/dev/null | tail -20 >&2
    exit 1; }
  echo "log: $RUNS/$BATCH/calib.log"
  ;;

fanout)
  IMAGES=("$@")
  [ -f "$RUNS/$BATCH/calib/calibration.json" ] || {
    echo "找不到 $RUNS/$BATCH/calib/calibration.json——段 0 還沒跑完" >&2; exit 1; }
  mapfile -t GPUS < <(free_gpus)
  [ ${#GPUS[@]} -ge ${#IMAGES[@]} ] || {
    echo "空閒卡 ${#GPUS[@]} 張，少於影像 ${#IMAGES[@]} 張" >&2; exit 1; }

  for i in "${!IMAGES[@]}"; do
    IMG=${IMAGES[$i]}; GPU=${GPUS[$i]}; D=$(shard_dir "$IMG")
    mkdir -p "$D/calib"
    # 共用同一份校準表：這是不可省的一步，理由見檔頭第 2 點。
    # 寫成 `cp -r <src>/. <dst>/` 而非 `cp -r <src> <dst>/`：後者在 <dst>/calib
    # 已存在時會複製成 `<dst>/calib/calib`。分片中斷後重跑 fanout 續跑是
    # 正常操作（`run_stage` 會跳過已完成的格），故這條路徑必須可重入。
    cp -r "$RUNS/$BATCH/calib/." "$D/calib/"
    LOG="$D/run.log"
    echo "分片 $IMG → GPU $GPU   $LOG"
    # session 名帶批次：兩組實驗的影像 id 相同，不帶批次就會在
    # `tmux new-session` 撞名，而第二組的失敗只表現為「session 沒有起來」。
    #
    # 退出碼必須自己記。原本寫的是 `… || break; done …; echo "[exit $?]"`，
    # 而以 `break` 結束的迴圈其狀態是 `break` 自己的 0，於是**失敗也印
    # `[exit 0]`**。2026-08-07 兩次實測都被這一點誤導：b3_bird_03 的 train
    # 有 2 格 OOM 失敗、v14_cat_02 的 rayscale 有 1 格失敗，兩者的 log 尾端
    # 都是 `[exit 0]`，而 `watch_remote.sh` 依 `[exit` 判定收工，於是把失敗
    # 回報成「收工」——最貴的一種誤報。
    #
    # `$?` 也不能寫在 `if ! cmd` 之後：`!` 會把狀態反相，取到的是 0。
    tmux new-session -d -s "wacv-$BATCH-$IMG" \
      "cd $WACV_ROOT && export PYTHONIOENCODING=utf-8 \
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
RC=0; \
for S in $STAGES; do \
  echo \"===== \$S =====\"; \
  $PY scripts/run_stage.py \$S --batch ${BATCH}_$IMG $COMMON --images $IMG; \
  RC=\$?; \
  if [ \$RC -ne 0 ]; then echo \"[fail-stage \$S rc=\$RC]\"; break; fi; \
done > $LOG 2>&1; echo \"[exit \$RC]\" >> $LOG"
  done
  sleep 5
  for IMG in "${IMAGES[@]}"; do
    tmux has-session -t "wacv-$BATCH-$IMG" 2>/dev/null || {
      echo "分片 $IMG 沒有起來。log：" >&2
      tail -20 "$(shard_dir "$IMG")/run.log" 2>/dev/null >&2
      exit 1; }
  done
  tmux ls
  ;;

watch)
  for D in "$RUNS/${BATCH}_"*; do
    [ -d "$D" ] || continue
    printf '%-28s ' "$(basename "$D")"
    $PY scripts/dashboard.py "$D" --json 2>/dev/null \
      | $PY -c 'import json,sys
d=json.load(sys.stdin)["summary"]
print(f"done={d[\"done\"]:5d} failed={d[\"failed\"]:3d} skipped={d[\"skipped\"]:4d} / {d[\"total\"]:5d}")' \
      || echo "(尚無進度)"
  done
  ;;

merge)
  IMAGES=("$@")
  OUT=$RUNS/${BATCH}_merged
  rm -rf "$OUT"; mkdir -p "$OUT/_cells"
  cp -r "$RUNS/$BATCH/calib" "$OUT/"
  for IMG in "${IMAGES[@]}"; do
    D=$(shard_dir "$IMG")
    cp "$D/_cells/"*.json "$OUT/_cells/"
    # 產物樹：<條件>/<影像>/…，逐影像互斥故直接疊上去
    for C in "$D"/*/; do
      B=$(basename "$C")
      case "$B" in _cells|calib) continue;; esac
      mkdir -p "$OUT/$B"; cp -r "$C"/* "$OUT/$B/" 2>/dev/null || true
    done
  done
  echo "合併完成：$(ls "$OUT/_cells" | wc -l) 格"
  PYTHONIOENCODING=utf-8 $PY scripts/run_stage.py report \
    --batch "${BATCH}_merged" $COMMON
  echo "compare.html → $OUT/compare.html"
  ;;

*) echo "未知模式 $MODE" >&2; exit 1;;
esac
