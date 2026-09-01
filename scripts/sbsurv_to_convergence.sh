#!/usr/bin/env bash
# 把 `sb_surv` 跑到早停真的觸發為止。
#
# 為什麼前兩次都停在上限
# ────────────────────────────────────────────────────────────────────
# `sb_surv` 跑了 6000 步、續跑再 6000 步，兩次都是 `stop_reason = max_steps`。
# 但看曲線它其實已經平了——`runs/ip2p_sbsurv_long/sb_surv_long/trace.csv` 的
# 最後六次固定評估是
#
#     11699   0.00162  0.00159  0.00158  0.00159  0.00159  0.00160
#     6205    0.00166  0.00160  0.00160  0.00164  0.00162  0.00164
#
# 也就是在第三位小數上來回震盪，沒有趨勢。**早停沒觸發是判準本身的問題，
# 不是模型還在學。**
#
# 病灶：`--min-delta 0.0002`（相對 0.02%）**遠低於評估自身的噪聲**。上面兩組
# 的相對標準差是 0.86% 與 1.49%，是 min_delta 的 43 倍與 75 倍。於是每隔幾次
# 評估就會有一次純粹因為抽樣而落到歷史最佳以下，`since_best` 被重置，
# `patience` 永遠數不滿。
#
# 兩個改動，都對準這一件事
# ────────────────────────────────────────────────────────────────────
#   `--eval-draws 16`   由 8 加倍。固定抽樣的噪聲按 1/sqrt(n) 降，預期由
#                       0.9–1.5% 降到 0.6–1.1%。評估每 400 步一次，成本可忽略。
#   `--min-delta 0.01`  由 0.02% 提到 1%，也就是**要求改善大於評估噪聲本身**。
#                       低於噪聲的「改善」不可與抽樣運氣區分，不該重置計數器。
#
# **這是收斂判準不是效果判準**（`CLAUDE.md` 的「訓練方法的實驗：不設判準」允許
# 前者、禁止後者）：它只決定「什麼時候停」，不決定「這個方法有沒有效」。
#
# `--patience 6` 配 `--eval-every 400` 表示連續 2400 步沒有超過 1% 的改善才停。
#
# 起點是已經跑完的 12000 步（6000 ＋ 續跑 6000），由 `--resume-weights` 載入。
# 上限再給 8000 步；**停在上限仍要照實報成未收斂**。
#
# 用法：bash scripts/sbsurv_to_convergence.sh "<卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SRC=runs/ip2p_sbsurv_long/sb_surv_long
ls "$SRC"/*__w.pt >/dev/null 2>&1 || {
  echo "錯誤：$SRC 沒有 __w.pt，沒有起點可續" >&2; exit 2; }

OUT=runs/ip2p_sbsurv_converged
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"

CUDA_VISIBLE_DEVICES="${DEVS[0]}" setsid nohup "$PY" scripts/ip2p_run.py \
    --out "$OUT/sb_surv_conv" \
    --data data/omniedit150 --conditions phase_gain --quantile 0 \
    --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
    --loss image_guidance --ig-zt diffuse_src --radius 2.5 \
    --spectral-floor 0.08 --floor-survival blur12 \
    --purify-aware eot_broad --eot-sigmas 0.5 1.0 2.0 3.0 \
    --resume-weights "$SRC" \
    --steps 8000 --step-size 0.01 \
    --eval-every 400 --eval-draws 16 --patience 6 --min-delta 0.01 \
    --save-weights --skip-existing \
    --images $IMGS < /dev/null >> "$OUT/sb_surv_conv.log" 2>&1 &
disown
echo "[conv] sb_surv_conv dev=${DEVS[0]}  上限 8000 步，早停 patience 6 x 400"
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
