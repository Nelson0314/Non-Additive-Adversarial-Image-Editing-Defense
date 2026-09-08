#!/usr/bin/env bash
# 極座標可分離內容場的掃描。**只動 `--patch-polar`，其餘與 `free` 臂逐項相同。**
#
# ██ 這一批尚未排進佇列 ██
# ────────────────────────────────────────────────────────────────────
# `scripts/band_and_purify_queue.sh` **沒有**引用這一支。要不要跑、跑哪幾個臂、
# 排在哪一段，由使用者決定。這裡先把它寫好，是為了遠端一恢復就能立刻送，
# 不必再花時間寫派工。
#
# 為什麼是這一批
# ────────────────────────────────────────────────────────────────────
# `runs/registration_ceiling/` 與 `runs/carrier_purify_response/` 兩份表收在
# 同一句話上：**內容場的粗細解決低通、解決不了未對位那一種讀法，而且已經
# 走到頭**——上限由支撐邊界與 `δ = c − x` 裡那份 −x 給，兩者都與 `c` 的
# 參數化無關。要動幾何那一欄必須換軸。
#
# 極座標下把場寫成只依一個變數就換到那個軸：
#
#     c = f(r)   只依半徑   繞影像中心旋轉任意角度 → 逐點不變
#     c = g(φ)   只依角度   繞影像中心縮放任意倍率 → 逐點不變
#
# **不變性對整個群成立，不是對某一個參數值。** 這是它與已否決的 log-periodic
# 候選（`runs/log_periodic_probe/`）的分界：那個構造只在 s = 1.2488 有值，
# 相鄰的 1.15 與 1.30 歸零，等於對評測算子的一個參數 co-adapt。
# `g(φ)` 對裁切的餘弦在比例 0.02–0.25 的八個點上全部是 1.000。
#
# 離線讀數（`runs/polar_carrier/`，十張影像、隨機剖面、未最佳化、RMS 對齊）：
#
#     場                rotate10   crop_resize0.1   參數量
#     帶限 S=32 ◂ 對照    +0.72        +0.51          768
#     f(r) 半徑          +0.81        +0.47            96
#     g(φ) 角度          +0.78        +0.59            96
#
# **代價未知**：自由度砍到 96 個數（`c = φ` 是 786 432），而
# `docs/DIRECTION.md` §3.4b 的階梯說效果來自逐像素的高頻自由度。
# 這一批就是去量那個代價。
#
# 四個臂
# ────────────────────────────────────────────────────────────────────
#   pr032 / pa032     f(r) / g(φ)，bins = 32
#   pa016             g(φ)，bins = 16——`bins` 是不變性的頻寬上限，
#                     16 的不變性更高（0.9992 對 0.9969）而自由度更低
#   pa032_rand        g(φ)、bins = 32、隨機剖面、零最佳化
#
# 隨機臂非有不可：`DIRECTION.md` §3.5 記過色彩網格的隨機解與最佳化解**看起來
# 一樣**。而且 `runs/polar_carrier/` 的三關全部是在隨機剖面上量的，
# 有這個臂才接得起來。
#
# 用法：bash scripts/polar_round.sh "<卡號…>" [smoke] ["<臂…>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-1 2 3 4 5})
MODE="${2:-full}"
[ ${#DEVS[@]} -gt 5 ] && DEVS=("${DEVS[@]:0:5}")

OUT=runs/ip2p_polar
LOG=runs/execution_logs/polar.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

arm_flags() {  # $1=臂 → "COND POLAR BINS"
  case "$1" in
    pr032)      echo "patch radial 32" ;;
    pa032)      echo "patch angular 32" ;;
    pa016)      echo "patch angular 16" ;;
    pa032_rand) echo "patch_rand angular 32" ;;
    *) return 1 ;;
  esac
}

NIMG="${NIMG:-6}"
IMGS=$(find runs/ip2p_face_defence -maxdepth 1 -type d -name 'clothing_plain_task_*' \
       2>/dev/null | sed 's|.*/clothing_plain_||' | sort | head -n "$NIMG")
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單" >&2; exit 2; }

COMMON="--data data/omniedit150 --attack-prompts data/attack_prompts.yaml \
--subject-source face --loss image_guidance --ig-zt diffuse_src \
--patch-carrier clothes --carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--patch-placement complement --radius 0.05 \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"

if [ "$MODE" = "smoke" ]; then
  IMG=$(echo $IMGS | awk '{print $1}')
  read -r COND POLAR BINS <<< "$(arm_flags pa032)"
  echo "=== [$(date '+%F %T')] 燒測 pa032（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --patch-polar "$POLAR" --patch-polar-bins "$BINS" \
    --steps 20 $COMMON 2>&1 | tee -a "$LOG" | tail -12
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  # 旗標若被靜默忽略，整批會退化成 `free` 的重跑，而每一欄看起來都正常。
  # 故直接讀回 CSV 的 patch_polar 欄，不只看檔案存不存在。
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="patch_polar") c=i} NR==2{print $c}' \
    "$OUT/_smoke/results.csv" | grep -qx angular || {
    echo "燒測的 results.csv 裡 patch_polar 不是 angular——旗標沒有生效" | tee -a "$LOG"
    exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="${3:-pr032 pa032 pa016 pa032_rand}"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] 極座標掃描：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND POLAR BINS <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[polar] $arm $POLAR/$BINS $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND \
        --patch-polar $POLAR --patch-polar-bins $BINS \
        --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND \
          --patch-polar $POLAR --patch-polar-bins $BINS \
          --steps 0 --resume-weights '$TRAIN_OUT' $COMMON
      done
    " > "$OUT/${arm}_$img.log" 2>&1 < /dev/null &
    sleep 4
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 收工，results $(find $OUT -name results.csv | wc -l) 份 ===" | tee -a "$LOG"
