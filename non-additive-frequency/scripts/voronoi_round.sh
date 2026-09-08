#!/usr/bin/env bash
# Voronoi 結構化圖樣的掃描。**只動 `--patch-seeds`，其餘與 `free` 臂逐項相同。**
#
# 為什麼是這一批
# ────────────────────────────────────────────────────────────────────
# 移植自 *Structured Adversarial Camouflage via Voronoi Diagrams*
# （arXiv:2606.17711）：最佳化種子點位置與每格顏色，胞結構由 Voronoi 圖決定、
# 調色盤固定且可印。該篇報跨 YOLOv9-12 的黑箱轉移，理由是「離散受限的色彩
# 空間降低對數位渲染的過擬合」。
#
# 本專案的離線讀數（`runs/carrier_survey/VORONOI.md`，同一張影像、真實衣物
# 載體、隨機起點、未最佳化；支撐內原圖 302 色）：
#
#     參數化          參數量      色數比    譜斜率 drop
#     free            786 432     10.81      +2.07
#     palette K=8   2 097 176      0.74      +2.39
#     voronoi K=24        120      0.64      +0.16
#     voronoi K=64        320      0.52      +0.17
#
# 兩件事讓它值得一批機時：
#
#   1. **參數量少 6 500 倍**（120 對 786 432）。`docs/PIPELINE.md` §5 列的唯一
#      不成立項是單張 50 分鐘，而參數量掉四個數量級時收斂步數**可能**跟著掉
#      ——這一批順便回答。
#   2. **它是第一個同時壓住色數與頻譜的參數化。** 調色盤壓色數但不動空間
#      頻率（+2.39），帶限壓頻帶但色數只到 4.56；Voronoi 兩邊都壓。
#
# **代價未知**：`docs/DIRECTION.md` §3.4b 的階梯說效果來自逐像素的高頻自由度，
# 而 Voronoi 砍得最兇。這一批就是去量那個代價。
#
# 三個臂
# ────────────────────────────────────────────────────────────────────
#   v024 / v064     K = 24 / 64 個種子點
#   v064_rand       K = 64、**共用同一組色票**、種子隨機、零最佳化
#
# 隨機臂非有不可：`DIRECTION.md` §3.5 記過色彩網格的隨機解與最佳化解**看起來
# 一樣**——外觀由參數化決定不由最佳化決定。離線讀數已經顯示同樣的傾向
# （色數比 0.52 對隨機的 0.60），要分辨就必須有這個臂。
#
# 用法：bash scripts/voronoi_round.sh "<卡號…>" [smoke]
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

OUT=runs/ip2p_voronoi
LOG=runs/execution_logs/voronoi.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

arm_flags() {  # $1=臂 → "COND K"
  case "$1" in
    v024)      echo "patch 24" ;;
    v064)      echo "patch 64" ;;
    v064_rand) echo "patch_rand 64" ;;
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
  read -r COND K <<< "$(arm_flags v064)"
  echo "=== [$(date '+%F %T')] 燒測 v064（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --patch-seeds "$K" --steps 20 $COMMON 2>&1 \
    | tee -a "$LOG" | tail -12
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  # 旗標若被靜默忽略，整批會退化成 `free` 的重跑，而每一欄看起來都正常。
  # 故直接讀回 CSV 的 patch_seeds 欄，不只看檔案存不存在。
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="patch_seeds") c=i} NR==2{print $c}' \
    "$OUT/_smoke/results.csv" | grep -qx 64 || {
    echo "燒測的 results.csv 裡 patch_seeds 不是 64——旗標沒有生效" | tee -a "$LOG"
    exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="${3:-v024 v064 v064_rand}"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] Voronoi 掃描：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND K <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[vor] $arm K=$K $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND --patch-seeds $K \
        --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND --patch-seeds $K \
          --steps 0 --resume-weights '$TRAIN_OUT' $COMMON
      done
    " > "$OUT/${arm}_$img.log" 2>&1 < /dev/null &
    sleep 4
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 收工，results $(find $OUT -name results.csv | wc -l) 份 ===" | tee -a "$LOG"

ENT=""
for arm in $ARMS; do
  for d in "$OUT"/${arm}_*/; do
    [ -d "$d" ] && [ -f "$d/results.csv" ] || continue
    ENT="$ENT --entry $(basename "$d")=$d"
  done
done
# shellcheck disable=SC2086
"$PY" scripts/identity_probe.py --out "$OUT/identity.csv" $ENT 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%F %T')] 身分讀數寫入 $OUT/identity.csv ===" | tee -a "$LOG"
