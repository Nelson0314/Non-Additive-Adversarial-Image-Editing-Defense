#!/usr/bin/env bash
# 色彩網格的**空間解析度 `G`** 掃描。半徑固定，只動 `G`。
#
# 為什麼掃 `G` 而不是半徑
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_colour_budget` 的答案是「半徑不是槓桿」：0.60 → 1.20 讓 DISTS
# 由 0.2018 動到 0.2044，兩個臂都跑滿 6000 步（不是早停），身分降幅在衣物
# 那一類只有 +0.014。**把預算開大，最佳化不會去用它。**
#
# `G` 則同時決定三件事（`docs/DIRECTION.md` §5.3）：
#
#   容量        12 · D · G²，`ip2p_color_hunt` 量到 8×8 → 16×16 讓位移由
#               0.4072 升到 0.5441——**容量是槓桿**
#   抗重取樣    色彩場落在 `f_n ≈ G/H`。`G` 小 → 低頻 → 重取樣抹不掉。
#               `G = 1` 是全域常數場，對裁切**精確等變**
#   產物外觀    `G = 16` 拉圖看到的是**霓虹色塊**不是「換色」（§3.5）。
#               `G = 1` 才是「整件衣服共用一個色彩變換」＝ 真正的換色
#
# 三個臂（`G = 16` 那一點由既有的 `r06` 提供，半徑同為 0.60，不重跑）
# ────────────────────────────────────────────────────────────────────
#   g01       G = 1   全域常數映射。容量 12·D = 192，裁切精確等變
#   g04       G = 4   中間點
#   g01_rand  G = 1、隨機、零最佳化。**「整件衣服隨機換一個顏色」**
#             ——這一臂本身就是一個有意義的對照：若它與 g01 打平，
#             那麼全域換色的效果與最佳化無關
#
# 用法：bash scripts/colour_grid_round.sh "<卡號…>" [smoke]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
# 權重在系統共用快取；`ssh <host> "bash script"` 不載入 profile。
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-1 2 3 4 5})
MODE="${2:-full}"
[ ${#DEVS[@]} -gt 5 ] && DEVS=("${DEVS[@]:0:5}")

OUT=runs/ip2p_colour_grid
LOG=runs/execution_logs/colour_grid.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"
# **半徑固定。** `ip2p_colour_budget` 已經證實它不是槓桿；兩個一起動就分不出
# 是誰的功勞。0.60 是那一批的最小值，也是 `G = 16` 那一點的既有設定。
RADIUS=0.60
LUMA=16

arm_flags() {  # $1=臂 → "COND GRID"
  case "$1" in
    g01)      echo "color_grid 1" ;;
    g04)      echo "color_grid 4" ;;
    g01_rand) echo "color_grid_rand 1" ;;
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
--color-luma-bins $LUMA \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"

if [ "$MODE" = "smoke" ]; then
  IMG=$(echo $IMGS | awk '{print $1}')
  read -r COND G <<< "$(arm_flags g01)"
  echo "=== [$(date '+%F %T')] 燒測 g01（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --radius "$RADIUS" --color-grid "$G" \
    --steps 20 $COMMON 2>&1 | tee -a "$LOG" | tail -12
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="g01 g04 g01_rand"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
# 只數 python，不數外殼（`docs/DEFECTS.md`）。
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] G 掃描：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張，半徑固定 $RADIUS ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND G <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[grid] $arm G=$G $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND --radius $RADIUS \
        --color-grid $G --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND --radius $RADIUS \
          --color-grid $G --steps 0 --resume-weights '$TRAIN_OUT' $COMMON
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
