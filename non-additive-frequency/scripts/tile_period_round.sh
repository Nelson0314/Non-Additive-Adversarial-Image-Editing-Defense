#!/usr/bin/env bash
#
# ██ 這一批不執行。 ████████████████████████████████████████████████████
#
# 兩個理由，都在 `docs/DIRECTION.md` §3.7／§6.2：
#
#   1. **平舖不等於帶限。** 磚裡面仍是逐像素自由的，鋪滿之後是基頻 1/T 的
#      梳狀譜、諧波一路到 Nyquist。本批原本被寄望的「抗低通」不成立；
#      要把能量搬到低通留得住的地方，機制是 `--patch-res`（帶限）。
#   2. **平舖單獨用時外觀沒有改掉。** 支撐內色數是原圖的 7.89 倍，落在
#      使用者已裁定排除的那一區（低頻替換 9.22、凍結亮度 9.11、
#      完全自由 10.81 都在同一個量級）。
#
# 平舖仍保留為**與其他形式組合時的一個成分**（例如有限色數 ＋ 週期 64），
# 但不單獨掃。腳本保留供重現，不要送。
#
# ████████████████████████████████████████████████████████████████████
# 平舖週期的單獨掃描。**只動磚長，其餘與 `free` 臂逐項相同。**
#
# 為什麼是這一批
# ────────────────────────────────────────────────────────────────────
# `docs/DIRECTION.md` §6 的現況：色彩載體已被否定（§6.0，身分讀數下未淨化時
# 就沒有效果），剩下的問題是**補丁載體有效但撐不住低通**：
#
#     無淨化 +0.538（14/25）  導向濾波 101%  8 階量化 93%
#     JPEG 75 只剩 12%   裁切 4%   重取樣 1%   模糊 1%
#
# 逐像素自由學的內容把能量散在整個頻譜，重取樣一次就毀。**規則週期圖樣的
# 能量集中在少數幾個空間頻率上**，若週期選在重取樣的截止之下，那些頻率
# 原則上留得住。這同時對應「像標記而不是像壞掉」——規則重複是浮水印的第一
# 視覺特徵。
#
# `--patch-tile` 是現成的旋鈕，但**從來沒有單獨掃過**：唯一跑過的
# `lowproj_tile`（5/25）是磚長 64 **與低頻替換一起開**的，分不出是誰的作用。
#
# 五個臂
# ────────────────────────────────────────────────────────────────────
#   t016 / t032 / t064 / t128   磚長 16 / 32 / 64 / 128 像素
#   t064_rand                   磚長 64、隨機內容、零最佳化
#
# 512² 上磚長 `T` 的基頻是 `f_n = 1/T`：
#
#     T=16 → 0.0625    T=32 → 0.031    T=64 → 0.016    T=128 → 0.0078
#
# `crop_resize0.1` 的重取樣率是 1.2488×、`resize_only` 是 0.8×，
# 截止分別在 `f_n ≈ 0.40` 與 `0.31` 附近——**四個磚長全部在截止之下**，
# 故這一批問的不是「有沒有越過截止」，而是**基頻低到什麼程度效果才留得住**，
# 以及**效果本身會不會隨磚長掉光**（磚越大、自由度越少）。
#
# 用法：bash scripts/tile_period_round.sh "<卡號…>" [smoke]
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

OUT=runs/ip2p_tile_period
LOG=runs/execution_logs/tile_period.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

arm_flags() {  # $1=臂 → "COND TILE"
  case "$1" in
    t016)      echo "patch 16" ;;
    t032)      echo "patch 32" ;;
    t064)      echo "patch 64" ;;
    t128)      echo "patch 128" ;;
    t064_rand) echo "patch_rand 64" ;;
    *) return 1 ;;
  esac
}

NIMG="${NIMG:-6}"
IMGS=$(find runs/ip2p_face_defence -maxdepth 1 -type d -name 'clothing_plain_task_*' \
       2>/dev/null | sed 's|.*/clothing_plain_||' | sort | head -n "$NIMG")
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單" >&2; exit 2; }

# **與 `free` 臂逐項相同**，只多 `--patch-tile`。這樣 `free`（磚長 0 = 不平舖）
# 就是這條掃描的第一個點，不必重跑。
COMMON="--data data/omniedit150 --attack-prompts data/attack_prompts.yaml \
--subject-source face --loss image_guidance --ig-zt diffuse_src \
--patch-carrier clothes --carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--patch-placement complement --radius 0.05 \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"

if [ "$MODE" = "smoke" ]; then
  IMG=$(echo $IMGS | awk '{print $1}')
  read -r COND T <<< "$(arm_flags t032)"
  echo "=== [$(date '+%F %T')] 燒測 t032（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --patch-tile "$T" --steps 20 $COMMON 2>&1 \
    | tee -a "$LOG" | tail -12
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="${3:-t016 t032 t064 t128 t064_rand}"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
# 只數 python，不數外殼（`docs/DEFECTS.md`）。
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] 平舖週期掃描：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND T <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[tile] $arm T=$T $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND --patch-tile $T \
        --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND --patch-tile $T \
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
