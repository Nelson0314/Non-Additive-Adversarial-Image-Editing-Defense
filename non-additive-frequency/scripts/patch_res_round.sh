#!/usr/bin/env bash
# 帶限參數化的單獨掃描。**只動 `--patch-res`，其餘與 `free` 臂逐項相同**，
# 故 `free`（res 1 = 全解析度）就是這條掃描的第一個點，不必重跑。
#
# 為什麼是這一批
# ────────────────────────────────────────────────────────────────────
# `docs/DIRECTION.md` §6.1b：補丁載體有效（+0.538、14/25），但
# **模糊 1%、重取樣 1%、裁切 4%**。§3.2 已經把分界釘出來——不在取景，
# 在有沒有把高頻抹掉。
#
# `--patch-tile` 常被當成這條路的旋鈕，但**平舖不等於帶限**：一塊磚的內容
# 仍然是逐像素自由的，鋪滿之後的頻譜是基頻 1/T 的梳狀譜，諧波一路延伸到
# Nyquist。平舖買到的是冗餘、參數量與「像標記」，不是把能量搬到低頻。
#
# `--patch-res S` 才是帶限：可學張量只有 1/S 大小，升取樣之後遠高於截止的
# 能量掉兩個數量級。實測（隨機內容、扣掉直流、128² 上量，
# `tests/test_patch_res.py::test_遠高於截止的能量掉兩個數量級`）：
#
#     S       E(f > 1/2)        自由度
#     1         0.806           3·H·W
#     4         0.007           3·H·W/16
#    16         0.004           3·H·W/256
#
# `f > 1/2` 是一次 2× 降取樣抹掉的那一整段，也就是重取樣與模糊真正吃掉的
# 東西。**恰好高於 1/S 的那一段另有 12–22% 的 sinc² 旁瓣洩漏**，不可宣稱
# 「依構造沒有細於 S 像素的分量」。
#
# 機制取自 IAM（arXiv:2402.16586）的 interpolation smoothing，但那篇動的是
# **更新步驟**（在半解析度上走一步再升回去），此處動的是**參數化本身**。
#
# 這條掃描同時是 §6.4 那條取捨曲線的自變數
# ────────────────────────────────────────────────────────────────────
# `S` 一動，三件事同時動：自由度（下降 S²）、頻帶（收窄）、外觀（由逐像素
# 噪點變成柔和色塊）。**加性方法畫不出這條曲線**——δ 的頻帶由 ε 與最佳化
# 決定，不能指定。故無論效果是升是降，這一批都落在方向上。
#
# 五個臂
# ────────────────────────────────────────────────────────────────────
#   s08 / s16 / s24 / s32   解析度倒數 8 / 16 / 24 / 32（512² 上是 64² … 16²）
#   s16_rand                S=16、隨機內容、零最佳化
#
# **格點為什麼由 {2,4,8,16} 改成 {8,16,24,32}**
# ────────────────────────────────────────────────────────────────────
# `runs/purifier_transfer/` 量到十四個算子的頻率響應（純 CPU、帶限雜訊探針）。
# 模糊那一欄的存活率：
#
#     頻帶（週期）   16 px   8 px   4 px
#     blur1.5        0.800  0.455  0.089
#     resize_only    0.973  0.848  0.366
#
# 要讓 `blur1.5` 留下八成以上，能量必須落在**粗於 16 像素**的結構上。
# `S = 8` 的截止在 1/8（週期 8 px），仍然偏高；`S = 16` 的截止在 1/16 才落在
# 那裡。**原本排的 S = 2 與 4（截止在週期 2 與 4 px）整段落在低通留不住的
# 帶上**，跑了也只會重現 `free` 的結果。
#
# 隨機臂非有不可：`docs/DIRECTION.md` §3.4b 的階梯是拿隨機當地板量出來的，
# 而帶限會同時改變**最佳化解**與**隨機解**的形狀，兩者要一起看。
#
# 用法：bash scripts/patch_res_round.sh "<卡號…>" [smoke]
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
# 預設仍是五張（`CLAUDE.md`：多人共用，佔滿會讓別人排不進來）。
# 超過五張**必須使用者明確授意**，故做成要另外給 `MAX_DEVS` 才放行，
# 不改預設值——改預設會讓之後每一批都靜默佔六張。
MAX_DEVS="${MAX_DEVS:-5}"
[ ${#DEVS[@]} -gt "$MAX_DEVS" ] && DEVS=("${DEVS[@]:0:$MAX_DEVS}")

OUT=runs/ip2p_patch_res
LOG=runs/execution_logs/patch_res.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

arm_flags() {  # $1=臂 → "COND S"
  case "$1" in
    s08)      echo "patch 8" ;;
    s16)      echo "patch 16" ;;
    s24)      echo "patch 24" ;;
    s32)      echo "patch 32" ;;
    s16_rand) echo "patch_rand 16" ;;
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
  read -r COND S <<< "$(arm_flags s16)"
  echo "=== [$(date '+%F %T')] 燒測 s16（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --patch-res "$S" --steps 20 $COMMON 2>&1 \
    | tee -a "$LOG" | tail -12
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  # 旗標若被靜默忽略，整批會退化成 `free` 的重跑，而每一欄看起來都正常。
  # 故直接讀回 CSV 的 patch_res 欄，不只看檔案存不存在。
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="patch_res") c=i} NR==2{print $c}' \
    "$OUT/_smoke/results.csv" | grep -qx 16 || {
    echo "燒測的 results.csv 裡 patch_res 不是 16——旗標沒有生效" | tee -a "$LOG"
    exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="${3:-s08 s16 s24 s32 s16_rand}"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] 帶限參數化掃描：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND S <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[res] $arm S=$S $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND --patch-res $S \
        --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND --patch-res $S \
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
