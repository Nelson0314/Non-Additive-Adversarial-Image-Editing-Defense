#!/usr/bin/env bash
# 抗淨化：把已存的防禦圖淨化一遍再讓攻擊方編輯，看身分讀數掉多少。
#
# 為什麼是現在跑這一批
#   使用者的主張有一項是抗淨化，而新威脅模型（臉不動、擾動全在衣物上、
#   讀數是身分）底下**一格都還沒跑過**。這一批只讀已存的防禦圖，不重訓，
#   故成本全在編輯。
#
# 四個臂
#   free    完全自由的補丁，內容軸最強的一格（14/25）
#   tint    低頻懲罰 λ=3，唯一兩軸同時改善的一格（13/25、DISTS 0.140）
#   rand    同幾何、隨機內容、零最佳化（3/25）。**這一臂不可省**——
#           FaceLock／EditShield／FaceShield 報的抗淨化都沒有等幾何的隨機
#           對照，沒有它就分不出「防禦守住了」與「衣服上有一塊很吵的東西」。
#   floor   防禦圖改用原圖本身，量的是淨化算子自己造成的位移（空白地板）
#
# 十二個算子的出處見 `scripts/purify_identity.py` 的 LITERATURE_SET，
# 逐項對照見 `docs/reference/SURVEY_IDENTITY_EDITING.md` §5。
#
# 用法：bash scripts/purify_identity_round.sh "<卡號…>" [smoke|full] ["<臂…>"]
#       第二個參數給 `smoke` 時只跑一個臂、一類、一張圖、三個算子。
#       第三個參數限定要跑哪幾個臂（預設四個全跑），供分批送出——卡不夠時
#       先送 floor 與 tint，那兩個是「地板 ＋ 主角」的最小可解讀組合。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-0 6 7})
MODE="${2:-full}"
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [smoke]" >&2; exit 2; }

# **一次最多五張卡**（CLAUDE.md）。多給的裁掉，不照單全收。
if [ ${#DEVS[@]} -gt 5 ]; then
  echo "提醒：給了 ${#DEVS[@]} 張卡，裁到前五張（${DEVS[*]:0:5}）"
  DEVS=("${DEVS[@]:0:5}")
fi

OUT=runs/ip2p_purify_identity
LOG=runs/execution_logs/purify_identity.log
mkdir -p "$OUT" runs/execution_logs

# 卡是多人共用的。指定的卡上有別人就整批拒絕啟動，不擠。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

cells_for() {  # $1=臂 $2=類別
  case "$1" in
    free|floor) echo "runs/ip2p_face_defence/$2_plain_task_*" ;;
    *)          echo "runs/ip2p_content_constraint/$1_$2_task_*" ;;
  esac
}

if [ "$MODE" = "smoke" ]; then
  echo "=== [$(date '+%F %T')] 燒測 ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/purify_identity.py \
    --arm tint --cells "runs/ip2p_content_constraint/tint_clothing_task_attr_mod_color_123744" \
    --purifiers identity jpeg75 rotate10 --out "$OUT/_smoke" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="${3:-floor free tint rand}"
CATS="clothing accessory background"
for a in $ARMS; do
  case "$a" in floor|free|tint|rand) ;;
    *) echo "錯誤：未知的臂 '$a'，可用：floor free tint rand" >&2; exit 2 ;;
  esac
done
echo "要跑的臂：$ARMS"

# 先把每一個臂的來源目錄檢查過，缺哪一個就整批拒絕——跑到第九個 job 才發現
# 目錄名寫錯的話，前面八個的機時就白花了。preflight 不載權重。
for arm in $ARMS; do
  for cat in $CATS; do
    "$PY" scripts/purify_identity.py --arm "$arm" \
      --cells "$(cells_for "$arm" "$cat")" --out "$OUT" --preflight >/dev/null \
      || { echo "錯誤：preflight 在 $arm/$cat 上失敗" >&2; exit 2; }
  done
done
NJOB=$(( $(echo $ARMS | wc -w) * $(echo $CATS | wc -w) ))
echo "preflight $NJOB 個 job 全部通過" | tee -a "$LOG"

SLOTS=$(( ${#DEVS[@]} * 2 ))   # 每卡最多兩個 process
running() { ps -u "$USER" -o cmd | grep -c "[p]urify_identity.py"; }

echo "=== [$(date '+%F %T')] 送出 ${#DEVS[@]} 張卡、$SLOTS 個並行位 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  for cat in $CATS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 30; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    echo "[purify] $arm/$cat dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/purify_identity.py \
      --arm "$arm" --cells "$(cells_for "$arm" "$cat")" --out "$OUT" \
      --skip-existing \
      > "$OUT/${arm}_${cat}.log" 2>&1 < /dev/null &
    sleep 5
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] $NJOB 個 job 全部收工，卡已釋出 ===" | tee -a "$LOG"

# 身分讀數在 CPU 上補算，不佔卡。
ENTRIES=""
for arm in $ARMS; do
  for d in "$OUT"/${arm}_*/; do
    [ -d "$d" ] || continue
    ENTRIES="$ENTRIES --entry $(basename "$d")=$d"
  done
done
# shellcheck disable=SC2086
TAG=$(echo $ARMS | tr ' ' '-')
"$PY" scripts/identity_probe.py --out "$OUT/identity_$TAG.csv" $ENTRIES 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%F %T')] 身分讀數寫入 $OUT/identity_$TAG.csv ===" | tee -a "$LOG"
