#!/usr/bin/env bash
set -uo pipefail
source ~/env.sh >/dev/null 2>&1
REPO=/nfs/home/nelson0314/WACV-s4
cd "$REPO"
export PYTHONPATH="$REPO"

CARD=${1:?用法: bash scripts/patch_pipeline_run.sh <卡號>}
DEF="$REPO/runs/${BATCH:-patch_free}"
EVAL="$REPO/runs/${BATCH:-patch_free}_eval"

# **同一個輸出目錄只准一個實例。** 重試迴圈或手滑重送會讓兩個行程寫同一份
# 結果，而且後啟動的那個會先 rm -rf 掉前一個正在寫的目錄。pgrep -f 不能用來
# 判斷——它會匹配到 ssh 自己的指令字串——所以用 PID 檔加 kill -0 驗活。
mkdir -p "$REPO/runs"
LOCK="$REPO/runs/.${BATCH:-patch_free}.pid"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[patch] !! 已有實例在跑（pid $(cat "$LOCK")），拒絕啟動 $(date -Is)"
  exit 3
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ "${FRESH:-0}" = "1" ]; then
  rm -rf "$DEF" "$EVAL" "$REPO/runs/${BATCH:-patch_free}_vqa"
  echo "[patch] FRESH=1，已清掉舊輸出"
fi

echo "[patch] 開始 $(date -Is) 卡 $CARD 主機 $(hostname) pid $$"

if [ -d "$DEF" ] && [ -n "$(ls -A "$DEF" 2>/dev/null)" ]; then
  echo "[patch] $DEF 已有輸出，跳過產圖"
else
  CUDA_VISIBLE_DEVICES=$CARD $PY "$REPO/scripts/immunise_patch.py" \
    --config "$REPO/configs/${CFG:-immunise_patch}.json" --out "$DEF" \
    > "$REPO/runs/${BATCH:-patch_free}.log" 2>&1
  echo "[patch] 產圖結束 $(date -Is) 回傳碼 $?"
  grep -E 'Traceback|Error|error' "$REPO/runs/${BATCH:-patch_free}.log" | head -3
fi

n=$(ls "$DEF"/*__defended.png 2>/dev/null | wc -l)
echo "[patch] 防禦圖 $n 張（期望 ${WANT_IMG:-4}）"
if [ "$n" -ne "${WANT_IMG:-4}" ]; then
  echo "[patch] !! 張數不符，不接評估"
  tail -n 8 "$REPO/runs/${BATCH:-patch_free}.log"
  exit 1
fi

mkdir -p "$EVAL"
for f in "$DEF"/*__defended.png; do
  b=$(basename "$f" .png)
  img=${b%%__*}
  v=${b#*__}; v=${v%__defended}
  RUN="$DEF/by_variant/$v"
  mkdir -p "$RUN"
  cp "$f" "$RUN/${img}__immunised.png"
done
echo "[patch] 攤成 $(ls -d "$DEF"/by_variant/*/ | wc -l) 個臂"

for V in $(ls "$DEF/by_variant"); do
  D="$EVAL/$V"
  if [ -d "$D" ] && [ -n "$(ls -A "$D" 2>/dev/null)" ]; then
    echo "[patch] $V 已有評估，跳過"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$CARD $PY "$REPO/scripts/evaluate_defence.py" \
    --config "$REPO/configs/evaluate_patch.json" \
    --run "$DEF/by_variant/$V" --out "$D" > "$EVAL/${V}.log" 2>&1
  rows=$(cat "$D"/*.csv 2>/dev/null | grep -vc '^arm,')
  echo "[patch] $V 列數 $rows（期望 2） $(date -Is)"
  [ "$rows" -ne 2 ] && { echo "[patch] !! $V 列數不符"; tail -n 4 "$EVAL/${V}.log"; }
done

echo "[patch] 評估結束 $(date -Is) 總列數 $(cat $EVAL/*/*.csv 2>/dev/null | grep -vc '^arm,')（期望 ${WANT_ROWS:-8}）"

CUDA_VISIBLE_DEVICES=$CARD $PY "$REPO/scripts/instruction_vqa.py" \
  --config "$REPO/configs/evaluate_patch.json" \
  --edits "$EVAL" --out "$REPO/runs/${BATCH:-patch_free}_vqa" \
  > "$REPO/runs/${BATCH:-patch_free}_vqa.log" 2>&1
echo "[patch] 指令完成度結束 $(date -Is)"
