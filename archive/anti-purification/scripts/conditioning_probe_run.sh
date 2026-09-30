#!/usr/bin/env bash
set -uo pipefail
source ~/env.sh >/dev/null 2>&1
REPO=/nfs/home/nelson0314/WACV-s4
cd "$REPO"
export PYTHONPATH="$REPO"

CARD=${1:?用法: bash scripts/conditioning_probe_run.sh <卡號>}
FLAT="$REPO/runs/plateau_probe_flat"
OUT="$REPO/runs/conditioning_probe.csv"

rm -rf "$FLAT"
mkdir -p "$FLAT"
n=0
for d in "$REPO"/runs/plateau_edit/by_variant/*/; do
  v=$(basename "$d")
  f="$d/task_env_weather_121086__immunised.png"
  [ -e "$f" ] || continue
  cp "$f" "$FLAT/$v.png"
  n=$((n+1))
done
echo "[probe] 攤平 $n 張防禦圖 $(date -Is)"
if [ "$n" -ne 20 ]; then
  echo "[probe] !! 不是 20 張，停"
  exit 1
fi

rm -f "$OUT"
CUDA_VISIBLE_DEVICES=$CARD $PY "$REPO/scripts/conditioning_probe.py" \
  --reference "$REPO/data/omniedit150/task_env_weather_121086/task_env_weather_121086.png" \
  --images "$FLAT" \
  --out "$OUT" \
  --sweep-images luma_gain__hw0.02__flat luma_gain__hw0.06__flat \
                 luma_gain__hw0.13__flat per_channel__hw0.13__flat
echo "[probe] 結束 $(date -Is) 回傳碼 $?"
