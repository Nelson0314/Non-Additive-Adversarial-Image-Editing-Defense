#!/usr/bin/env bash
# 載體變體的篩選評估：一個變體吃滿五張卡，跑完再換下一個。
#
# 為什麼是鏈式而不是一次全撒：一個變體跑完就有一份完整的 CSV。時間不夠時
# 損失的是排在後面的變體，不是每個變體都只跑了一半。
#
# 順序是優先序。flow_02 排第一：位移最小、最可能通過自然度那一關。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s4
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s4

CARDS=(2 3 4 5 6)
ORDER=(flow_02 flow_04 flow_08 lab_field flow_16)

echo "[chain] 等產圖跑完"
while pgrep -u "$(whoami)" -f immunise_field.py > /dev/null; do sleep 60; done
echo "[chain] 產圖已結束 $(date -Is)"

for V in "${ORDER[@]}"; do
  RUN=runs/field_ladder/by_variant/$V
  mkdir -p "$RUN"
  n=0
  for f in runs/field_ladder/shard*/*__${V}__immunised.png; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); img=${b%__${V}__immunised.png}
    ln -sf "$(readlink -f "$f")" "$RUN/${img}__immunised.png"
    n=$((n+1))
  done
  echo "[chain] $V 有 $n 張防禦圖 $(date -Is)"
  if [ "$n" -eq 0 ]; then echo "[chain] $V 沒有產物，跳過"; continue; fi

  mkdir -p runs/field_screen/$V
  for i in 1 2 3 4 5; do
    G=${CARDS[$((i-1))]}
    CUDA_VISIBLE_DEVICES=$G $PY scripts/evaluate_defence.py       --config configs/evaluate_screen.json --run "$RUN"       --out runs/field_screen/$V/shard$i --shard $i/5       > runs/field_screen/${V}_shard$i.log 2>&1 &
  done
  wait
  echo "[chain] $V 完成 $(date -Is) 列數 $(cat runs/field_screen/$V/shard*/*.csv 2>/dev/null | grep -vc '^arm,')"
done
echo "[chain] 全部結束 $(date -Is)"
