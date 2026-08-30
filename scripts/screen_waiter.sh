#!/usr/bin/env bash
# 等強化批次的訓練收工，把騰出來的卡交給視覺篩選批。**卡最多五張。**
# 等待器只負責等；卡空不空由 `visual_screen.sh` 裡的 `free_cards.sh --assert`
# 決定，那一道會 exit。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s3
export PYTHONIOENCODING=utf-8
echo "$(date +%H:%M) 篩選等待器啟動：等 ip2p_ig_harden 的訓練收工"
while true; do
  n=$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py --out runs/ip2p_ig_harden')
  [ "$n" -eq 0 ] && break
  sleep 120
done
echo "$(date +%H:%M) 強化批訓練收工，取空卡"
FREE=($(bash scripts/free_cards.sh))
echo "$(date +%H:%M) 空卡：${FREE[*]:-（沒有）}"
if [ "${#FREE[@]}" -lt 4 ]; then
  echo "空卡不足四張，不送。" >&2; exit 4
fi
bash scripts/visual_screen.sh "${FREE[0]} ${FREE[1]} ${FREE[2]} ${FREE[3]}"
