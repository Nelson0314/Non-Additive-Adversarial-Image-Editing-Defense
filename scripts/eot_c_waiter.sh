#!/usr/bin/env bash
# 乙的訓練一收工，同時做兩件事：把丙的訓練送上四張卡，把乙的抗淨化跑在第五張。
# **合計五張，不超過上限。** 丙收工後再跑丙的抗淨化。
# 卡空不空一律由各腳本裡的 `free_cards.sh --assert` 決定，那一道會 exit。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s3
export PYTHONIOENCODING=utf-8

wait_for () {
  echo "$(date +%H:%M) 等 $1 收工"
  while true; do
    n=$(ps -u "$USER" -o cmd | grep -c "[i]p2p_run.py --out runs/$1")
    [ "$n" -eq 0 ] && break
    sleep 120
  done
  echo "$(date +%H:%M) $1 收工"
}

wait_for ip2p_eot_matched
FREE=($(bash scripts/free_cards.sh))
echo "$(date +%H:%M) 空卡：${FREE[*]:-（沒有）}"
if [ "${#FREE[@]}" -lt 5 ]; then
  echo "空卡不足五張，先只跑乙的抗淨化" >&2
else
  bash scripts/eot_blur_round.sh "${FREE[0]} ${FREE[1]} ${FREE[2]} ${FREE[3]}" || true
fi
sleep 30
FREE=($(bash scripts/free_cards.sh))
[ "${#FREE[@]}" -lt 1 ] && { echo "沒有卡跑乙的抗淨化" >&2; exit 4; }
SRC=runs/ip2p_eot_matched OUT=runs/ip2p_eot_matched_purify \
TAGS="eb_off eb_r20 eb_r22 eb_r25" \
bash scripts/purify_points.sh "${FREE[0]}"

wait_for ip2p_eot_blur
FREE=($(bash scripts/free_cards.sh))
[ "${#FREE[@]}" -lt 1 ] && { echo "沒有卡跑丙的抗淨化" >&2; exit 4; }
SRC=runs/ip2p_eot_blur OUT=runs/ip2p_eot_blur_purify \
TAGS="bl_wide bl_nocrop bl_both bl_s40" \
bash scripts/purify_points.sh "${FREE[0]}"
echo "$(date +%H:%M) 乙丙都跑完"
