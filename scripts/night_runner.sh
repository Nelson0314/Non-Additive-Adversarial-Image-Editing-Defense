#!/usr/bin/env bash
# 夜間排程：等目前的批次收工 → 送下一輪訓練（四卡）＋ 上一輪的抗淨化（一卡）。
# **合計五張，不超過上限。** 卡空不空一律由各腳本的 `free_cards.sh --assert`
# 決定，那一道會 exit。
#
# **隊列在「等完」之後才讀**（`runs/night_queue.txt`），所以看完圖之後插隊或
# 砍掉後面的輪次，改那個檔即可。**這支執行中不可覆寫。**
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s3
export PYTHONIOENCODING=utf-8
done_file=runs/night_queue_done.txt
touch "$done_file"

idle () {
  while true; do
    n=$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py')
    [ "$n" -eq 0 ] && break
    sleep 120
  done
}

purify () {   # $1=SRC 目錄名  $2=tags
  FREE=($(bash scripts/free_cards.sh))
  [ "${#FREE[@]}" -lt 1 ] && { echo "$(date +%H:%M) 沒有卡跑 $1 的抗淨化" >&2; return 1; }
  SRC="runs/$1" OUT="runs/${1}_purify" TAGS="$2" \
  setsid nohup bash scripts/purify_points.sh "${FREE[-1]}" \
      >> "/tmp/purify_$1.log" 2>&1 &
  disown
  echo "$(date +%H:%M) $1 的抗淨化已送（卡 ${FREE[-1]}）"
}

echo "$(date +%H:%M) 夜間排程啟動"
idle
purify ip2p_eot_matched "eb_off eb_r20 eb_r22 eb_r25"

while true; do
  next=""
  while read -r r; do
    [ -z "$r" ] && continue
    grep -qxF "$r" "$done_file" || { next="$r"; break; }
  done < runs/night_queue.txt
  [ -z "$next" ] && { echo "$(date +%H:%M) 隊列跑完"; break; }

  # 等到訓練的卡空出來（抗淨化在另一張卡上，不擋）
  while true; do
    n=$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py')
    [ "$n" -eq 0 ] && break
    sleep 120
  done
  FREE=($(bash scripts/free_cards.sh))
  echo "$(date +%H:%M) 下一輪 $next，空卡：${FREE[*]:-（沒有）}"
  if [ "${#FREE[@]}" -lt 4 ]; then
    echo "$(date +%H:%M) 空卡不足四張，等十分鐘" >&2; sleep 600; continue
  fi
  if bash "scripts/$next.sh" "${FREE[0]} ${FREE[1]} ${FREE[2]} ${FREE[3]}"; then
    echo "$next" >> "$done_file"
    sleep 120
    idle
    case "$next" in
      eot_blur_round)   purify ip2p_eot_blur   "bl_wide bl_nocrop bl_both bl_s40" ;;
      blend_loss_round) purify ip2p_blend_loss "bl_w000 bl_w025 bl_w100 bl_w400" ;;
    esac
  else
    echo "$(date +%H:%M) $next 派工失敗，等十分鐘" >&2; sleep 600
  fi
done
echo "$(date +%H:%M) 夜間排程結束"
