#!/usr/bin/env bash
# color 佇列結束後自動恢復 Pexels 長跑（basic-1 與 basic-2 各執行一份，NFS 共用狀態）。
# 每 5 分鐘：寫本機的 color lane 行程數到 color_lanes_<主機>；兩台的紀錄都在 10 分鐘內、皆為 0、
# 且 ~/gpu_leases 沒有名稱以 color_ 開頭的租約，連續兩次成立即恢復：
#   第一個成立的主機（mkdir resume_after_color.done 成功者）把 worker_a.lock、worker_b.lock 移到 superseded/；
#   兩台都掛上 worker a、b 的等待行程（run_worker_when_card_free.sh，以鎖目錄保證每個 worker 只啟動一次）。
# 紀錄：runtime/pexels_portraits/resume_after_color.log
B=~/image-immunization/baseline
R=$B/runtime/pexels_portraits
host=$(hostname)
log=$R/resume_after_color.log
ok=0
echo "$(date -Is) $host watcher started" >> "$log"
while :; do
  echo "$(date +%s) $(pgrep -f "run_lut_queue.sh lane" | wc -l)" > "$R/color_lanes_$host.tmp" && mv "$R/color_lanes_$host.tmp" "$R/color_lanes_$host"
  now=$(date +%s); fresh=1; lanes=0
  for h in basic-1 basic-2; do
    f=$R/color_lanes_$h
    [ -f "$f" ] || { fresh=0; continue; }
    read -r t n < "$f"
    [ $((now - t)) -le 600 ] || fresh=0
    lanes=$((lanes + n))
  done
  color_leases=$(cat ~/gpu_leases/*-* 2>/dev/null | awk '$3 ~ /^color_/' | wc -l)
  if [ "$fresh" -eq 1 ] && [ "$lanes" -eq 0 ] && [ "$color_leases" -eq 0 ]; then ok=$((ok + 1)); else ok=0; fi
  if [ "$ok" -ge 2 ]; then
    if mkdir "$R/resume_after_color.done" 2>/dev/null; then
      for w in a b; do
        [ -d "$R/worker_$w.lock" ] && mv "$R/worker_$w.lock" "$R/superseded/worker_$w.lock.paused_$(date +%m%d%H%M)"
      done
      echo "$(date -Is) $host condition met (lanes=0 on both hosts, no color leases); worker locks moved" >> "$log"
    fi
    for w in a b; do
      setsid nohup bash "$B/scripts/run_worker_when_card_free.sh" "$w" $(cat "$R/worker_$w.sets") > /dev/null 2>&1 < /dev/null &
    done
    echo "$(date -Is) $host waiters for a and b launched" >> "$log"
    exit 0
  fi
  sleep 300
done
