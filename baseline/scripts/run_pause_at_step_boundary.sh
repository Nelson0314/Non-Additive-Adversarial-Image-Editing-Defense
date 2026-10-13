#!/usr/bin/env bash
# 在 Pexels worker 的下一個步驟邊界暫停：等 log 出現新的 [STEP]（前一步已寫完完成標記），立即終止該 worker。
# 用法：run_pause_at_step_boundary.sh <worker 名稱> <evaluate_pexels_portraits.sh 的 PID>
R=~/image-immunization/baseline/runtime/pexels_portraits
w=$1; pid=$2
log=$R/worker_$w.log
n0=$(grep -c "^\[STEP\]" "$log")
while kill -0 "$pid" 2>/dev/null; do
  [ "$(grep -c "^\[STEP\]" "$log")" -gt "$n0" ] && break
  sleep 5
done
step=$(grep "^\[STEP\]" "$log" | tail -1)
pkill -TERM -P "$pid"
kill -TERM "$pid" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
echo "$(date -Is) worker $w paused at: $step; alive=$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)" >> "$R/pause.log"
