#!/usr/bin/env bash
# 為 color 佇列讓卡（basic-1 與 basic-2 各執行一份，NFS 共用狀態；以 setsid nohup 在專案根背景執行）。
# 用法：[IMMEDIATE=1] [ALIGNED_ONLY=1] [EXTRA_ALIGNED_LOOPS="a b"] run_pause_for_color_queue.sh <標籤> <log>[=<標記>]...
#   log 為 color trial logs/ 下的檔名；標記預設 [ALL-DONE]，恢復條件為每個 log 的標記行數都比啟動時多。
#   <log> 寫成 @<檔名> 時，條件為該檔存在。ALIGNED_ONLY=1 時只暫停與恢復等失真 worker，原生 Pexels 不動；
#   NATIVE_WORKERS（預設 "a b"）限定恢復時移鎖與重掛等待行程的原生 worker；NO_PAUSE=1 時略過暫停步驟
#   （由呼叫端先行停止），只負責恢復。
#   1. 停止本機原生 Pexels 的等待行程；本機原生 worker 與等失真 worker 在下一個步驟邊界暫停（IMMEDIATE=1 時立即終止，
#      中斷的步驟沒有完成標記，恢復後整步重跑）；等失真取卡迴圈先停止。
#   2. 等每個 color log 都新增其標記行。
#   3. 恢復：第一個到達的主機把原生 worker 的鎖移到 superseded/，兩台各掛原生 a、b 的等待行程；
#      移除本機被終止的等失真 driver 的影像組鎖，重新啟動本機原有的等失真 worker 迴圈。
# 紀錄：runtime/pexels_portraits/pause_for_color_<標籤>.log
TAG=$1; shift; SPECS=("$@")
B=~/image-immunization/baseline
R=$B/runtime/pexels_portraits
A=$R/aligned
C=~/image-immunization/color/trials/lut3d_defense_and_naturalness/logs
host=$(hostname)
LOG=$R/pause_for_color_$TAG.log
log() { echo "$(date -Is) $host $*" >> "$LOG"; }
marker_count() {  # <log>[=<標記>] 或 @<檔名>
  if [ "${1:0:1}" = "@" ]; then [ -e "$C/${1:1}" ] && echo 1 || echo 0; return; fi
  local f=${1%%=*} m="[ALL-DONE]" c
  [ "$1" != "$f" ] && m=${1#*=}
  c=$(grep -cF -- "$m" "$C/$f" 2>/dev/null); echo "${c:-0}"
}
declare -A start_count
for sp in "${SPECS[@]}"; do start_count[$sp]=$(marker_count "$sp"); done
all_advanced() {
  local sp
  for sp in "${SPECS[@]}"; do [ "$(marker_count "$sp")" -gt "${start_count[$sp]}" ] || return 1; done
}

pause_at_boundary() {  # <worker log> <driver PID> <標籤>
  local wlog=$1 pid=$2 tag=$3 n0 step
  n0=$(grep -c "^\[STEP\]" "$wlog")
  while [ -z "${IMMEDIATE:-}" ] && kill -0 "$pid" 2>/dev/null; do
    [ "$(grep -c "^\[STEP\]" "$wlog")" -gt "$n0" ] && break
    sleep 5
  done
  step=$(grep "^\[STEP\]" "$wlog" | tail -1)
  pkill -TERM -P "$pid"
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  log "$tag paused at: $step; alive=$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)"
}

NATIVE_WORKERS=${NATIVE_WORKERS-a b}
[ -n "${ALIGNED_ONLY:-}" ] && NATIVE_WORKERS=""
if [ -z "${ALIGNED_ONLY:-}" ] && [ -z "${NO_PAUSE:-}" ]; then
  waiters=$(pgrep -f "^bash ($B/)?scripts/run_worker_when_card_free.sh")
  [ -n "$waiters" ] && kill -TERM $waiters && log "native waiters stopped: $(echo $waiters)"
fi
# 等失真 driver → 其取卡迴圈的 worker 名稱（driver 的祖先行程為 run_pexels_aligned_worker.sh <名稱>）
declare -A aligned_name
aligned_pids=$([ -z "${NO_PAUSE:-}" ] && pgrep -f "^bash scripts/evaluate_pexels_aligned.sh")
for pid in $aligned_pids; do
  q=$pid
  for i in 1 2 3 4; do
    q=$(ps -o ppid= -p "$q" | tr -d ' ')
    [ -z "$q" ] && break
    name=$(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null | awk '/run_pexels_aligned_worker.sh/ {print $NF}')
    [ -n "$name" ] && { aligned_name[$pid]=$name; break; }
  done
done
loops=""; for w in $([ -z "${NO_PAUSE:-}" ] && echo a b); do
  p=$(pgrep -f "^bash scripts/run_pexels_aligned_worker.sh $w")
  [ -n "$p" ] && { kill -TERM $p; loops="$loops $w"; log "aligned $w card-acquisition loop stopped"; }
done
# 先前暫停時已停止、恢復時須一併重啟的等失真迴圈（環境變數 EXTRA_ALIGNED_LOOPS，例如 "b"）
for w in ${EXTRA_ALIGNED_LOOPS:-}; do case " $loops " in *" $w "*) ;; *) loops="$loops $w";; esac; done
log "start: marker counts $(for sp in "${SPECS[@]}"; do echo -n "$sp=${start_count[$sp]} "; done); aligned drivers: $(for p in $aligned_pids; do echo -n "$p=${aligned_name[$p]:-?} "; done)"
for pid in $([ -z "${ALIGNED_ONLY:-}" ] && [ -z "${NO_PAUSE:-}" ] && pgrep -f "^bash scripts/evaluate_pexels_portraits.sh "); do
  card=$(tr '\0' ' ' < /proc/$pid/cmdline | awk '{print $3}')
  for w in a b; do
    head -1 "$R/worker_$w.lock/owner" 2>/dev/null | grep -q "^$host card $card " && \
      pause_at_boundary "$R/worker_$w.log" "$pid" "native $w" &
  done
done
for pid in $aligned_pids; do
  name=${aligned_name[$pid]:-}
  [ -n "$name" ] || { log "aligned driver $pid: worker name not found; not paused"; continue; }
  pause_at_boundary "$A/worker_$name.log" "$pid" "aligned $name" &
done
wait
sleep 5
log "leases now: $(ls ~/gpu_leases | tr '\n' ' ')"

until all_advanced; do sleep 300; done
log "color markers seen: ${SPECS[*]}"
if [ -n "$NATIVE_WORKERS" ]; then
if mkdir "$R/resume_for_color_$TAG.done" 2>/dev/null; then
  for w in $NATIVE_WORKERS; do
    [ -d "$R/worker_$w.lock" ] && mv "$R/worker_$w.lock" "$R/superseded/worker_$w.lock.paused_$(date +%m%d%H%M)"
  done
  touch "$R/resume_for_color_$TAG.moved"
  log "native worker locks moved"
fi
until [ -f "$R/resume_for_color_$TAG.moved" ]; do sleep 10; done
for w in $NATIVE_WORKERS; do
  setsid nohup bash "$B/scripts/run_worker_when_card_free.sh" "$w" $(cat "$R/worker_$w.sets") > /dev/null 2>&1 < /dev/null &
done
log "native waiters launched: $NATIVE_WORKERS"
fi
for pid in $aligned_pids; do
  for l in "$A"/locks/*/; do
    grep -q "pid $pid " "$l/owner" 2>/dev/null && { rm -r "$l"; log "removed aligned lock ${l%/}"; }
  done
done
for w in $loops; do
  (cd "$B" && setsid nohup bash scripts/run_pexels_aligned_worker.sh $w > /dev/null 2>&1 < /dev/null &)
  log "aligned worker $w relaunched"
done
