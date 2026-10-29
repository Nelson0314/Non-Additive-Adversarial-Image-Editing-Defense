#!/usr/bin/env bash
# Pexels 佇列的監督：兩台主機的 crontab 每 30 分鐘各執行一次（本機的行程只由本機的監督管理）。
#   1. 移除本機已消失行程留下的認領（runtime/pexels_portraits/queue/claims/<工作>/owner）。
#   2. baseline 可用卡數 = 6 − max(~/gpu_leases 中名稱不以 baseline_ 開頭的租約數, color 預留張數)，
#      color 預留張數讀 runtime/pexels_portraits/queue/color_reserve（0–6 的整數；缺檔為 0）。
#      已認領的工作多於可用卡數時，停止本機最新認領的 worker（中斷的步驟之後依完成標記重做）；
#      已認領的工作數（兩台合計）低於可用卡數時，於本機以 setsid nohup 啟動 worker
#      （scripts/run_pexels_job.sh），每次至多啟動本機空卡數（vendor/scripts/measure_free_gpus.sh）。
#   color 要取卡：寫入 color_reserve 後於兩台各執行一次本腳本，不必等下一個 cron 週期（見 docs/PEXELS_PORTRAITS.md「操作」）。
#   停止監督：建立 runtime/pexels_portraits/queue/PAUSED。
# 紀錄：runtime/pexels_portraits/queue/supervisor.log
# crontab：*/30 * * * * bash $HOME/image-immunization/baseline/scripts/run_pexels_supervisor.sh
set -u
B=~/image-immunization/baseline
Q=$B/runtime/pexels_portraits/queue
mkdir -p $Q/claims
log() { echo "$(date -Iseconds) $(hostname) $*" >> $Q/supervisor.log; }
[ -e $Q/PAUSED ] && { log "paused"; exit 0; }
[ -f $Q/jobs.txt ] || { log "[FATAL] 缺 $Q/jobs.txt"; exit 1; }

host=$(hostname)
for c in $Q/claims/*; do
  [ -f $c/owner ] || continue
  read -r h pid < $c/owner
  [ "$h" = "$host" ] || continue
  ps -p "$pid" > /dev/null 2>&1 || { log "reap stale claim $(basename $c) (pid $pid)"; rm -rf "$c"; }
done

others=0
for f in ~/gpu_leases/*; do
  [ -f "$f" ] || continue
  read -r _ _ name _ < "$f"
  case "$name" in baseline_*) ;; *) others=$((others + 1)) ;; esac
done
reserve=$(cat $Q/color_reserve 2>/dev/null || echo 0)
[[ "$reserve" =~ ^[0-6]$ ]] || { log "[FATAL] $Q/color_reserve 須為 0–6 的整數，收到 '$reserve'"; exit 1; }
held=$others; [ $reserve -gt $held ] && held=$reserve
allowed=$((6 - held)); [ $allowed -lt 0 ] && allowed=0
claimed=$(find $Q/claims -mindepth 1 -maxdepth 1 -type d | wc -l)
# 已認領的工作多於可用卡數（color 新增租約或預留）：停止本機最新認領的 worker，交還其卡。
excess=$((claimed - allowed))
if [ $excess -gt 0 ]; then
  for c in $(ls -td $Q/claims/*/ 2>/dev/null); do
    [ $excess -gt 0 ] || break
    read -r h pid < $c/owner || continue
    [ "$h" = "$host" ] || continue
    kill -TERM -- -"$pid" 2>/dev/null
    rm -rf "$c"
    log "stop $(basename $c) (pid $pid) to free a card for color (reserve=$reserve others=$others)"
    excess=$((excess - 1)); claimed=$((claimed - 1))
  done
fi
pending=0
while read -r id _; do
  [ -n "$id" ] && [ "${id#\#}" = "$id" ] || continue
  [ -e $Q/done/$id ] || [ -e $Q/failed/$id ] || [ -d $Q/claims/$id ] || pending=$((pending + 1))
done < $Q/jobs.txt
free=$(bash $B/vendor/scripts/measure_free_gpus.sh 2>/dev/null | wc -w)
start=$((allowed - claimed)); [ $start -gt $free ] && start=$free; [ $start -gt $pending ] && start=$pending
[ $start -lt 0 ] && start=0
log "others=$others reserve=$reserve allowed=$allowed claimed=$claimed pending=$pending free_local=$free start=$start"
for _ in $(seq 1 $start); do
  (cd $B && setsid nohup bash scripts/run_pexels_job.sh > /dev/null 2>&1 < /dev/null &)
  sleep 20   # 讓前一個 worker 先完成認領與取卡，避免同時取卡互搶
done
