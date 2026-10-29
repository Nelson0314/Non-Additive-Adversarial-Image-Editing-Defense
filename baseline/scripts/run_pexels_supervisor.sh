#!/usr/bin/env bash
# Pexels 佇列的監督：兩台主機的 crontab 每 30 分鐘各執行一次（本機的行程只由本機的監督管理）。
#   1. 移除本機已消失行程留下的認領（runtime/pexels_portraits/queue/claims/<工作>/owner）。
#   2. baseline 可用卡數 = 6 − max(~/gpu_leases 中名稱不以 baseline_ 開頭的租約數, color 預留張數)，
#      color 預留張數讀 runtime/pexels_portraits/queue/color_reserve（0–6 的整數；缺檔為 0）。
#      已認領的工作多於可用卡數時，停止本機最新認領的 worker（中斷的步驟之後依完成標記重做）；
#      已認領的工作數（兩台合計）低於可用卡數時，於本機以 setsid nohup 啟動 worker
#      （scripts/run_pexels_job.sh），每次至多啟動本機空卡數（vendor/scripts/measure_free_gpus.sh）。
#   3. CPU 工作（jobs.txt 中標 CPU 者）不佔卡、不計入上述認領數：本機進行中的 CPU 工作少於 CPU_PER_HOST（預設 2）
#      且有可認領的 CPU 工作時，啟動 `run_pexels_job.sh cpu`。
#   可認領：沒有完成或失敗標記、未被認領，且 AFTER= 所列工作皆已完成（格式見 scripts/run_pexels_job.sh）。
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
CPU_PER_HOST=${CPU_PER_HOST:-2}
claim_kind() { local _h _p k; read -r _h _p k < "$1/owner" || return 1; echo "${k:-gpu}"; }
for c in $Q/claims/*; do
  [ -f $c/owner ] || continue
  read -r h pid _ < $c/owner
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
claimed=0 cpu_local=0
for c in $Q/claims/*/; do
  [ -f $c/owner ] || continue
  if [ "$(claim_kind $c)" = cpu ]; then
    read -r h _ < $c/owner; [ "$h" = "$host" ] && cpu_local=$((cpu_local + 1))
  else
    claimed=$((claimed + 1))
  fi
done
# 已認領的工作多於可用卡數（color 新增租約或預留）：停止本機最新認領的 worker，交還其卡。
excess=$((claimed - allowed))
if [ $excess -gt 0 ]; then
  for c in $(ls -td $Q/claims/*/ 2>/dev/null); do
    [ $excess -gt 0 ] || break
    read -r h pid _ < $c/owner || continue
    [ "$h" = "$host" ] || continue
    [ "$(claim_kind $c)" = cpu ] && continue
    kill -TERM -- -"$pid" 2>/dev/null
    rm -rf "$c"
    log "stop $(basename $c) (pid $pid) to free a card for color (reserve=$reserve others=$others)"
    excess=$((excess - 1)); claimed=$((claimed - 1))
  done
fi
pending=0 cpu_pending=0
while read -r id cmd; do
  [ -n "$id" ] && [ "${id#\#}" = "$id" ] || continue
  [ -e $Q/done/$id ] || [ -e $Q/failed/$id ] || [ -d $Q/claims/$id ] && continue
  deps=""
  case $cmd in AFTER=*) deps=${cmd%% *}; deps=${deps#AFTER=}; cmd=${cmd#* } ;; esac
  for d in ${deps//,/ }; do [ -e $Q/done/$d ] || continue 2; done
  case $cmd in "CPU "*) cpu_pending=$((cpu_pending + 1)) ;; *) pending=$((pending + 1)) ;; esac
done < $Q/jobs.txt
free=$(bash $B/vendor/scripts/measure_free_gpus.sh 2>/dev/null | wc -w)
start=$((allowed - claimed)); [ $start -gt $free ] && start=$free; [ $start -gt $pending ] && start=$pending
[ $start -lt 0 ] && start=0
cpu_start=$((CPU_PER_HOST - cpu_local)); [ $cpu_start -gt $cpu_pending ] && cpu_start=$cpu_pending
[ $cpu_start -lt 0 ] && cpu_start=0
log "others=$others reserve=$reserve allowed=$allowed claimed=$claimed pending=$pending free_local=$free start=$start cpu_local=$cpu_local cpu_pending=$cpu_pending cpu_start=$cpu_start"
for _ in $(seq 1 $start); do
  (cd $B && setsid nohup bash scripts/run_pexels_job.sh > /dev/null 2>&1 < /dev/null &)
  sleep 20   # 讓前一個 worker 先完成認領與取卡，避免同時取卡互搶
done
for _ in $(seq 1 $cpu_start); do
  (cd $B && setsid nohup bash scripts/run_pexels_job.sh cpu > /dev/null 2>&1 < /dev/null &)
  sleep 5
done
