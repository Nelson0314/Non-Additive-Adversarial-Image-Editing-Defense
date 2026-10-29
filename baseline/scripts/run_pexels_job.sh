#!/usr/bin/env bash
# Pexels 佇列的 worker：自 runtime/pexels_portraits/queue/jobs.txt 認領第一個尚未完成、尚未被認領的工作，
# 取得一張卡的租約（名稱 baseline_pexels_<工作>，全局上限 6）後執行，結束即退出；是否再開 worker 由
# scripts/run_pexels_supervisor.sh 決定。
#
# jobs.txt 每行「<工作 ID> <指令…>」，指令在 baseline 專案根執行、由租約設定 CUDA_VISIBLE_DEVICES。
# 狀態（Q=runtime/pexels_portraits/queue）：
#   $Q/claims/<工作>/owner   認領者「<主機> <PID>」；行程消失後由同主機的 supervisor 移除
#   $Q/done/<工作>           完成標記
#   $Q/failures/<工作>       失敗次數；達 3 次即記 $Q/failed/<工作>，不再認領，需人工處理
#   $Q/logs/<工作>.log       工作輸出；$Q/worker.log 為認領與結束紀錄
# 取卡失敗（沒有空卡）不計失敗次數。指令以 NOLEASE 開頭時不經租約包裝，由指令自行取卡。
# 用法：bash scripts/run_pexels_job.sh
set -u
B=~/image-immunization/baseline
Q=$B/runtime/pexels_portraits/queue
export ENV_FILE=~/env.sh
mkdir -p $Q/claims $Q/done $Q/failures $Q/failed $Q/logs
log() { echo "$(date -Iseconds) $(hostname) $$ $*" >> $Q/worker.log; }

job=""
while read -r id cmd; do
  [ -n "$id" ] && [ "${id#\#}" = "$id" ] || continue
  [ -e $Q/done/$id ] || [ -e $Q/failed/$id ] && continue
  mkdir $Q/claims/$id 2>/dev/null || continue
  echo "$(hostname) $$" > $Q/claims/$id/owner
  job=$id; break
done < $Q/jobs.txt
[ -n "$job" ] || { log "no job to claim"; exit 0; }
trap 'rm -rf $Q/claims/$job' EXIT
log "claim $job: $cmd"

if [ "${cmd%% *}" = NOLEASE ]; then
  # 自行取租約的指令（scripts/evaluate_pexels_portraits.sh 以卡號為參數、租約名 pexels_portraits）
  (cd $B && bash -c "${cmd#NOLEASE }") >> $Q/logs/$job.log 2>&1
else
  bash $B/vendor/scripts/run_with_gpu_lease.sh --work-dir $B --env-file $B/scripts/env.sh --limit 6 "baseline_pexels_$job" \
    bash -c "$cmd" >> $Q/logs/$job.log 2>&1
fi
rc=$?
if [ $rc -eq 0 ]; then
  touch $Q/done/$job; log "done $job"
elif tail -n 3 $Q/logs/$job.log | grep -q "沒有空卡\|NOT FREE OR CAP REACHED\|NO FREE CARD"; then
  log "no free card for $job"
else
  n=$(( $(cat $Q/failures/$job 2>/dev/null || echo 0) + 1 ))
  echo $n > $Q/failures/$job
  log "exit $rc $job (failure $n)"
  [ $n -lt 3 ] || { touch $Q/failed/$job; log "[FAILED] $job after $n failures"; }
fi
