#!/usr/bin/env bash
# Pexels 等失真臂的 worker：取得本機一張卡的租約後執行 scripts/evaluate_pexels_aligned.sh。
# 沒有空卡時 10 分鐘後重試；沒有可處理的影像組（結束碼 4）時 30 分鐘後重試；全部完成或其他錯誤即結束。
# 用法：bash scripts/run_pexels_aligned_worker.sh <名稱>
set -u
NAME=${1:?需要 worker 名稱}
B=~/image-immunization/baseline
D=$B/runtime/pexels_portraits/aligned
SETS=$(cd $B && python3 -c "import json; print(' '.join(s['name'] for s in json.load(open('data/pexels_portraits/image_sets.json'))['sets']))")
export ENV_FILE=~/env.sh
log() { echo "$(date -Iseconds) $(hostname) $NAME $*" >> $D/run.log; }
log "start"
while true; do
  bash $B/vendor/scripts/run_with_gpu_lease.sh --work-dir $B --env-file $B/scripts/env.sh pexels_aligned \
    bash scripts/evaluate_pexels_aligned.sh $SETS >> $D/worker_$NAME.log 2>&1
  rc=$?
  if [ $rc -ne 0 ] && tail -n 3 $D/worker_$NAME.log | grep -q "沒有空卡"; then
    log "no free card; retry in 600 s"; sleep 600; continue
  fi
  if [ $rc -eq 4 ]; then log "no ready image set; retry in 1800 s"; sleep 1800; continue; fi
  log "exit $rc"; exit $rc
done
