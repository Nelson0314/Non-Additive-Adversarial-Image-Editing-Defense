#!/usr/bin/env bash
# 等到本機有一張沒有 compute app、也沒有租約的卡，再以該卡啟動 Pexels 評測的一個 worker。
# 用法：run_worker_when_card_free.sh <worker 名稱> <影像組...>
# 同名 worker 以 NFS 上的鎖目錄保證只啟動一次（兩台主機各有一個等待行程）。worker 結束後不重啟：
# 租約被拒（結束碼 3）時釋放鎖並繼續等待，其餘結束碼原樣結束並留下鎖與紀錄。
B=~/image-immunization/baseline
S=$B/runtime/pexels_portraits
name=$1; shift
lock=$S/worker_$name.lock
while :; do
  [ -d "$lock" ] && exit 0
  for c in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    uuid=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid" && continue
    [ -e ~/gpu_leases/"$(hostname)-$c" ] && continue
    mkdir "$lock" 2>/dev/null || exit 0
    echo "$(hostname) card $c started $(date -Is)" > "$lock/owner"
    cd "$B" || exit 1
    ENV_FILE=~/env.sh bash scripts/evaluate_pexels_portraits.sh "$c" "$@" >> "$S/worker_$name.log" 2>&1
    rc=$?
    echo "exit $rc $(date -Is)" >> "$lock/owner"
    if [ "$rc" -eq 3 ]; then rm -rf "$lock"; break; fi
    exit "$rc"
  done
  sleep 300
done
