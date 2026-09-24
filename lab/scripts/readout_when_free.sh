#!/usr/bin/env bash
# 等到有一張真正空的卡，拿一份租約，跑跨臂讀數，跑完放掉。
#
# 為什麼不是直接跑 `readout.sh`：卡是多人共用的，開工時常常一張都沒有。
# 輪詢寫在遠端這一側，等待不產生任何本機往返；租約與 `dispatch.sh` 用同一個
# 目錄與同一個格式（`<主機> <pid> <名稱>`），所以另一個 session 的排程器
# 看得到這張卡被佔著。
#
# 用法：
#   nohup setsid bash lab/scripts/readout_when_free.sh > log 2>&1 < /dev/null &
set -uo pipefail
source ~/env.sh
R=/nfs/home/nelson0314/image-immunization
cd "$R" || { echo "[FATAL] 進不去 $R"; exit 1; }

HOST=$(hostname)
LEASE=$HOME/lab_leases
POLL=180
CAP=5
mkdir -p "$LEASE"

echo "[WAIT-START] $(date -Is) host=$HOST 等一張空卡跑跨臂讀數"
GPU=""
while [ -z "$GPU" ]; do
  held=$(ls -1 "$LEASE" 2>/dev/null | wc -l)
  if [ "$held" -ge "$CAP" ]; then
    echo "[WAIT] $(date -Is) 租約已滿（$held/$CAP）"; sleep "$POLL"; continue
  fi
  for c in $(bash scripts/free_cards.sh 2>/dev/null); do
    [ -e "$LEASE/${HOST}-${c}" ] && continue
    bash scripts/free_cards.sh --assert "$c" >/dev/null 2>&1 || continue
    GPU="$c"; break
  done
  [ -z "$GPU" ] && { echo "[WAIT] $(date -Is) 沒有空卡"; sleep "$POLL"; }
done

LEASEFILE="$LEASE/${HOST}-${GPU}"
echo "$HOST $$ lab_readout" > "$LEASEFILE"
trap 'rm -f "$LEASEFILE"' EXIT
echo "[LAUNCH] $(date -Is) gpu=$GPU"

bash lab/scripts/readout.sh "$GPU"
rc=$?
echo "[READOUT-EXIT] $(date -Is) rc=$rc"
wc -l lab/results/*.csv
