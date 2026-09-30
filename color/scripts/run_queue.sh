#!/usr/bin/env bash
# color 的工作佇列：以 vendor/scripts/queue_worker.sh 排程，注入本專案的工作語法。
#
# 用法（color 專案根）：nohup setsid bash scripts/run_queue.sh <佇列名> <工作>... &
#
# 工作的寫法與相依（相依由 scripts/queue_depends.sh 定義）：
#   pilot:<條件>:<影像>:<步數>  單張短步數試跑，輸出到 artifacts/defense_pilots/<條件>/<影像>
#   def:<條件>:<影像>           單張防禦圖，輸出到 artifacts/defense_shards/<條件>/<影像>
#                               相依：同一條件的 pilot（若有列出）
#   chain:<條件>                併分片（若有）後跑 evaluate_condition.sh（只跑 ip2p）
#                               相依：同一條件全部 def（若有列出）
#   readout                     跨條件讀數。相依：全部 chain
#   fid                         防禦圖失真（measure_defense_fidelity，需 FID_ARMS）
# 佇列外的相依：readout 與 fid 等待 WAIT_ARMS 中每個條件的
# runtime/state/<條件>.pedit_blur2_ip2p.done。
#
# 狀態寫在 runtime/queues/<佇列名>/，逐工作的 log 在 runtime/logs/queue_<佇列名>/。
# QUEUE_CAP 可另外降低本類佇列的合計租約數；全局卡數見 vendor/scripts/gpu_policy.sh。
set -uo pipefail
[ $# -ge 2 ] || { echo "用法：run_queue.sh <佇列名> <工作>..." >&2; exit 2; }
QNAME="$1"; shift
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPTS/env.sh" || exit 1
for job in "$@"; do
  if [ "$job" = fid ] && [ -z "${FID_ARMS:-}" ]; then
    echo "[FATAL] fid 工作必須明確指定 FID_ARMS" >&2; exit 2
  fi
done
export PY FID_ARMS="${FID_ARMS:-}" WAIT_ARMS="${WAIT_ARMS:-}"
exec bash "$GPU_TOOLS/queue_worker.sh" --workdir "$COLOR_ROOT" \
  --state "$COLOR_ROOT/runtime/queues/$QNAME" --logs "$COLOR_ROOT/runtime/logs/queue_$QNAME" \
  --runner "$SCRIPTS/queue_job.sh" --validator "$SCRIPTS/queue_validate.sh" \
  --depends "$SCRIPTS/queue_depends.sh" "$QNAME" "$@"
