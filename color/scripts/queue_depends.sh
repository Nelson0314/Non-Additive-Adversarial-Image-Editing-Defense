#!/usr/bin/env bash
# 佇列相依：`queue_depends.sh <工作> <全部工作>...`，每行印出一件佇列內的上游工作；
# 結束碼 0 表示佇列外的前置條件已滿足、1 表示尚未滿足。
set -uo pipefail
job="$1"; shift
kind=${job%%:*}; arm=$(echo "$job" | cut -d: -f2)
for j in "$@"; do
  case "$kind" in
    def)     [[ "$j" == pilot:"$arm":* ]] && echo "$j" ;;
    chain)   [[ "$j" == def:"$arm":* ]] && echo "$j" ;;
    readout) [[ "$j" == chain:* ]] && echo "$j" ;;
  esac
done
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$kind" in
  readout|fid)
    for a in ${WAIT_ARMS:-}; do
      [ -e "$ROOT/runtime/state/$a.chain.done" ] || exit 1
    done ;;
esac
exit 0
