#!/usr/bin/env bash
# 阻塞到「全部跑完」或「有一個階段失敗」為止，然後**只回報一次**。
#
# 為什麼不是逐次輪詢回報：每一次回報都要重送整段對話，等一小時就是十幾次
# 零資訊的往返。這一支在遠端等，條件成立才印出一段摘要並結束。
#
# 用法：bash lab/scripts/wait_done.sh <期望的 sentinel 數> [輪詢秒數]
set -uo pipefail
WANT="${1:-90}"
POLL="${2:-300}"
R=/nfs/home/nelson0314/image-immunization
cd "$R" || exit 1

while true; do
  N=$(ls -1 lab/runs/state/*.done 2>/dev/null | wc -l)
  BAD=$(grep -h "^\[EXIT\]" lab/runs/logs/*.log 2>/dev/null | grep -v "rc=0" | tail -3)
  if [ -n "$BAD" ]; then
    echo "[FAILED] $(date -Is) 有階段回傳非零："
    echo "$BAD"
    break
  fi
  if [ "$N" -ge "$WANT" ]; then
    echo "[ALL-DONE] $(date -Is) sentinel $N/$WANT"
    break
  fi
  sleep "$POLL"
done

echo "--- sentinel 逐臂 ---"
for a in style_random style_low style_filter style_filter_guided curve_dual_spatial curve_dual_chroma; do
  printf "%-20s %s/18\n" "$a" "$(ls -1 lab/runs/state/${a}.*.done 2>/dev/null | wc -l)"
done
echo "--- 租約 ---"; ls ~/lab_leases/ 2>/dev/null
