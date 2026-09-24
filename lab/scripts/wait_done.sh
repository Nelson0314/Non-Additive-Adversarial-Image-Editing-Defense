#!/usr/bin/env bash
# 阻塞到「全部跑完」或「真的卡住」為止，然後**只回報一次**。
#
# 為什麼不是逐次輪詢回報：每一次回報都要重送整段對話，等一小時就是十幾次
# 零資訊的往返。這一支在遠端等，條件成立才印出一段摘要並結束。
#
# 「失敗」的判準不是「有 rc≠0 的階段」
# ────────────────────────────────────────────────────────────────────
# 排程器會換一張卡重試，所以單一個 rc≠0 是正常的（別人在 assert 通過後的
# 二十秒內佔走同一張卡，就會是這個形狀）。真正該回報的是兩件事：
#   1. 某個臂 `[GIVE-UP]`——重試次數用完；
#   2. **全停了**：沒有任何 arm_chain 在跑，也沒有任何 dispatcher 在等，
#      而 sentinel 還沒滿。那時候沒有人會再推進，等下去只是空等。
#
# 用法：bash lab/scripts/wait_done.sh <期望的 sentinel 數> [輪詢秒數]
set -uo pipefail
WANT="${1:-60}"
POLL="${2:-600}"
R=/nfs/home/nelson0314/image-immunization
cd "$R" || exit 1
ARMS="inpaint_outside_face inpaint_bg style_affine style_opt curve_dual_chroma ab_warp"

alive() {
  ps -u "$USER" -o args= | grep -cE "arm_chain.sh|dispatch.sh" || true
}

while true; do
  N=$(ls -1 lab/runs/state/*.done 2>/dev/null | wc -l)
  if [ "$N" -ge "$WANT" ]; then
    echo "[ALL-DONE] $(date -Is) sentinel $N/$WANT"; break
  fi
  GIVEUP=$(grep -h "^\[GIVE-UP\]" lab/runs/logs/*.log 2>/dev/null | tail -3)
  if [ -n "$GIVEUP" ]; then
    echo "[GAVE-UP] $(date -Is)"; echo "$GIVEUP"; break
  fi
  if [ "$(alive)" -eq 0 ]; then
    echo "[STALLED] $(date -Is) 本機沒有任何 arm_chain 或 dispatcher 在跑，"
    echo "          sentinel $N/$WANT。另一台可能還有，登入 basic-1 再查一次。"
    break
  fi
  sleep "$POLL"
done

echo "--- sentinel 逐臂 ---"
for a in $ARMS; do
  printf "%-22s %s/10\n" "$a" "$(ls -1 lab/runs/state/${a}.*.done 2>/dev/null | wc -l)"
done
echo "--- 租約 ---"; ls ~/lab_leases/ 2>/dev/null
echo "--- 最近的重試 ---"; grep -h "^\[RETRY\]\|^\[REGAIN\]" lab/runs/logs/*.log 2>/dev/null | tail -5
