#!/usr/bin/env bash
# 等損失軸收工，接手它那五張卡跑抗淨化。
#
# 為什麼要有這一支：卡是多人共用的，而現在八張全滿——0／6／7 是別人的，
# 1–5 是 `queue_loss_axis.sh` 的。硬送會擠到別人或跟自己的佇列搶。這一支
# 只做一件事：**等到損失軸自己打出最終標記、卡真的釋出，才 assert 並送出**。
#
# 兩個等待條件缺一不可：
#   1. `queue_loss.log` 出現「損失軸四個臂全部收工」——只看 process 歸零不夠，
#      那在兩個臂之間也會成立，會搶在 `attn` 燒測前面。
#   2. `ip2p_run.py` 歸零之後再等 90 秒——process 從 `ps` 消失之後 GPU 記憶體
#      還沒釋放，不等會被自己剛結束的那一批擋下（`docs/OPERATIONS.md`）。
#
# 用法：bash scripts/chain_purify_after_loss.sh ["<卡號…>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || exit 2

CARDS="${1:-1 2 3 4 5}"
QLOG=runs/execution_logs/queue_loss.log
LOG=runs/execution_logs/chain_purify.log
DONE_MARK="損失軸四個臂全部收工"

echo "=== [$(date '+%F %T')] 等損失軸打出最終標記 ===" | tee -a "$LOG"
while ! grep -q "$DONE_MARK" "$QLOG" 2>/dev/null; do sleep 120; done
echo "=== [$(date '+%F %T')] 標記出現，等 process 歸零 ===" | tee -a "$LOG"
while [ "$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py')" -gt 0 ]; do sleep 60; done
# 記憶體釋放的延遲。
sleep 90

if ! bash scripts/free_cards.sh --assert "$CARDS"; then
  echo "=== [$(date '+%F %T')] 卡不是空的（別人進來了），不擠，結束 ===" | tee -a "$LOG"
  exit 3
fi
echo "=== [$(date '+%F %T')] 接手 $CARDS，送出抗淨化四個臂 ===" | tee -a "$LOG"
exec bash scripts/purify_identity_round.sh "$CARDS" full "floor tint free rand"
