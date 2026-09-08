#!/usr/bin/env bash
# 等色彩族的抗淨化收工，接手同一批卡跑 `G` 掃描。
#
# 排在抗淨化後面的理由：`G` 掃描要跑 6000 步 × 18 格，而抗淨化只讀已存的
# 防禦圖（每 job 約一小時但不重訓）。先把便宜且會決定方向的那一批跑完
# （`DIRECTION.md` §5.2：色彩族在身分讀數下平不平，是分水嶺），
# 再花機時在新的掃描上。
#
# 用法：bash scripts/chain_colour_grid.sh ["<卡號…>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || exit 2

CARDS="${1:-1 2 3 4 5}"
PLOG=runs/execution_logs/colour_purify.log
LOG=runs/execution_logs/chain_colour_grid.log

echo "=== [$(date '+%F %T')] 等色彩族抗淨化收工 ===" | tee -a "$LOG"
# 兩個條件：抗淨化打出收工標記，且 `purify_identity.py` 歸零。
while ! grep -q "收工" "$PLOG" 2>/dev/null; do sleep 120; done
while [ "$(ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /purify_identity\.py/' | wc -l)" -gt 0 ]; do
  sleep 60
done
sleep 90

if ! bash scripts/free_cards.sh --assert "$CARDS"; then
  echo "=== [$(date '+%F %T')] 卡不是空的，不擠，結束 ===" | tee -a "$LOG"
  exit 3
fi
echo "=== [$(date '+%F %T')] 接手 $CARDS，跑 G 掃描 ===" | tee -a "$LOG"
exec bash scripts/colour_grid_round.sh "$CARDS"
