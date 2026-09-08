#!/usr/bin/env bash
# 等色彩預算掃描收工，接手同一批卡量它的抗淨化。
#
# 為什麼要接這一段
#   `docs/DIRECTION.md` §3.1：補丁族與色彩族的抗淨化形狀互補，但那個比較目前
#   踩在兩個不同的協定上——補丁族是「衣物載體、身分讀數、十二個算子」
#   （`runs/ip2p_purify_identity`），色彩族是「全圖、半徑 0.10、舊讀數」
#   （`runs/ip2p_color_capacity_purify`）。**兩張表不可並列。**
#
#   這一支用**同一支 `purify_identity.py`、同一組十二個算子、同一個空白地板**
#   量新的色彩工作點，讓互補性這個主張有一張可並列的表。
#
# 兩個等待條件（缺一不可，理由同 `chain_purify_after_loss.sh`）：
#   1. 色彩掃描打出「收工」標記——只看 process 歸零不夠，臂與臂之間也會歸零
#   2. `ip2p_run.py` 歸零之後再等 90 秒，GPU 記憶體才真的釋出
#
# 用法：bash scripts/chain_colour_purify.sh ["<卡號…>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || exit 2

CARDS="${1:-1 2 3 4 5}"
CLOG=runs/execution_logs/colour_budget.log
LOG=runs/execution_logs/chain_colour_purify.log
DONE_MARK="收工，results"

echo "=== [$(date '+%F %T')] 等色彩預算掃描收工 ===" | tee -a "$LOG"
while ! grep -q "$DONE_MARK" "$CLOG" 2>/dev/null; do sleep 120; done
echo "=== [$(date '+%F %T')] 標記出現，等 process 歸零 ===" | tee -a "$LOG"
while [ "$(ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l)" -gt 0 ]; do
  sleep 60
done
sleep 90

if ! bash scripts/free_cards.sh --assert "$CARDS"; then
  echo "=== [$(date '+%F %T')] 卡不是空的（別人進來了），不擠，結束 ===" | tee -a "$LOG"
  exit 3
fi
echo "=== [$(date '+%F %T')] 接手 $CARDS，量色彩族的抗淨化 ===" | tee -a "$LOG"
exec bash scripts/colour_purify_round.sh "$CARDS"
