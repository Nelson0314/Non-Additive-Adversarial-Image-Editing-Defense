#!/usr/bin/env bash
# 等 `jpeg_and_band_queue.sh` 兩批收工，接手同一組卡量它們的抗淨化。
#
# 為什麼要接這一段：那兩批的存在理由都在淨化之後
# （見 `scripts/jpeg_band_purify_round.sh` 的檔頭）。只跑派工不跑這一段，
# 兩批等於沒有回答任何問題。
#
# 兩個等待條件（缺一不可）：
#   1. 佇列打出「兩批全部收工」——只看 process 歸零不夠，批與批之間也會歸零；
#   2. `ip2p_run.py` 歸零之後再等 90 秒，GPU 記憶體才真的釋出。
#
# 用法：bash scripts/chain_jpeg_band_purify.sh ["<卡號…>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || exit 2

CARDS="${1:-1 2 3 4 5}"
QLOG=runs/execution_logs/jpeg_and_band_queue.log
LOG=runs/execution_logs/chain_jpeg_band_purify.log
DONE_MARK="兩批全部收工"
mkdir -p runs/execution_logs

echo "=== [$(date '+%F %T')] 等 jpeg_and_band_queue 收工 ===" | tee -a "$LOG"
while ! grep -q "$DONE_MARK" "$QLOG" 2>/dev/null; do
  # 佇列自己在燒測失敗或卡被別人佔走時會中止。**那時要跟著結束**，
  # 否則這一支會在這裡等到 session 結束，而看起來像是還在跑。
  if grep -qE "佇列中止|佇列停在" "$QLOG" 2>/dev/null; then
    echo "=== [$(date '+%F %T')] 佇列中止，抗淨化不接手 ===" | tee -a "$LOG"
    exit 4
  fi
  sleep 120
done

echo "=== [$(date '+%F %T')] 標記出現，等 process 歸零 ===" | tee -a "$LOG"
while [ "$(ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l)" -gt 0 ]; do
  sleep 60
done
sleep 90

for which in deliver_jpeg patch_res; do
  if ! bash scripts/free_cards.sh --assert "$CARDS"; then
    echo "=== [$(date '+%F %T')] 卡不是空的（別人進來了），不擠，停在 $which 之前 ===" | tee -a "$LOG"
    exit 3
  fi
  echo "=== [$(date '+%F %T')] 接手 $CARDS，量 $which 的抗淨化 ===" | tee -a "$LOG"
  bash scripts/jpeg_band_purify_round.sh "$CARDS" "$which" || {
    echo "=== [$(date '+%F %T')] $which 抗淨化失敗，中止 ===" | tee -a "$LOG"; exit 5; }
done
echo "=== [$(date '+%F %T')] 兩批的抗淨化全部收工 ===" | tee -a "$LOG"
