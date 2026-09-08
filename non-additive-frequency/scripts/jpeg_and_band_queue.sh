#!/usr/bin/env bash
# 兩批依序跑完：量化交付的 QD 掃描，然後帶限參數化的 S 掃描。
#
# **一次只送一批**，因為每一批自己就會把「五張卡 × 每卡兩個 process」用滿；
# 同時送兩批會讓每張卡變成四個 process。
#
# 順序是刻意的，兩個理由
# ────────────────────────────────────────────────────────────────────
# 1. 量化交付騎在一個**已經驗證過的機制**上（相位族：jpeg75 由 0.362 到
#    0.493、blur 由 0.118 到 0.188、人眼 0/13 到 5/13），只是此前被守門擋在
#    補丁族之外。帶限參數化是新的東西，先驗證的先跑。
# 2. 兩批共用同一組影像與同一個 `free` 起點，任何一批先跑完都能單獨解讀。
#
# 每一批送出前重新確認卡是空的：前一批剛結束，別人可能在這個空檔排進來。
# **印了不擋等於沒擋**，故用 `--assert`（回傳 3 就中止整個佇列）。
#
# 用法：bash scripts/jpeg_and_band_queue.sh "<卡號…>"
#       送出後可以直接斷線（每一批自己會 setsid nohup）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

CARDS="${1:-}"
[ -z "$CARDS" ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }

LOG=runs/execution_logs/jpeg_and_band_queue.log
mkdir -p runs/execution_logs

# **變數名不可用 GROUPS／UID／RANDOM 等 bash 內建**（`docs/DEFECTS.md`：
# 指派給 GROUPS 會被靜默忽略，展開成群組 ID）。
BATCH_SCRIPTS="deliver_jpeg_patch_round patch_res_round"

running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

wait_idle() {
  while [ "$(running)" -gt 0 ]; do sleep 60; done
}

for s in $BATCH_SCRIPTS; do
  echo "=== [$(date '+%F %T')] $s 燒測 ===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "$CARDS" || {
    echo "卡不再是空的，佇列停在 $s 之前。" | tee -a "$LOG" >&2; exit 3; }
  # 燒測失敗就**中止整個佇列**而不是跳過這一批：兩批共用同一條程式路徑
  # （補丁載體 ＋ image_guidance），一批燒不過表示另一批多半也燒不過，
  # 繼續送只會把卡佔著跑出一批沒有人要的東西。
  bash "scripts/$s.sh" "$CARDS" smoke || {
    echo "$s 燒測失敗，佇列中止。" | tee -a "$LOG" >&2; exit 4; }
  wait_idle
  echo "=== [$(date '+%F %T')] $s 正式送出 ===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "$CARDS" || {
    echo "卡不再是空的，佇列停在 $s 正式批之前。" | tee -a "$LOG" >&2; exit 3; }
  bash "scripts/$s.sh" "$CARDS" || {
    echo "送出 $s 失敗，佇列中止。" | tee -a "$LOG" >&2; exit 5; }
  wait_idle
  echo "=== [$(date '+%F %T')] $s 收工 ===" | tee -a "$LOG"
done

echo "=== [$(date '+%F %T')] 兩批全部收工 ===" | tee -a "$LOG"
for d in runs/ip2p_deliver_jpeg_patch runs/ip2p_patch_res; do
  echo "$d: $(find "$d" -name results.csv 2>/dev/null | wc -l) 份 results.csv" | tee -a "$LOG"
done
