#!/usr/bin/env bash
# 八個臂依序跑完。**一次只送一個臂**，因為十張影像正好用滿「五張卡 × 每卡兩個
# process」；同時送兩個臂會讓每張卡變成四個 process。
#
# 順序是刻意的：三個不最佳化的臂先跑。它們一格約一分鐘，十五分鐘之內就能回答
# 「那六千步到底買到什麼」——若隨機打平，後面五個臂的前提就要重寫，先跑它們
# 等於先付五小時再發現問題。
#
# 用法：bash scripts/content_constraint_queue.sh "<卡號…>"
#       送出後可以直接斷線（腳本自己會 setsid nohup）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

CARDS="${1:-}"
[ -z "$CARDS" ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }

# 便宜的先，貴的後。**變數名不可用 GROUPS／UID／RANDOM 等 bash 內建**
# （`docs/DEFECTS.md`：指派給 GROUPS 會被靜默忽略，展開成群組 ID）。
CHEAP_ARMS="rand bg_rand colorgrid_rand"
COSTLY_ARMS="lowproj chroma tint colorgrid bg"

# **只數 python，不數外殼**（見 `docs/DEFECTS.md`）：每一格是
# `bash -c "python ip2p_run.py …"`，外殼的指令字串裡同樣含 `ip2p_run.py`，
# `grep -c` 會把一格算成兩個，並行度變成設定值的一半。
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

wait_idle() {
  # 送出後 `content_constraint_round.sh` 自己會 sleep 25 再回來，故這裡直接
  # 開始輪詢。**比對腳本路徑而不是關鍵字**（`docs/OPERATIONS.md`）。
  local n
  while :; do
    n=$(running)
    [ "$n" -eq 0 ] && break
    sleep 60
  done
}

for arm in $CHEAP_ARMS $COSTLY_ARMS; do
  echo "=== [$(date '+%F %T')] 送出 $arm ==="
  # 每一個臂送出前重新確認卡是空的：前一個臂剛結束，而別人可能在這個空檔
  # 排了工作進來。**印了不擋等於沒擋**，故用 --assert（回傳 3 就中止整個佇列）。
  bash scripts/free_cards.sh --assert "$CARDS" || {
    echo "卡不再是空的，佇列停在 $arm 之前。" >&2; exit 3; }
  bash scripts/content_constraint_round.sh "$CARDS" "$arm" || {
    echo "送出 $arm 失敗，佇列中止。" >&2; exit 4; }
  wait_idle
  echo "=== [$(date '+%F %T')] $arm 收工，目錄數 $(ls -d runs/ip2p_content_constraint/${arm}_* 2>/dev/null | wc -l) ==="
done

echo "=== [$(date '+%F %T')] 八個臂全部收工 ==="
ls -d runs/ip2p_content_constraint/*/ 2>/dev/null | wc -l
