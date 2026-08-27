#!/usr/bin/env bash
# 等分階段訓練那一批（`runs/ip2p_stage2/`）收工，再送整併版的掃描。
#
# 為什麼要一支排隊腳本
# ────────────────────────────────────────────────────────────────────
# 整台機器只有兩張卡是空的，每卡最多 2 個 process（第 3 個實測會 OOM，
# `docs/OPERATIONS.md`）。分階段訓練的防禦與抗淨化正佔著這四個槽，整併版的
# 四個工作點必須等它們讓出來。**不可以先送再讓兩批互相搶**——別人的卡一張
# 都不能碰，自己的卡擠爆了一樣是重跑。
#
# 收工的判準有兩個，缺一不可
# ────────────────────────────────────────────────────────────────────
#   1. 三個抗淨化的 CSV 都寫出來了（`runs/ip2p_stage2/purify/*.csv`）。
#   2. 自己名下沒有任何 `ip2p_run.py` 或 `phase_retention.py` 還在跑。
#
# 只看第一條不夠：CSV 是逐列寫的，最後一個 process 可能還在收尾。只看第二條
# 也不夠：抗淨化那一輪還沒被送出去時，process 數本來就是零。
#
# 逾時不硬送
# ────────────────────────────────────────────────────────────────────
# 等超過 `MAX_WAIT_MIN` 分鐘就**放棄並在 log 裡說明**，不改判準硬送。前一批
# 若是掛掉而不是跑完，硬送只會把兩個問題疊在一起。
#
# 用法（一律以 nohup 背景執行）：
#   nohup bash scripts/dct_unified_queue.sh "<卡號>" > runs/ip2p_dct_unified/queue.log 2>&1 &
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT"

DEVS="${1:-}"
[ -z "$DEVS" ] && { echo "用法：$0 \"<卡號>\"" >&2; exit 2; }
MAX_WAIT_MIN="${MAX_WAIT_MIN:-360}"
POLL_SEC="${POLL_SEC:-120}"

mkdir -p runs/ip2p_dct_unified
NEED="s2_tight s2_hard s2_null"

waited=0
while :; do
  ready=1
  for t in $NEED; do
    [ -s "runs/ip2p_stage2/purify/$t.csv" ] || ready=0
  done
  busy=$(pgrep -u "$(id -u)" -f '[i]p2p_run\.py|[p]hase_retention\.py' | wc -l)
  if [ "$ready" -eq 1 ] && [ "$busy" -eq 0 ]; then
    echo "[queue] 分階段訓練那一批收工（等了 ${waited} 分鐘），開始送整併版"
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "[queue] 等了 ${waited} 分鐘仍未收工（csv 齊=$ready、在跑的 process=$busy）。" >&2
    echo "[queue] **不硬送**：前一批若是掛掉而不是跑完，硬送會把兩個問題疊在一起。" >&2
    exit 4
  fi
  sleep "$POLL_SEC"
  waited=$(( waited + POLL_SEC / 60 ))
done

# 派工腳本自己會再跑一次 `free_cards.sh --assert`，那一道是強制的：
# 等待期間別人可能已經把卡接走。
exec bash scripts/dct_unified_sweep.sh "$DEVS"
