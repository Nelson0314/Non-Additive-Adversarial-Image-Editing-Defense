#!/usr/bin/env bash
# 排程器：手上有幾張空卡就派幾個臂，沒有空卡就等。
#
# 為什麼要有它
# ────────────────────────────────────────────────────────────────────
# 卡是多人共用的，開工的時候常常一張都沒有。輪詢寫在遠端這一側，等待就不會
# 產生任何本機的往返；等到卡空出來的那一刻立刻派工，不需要有人守著。
#
# 全域五張的上限用 NFS 上的租約目錄實作（$LEASE），一張卡一個檔，檔名是
# `<主機>-<卡號>`。**兩台機器共用同一個 home，所以這個計數是跨機的。**
# 租約在鏈結束時由 wrapper 移除；wrapper 被砍掉時租約會留下，
# `--reap` 會把持有者已不存在的租約清掉。
#
# 用法：
#   bash lab/scripts/dispatch.sh <臂名> [臂名...]      # 前景，會一直等
#   nohup setsid bash lab/scripts/dispatch.sh ... > log 2>&1 < /dev/null &
set -uo pipefail
source ~/env.sh
R=/nfs/home/nelson0314/image-immunization
cd "$R" || { echo "[FATAL] 進不去 $R"; exit 1; }

HOST=$(hostname)
LEASE=$HOME/lab_leases
CAP=5
POLL=180
RETRIES=8
LOGDIR=lab/runs/logs
mkdir -p "$LEASE" "$LOGDIR" lab/runs/state

reap() {
  # 持有者已經不在了的租約要清掉，否則一次崩潰會永久吃掉一個名額。
  for f in "$LEASE"/*; do
    [ -e "$f" ] || continue
    read -r h p _ < "$f" 2>/dev/null || continue
    [ "$h" = "$HOST" ] || continue          # 別台的租約不能在這裡判定
    if ! ps -p "$p" > /dev/null 2>&1; then
      echo "[REAP] $(date -Is) $(basename "$f")（pid $p 已不在）"
      rm -f "$f"
    fi
  done
}

held() { ls -1 "$LEASE" 2>/dev/null | wc -l; }

# 鏈失敗要重試，而且要換一張卡。
# 實際踩過：`--assert` 通過之後、權重還沒載完的那二十秒裡，另一個使用者把
# 同一張卡佔走 19.65 GiB，我的工作 OOM 而死。那不是程式的錯，但排程器如果
# 不重試，那個臂就永遠停在那裡而 log 只有一行 rc=1。
run_arm() {
  local gpu="$1" arm="$2"
  local lease="$LEASE/${HOST}-${gpu}"
  (
    echo "$HOST $$ $arm" > "$lease"
    trap 'rm -f "$lease"' EXIT
    for try in $(seq 1 "$RETRIES"); do
      bash lab/scripts/arm_chain.sh "$gpu" "$arm" && exit 0
      echo "[RETRY] $(date -Is) arm=$arm try=$try 原卡=$gpu"
      rm -f "$lease"
      while true; do
        for c in $(bash scripts/free_cards.sh 2>/dev/null); do
          [ -e "$LEASE/${HOST}-${c}" ] && continue
          bash scripts/free_cards.sh --assert "$c" >/dev/null 2>&1 || continue
          gpu="$c"; lease="$LEASE/${HOST}-${gpu}"
          echo "$HOST $$ $arm" > "$lease"
          echo "[REGAIN] $(date -Is) arm=$arm 新卡=$gpu"
          break 2
        done
        sleep "$POLL"
      done
    done
    echo "[GIVE-UP] $(date -Is) arm=$arm 重試 $RETRIES 次仍失敗"
    exit 1
  ) > "$LOGDIR/${arm}.log" 2>&1 < /dev/null &
  echo "[LAUNCH] $(date -Is) arm=$arm gpu=$gpu host=$HOST pid=$!"
}

PENDING=("$@")
[ "${#PENDING[@]}" -eq 0 ] && { echo "沒有給臂名"; exit 2; }
echo "[DISPATCH-START] $(date -Is) host=$HOST 臂：${PENDING[*]}"

while [ "${#PENDING[@]}" -gt 0 ]; do
  reap
  n=$(held)
  if [ "$n" -ge "$CAP" ]; then
    echo "[WAIT] $(date -Is) 租約已滿（$n/$CAP）"
    sleep "$POLL"; continue
  fi
  # `free_cards.sh` 兩個方向都會判錯，所以派工前再 --assert 一次。
  FREE=$(bash scripts/free_cards.sh 2>/dev/null)
  LAUNCHED=0
  for c in $FREE; do
    [ -e "$LEASE/${HOST}-${c}" ] && continue
    [ "$(held)" -ge "$CAP" ] && break
    [ "${#PENDING[@]}" -eq 0 ] && break
    if ! bash scripts/free_cards.sh --assert "$c" >/dev/null 2>&1; then
      echo "[SKIP] 卡$c 複驗不空"; continue
    fi
    run_arm "$c" "${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    LAUNCHED=1
    sleep 20      # 兩個工作同時起來會同時看到同一張空卡
  done
  if [ "$LAUNCHED" -eq 0 ]; then
    echo "[WAIT] $(date -Is) 沒有空卡，剩 ${#PENDING[@]} 個臂：${PENDING[*]}"
    sleep "$POLL"
  fi
done

echo "[DISPATCH-ALL-LAUNCHED] $(date -Is) 等所有鏈結束"
while [ "$(ls -1 "$LEASE" 2>/dev/null | wc -l)" -gt 0 ]; do reap; sleep 60; done
echo "[DISPATCH-DONE] $(date -Is)"
