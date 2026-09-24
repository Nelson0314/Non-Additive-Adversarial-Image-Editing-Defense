#!/usr/bin/env bash
# 共用工作佇列的 worker。每台主機各跑一個，依租約目錄計五卡總數。
#
# 用法（遠端）：nohup setsid bash lab/scripts/queue_worker.sh <佇列名> <工作>... &
#
# 工作的寫法與相依：
#   pilot:<臂>:<影像>:<步數>   單張短步數試跑，輸出到 lab/runs/defence_pilot/<臂>
#   def:<臂>:<影像>            單張防禦圖，輸出到 lab/runs/defence_shards/<臂>/<影像>
#                              相依：同一臂的 pilot（若有列出）
#   chain:<臂>                 併分片（若有）後跑 arm_chain.sh（只跑 ip2p）
#                              相依：同一臂全部 def（若有列出）
#   gen:<臂>                   跑一次 defence_cmd（隨機對照一次產出所有 replicate），
#                              並為產滿 8 張的 <臂前綴>* 目錄寫 defence sentinel
#                              chain 工作相依於全部 gen
#   readout                    跨臂讀數。相依：全部 chain
#
# 狀態寫在 lab/runs/queue/<佇列名>/：<工作>.lock（mkdir，跨機原子）、
# <工作>.done、<工作>.fails。失敗的工作解鎖重試，同一工作失敗 3 次即停住，
# 寫 <工作>.GIVEUP，不再派相依於它的工作。
set -uo pipefail
QNAME="$1"; shift
JOBS=("$@")
source ~/env.sh
R=/nfs/home/nelson0314/image-immunization
cd "$R" || { echo "[FATAL] 進不去 $R" >&2; exit 1; }

HOST=$(hostname)
LEASE=$HOME/lab_leases
CAP=5          # 兩個 session 合計
MYCAP=4        # lab 自己最多四張，留一張給另一個 session
POLL=120
MAXFAIL=3
Q=lab/runs/queue/$QNAME
LOGDIR=lab/runs/logs/queue_$QNAME
mkdir -p "$LEASE" "$Q" "$LOGDIR" lab/runs/state

log() { echo "[$1] $(date -Is) host=$HOST ${*:2}" >&2; }

key() { echo "$1" | tr ':/' '__'; }
done_() { [ -e "$Q/$(key "$1").done" ]; }
dead() { [ -e "$Q/$(key "$1").GIVEUP" ]; }

reap() {
  for f in "$LEASE"/*; do
    [ -e "$f" ] || continue
    read -r h p _ < "$f" 2>/dev/null || continue
    [ "$h" = "$HOST" ] || continue
    ps -p "$p" > /dev/null 2>&1 || { log REAP "$(basename "$f") pid $p"; rm -f "$f"; }
  done
}
held() { ls -1 "$LEASE" 2>/dev/null | wc -l; }
mine() { grep -l " q_" "$LEASE"/* 2>/dev/null | wc -l; }
full() { [ "$(held)" -ge "$CAP" ] || [ "$(mine)" -ge "$MYCAP" ]; }

# 相依是否滿足：0＝可派、1＝還不能、2＝永遠不能（上游放棄）
ready() {
  local job="$1" kind arm j
  kind=${job%%:*}; arm=$(echo "$job" | cut -d: -f2)
  local deps=()
  for j in "${JOBS[@]}"; do
    case "$kind" in
      def)     [[ "$j" == pilot:"$arm":* ]] && deps+=("$j") ;;
      chain)   [[ "$j" == def:"$arm":* || "$j" == gen:* ]] && deps+=("$j") ;;
      readout) [[ "$j" == chain:* ]] && deps+=("$j") ;;
    esac
  done
  local r
  for j in "${deps[@]}"; do
    dead "$j" && return 2
    done_ "$j" && continue
    ready "$j"; r=$?
    [ "$r" -eq 2 ] && return 2
    return 1
  done
  return 0
}

merge_shards() {
  local arm="$1" src="lab/runs/defence_shards/$1" dst="lab/runs/defence/$1"
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  cp -f "$src"/*/*.png "$dst"/
  "$PY" - "$src" "$dst/results.csv" <<'EOF'
import csv, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/results.csv")):
    rows += list(csv.DictReader(open(f, encoding="utf-8")))
keys = sorted({k for r in rows for k in r})
with open(sys.argv[2], "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} rows", file=sys.stderr)
EOF
  local n; n=$(ls -1 "$dst"/*__"$arm"__def.png 2>/dev/null | wc -l)
  [ "$n" -eq 8 ] || { log FATAL "merge $arm 只有 $n 張防禦圖"; return 1; }
  touch "lab/runs/state/$arm.defence.done"
}

run_job() {
  local job="$1" gpu="$2" kind arm img steps
  kind=${job%%:*}; arm=$(echo "$job" | cut -d: -f2)
  img=$(echo "$job" | cut -d: -f3); steps=$(echo "$job" | cut -d: -f4)
  export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTHONPATH="$R" PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  [[ "$gpu" =~ ^[0-9]+$ ]] || { log FATAL "卡號不是純數字：$gpu"; return 1; }
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { log FATAL "卡 $gpu 上 torch.cuda.is_available() 為假"; return 1; }
  case "$kind" in
    pilot) DEF_OUT="lab/runs/defence_pilot/$arm" \
             bash lab/scripts/defence_cmd.sh "$arm" --images "$img" --steps "$steps" ;;
    def)   DEF_OUT="lab/runs/defence_shards/$arm/$img" \
             bash lab/scripts/defence_cmd.sh "$arm" --images "$img" ;;
    gen)   bash lab/scripts/defence_cmd.sh "$arm" || return 1
           local pre=${arm%_r[0-9]*} d n
           for d in lab/runs/defence/"$pre"_r*; do
             n=$(ls -1 "$d"/*__"$(basename "$d")"__def.png 2>/dev/null | wc -l)
             [ "$n" -eq 8 ] && touch "lab/runs/state/$(basename "$d").defence.done"
           done ;;
    chain) merge_shards "$arm" && bash lab/scripts/arm_chain.sh "$gpu" "$arm" ;;
    readout) bash lab/scripts/readout.sh "$gpu" ;;
    *) log FATAL "未知的工作：$job"; return 2 ;;
  esac
}

launch() {
  local job="$1" gpu="$2" k; k=$(key "$job")
  local lease="$LEASE/${HOST}-${gpu}"
  echo "$HOST $$ q_$QNAME" > "$lease"
  (
    trap 'rm -f "$lease"' EXIT
    echo "$HOST $BASHPID q_$QNAME" > "$lease"
    log START "$job gpu=$gpu"
    run_job "$job" "$gpu" > "$LOGDIR/$k.log" 2>&1 < /dev/null
    rc=$?
    log EXIT "$job gpu=$gpu rc=$rc"
    if [ "$rc" -eq 0 ]; then touch "$Q/$k.done"
    else
      echo x >> "$Q/$k.fails"
      if [ "$(wc -l < "$Q/$k.fails")" -ge "$MAXFAIL" ]; then
        touch "$Q/$k.GIVEUP"; log GIVE-UP "$job"
      fi
    fi
    rmdir "$Q/$k.lock"
  ) &
  sleep 20
}

log WORKER-START "佇列 $QNAME 工作 ${#JOBS[@]} 個"
while true; do
  reap
  remaining=0; runnable=()
  for j in "${JOBS[@]}"; do
    done_ "$j" && continue
    dead "$j" && continue
    ready "$j"; r=$?
    [ "$r" -eq 2 ] && continue
    remaining=$((remaining + 1))
    [ "$r" -eq 0 ] && [ ! -d "$Q/$(key "$j").lock" ] && runnable+=("$j")
  done
  if [ "$remaining" -eq 0 ]; then log WORKER-DONE "佇列清空或上游放棄"; break; fi
  if [ "${#runnable[@]}" -eq 0 ] || full; then sleep "$POLL"; continue; fi
  for c in $(bash scripts/free_cards.sh 2>/dev/null); do
    [ "${#runnable[@]}" -eq 0 ] && break
    full && break
    [ -e "$LEASE/${HOST}-${c}" ] && continue
    bash scripts/free_cards.sh --assert "$c" >/dev/null 2>&1 || continue
    # 別人的 compute app 不在記憶體門檻裡：有任何不是本使用者的 pid 就跳過
    uuid=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader)
    others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v u="$uuid" '$1==u{print $2}' | wc -l)
    [ "$others" -eq 0 ] || continue
    job="${runnable[0]}"
    mkdir "$Q/$(key "$job").lock" 2>/dev/null || { runnable=("${runnable[@]:1}"); continue; }
    launch "$job" "$c"
    runnable=("${runnable[@]:1}")
  done
  sleep "$POLL"
done
wait
log WORKER-EXIT "全部子工作結束"
