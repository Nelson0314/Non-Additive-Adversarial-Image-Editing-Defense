#!/usr/bin/env bash
# 共用工作佇列的 worker。取卡經 gpu_lease.sh，容量政策見 gpu_policy.sh。
#
# 用法：bash queue_worker.sh --work-dir <目錄> --state-dir <目錄> --log-dir <目錄>
#           --runner <指令> --validator <指令> [--depends <指令>] [--env-file <檔案>]
#           <佇列名> <工作>...
#
# 工作語法與相依由專案注入：
#   --runner <指令>     以 `<指令> <工作> <卡號>` 執行一件工作；結束碼 0 表示執行完成。
#   --validator <指令>  以 `<指令> <工作>` 驗收輸出；結束碼 0 才寫入 .done，
#                       既有 .done 亦須重新通過驗收才算完成。
#   --depends <指令>    以 `<指令> <工作> <全部工作>...` 每行印出一件佇列內的上游
#                       工作；結束碼 0 表示佇列外的前置條件已滿足、1 表示尚未
#                       滿足，其他值使 worker 中止。未提供時沒有相依。
#
# 狀態寫在 --state-dir：<工作>.lock（mkdir，跨機原子）、<工作>.done、<工作>.fails。
# 失敗的工作解鎖重試；同一工作失敗 MAXFAIL 次寫 <工作>.GIVEUP，不再派送相依於
# 它的工作。佇列結束時任一工作未完成或未通過驗收，worker 回傳 1。
# 環境變數 QUEUE_CAP 可另外降低本類佇列的合計租約數，不超過全局授權；POLL 與
# LAUNCH_GAP 為輪詢與相鄰派工的間隔秒數（預設 120、20）。
set -uo pipefail
WORKDIR=""; STATE=""; LOGDIR=""; RUNNER=""; VALIDATOR=""; DEPENDS=""; ENV_FILE=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --work-dir) WORKDIR=$2; shift 2 ;;
    --state-dir) STATE=$2; shift 2 ;;
    --log-dir) LOGDIR=$2; shift 2 ;;
    --runner) RUNNER=$2; shift 2 ;;
    --validator) VALIDATOR=$2; shift 2 ;;
    --depends) DEPENDS=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    *) echo "未知參數 $1" >&2; exit 2 ;;
  esac
done
for required in work-dir:WORKDIR state-dir:STATE log-dir:LOGDIR runner:RUNNER validator:VALIDATOR; do
  variable=${required#*:}
  [ -n "${!variable}" ] || { echo "[FATAL] 缺少必要參數 --${required%%:*}" >&2; exit 2; }
done
[ $# -ge 2 ] || { echo "[FATAL] 需要佇列名與至少一件工作" >&2; exit 2; }
QNAME="$1"; shift
JOBS=("$@")
if [ -n "$ENV_FILE" ]; then
  source "$ENV_FILE" || { echo "[FATAL] 無法載入環境檔 $ENV_FILE" >&2; exit 1; }
fi
PY=${PY:-python}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$WORKDIR" || { echo "[FATAL] 無法進入 $WORKDIR" >&2; exit 1; }

HOST=$(hostname)
source "$SCRIPTS/gpu_lease.sh"
gpu_policy_init || exit $?
MYCAP=${QUEUE_CAP:-$(gpu_global_cap)}
gpu_valid_cap "$MYCAP" || { echo "[FATAL] QUEUE_CAP 必須是正整數" >&2; exit 2; }
POLL=${POLL:-120}
LAUNCH_GAP=${LAUNCH_GAP:-20}
MAXFAIL=${MAXFAIL:-3}
Q=$STATE
mkdir -p "$LEASE" "$Q" "$LOGDIR"

log() { echo "[$1] $(date -Is) host=$HOST ${*:2}" >&2; }

key() { echo "$1" | tr ':/' '__'; }
validate_job() { "$VALIDATOR" "$1"; }
done_() { [ -e "$Q/$(key "$1").done" ] && validate_job "$1" >/dev/null 2>&1; }
dead() { [ -e "$Q/$(key "$1").GIVEUP" ]; }

reap() { lease_reap; }
held() { lease_count; }
mine() { lease_group_count q_; }
full() { [ "$(held)" -ge "$(gpu_global_cap)" ] || [ "$(mine)" -ge "$MYCAP" ]; }

# 相依是否滿足：0＝可派、1＝還不能、2＝永遠不能（上游放棄）
ready() {
  local job="$1" j deps=() external=0 listed
  if [ -n "$DEPENDS" ]; then
    listed=$("$DEPENDS" "$job" "${JOBS[@]}")
    external=$?
    if [ "$external" -gt 1 ]; then
      log FATAL "相依指令對 $job 回傳 $external"; exit 2
    fi
    while IFS= read -r j; do [ -n "$j" ] && deps+=("$j"); done <<< "$listed"
  fi
  local r
  for j in "${deps[@]}"; do
    dead "$j" && return 2
    done_ "$j" && continue
    ready "$j"; r=$?
    [ "$r" -eq 2 ] && return 2
    return 1
  done
  return "$external"
}

run_job() {
  local job="$1" gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  [[ "$gpu" =~ ^[0-9]+$ ]] || { log FATAL "卡號不是純數字：$gpu"; return 1; }
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { log FATAL "卡 $gpu 上 torch.cuda.is_available() 為假"; return 1; }
  "$RUNNER" "$job" "$gpu"
}

launch() {
  local job="$1" gpu="$2" k; k=$(key "$job")
  (
    lease_acquire "$gpu" "q_$QNAME" "$(gpu_global_cap)" q_ "$MYCAP" || {
      rmdir "$Q/$k.lock"; exit 4;
    }
    trap 'lease_release "$gpu"; rmdir "$Q/$k.lock"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    log START "$job gpu=$gpu"
    run_job "$job" "$gpu" > "$LOGDIR/$k.log" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ]; then
      validate_job "$job" >> "$LOGDIR/$k.log" 2>&1
      rc=$?
    fi
    log EXIT "$job gpu=$gpu rc=$rc"
    if [ "$rc" -eq 0 ]; then touch "$Q/$k.done"
    else
      echo x >> "$Q/$k.fails"
      if [ "$(wc -l < "$Q/$k.fails")" -ge "$MAXFAIL" ]; then
        touch "$Q/$k.GIVEUP"; log GIVE-UP "$job"
      fi
    fi
  ) &
  sleep "$LAUNCH_GAP"
}

main() {
  log WORKER-START "佇列 $QNAME 工作 ${#JOBS[@]} 個"
  local remaining runnable j r c job
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
    for c in $(bash "$SCRIPTS/free_cards.sh" 2>/dev/null); do
      [ "${#runnable[@]}" -eq 0 ] && break
      full && break
      job="${runnable[0]}"
      runnable=("${runnable[@]:1}")
      mkdir "$Q/$(key "$job").lock" 2>/dev/null || continue
      # 掃描時的完成判定與鎖的檢查不是同一時刻：工作可能在兩者之間（例如相依檢查執行中）
      # 完成並釋放鎖。執行端先寫 .done／.GIVEUP 再釋放鎖，故取得鎖後重新判定即可排除重派。
      if done_ "$job" || dead "$job"; then
        rmdir "$Q/$(key "$job").lock"
        continue
      fi
      launch "$job" "$c"
    done
    sleep "$POLL"
  done
  wait
  log WORKER-EXIT "全部子工作結束"
  for job in "${JOBS[@]}"; do
    done_ "$job" || { log FATAL "工作未完成或驗收失敗：$job"; exit 1; }
  done
}

main
