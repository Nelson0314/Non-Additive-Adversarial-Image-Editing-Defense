#!/usr/bin/env bash
# 共用 NFS 租約；檢查、取卡、擁有者驗證與釋放都在同一個 mkdir 鎖內。
# 租約目錄 LEASE 的預設值見 gpu_policy.sh。
LEASE_HOST=${LEASE_HOST:-$(hostname)}
GPU_SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$GPU_SCRIPTS/gpu_policy.sh"

lease_locked() (
  mkdir -p "$LEASE" || exit 1
  local attempt
  for ((attempt=0; attempt<100; attempt++)); do
    if mkdir "$LEASE/.guard" 2>/dev/null; then
      trap 'rmdir "$LEASE/.guard"' EXIT
      trap 'exit 130' INT
      trap 'exit 143' TERM
      "$@"
      exit $?
    fi
    sleep 0.1
  done
  echo "[FATAL] 租約鎖未釋放：$LEASE/.guard；請確認持有程序後處理" >&2
  exit 1
)

lease_reap_locked() {
  local file host pid name token
  for file in "$LEASE"/*; do
    [ -f "$file" ] || continue
    read -r host pid name token < "$file" || continue
    [ "$host" = "$LEASE_HOST" ] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -0 "$pid" 2>/dev/null || {
      # ps 可看見但無權傳 signal 的程序仍視為存活。
      ps -p "$pid" >/dev/null 2>&1 || rm -f -- "$file"
    }
  done
}

lease_reap() { lease_locked lease_reap_locked; }
lease_count() {
  local file count=0
  for file in "$LEASE"/*; do [ -f "$file" ] && count=$((count + 1)); done
  echo "$count"
}
lease_group_count() {
  local file host pid name token count=0
  for file in "$LEASE"/*; do
    [ -f "$file" ] || continue
    read -r host pid name token < "$file" || continue
    [[ "$name" == "$1"* ]] && count=$((count + 1))
  done
  echo "$count"
}
lease_running() {
  local file host pid name token
  for file in "$LEASE"/*; do
    [ -f "$file" ] || continue
    read -r host pid name token < "$file" || continue
    [ "$name" = "$1" ] && return 0
  done
  return 1
}

lease_card_available() {
  local gpu=$1 uuid apps others
  bash "$GPU_SCRIPTS/measure_free_gpus.sh" --assert "$gpu" >/dev/null 2>&1 || return 1
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader) || return 1
  [ -n "$uuid" ] || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits) || return 1
  others=$(printf '%s\n' "$apps" | awk -F', ' -v u="$uuid" '$1==u{s+=$2} END{print s+0}')
  [ "$others" -lt 1024 ]
}

lease_acquire_locked() {
  local gpu=$1 name=$2 owner=$3 token=$4 cap=$5 prefix=$6 group_cap=$7
  lease_reap_locked
  local global_cap
  global_cap=$(gpu_global_cap) || return $?
  [ "$cap" -le "$global_cap" ] || cap=$global_cap
  [ ! -e "$LEASE/$LEASE_HOST-$gpu" ] || return 4
  [ "$(lease_count)" -lt "$cap" ] || return 4
  if [ -n "$prefix" ]; then
    [ "$(lease_group_count "$prefix")" -lt "$group_cap" ] || return 4
  fi
  lease_card_available "$gpu" || return 4
  # noclobber 同時拒絕未採用共用協定的舊 writer 已建立的租約。
  (set -o noclobber; printf '%s %s %s %s\n' "$LEASE_HOST" "$owner" "$name" "$token" \
      > "$LEASE/$LEASE_HOST-$gpu") || return 4
}
lease_acquire() {
  local gpu=$1 name=$2 cap=$3 prefix=${4:-} group_cap=${5:-$3} owner=$BASHPID
  [[ "$gpu" =~ ^[0-9]+$ && "$cap" =~ ^[1-9][0-9]*$ && "$group_cap" =~ ^[1-9][0-9]*$ ]] || return 2
  [[ "$name" != *[[:space:]]* && -n "$name" ]] || return 2
  LEASE_TOKEN="$LEASE_HOST-$owner-$RANDOM-$RANDOM"
  lease_locked lease_acquire_locked "$gpu" "$name" "$owner" "$LEASE_TOKEN" "$cap" "$prefix" "$group_cap"
}

lease_release_locked() {
  local gpu=$1 token=$2 host pid name recorded file="$LEASE/$LEASE_HOST-$1"
  [ -f "$file" ] || return 0
  read -r host pid name recorded < "$file" || return 1
  [ "$recorded" = "$token" ] && [ -n "$token" ] || return 1
  rm -f -- "$file"
}
lease_release() { lease_locked lease_release_locked "$1" "${2:-${LEASE_TOKEN:-}}"; }
