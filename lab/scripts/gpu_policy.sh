#!/usr/bin/env bash
# 預設值只在此定義。全局租約合計；明確授權以參數或 LAB_CAP 指定。
GPU_DEFAULT_CAP=6
LEASE=${LEASE:-$HOME/lab_leases}

gpu_valid_cap() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
gpu_global_cap() {
  local cap=$GPU_DEFAULT_CAP
  if [ -f "$LEASE/.capacity" ]; then
    read -r cap < "$LEASE/.capacity" || return 2
  fi
  gpu_valid_cap "$cap" || { echo "[FATAL] 全局卡數必須是正整數：$cap" >&2; return 2; }
  echo "$cap"
}
gpu_policy_init_locked() {
  local requested=$1
  [ "$requested" != default ] || requested=$GPU_DEFAULT_CAP
  if [ -n "$requested" ]; then
    gpu_valid_cap "$requested" || { echo "[FATAL] 授權卡數必須是正整數：$requested" >&2; return 2; }
    printf '%s\n' "$requested" > "$LEASE/.capacity.tmp" || return 1
    mv -- "$LEASE/.capacity.tmp" "$LEASE/.capacity" || return 1
  fi
  gpu_global_cap >/dev/null
}
gpu_policy_init() { lease_locked gpu_policy_init_locked "${1:-${LAB_CAP:-}}"; }
