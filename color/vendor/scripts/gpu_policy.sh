#!/usr/bin/env bash
# 全局 GPU 卡數政策。預設值只在此定義；所有主機、session 與排程的租約合計。
# 明確授權以參數或環境變數 GPU_CAP 指定，參數優先；未指定時沿用租約目錄中
# 已記錄的授權值，沒有紀錄時為 6。
GPU_DEFAULT_CAP=6
# 租約目錄與既有排程共用；改名須所有取卡入口同時切換（第 10 項）。
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
gpu_policy_init() { lease_locked gpu_policy_init_locked "${1:-${GPU_CAP:-}}"; }
