#!/usr/bin/env bash
# 取一張空卡、登記租約、確認 CUDA 可用後執行一條指令；結束時釋放租約。
# 用法：bash lab/scripts/run_on_card.sh [--cap <全局授權|default>] [--limit <派工限制>] <名稱> <指令...>
# 取卡邏輯由 gpu_lease.sh 與所有排程共用。
set -uo pipefail
AUTHORIZED_CAP=""; LAUNCH_LIMIT=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --cap) AUTHORIZED_CAP=$2; shift 2 ;;
    --limit) LAUNCH_LIMIT=$2; shift 2 ;;
    *) echo "未知參數 $1" >&2; exit 2 ;;
  esac
done
NAME="$1"; shift
source ~/env.sh >/dev/null 2>&1
R=/nfs/home/nelson0314/image-immunization; cd "$R" || exit 1
source lab/scripts/gpu_lease.sh
gpu_policy_init "$AUTHORIZED_CAP" || exit $?
CAP=${LAUNCH_LIMIT:-$(gpu_global_cap)}
gpu_valid_cap "$CAP" || { echo "[FATAL] 派工限制必須是正整數" >&2; exit 2; }
GPU=""
for c in $(bash scripts/free_cards.sh 2>/dev/null); do
  lease_acquire "$c" "$NAME" "$CAP" || continue
  GPU=$c; break
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "[FATAL] 沒有空卡" >&2; exit 1; }
trap 'lease_release "$GPU"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$R PYTHONIOENCODING=utf-8
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] 卡 $GPU 上 CUDA 不可用" >&2; exit 1; }
echo "[CARD] $(hostname) gpu=$GPU" >&2
"$@"
