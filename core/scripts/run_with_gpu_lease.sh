#!/usr/bin/env bash
# 取得一張空卡的租約、確認 CUDA 可用後執行一條指令；結束時釋放租約。
#
# 用法：bash run_with_gpu_lease.sh --work-dir <目錄> [--env-file <檔案>]
#           [--cap <全局授權|default>] [--limit <派工限制>] <租約名稱> <指令...>
#
# --work-dir  執行指令的工作目錄（必填）。
# --env-file      先 source 的環境檔（例如設定 PY 與 PYTHONPATH）；source 失敗即中止。
# --cap      全局授權卡數，寫入共用租約目錄；未給時依 GPU_CAP 或既有紀錄。
# --limit    單次派工計入的租約上限，不超過全局授權。
# CUDA 檢查使用 $PY，未設定時為 python。
set -uo pipefail
AUTHORIZED_CAP=""; LAUNCH_LIMIT=""; WORKDIR=""; ENV_FILE=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --cap) AUTHORIZED_CAP=$2; shift 2 ;;
    --limit) LAUNCH_LIMIT=$2; shift 2 ;;
    --work-dir) WORKDIR=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    *) echo "未知參數 $1" >&2; exit 2 ;;
  esac
done
[ -n "$WORKDIR" ] || { echo "[FATAL] 必須指定 --work-dir" >&2; exit 2; }
[ $# -ge 2 ] || { echo "[FATAL] 需要租約名稱與指令" >&2; exit 2; }
NAME="$1"; shift
if [ -n "$ENV_FILE" ]; then
  source "$ENV_FILE" || { echo "[FATAL] 無法載入環境檔 $ENV_FILE" >&2; exit 1; }
fi
PY=${PY:-python}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$WORKDIR" || { echo "[FATAL] 無法進入 $WORKDIR" >&2; exit 1; }
source "$SCRIPTS/gpu_lease.sh"
gpu_policy_init "$AUTHORIZED_CAP" || exit $?
CAP=${LAUNCH_LIMIT:-$(gpu_global_cap)}
gpu_valid_cap "$CAP" || { echo "[FATAL] 派工限制必須是正整數" >&2; exit 2; }
GPU=""
for c in $(bash "$SCRIPTS/free_cards.sh" 2>/dev/null); do
  lease_acquire "$c" "$NAME" "$CAP" || continue
  GPU=$c; break
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "[FATAL] 沒有空卡" >&2; exit 1; }
trap 'lease_release "$GPU"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONIOENCODING=utf-8
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] 卡 $GPU 上 CUDA 不可用" >&2; exit 1; }
echo "[CARD] $(hostname) gpu=$GPU" >&2
"$@"
