#!/usr/bin/env bash
# 取一張空卡、登記租約、確認 CUDA 可用後執行一條指令；結束時釋放租約。
# 用法：bash lab/scripts/run_on_card.sh <名稱> <指令...>
# 取卡邏輯與 queue_worker.sh 相同：free_cards 放行、租約目錄沒有、別人的記憶體 < 1 GB。
set -uo pipefail
NAME="$1"; shift
source ~/env.sh >/dev/null 2>&1
R=/nfs/home/nelson0314/image-immunization; cd "$R" || exit 1
LEASE=$HOME/lab_leases; mkdir -p "$LEASE"
GPU=""
for c in $(bash scripts/free_cards.sh 2>/dev/null); do
  [ -e "$LEASE/$(hostname)-$c" ] && continue
  bash scripts/free_cards.sh --assert "$c" >/dev/null 2>&1 || continue
  uuid=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader)
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits \
           | awk -F', ' -v u="$uuid" '$1==u{s+=$2} END{print s+0}')
  [ "$others" -lt 1024 ] || continue
  GPU=$c; break
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "[FATAL] 沒有空卡" >&2; exit 1; }
echo "$(hostname) $$ $NAME" > "$LEASE/$(hostname)-$GPU"
trap 'rm -f "$LEASE/$(hostname)-$GPU"' EXIT
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$R PYTHONIOENCODING=utf-8
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] 卡 $GPU 上 CUDA 不可用" >&2; exit 1; }
echo "[CARD] $(hostname) gpu=$GPU" >&2
"$@"
