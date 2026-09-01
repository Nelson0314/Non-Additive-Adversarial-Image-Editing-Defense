#!/usr/bin/env bash
# 公平比較 · 階段 A（校準曲線）· DCT-Shield 側，卡 0–3。
#
# 一張要 575 s（1000 步 PGD），四個強度共 100 張份，故每個強度切三片、
# 12 個 process 分四張卡。強度的取法：
#
#   ε=1／Q_alg=0.95  論文正文 §5.4 的原生工作點，公平比較的錨點
#   ε=2, ε=3         往上，仍滿足 §4.2 的 ε≥1 抗 JPEG 條件
#   ε=1／Q_alg=0.98  往下。**改的是量化步長不是 ε**，故抗 JPEG 條件仍成立；
#                    論文自己也按場景換 Q_alg（inpainting 0.9、JPEG 圖 0.85），
#                    是該方法既有的旋鈕，不是我們發明的
#
# 曲線要涵蓋錨點兩側才能內插；只往上會變成外插，`tradeoff_curve.py` 會拒絕。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a_dct
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

while ps -u $USER -o cmd | grep "[i]p2p_run" | grep -q theta0820; do sleep 30; done

mapfile -t ALL < runs/fair0820/images25.txt
[ "${#ALL[@]}" -eq 25 ] || { log "影像清單是 ${#ALL[@]} 張，不是 25，中止"; exit 1; }
log "階段 A DCT 側開始，${#ALL[@]} 張"

shard() {   # shard <k> —— 取第 k 片（共 3 片）
  local k=$1 i=0
  for n in "${ALL[@]}"; do
    [ $((i % 3)) -eq "$k" ] && printf '%s ' "$n"
    i=$((i+1))
  done
}

# tag  eps  q_alg
SPECS="e10_q95 1 0.95
e20_q95 2 0.95
e30_q95 3 0.95
e10_q98 1 0.98"

g=0
while read -r tag eps q; do
  [ -n "${tag:-}" ] || continue
  for k in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/ip2p_run.py \
      --out "$ROOT/${tag}_$k" --data data/omniedit150 --images $(shard $k) \
      --conditions dct_shield --eps "$eps" --q-alg "$q" \
      > "$ROOT/${tag}_$k.log" 2>&1 &
  done
  g=$(((g+1) % 4))
done <<< "$SPECS"
wait
for t in e10_q95 e20_q95 e30_q95 e10_q98; do
  log "$t：$(cat $ROOT/${t}_*/results.csv 2>/dev/null | grep -c '^task_') 列"
done
log "階段 A DCT 側完成"
