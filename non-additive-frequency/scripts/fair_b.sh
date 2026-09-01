#!/usr/bin/env bash
# 公平比較 · 階段 B（正式表）· 75 張，三個條件都錨在同一個失真點。
#
# 錨點由階段 A 的取捨曲線內插求得（runs/fair0820/curve_fid_dists_*_anchors.csv）：
#
#   DCT-Shield 原生 ε=1／Q_alg=0.95  →  DISTS 0.0538（參照點，不動）
#   紋理重相位  θ=1.27               →  由 θ=1.2（0.0502）與 1.4（0.0605）內插
#   隨機相位    θ=1.08               →  由 θ=1.0（0.0477）與 1.3（0.0720）內插
#
# **只有等失真錨點，沒有等效果錨點。** 相位側四個 θ 的位移是 0.251–0.311，
# DCT 在 ε=1 就已經是 0.658，等效果的內插一律落在掃描範圍外，
# `tradeoff_curve.py` 正確地判為外插並拒絕給值。這是結果不是缺漏。
#
# σ 取 0（不開逐像素閘）：同一失真預算下 σ=2 的位移是 0.2314、σ=0 是 0.2741，
# 逐像素閘掉 16%，且先前的臉部放大圖顯示它沒有解決人臉損傷。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/b
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

while ps -u $USER -o cmd | grep "[i]p2p_run" | grep -q fair0820/a_; do sleep 30; done

mapfile -t ALL < runs/fair0820/images75.txt
[ "${#ALL[@]}" -eq 75 ] || { log "影像清單是 ${#ALL[@]} 張，不是 75，中止"; exit 1; }
log "階段 B 開始，${#ALL[@]} 張"

shard() {   # shard <k> <n> —— 取 n 片中的第 k 片
  local k=$1 n=$2 i=0
  for x in "${ALL[@]}"; do
    [ $((i % n)) -eq "$k" ] && printf '%s ' "$x"
    i=$((i+1))
  done
}

# DCT-Shield：單張 575 s，是瓶頸，給五張卡十五片
for k in $(seq 0 14); do
  CUDA_VISIBLE_DEVICES=$((k % 5)) nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/dct_$k" --data data/omniedit150 --images $(shard $k 15) \
    --conditions dct_shield --eps 1 --q-alg 0.95 \
    > "$ROOT/dct_$k.log" 2>&1 &
done
# 相位與隨機相位：單張 122 s，共用卡 5–7
i=0
for spec in "phase 1.27" "phase_rand 1.08"; do
  set -- $spec
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$((5 + i % 3)) nohup $PY scripts/ip2p_run.py \
      --out "$ROOT/$1_$k" --data data/omniedit150 --images $(shard $k 4) \
      --conditions "$1" --radius "$2" --pixel-gate-sigma 0 \
      > "$ROOT/$1_$k.log" 2>&1 &
    i=$((i+1))
  done
done
wait
# `phase_*` 會連 `phase_rand_*` 一起吃到，故分片編號用 [0-9] 收斂
for c in 'dct_[0-9]' 'phase_[0-9]' 'phase_rand_[0-9]'; do
  log "$c：$(cat $ROOT/${c}*/results.csv 2>/dev/null | grep -c '^task_') 列"
done
log "階段 B 完成"
