#!/usr/bin/env bash
# 逐像素閘定案後的 θ 重定（2026-08-20，使用者裁定的 A→B 順序的 B）。
#
# σ 掃描的結果：逐像素閘把失真與防禦位移**同時**壓下去（σ=1 時 DISTS
# 0.0551 → 0.0209、位移 0.3004 → 0.1851）。兩者都降，故「閘有沒有用」
# 不能由 σ 掃描本身回答——必須把 θ 調回同一個失真預算再比位移。
#
# 目標預算是 σ=0／θ=1.30 的 DISTS 0.0551。若比例外推成立，σ=1 需要
# θ≈3.4、σ=2 需要 θ≈2.5；相位旋轉會飽和，故兩側各掃四點再內插，不外推。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/theta0820
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

while [ "$(ps -u $USER -o cmd | grep -c '[i]p2p_run')" -gt 0 ]; do sleep 30; done
log "θ 重定開始"

# 與 σ 掃描逐字相同的 12 張，否則兩批不可比
IMGS=$($PY -c "
import pathlib
names = sorted(p.name for p in pathlib.Path('data/omniedit150').iterdir() if p.is_dir())
face = 'task_attr_mod_color_11699'
rest = [n for n in names if n != face][::13][:11]
print(' '.join([face] + rest))
")
log "影像 $(echo $IMGS | wc -w) 張"

launch() {   # launch <sigma> <theta> <gpu0> <gpu1> <gpu2> <gpu3>
  local S=$1 T=$2; shift 2
  local gpus=("$@") k=0
  local tag="s${S}_t$(echo $T | tr -d .)"
  for g in "${gpus[@]}"; do
    for w in 0 1 2; do
      sub=$($PY -c "
names='''$IMGS'''.split()
print(' '.join(names[$k::12]))
")
      [ -z "$sub" ] || CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/ip2p_run.py \
        --out "$ROOT/${tag}_$k" --data data/omniedit150 --images $sub \
        --conditions phase --radius $T --pixel-gate-sigma $S \
        > "$ROOT/${tag}_$k.log" 2>&1 &
      k=$((k+1))
    done
  done
  echo "$tag"
}

# 每一波兩個工作點，各 12 個 worker，共 24 個分到 8 張卡（每卡 3 個，
# 與 σ 掃描同密度，已知不會 OOM）
run_wave() {
  a=$(launch $1 $2 0 1 2 3)
  b=$(launch $3 $4 4 5 6 7)
  wait
  for t in $a $b; do
    log "$t 完成，$(cat $ROOT/${t}_*/results.csv 2>/dev/null | grep -c '^task_') 列"
  done
}

run_wave 1 2.0  2 1.8
run_wave 1 3.0  2 2.2
run_wave 1 3.5  2 2.6
run_wave 1 4.5  2 3.2
log "θ 重定全部完成"
