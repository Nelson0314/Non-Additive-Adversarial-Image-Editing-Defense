#!/usr/bin/env bash
# A2 的補點，接著直接開階段 C。
#
# 補點的理由：A2 在 DISTS 軸與 PSNR 軸上都答完了（r_min=0.06 的位移是 0.12 的
# 0.612 與 0.730 倍，兩軸都輸），但 LPIPS 軸落在掃描範圍外——r_min=0.06 的
# LPIPS 只掃到 0.1527，基準點是 0.2051。**那正好是唯一可能對 r_min 有利的軸**
# （等 DISTS 的對照圖顯示 LPIPS 才抓得到 DCT 那面牆的彩色雜點），所以不能靠
# 外插打發。補 θ=0.80 與 1.00 兩點把範圍蓋過去。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a2_rmin
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

mapfile -t ALL < runs/fair0820/images25.txt
[ "${#ALL[@]}" -eq 25 ] || { log "影像清單是 ${#ALL[@]} 張，不是 25，中止"; exit 1; }
shard() { local k=$1 i=0; for x in "${ALL[@]}"; do
  [ $((i % 4)) -eq "$k" ] && printf '%s ' "$x"; i=$((i+1)); done; }

log "A2 補點開始（r_min=0.06，θ=0.80／1.00，蓋住 LPIPS 0.2051）"
g=0
for th in 0.80 1.00; do
  tag=rm06_t$(echo $th | tr -d .)
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/ip2p_run.py \
      --out "$ROOT/${tag}_$k" --data data/omniedit150 --images $(shard $k) \
      --conditions phase --radius "$th" --r-min 0.06 --pixel-gate-sigma 0 \
      > "$ROOT/${tag}_$k.log" 2>&1 &
    g=$(((g+1) % 8))
  done
done
wait
for th in 080 100; do
  log "rm06_t$th：$(cat $ROOT/rm06_t${th}_*/results.csv 2>/dev/null | grep -c '^task_') 列"
done
log "A2 補點完成"

bash scripts/fair_c.sh runs/fair0820/b_merged
