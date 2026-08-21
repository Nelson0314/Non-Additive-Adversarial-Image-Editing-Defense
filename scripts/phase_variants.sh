#!/usr/bin/env bash
# 相位臂的三個「便宜改動」（2026-08-21 的改動二與四），25 張，卡 2–7。
#
# 全部錨在同一個失真點比較：DCT-Shield 原生 ε=1 的 DISTS 0.0538。既有的
# r_min=0.12 基準曲線（runs/fair0820/a_phase/p00_t1*）在該處內插出的位移是
# 0.2742，這一輪的每個變體都要跟那個數字比。
#
#   V1 損失換成 ‖E(x')‖₂     參數化完全不動，只換目標。DCT-Shield 用的就是這個，
#                            它單調、無方向要求；我們原本的 encoder-target 要
#                            同時對長度與方向。θ 網格與基準相同（映射未變）
#   V2 gl_iters = 3          Griffin-Lim 迭代投影。單張探針：θ=1.2 時 DISTS 由
#                            0.0307 掉到 0.0114，故 θ 要拉到 2.4–4.0 才回得到
#                            同一個預算
#   V3 block = 64            區塊大小從未掃過。單張探針：θ=0.8 就有 DISTS
#                            0.0486，比基準強很多，故 θ 網格改成 0.4–0.7
#
# 每個變體給足三個 θ（V1 兩個就夠，映射與基準相同），確保錨點落在範圍內、
# 不必外插。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a4_variants
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

IMGS=$(tr '\n' ' ' < runs/fair0820/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "相位變體開始，$(echo $IMGS | wc -w) 張"

# tag  gpu  額外旗標...
run() {
  local tag=$1 gpu=$2 th=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
    --conditions phase --radius "$th" "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}

run ln_t12   2 1.2  --loss latent_norm
run ln_t14   3 1.4  --loss latent_norm
run gl_t24   4 2.4  --gl-iters 3
run gl_t32   5 3.2  --gl-iters 3
run gl_t40   6 4.0  --gl-iters 3
run b64_t040 7 0.40 --block 64
wait
log "第一波完成（V1 兩點、V2 三點、V3 一點）"

run b64_t055 2 0.55 --block 64
run b64_t070 3 0.70 --block 64
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "相位變體全部完成"
