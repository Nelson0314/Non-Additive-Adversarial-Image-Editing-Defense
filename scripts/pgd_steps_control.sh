#!/usr/bin/env bash
# 最佳化預算的對照：本方法 100 步、DCT-Shield 1000 步，而頭對頭表把兩者並排。
# basic-2 的 8 張卡。
#
# 為什麼要這一批
# ────────────────────────────────────────────────────────────────────
# `scripts/ip2p_run.py` 的 `--steps` 預設 100（本方法），`--dct-steps` 預設
# 1000（DCT-Shield 論文 §5.4）。**預算差十倍這件事此前不在任何欄位裡**，於是
# 「誰比較強」與「誰跑比較久」在報表上分不開。本批把它變成一個受控的變因，
# 並在兩個方向上各走一遍：把我們加上去、把它降下來。
#
# 這是乾淨的檢定，因為步長會跟著步數縮
# ────────────────────────────────────────────────────────────────────
# `run_param_pgd` 的步長是 `alpha = radius / (steps * saturate_at)`，故總行程
# 恆等於 radius，與步數無關。步數變多換到的是**細修的迭代數**而不是更大的
# 預算——半徑與失真的對應不會被步數平移。DCT-Shield 走的是它自己的
# `run_dct_shield`，該篇的 gamma 固定 0.1，降步數等於減少總行程，故它那三點
# 要與 eps 掃描一起讀。
#
# 判準：若 steps=1000 把本方法在等 LPIPS 上的效果拉近 DCT-Shield 一半以上，
# `runs/ip2p_fair_comparison/b` 的 75 張頭對頭表必須在對齊的預算上重跑。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_pgd_steps
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "步數對照開始，$(echo $IMGS | wc -w) 張"

# 實測單張成本（runs/ip2p_fair_comparison 的 total_seconds 中位數）：
# phase 100 步 39 s、dct_shield 1000 步 587 s。編輯本身約佔 19 s，其餘按步數
# 線性外推來排卡，故最長的一格約 91 分鐘。
run() {
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}

# 本方法：等 DISTS 錨點的 theta=1.27 上把步數拉滿
run phase_s0100 0 --conditions phase --radius 1.27 --steps 100
run phase_s0200 0 --conditions phase --radius 1.27 --steps 200
run phase_s0400 1 --conditions phase --radius 1.27 --steps 400
run phase_s0700 2 --conditions phase --radius 1.27 --steps 700
run phase_s1000 3 --conditions phase --radius 1.27 --steps 1000
# 幅度增益那一臂也要有，否則階段一的最佳工作點沒有對應的步數讀數
run gain_s0100  4 --conditions phase_gain --radius 1.5 --gain-ratio 0.25 --steps 100
run gain_s1000  5 --conditions phase_gain --radius 1.5 --gain-ratio 0.25 --steps 1000
# DCT-Shield：反方向。eps=1 固定，只降步數。1000 步那點已有
# runs/ip2p_fair_comparison/a_dct/e10_q95_*（同一批 25 張），不重跑。
run dct_s0100   6 --conditions dct_shield --dct-steps 100
run dct_s0250   6 --conditions dct_shield --dct-steps 250
run dct_s0500   7 --conditions dct_shield --dct-steps 500
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "步數對照完成"
