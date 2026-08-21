#!/usr/bin/env bash
# 閘開度 × 高增益 × 步數三者疊加，把等 LPIPS 錨點真正做出來。basic-1。
#
# 三個已經各自量到的事實，本批測它們疊不疊加
# ────────────────────────────────────────────────────────────────────
# 1. 高增益能走到 LPIPS 0.4303（`ip2p_reach_lpips_ext/pgx_g78`），距離
#    DCT-Shield 在同一批 25 張上的最低點 0.4450 只差 3.4%，但仍在範圍外，
#    協定拒絕外插，故等 LPIPS 錨點還是算不出來。
# 2. 把紋理閘的**能量因子**全開（`--quantile 0`）在三個 radius 上都給出該
#    radius 的最佳 L/D：0.9 → 5.17、1.27 → 4.26、1.8 → 3.53，各比同 radius
#    的其他設定高 15–20%。**放行邊緣（`--gate-edge-power 0`）方向相反**，
#    L/D 反而由 3.70 掉到 3.35，故本批把邊緣那一項留在 1.0。
# 3. 1000 步把 L/D 由 3.70 抬到 4.66（+26%），且位移／LPIPS 由 1.369 升到
#    1.407。代價是絕對失真與絕對位移同時下降，要靠更大的 radius 補回來。
#
# 三者都指向同一個工作點：**quantile=0 ＋ 高 gain ＋ 1000 步**。本批的問題是
# 它們是相加還是互相抵銷。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_reach_lpips_gated
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "閘開×增益×步數開始，$(echo $IMGS | wc -w) 張"

run() {
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
    --conditions phase_gain --radius 3.0 "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}
# 閘全開的能量因子 ＋ 100 步：直接看疊加後能不能越過 0.4450
run gq0_g30_s100  0 --gain-ratio 1.0 --quantile 0.0 --steps 100
run gq0_g54_s100  1 --gain-ratio 1.8 --quantile 0.0 --steps 100
run gq0_g78_s100  2 --gain-ratio 2.6 --quantile 0.0 --steps 100
# 同樣三點 ＋ 1000 步：步數在高增益區還有沒有那 26%
run gq0_g30_s1000 3 --gain-ratio 1.0 --quantile 0.0 --steps 1000
run gq0_g54_s1000 4 --gain-ratio 1.8 --quantile 0.0 --steps 1000
run gq0_g78_s1000 5 --gain-ratio 2.6 --quantile 0.0 --steps 1000
# 對照：閘維持定案值、只加步數。用來把「閘」與「步數」的貢獻分開
run gq5_g78_s1000 6 --gain-ratio 2.6 --quantile 0.5 --steps 1000
# 再往上推一階，確保曲線越過 DCT-Shield 的最低點而不是剛好停在下面
run gq0_g104_s100 7 --gain-ratio 3.4 --quantile 0.0 --steps 100
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "閘開×增益×步數完成"
