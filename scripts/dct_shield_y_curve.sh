#!/usr/bin/env bash
# DCT-Shield 的 Y-only 變體上 IP2P 線，外加把 base 變體的曲線往低失真延伸。
# basic-2，逐影像分片。
#
# 為什麼這批是必要的而不是補充
# ────────────────────────────────────────────────────────────────────
# `docs/BASELINES.md` 寫著「頭對頭表上**不可只放 base 變體**」，而 IP2P 線至今
# 只跑過 base。SDEdit 線 7 張上，Y-only 的抗淨化淨增益是 base 的數倍：
# JPEG-75 +0.5185 對 +0.0472（差 11 倍）、GrIDPure +0.3439 對 +0.0844。
# 拿 base 當門檻去追 universal 抗淨化，追到的會是一條假的線。
#
# 同時把 base 變體的步數往下延伸到 25／50 步。理由是曲線重疊：本方法目前最高
# 到 LPIPS 0.4303，而 base 變體在同一批 25 張上的最低點是 100 步的 0.4450。
# 兩條曲線差 3.4% 而不重疊，協定拒絕外插，等 LPIPS 錨點就算不出來。降步數會
# 讓它的失真往下走，從它那一側把區間接起來。
#
# 分片：DCT-Shield 1000 步實測 587 秒／張，25 張單 process 要 4.1 小時。
# 每個工作點切成兩片、各 13/12 張，壓到約 2 小時。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_dct_shield_y
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

LIST=runs/ip2p_fair_comparison/images25.txt
A=$(head -13 $LIST | tr '\n' ' ')
B=$(tail -12 $LIST | tr '\n' ' ')
[ -n "$A" ] && [ -n "$B" ] || { log "影像清單是空的，中止"; exit 1; }
log "Y-only 曲線開始"

run() {
  local tag=$1 gpu=$2 imgs=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $imgs "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}
# Y-only，論文定案 1000 步，三個 eps 畫出它自己的取捨曲線
run y_e10_a 0 "$A" --conditions dct_shield_y --eps 1.0
run y_e10_b 1 "$B" --conditions dct_shield_y --eps 1.0
run y_e20_a 2 "$A" --conditions dct_shield_y --eps 2.0
run y_e20_b 3 "$B" --conditions dct_shield_y --eps 2.0
run y_e30_a 4 "$A" --conditions dct_shield_y --eps 3.0
run y_e30_b 5 "$B" --conditions dct_shield_y --eps 3.0
# base 變體往低失真延伸，把 LPIPS 軸上的區間從它那一側接起來
run base_s025 6 "$A $B" --conditions dct_shield --dct-steps 25
run base_s050 7 "$A $B" --conditions dct_shield --dct-steps 50
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "Y-only 曲線完成"
