#!/usr/bin/env bash
# Y-only 的抗淨化。basic-1，補完最強 baseline 那一格。
#
# 為什麼是它：SDEdit 線 7 張上 Y-only 的抗淨化淨增益是 base 的數倍
# （JPEG-75 +0.5185 對 +0.0472），而 IP2P 線從未測過。今晚已量到它在
# DISTS 0.0275（三個條件裡最低）上視覺擋下 4/13，是失真效率最高的一格。
# 抗淨化那一軸若它也贏，本專案的主主張就沒有立足處；若它輸，那是我們僅剩
# 的方向。
#
# 與 `retention_matched_blocking.sh` 同一組算子與同一批 13 張，故兩份表可
# 直接並排。地板由那一批提供，此處不重跑。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_retention_y_only
mkdir -p $ROOT/gallery
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

OBEYED=$(awk -F, 'NR>1 && $4=="obeyed" {print $2}' \
         runs/obedience_audit/undefended_obedience.csv | tr '\n' ' ')
[ -n "$OBEYED" ] || { log "服從清單是空的，中止"; exit 1; }
OPS="identity blur1 jpeg75 crop_resize0.1 jpeg_then_resize75 adverse_cleaner"
log "Y-only 抗淨化開始，$(echo $OBEYED | wc -w) 張"

# Y-only 的 25 張分成兩個分片跑（y_e10_a 13 張、y_e10_b 12 張），故兩個目錄
# 各跑一次；`--images` 會把交集取出來。
gpu=0
for d in y_e10_a y_e10_b y_e20_a y_e20_b; do
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/phase_retention.py \
    --run "runs/ip2p_dct_shield_y/$d" --data data/omniedit150 --attacker ip2p \
    --conditions dct_shield_y --images $OBEYED --purifiers $OPS \
    --gallery "$ROOT/gallery" --out "$ROOT/$d.csv" \
    > "$ROOT/$d.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait
for f in $ROOT/*.csv; do
  [ -f "$f" ] && log "$(basename $f .csv)：$(( $(wc -l < $f) - 1 )) 列"
done
log "Y-only 抗淨化完成"
