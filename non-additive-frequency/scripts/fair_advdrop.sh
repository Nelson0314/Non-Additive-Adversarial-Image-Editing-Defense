#!/usr/bin/env bash
# AdvDrop 移植到本專案場景（IP2P ＋ OmniEdit）的強度掃描，25 張。
#
# 先在它自己的威脅模型上重現過（runs/advdrop_repro）：未定向成功率與 Bit-6
# 兩欄在步長 4 之下對得上論文，JPEG-30 那一欄對不上。移植時沿用步長 4。
#
# 三處必須寫進報表的事
# ────────────────────────────────────────────────────────────────
# 1. **損失換掉了**：原文是分類的交叉熵 log p_y，本威脅模型沒有分類器。
#    此處用的是本專案共用的 encoder-target 損失——與相位臂**完全相同**，
#    於是這一輪是乾淨的「參數化」消融：同一個損失、同一個攻擊方、同一批
#    影像，唯一的差別是擾動由「旋轉相位」換成「丟棄係數」。
# 2. **步長 4 而非論文式 (7) 隱含的 1**，理由見 runs/advdrop_repro。
# 3. **強度的旋鈕不是 eps，是步數**。本機實測（單張）：eps 由 100 加到 1600，
#    DISTS 只由 0.0192 動到 0.0209——因為 50 步 × 步長 4 最多把量化表推到
#    201，eps 根本沒有binding。改掃步數之後 DISTS 才動起來：
#      50 步 0.0200 / 150 步 0.0310 / 400 步 0.0505 / 1000 步 0.0698
#    故此處固定 eps=1600（刻意不 binding），掃步數。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a3_advdrop
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

mapfile -t ALL < runs/fair0820/images25.txt
[ "${#ALL[@]}" -eq 25 ] || { log "影像清單是 ${#ALL[@]} 張，不是 25，中止"; exit 1; }
shard() { local k=$1 i=0; for x in "${ALL[@]}"; do
  [ $((i % 3)) -eq "$k" ] && printf '%s ' "$x"; i=$((i+1)); done; }

log "AdvDrop 移植掃描開始，${#ALL[@]} 張，固定 eps=1600、步長 4，掃步數"
g=0
for st in 150 400 700 1000; do
  for k in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/ip2p_run.py \
      --out "$ROOT/s${st}_$k" --data data/omniedit150 --images $(shard $k) \
      --conditions advdrop --advdrop-eps 1600 --advdrop-steps "$st" \
      > "$ROOT/s${st}_$k.log" 2>&1 &
    g=$(((g+1) % 8))
  done
done
wait
for st in 150 400 700 1000; do
  log "steps=$st：$(cat $ROOT/s${st}_*/results.csv 2>/dev/null | grep -c '^task_') 列"
done
log "AdvDrop 移植掃描完成"
