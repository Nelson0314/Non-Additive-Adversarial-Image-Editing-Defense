#!/usr/bin/env bash
# DJSMA 在它自己的威脅模型上的重現，200 張 × 兩個模型。
#
# 診斷（8 張、VGG19）已經確認方法接對了，且**目標標籤取 least_likely 時
# 對得上論文**：
#   least_likely τ=800  未壓縮成功 0.88、壓縮後 0.62
#   random       τ=800  未壓縮成功 0.88、壓縮後 0.38
# 論文 Table 1 的 VGG19 成功率是 63.4 − 1.2 = 62.2%。目標標籤的選法論文未載，
# least_likely 是本專案指定，這個對照是選它的理由，必須寫進報表。
#
# 分片方式：一張卡吃一整批（batch=25），1500 次迭代整批一起跑，故 8 張卡
# 約 15 分鐘就能吃完 200 張。逐張跑會慢一個數量級。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/djsma_repro
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 等相位那幾批把卡讓出來
while ps -u $USER -o cmd | grep -q "[i]p2p_run"; do sleep 30; done
log "DJSMA 重現開始"

for m in vgg19 resnet101; do
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/djsma_repro.py \
      --out "$ROOT/${m}_g$g" --models "$m" --batch 25 --tau 1500 \
      --target least_likely --eval-quality 75 85 \
      --shard "$g" --shards 8 \
      > "$ROOT/${m}_g$g.log" 2>&1 &
  done
  wait
  log "$m 完成"
done
log "DJSMA 重現結束"
