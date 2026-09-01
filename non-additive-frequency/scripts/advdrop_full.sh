#!/usr/bin/env bash
# AdvDrop 重現的補完（2026-08-21）。第一輪三欄裡對上兩欄：
#
#   無防禦  0.960 對論文 1.000     Bit-6 0.945 對 0.957     JPEG-30 0.200 對 0.826
#
# 這一輪補四件事，各自回答一個具體的假設：
#
#   A 成功率的定義  論文未明說是 f(x')!=y 還是 f(D(x'))!=f(D(x))。後者自動扣掉
#                   「JPEG 自己就把 12% 的乾淨影像分錯」那一塊。**這是目前
#                   JPEG-30 差四倍最可能的解釋**
#   B 張數          由 200 拉到 1000，把 ±1.5% 的抽樣誤差壓到 ±0.7%
#   C 步長          4/6/8，看無防禦那一欄能不能由 0.960 推到 1.000
#   D 量化方法      硬四捨五入（§4.5 報 5.00±0.98%）與定向攻擊（Table 1 第二列）
#
# 卡 0–1；相位臂的四個改動同時在卡 2–7 上跑。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/advdrop_repro
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/full_log.txt; }

[ -f data/imagenet_advdrop/provenance.json ] || {
  log "缺 data/imagenet_advdrop，中止"; exit 1; }

log "補完開始"

# A ＋ C：兩種成功率定義 × 三個步長，200 張（先看方向）
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/advdrop_repro.py \
  --out $ROOT/def_label --eps 100 --step-size 4 6 8 --succ-def label \
  > $ROOT/def_label.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/advdrop_repro.py \
  --out $ROOT/def_defended --eps 20 60 100 --step-size 4 6 8 \
  --succ-def defended > $ROOT/def_defended.log 2>&1 &
wait
log "A＋C 完成"

# D：硬四捨五入的消融（§4.5）與定向攻擊（Table 1 第二列）
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/advdrop_repro.py \
  --out $ROOT/hard_quant --eps 100 --step-size 4 --hard-quant \
  > $ROOT/hard_quant.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/advdrop_repro.py \
  --out $ROOT/targeted --eps 20 60 100 --step-size 4 --targeted --limit 100 \
  > $ROOT/targeted.log 2>&1 &
wait
log "D 完成"
log "補完結束"
