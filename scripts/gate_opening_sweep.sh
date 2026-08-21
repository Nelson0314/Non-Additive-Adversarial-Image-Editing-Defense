#!/usr/bin/env bash
# 把三個閘打開：測「擾動被允許出現在哪裡」對 LPIPS／DISTS 比值的影響。
# basic-1 的 8 張卡，接在強度延伸掃描之後。
#
# 假設
# ────────────────────────────────────────────────────────────────────
# `runs/distortion_axis_analysis/axis_fits.csv` 量到 DISTS 對不同方法開的價
# 差 3.4 倍，而 LPIPS 只差 1.27 倍。逐工作點的比值與「擾動放在哪裡」對得上：
#
#   DCT-Shield（全影像、每個 8x8 塊、所有通道）      L/D 9.13
#   DJSMA（全影像、稀疏 ±1）                         L/D 6.22
#   紋理重相位（高通、只在紋理區，有效面積 0.450）   L/D 3.0–4.6
#   帶通 r_max=0.4（更集中在中頻紋理）               L/D 1.53
#
# 也就是說本方法的三個閘剛好把擾動趕進 DISTS 標價最貴的地方。DISTS 量的是
# 紋理統計量，而本方法保留幅度譜、只重排紋理。
#
# 另一個獨立的理由：邊緣（coherence 高）目前被閘設為 0，而邊緣正是導向濾波、
# 雙邊濾波、TV 去噪這一類算子的**不變集**。把擾動趕出邊緣，等於主動放棄那
# 幾個淨化算子底下唯一活得下來的位置。
#
# 兩個旋鈕（兩者的預設都逐位元重現現行行為，由 tests 釘住）
#   --quantile          紋理閘的能量參考分位數。調低放行更多低能量區塊
#   --gate-edge-power   (1 - coherence^2) ** p。p=0 完全不壓制邊緣
#
# **頻譜加性項不在本批**：那會讓平坦區變成可達，但也會使方法不再是非加性。
# 使用者裁定先不做。
#
# 判準：主讀數是等 LPIPS 錨點上的 DISTS（越低越好），副讀數是效果不得下降。
# 每組三個 radius 才能內插，單點不算數。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_gate_opening
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 等強度延伸那批把卡讓出來。比對腳本路徑而非關鍵字，且用中括號寫法使樣式
# 字串本身不匹配（否則會匹配到自己的 ssh 連線）。
while ps -u $USER -o cmd | grep -q "[i]p2p_run.py --out runs/ip2p_reach_lpips"; do
  sleep 30
done

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "閘開度掃描開始，$(echo $IMGS | wc -w) 張"

# 五組設定 x 三個 radius。`q50_e10` 是現行定案，已有既有批次，不重跑。
#   q50_e05  只放行邊緣一半
#   q50_e00  完全不壓制邊緣
#   q00_e10  閘的能量因子全開（每個區塊都拿滿）
#   q00_e00  兩者全開 = 這個構造的閘上界
#   q25_e05  中間點，用來看是不是單調
CFGS="
q50_e05  0.5   0.5
q50_e00  0.5   0.0
q00_e10  0.0   1.0
q00_e00  0.0   0.0
q25_e05  0.25  0.5
"
gpu=0
while read -r name q ep; do
  [ -n "${name:-}" ] || continue
  for rad in 0.9 1.27 1.8; do
    tag="${name}_r$(echo $rad | tr -d '.')"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
      --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
      --conditions phase --radius "$rad" \
      --quantile "$q" --gate-edge-power "$ep" \
      > "$ROOT/$tag.log" 2>&1 &
    gpu=$(( (gpu + 1) % 8 ))
  done
done <<< "$CFGS"
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "閘開度掃描完成"
