#!/usr/bin/env bash
# DCT 對抗浮水印（摘要重建）的強度掃描，25 張。
#
# **這不是重現。** 那篇（The Imaging Science Journal 2026,
# doi:10.1080/13682199.2026.2644653）在付費牆後、無 arXiv、無公開程式碼；
# 本專案只實作它摘要明載的三個機制（迴圈內模擬重壓 ＋ 直通估計、顯著圖選
# 位置、係數個數的稀疏約束），全部超參數為本專案指定並逐列寫進 CSV。
#
# 強度旋鈕取 `topk`（每個區塊最多改幾個係數），eps 固定 2.0。單張探針：
#   topk=8  DISTS 0.0271 LPIPS 0.2318
#   topk=16 DISTS 0.0369 LPIPS 0.2786
# 該張的 DISTS 約為 25 張平均的 0.61 倍，故 topk 8–64 罩得住錨點 0.0538。
#
# 一併記下探針看到的事：它的 LPIPS/DISTS 比值約是相位臂的 2.4 倍——與
# DCT-Shield 同一個特徵（稀疏的整階改動在 LPIPS 上很顯眼、DISTS 看不太到）。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/b1_wm
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 等帶通那一批把卡讓出來，避免同時佔滿超過一小時
while ps -u $USER -o cmd | grep "[i]p2p_run" | grep -q a9_rmax; do sleep 30; done
log "浮水印掃描開始"

mapfile -t ALL < runs/fair0820/images25.txt
[ "${#ALL[@]}" -eq 25 ] || { log "影像清單是 ${#ALL[@]} 張，中止"; exit 1; }
shard() { local k=$1 i=0; for x in "${ALL[@]}"; do
  [ $((i % 3)) -eq "$k" ] && printf '%s ' "$x"; i=$((i+1)); done; }

g=0
for k in 8 16 32 64; do
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/ip2p_run.py \
      --out "$ROOT/k${k}_$s" --data data/omniedit150 --images $(shard $s) \
      --conditions dct_wm --wm-topk "$k" --wm-eps 2.0 --wm-steps 300 \
      > "$ROOT/k${k}_$s.log" 2>&1 &
    g=$(((g+1) % 8))
  done
done
wait
for k in 8 16 32 64; do
  log "topk=$k：$(cat $ROOT/k${k}_*/results.csv 2>/dev/null | grep -c '^task_') 列"
done
log "浮水印掃描完成"
