#!/usr/bin/env bash
# 改動一（可學幅度）與改動三（針對淨化最佳化）的掃描，25 張。
#
# 全部錨在同一個失真點：DCT-Shield 原生 ε=1 的 DISTS 0.0538。基準曲線
# （runs/fair0820/a_phase/p00_t1*）在該處內插出的位移是 0.2742。
#
# 第一波 · 改動一
#   gain_only   theta 凍結在 0，只學對數增益。ratio=1 故 radius 即 gain_max
#   phase_gain  兩者都學，ratio=0.25（增益上界是相位半徑的四分之一）
#
#   theta 網格由本機單張探針選出（該張的 DISTS 約為 25 張平均的 0.61 倍）：
#     gain_only  r=0.8 → 0.0260   phase_gain r=1.0 → 0.0312（基準 theta=1.2 是 0.0307）
#   故兩者的網格都罩住錨點兩側。
#
#   單張探針上的預告：**phase_gain 在同一個 DISTS 上的 effect 比基準高 10%，
#   gain_only 比基準低**。25 張才算數。
#
# 第二波 · 改動三
#   在基準的錨點 theta=1.27 上，把可微分淨化算子放進最佳化迴圈：
#     curriculum  JPEG 品質 95→50 線性（MetaCloak-JPEG 的作法）
#     eot_ops     每步隨機抽 identity／模糊／裁切縮放／JPEG75 之一
#   目的是鞏固「過 JPEG 之後贏 2.77 倍」那一格，並看模糊與裁切能不能救回來。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a5_gain
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 等相位變體那一批把卡讓出來
while ps -u $USER -o cmd | grep "[i]p2p_run" | grep -q a4_variants; do sleep 30; done

IMGS=$(tr '\n' ' ' < runs/fair0820/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "第一波（改動一）開始，$(echo $IMGS | wc -w) 張"

run() {
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}

run go_r06 0 --conditions gain_only  --radius 0.6 --gain-ratio 1.0
run go_r08 1 --conditions gain_only  --radius 0.8 --gain-ratio 1.0
run go_r10 2 --conditions gain_only  --radius 1.0 --gain-ratio 1.0
run go_r13 3 --conditions gain_only  --radius 1.3 --gain-ratio 1.0
run pg_r08 4 --conditions phase_gain --radius 0.8 --gain-ratio 0.25
run pg_r10 5 --conditions phase_gain --radius 1.0 --gain-ratio 0.25
run pg_r12 6 --conditions phase_gain --radius 1.2 --gain-ratio 0.25
run pg_r15 7 --conditions phase_gain --radius 1.5 --gain-ratio 0.25
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "第一波完成"

# ---- 第二波 · 改動三 ----
ROOT2=runs/fair0820/a6_purify
mkdir -p $ROOT2
log2() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT2/log.txt; }
log2 "第二波（改動三）開始"
run2() {
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT2/$tag" --data data/omniedit150 --images $IMGS "$@" \
    > "$ROOT2/$tag.log" 2>&1 &
}
# 錨點 theta=1.27；另各給一個較低的 theta，因為把算子放進迴圈通常會改變
# theta 與失真的對應，只有單點時錨不住
run2 cur_t127 0 --conditions phase --radius 1.27 --purify-aware curriculum
run2 cur_t100 1 --conditions phase --radius 1.00 --purify-aware curriculum
run2 eot_t127 2 --conditions phase --radius 1.27 --purify-aware eot_ops
run2 eot_t100 3 --conditions phase --radius 1.00 --purify-aware eot_ops
run2 f75_t127 4 --conditions phase --radius 1.27 --purify-aware fixed75
run2 f75_t100 5 --conditions phase --radius 1.00 --purify-aware fixed75
wait
for d in $ROOT2/*/; do
  log2 "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log2 "第二波完成"
