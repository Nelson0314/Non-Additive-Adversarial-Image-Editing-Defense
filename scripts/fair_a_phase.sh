#!/usr/bin/env bash
# 公平比較 · 階段 A（校準曲線）· 相位側，卡 4–7。
#
# 一個 process 負責「一個強度 × 全部 25 張」，而不是「一個強度 × 一張」。
# 理由是實測：一張的純運算是 122 s，但一個 process 光載 IP2P 就要約 320 s
# （權重在 NFS 上）。分片越細，載模型的固定成本被重複越多次——σ 掃描那批
# 12 個 worker 各跑一張，7.4 分鐘裡有 5.3 分鐘在載模型。
#
# 12 個強度 = 12 個 process = 4 張卡各 3 個，與已知不會 OOM 的密度相同。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a_phase
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 只等 θ 重定那一批，不要等自己的姊妹腳本（DCT 側同時在跑）
while ps -u $USER -o cmd | grep "[i]p2p_run" | grep -q theta0820; do sleep 30; done

IMGS=$(tr '\n' ' ' < runs/fair0820/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "階段 A 相位側開始，$(echo $IMGS | wc -w) 張"

# tag  gpu  條件  σ  θ
JOBS="
p00_t10  4  phase       0  1.0
p00_t12  4  phase       0  1.2
p00_t14  4  phase       0  1.4
p00_t16  5  phase       0  1.6
r00_t07  5  phase_rand  0  0.7
r00_t10  5  phase_rand  0  1.0
r00_t13  6  phase_rand  0  1.3
r00_t16  6  phase_rand  0  1.6
p20_t18  6  phase       2  1.8
p20_t22  7  phase       2  2.2
p20_t26  7  phase       2  2.6
p20_t30  7  phase       2  3.0
"
while read -r tag gpu cond sig th; do
  [ -n "${tag:-}" ] || continue
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
    --conditions "$cond" --radius "$th" --pixel-gate-sigma "$sig" \
    > "$ROOT/$tag.log" 2>&1 &
done <<< "$JOBS"
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "階段 A 相位側完成"
