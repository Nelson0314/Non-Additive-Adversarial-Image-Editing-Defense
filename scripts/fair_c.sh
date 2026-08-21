#!/usr/bin/env bash
# 公平比較 · 階段 C（抗淨化 ＋ 空白地板）· 75 張。
#
# 不重跑攻擊：`phase_retention.py` 只讀階段 B 已存下的 `*__def.png`，
# 所以 DCT-Shield 那 1000 步 PGD 不會再跑一次。
#
# 兩趟。第一趟三個條件、第二趟空白地板（`--floor`，防禦圖就是原圖）。
# 分開跑是因為地板便宜、先落地就能先看；合在一起跑總機時一樣。
#
# **空白地板不是可選項**。淨化算子自己就會把編輯推開（t0820 實測高斯模糊
# 的地板是 0.1903），不扣掉它，「淨化後絕對位移較大」就無法排除「該算子本來
# 就把編輯推得比較開」這個平庸解釋——那正是 t0820 保留率 25/25 全勝、扣掉
# 地板後只剩 JPEG 勝的原因。
set -u
RUN=${1:?用法：bash scripts/fair_c.sh <合併後的批次目錄>}
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/c
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

[ -f "$RUN/results.csv" ] || { log "$RUN 沒有 results.csv，中止"; exit 1; }
mapfile -t ALL < runs/fair0820/images75.txt
[ "${#ALL[@]}" -eq 75 ] || { log "影像清單是 ${#ALL[@]} 張，不是 75，中止"; exit 1; }
log "階段 C 開始，來源 $RUN，${#ALL[@]} 張"

# 四個算子。identity 是 retention 的分母，`purifier_set` 會強制它在場。
PUR="identity blur1 jpeg75 crop_resize0.1"
SEEDS=3
N=24

shard() { local k=$1 i=0; for x in "${ALL[@]}"; do
  [ $((i % N)) -eq "$k" ] && printf '%s ' "$x"; i=$((i+1)); done; }

pass() {   # pass <名稱> <額外旗標...>
  local name=$1; shift
  for k in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$((k % 8)) nohup $PY scripts/phase_retention.py \
      --run "$RUN" --data data/omniedit150 --attacker ip2p \
      --images $(shard $k) --purifiers $PUR --seeds $SEEDS "$@" \
      --out "$ROOT/${name}_$k.csv" > "$ROOT/${name}_$k.log" 2>&1 &
  done
  wait
  local rows
  rows=$(cat $ROOT/${name}_*.csv 2>/dev/null | grep -vc '^image,' || true)
  log "$name 完成，$rows 列"
  [ "${rows:-0}" -gt 0 ] || { log "$name 零列，中止"; exit 1; }
}

pass ret
pass floor --floor
log "階段 C 完成"
