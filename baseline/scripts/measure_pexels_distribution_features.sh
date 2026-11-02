#!/usr/bin/env bash
# Pexels 分布指標（CLIP-IQA、InceptionV3 特徵）的預先抽取，CPU、不取卡：對目前已存在的影像平行抽取尚未抽過的特徵，
# 存成 runtime/pexels_portraits/summary/distribution_metrics/pexels_features_<標籤>_<i>.pt。
# scripts/measure_pexels_summary_inputs.sh distribution 合併時讀取同目錄全部分片，只補抽剩下的影像。
# 用法（遠端 baseline 專案根）：ENV_FILE=~/env.sh SHARDS=8 bash scripts/measure_pexels_distribution_features.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
SHARDS=${SHARDS:-8}
O=$BASELINE_ROOT/runtime/pexels_portraits/summary/distribution_metrics
mkdir -p "$O"
for f in "$BASELINE_ROOT"/runtime/distribution_metrics/pexels_features_*.pt; do
  [ -e "$O/$(basename "$f")" ] || ln -s "$f" "$O/$(basename "$f")" || exit 1
done
tag=prefetch_$(date +%s)
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  "$PY" scripts/measure_distribution_metrics.py "$O" --dataset pexels --shard "$i" --shards "$SHARDS" --tag "$tag" \
      > "$O/shard_${tag}_$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "[FATAL] 特徵分片失敗（行程 $p），見 $O/shard_${tag}_*.log" >&2; exit 1; }; done
echo "[ALLDONE] $tag"
