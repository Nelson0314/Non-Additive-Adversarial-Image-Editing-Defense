#!/usr/bin/env bash
# Pexels 主表彙整的 CPU 工作（遠端佇列，不取卡）：
#   arcface_pair <影像組>  scripts/measure_pexels_arcface_pair.py
#   distribution           CLIP-IQA、FID、KID：沿用 runtime/distribution_metrics/ 已抽取的特徵（符號連結），平行抽取其餘影像後合併，
#                          輸出 runtime/pexels_portraits/summary/distribution_metrics/；每組須為 306 張（--require-images）
#   summary                scripts/generate_pexels_main_table_summary.py
# 用法（遠端 baseline 專案根）：ENV_FILE=~/env.sh bash scripts/measure_pexels_summary_inputs.sh <工作> [<影像組>]
set -uo pipefail
[ $# -ge 1 ] || { echo "用法：$0 <arcface_pair|distribution|summary> [<影像組>]" >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
SHARDS=${SHARDS:-8}
case $1 in
  arcface_pair)
    [ $# -eq 2 ] || { echo "arcface_pair 需要影像組" >&2; exit 2; }
    "$PY" scripts/measure_pexels_arcface_pair.py "$2" || exit 1
    ;;
  distribution)
    O=$BASELINE_ROOT/runtime/pexels_portraits/summary/distribution_metrics
    mkdir -p "$O"
    for f in "$BASELINE_ROOT"/runtime/distribution_metrics/pexels_features_*.pt; do
      [ -e "$O/$(basename "$f")" ] || ln -s "$f" "$O/$(basename "$f")" || exit 1
    done
    # 每次執行用不同的標籤：分片檔以標籤命名，重跑時不覆寫前一次已抽取的分片
    tag=complete_$(date +%s)
    pids=()
    for i in $(seq 0 $((SHARDS - 1))); do
      "$PY" scripts/measure_distribution_metrics.py "$O" --dataset pexels --shard "$i" --shards "$SHARDS" --tag "$tag" \
          > "$O/shard_${tag}_$i.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "[FATAL] 特徵分片失敗（行程 $p），見 $O/shard_${tag}_*.log" >&2; exit 1; }; done
    "$PY" scripts/measure_distribution_metrics.py "$O" --dataset pexels --require-images 306 || exit 1
    ;;
  summary)
    "$PY" scripts/generate_pexels_main_table_summary.py || exit 1
    ;;
  *) echo "[FATAL] 未知的工作 $1" >&2; exit 2 ;;
esac
