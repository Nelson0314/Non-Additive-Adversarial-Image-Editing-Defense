#!/usr/bin/env bash
# 跨條件的讀數：編輯位移與淨化後保留率。所有條件的鏈都跑完之後才有意義。
# 兩支都要一張卡（LPIPS／SigLIP 在 GPU 上算）。
# 用法：bash scripts/measure_condition_results.sh <GPU>
set -uo pipefail
GPU="$1"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$COLOR_ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
A=artifacts
DATA=data/portraits
CLI=immunization_color.cli
mkdir -p results

echo "[START] $(date -Is) displacement"
"$PY" -m $CLI.measure_edit_displacement \
    --defended-root "$A/defended_edits" --preflight "$A/undefended_edits" \
    --data "$DATA" --out results/displacement.csv
rc=$?
echo "[EXIT] $(date -Is) displacement rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

echo "[START] $(date -Is) retention"
"$PY" -m $CLI.measure_purified_displacement \
    --purified-root "$A/purified_edits" --displacement results/displacement.csv \
    --data "$DATA" --out results/retention.csv
rc=$?
echo "[EXIT] $(date -Is) retention rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
echo "[READOUT-DONE] $(date -Is)"
