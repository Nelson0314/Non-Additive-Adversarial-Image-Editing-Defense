#!/usr/bin/env bash
# 跨臂的讀數：位移與保留率。所有臂的鏈都跑完之後才有意義。
#
# 兩支腳本都要一張卡（LPIPS／SigLIP 在 GPU 上算）。
# 用法：bash lab/scripts/readout.sh <GPU>
set -uo pipefail
GPU="$1"
source ~/env.sh
R=/nfs/home/nelson0314/image-immunization
cd "$R" || exit 1
export PYTHONPATH="$R" TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
L=lab/runs
DATA=lab/data/portraits
mkdir -p lab/results

echo "[START] $(date -Is) displacement"
"$PY" lab/code/edit_displacement.py \
    --defended-root "$L/edit_defended" --preflight "$L/edit_preflight" \
    --data "$DATA" --out lab/results/displacement.csv
echo "[EXIT] $(date -Is) displacement rc=$?"

echo "[START] $(date -Is) retention"
"$PY" lab/code/edit_retention.py \
    --purified-root "$L/edit_purified" --displacement lab/results/displacement.csv \
    --data "$DATA" --out lab/results/retention.csv
echo "[EXIT] $(date -Is) retention rc=$?"
echo "[READOUT-DONE] $(date -Is)"
