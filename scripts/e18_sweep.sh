#!/bin/bash
# E18 — latent_opt 的 lr x 步數掃描，延續 E17。
#
# 步數不另開格：latent_opt_history.json 每 25 步記一次 lpips，
# 400/800/1200/1600 的值由軌跡讀出，因此只需 3 個 lr 各跑滿 1600 步。
# --skip_asym：asym 兩臂與 lr/步數無關，E17 已量過，重跑只是浪費機時。
set -u
cd /work/nelson0314/WACV
source env.sh

for LR in 0.005 0.02 0.05; do
  OUT="runs/e18_lopt_lr${LR}"
  echo "=========== lr=${LR} -> ${OUT} ==========="
  python scripts/e17_vae_floor.py \
      --out "${OUT}" \
      --steps 1600 \
      --lr "${LR}" \
      --skip_asym
  echo "=========== lr=${LR} done rc=$? ==========="
done
echo "ALL_DONE"
