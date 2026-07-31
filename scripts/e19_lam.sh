#!/bin/bash
# E19 — lam（LPIPS 權重）x decoder（SD vs asymmetric）完全交叉。
#
# 步數固定 400、lr 固定 0.02：E18 已證步數不是瓶頸——同一 lr 下 400->1600
# 只降 1.6%（lr=0.02）與 2.6%（lr=0.005），兩個 lr 的落點相差 1.7%。
#
# lam 取 {0.1, 1, 10}：E18 完整六張顯示綁定約束是 LPIPS 本身（僅 1/6 低於
# 0.063），而 DISTS 平均反而改善（0.0427 -> 0.0394，3/6 逐張不退）。既然
# 擋住的是 LPIPS，就要往「更用力優化 LPIPS」的方向探。DISTS 那一關會負責
# 攔下純粹靠攻擊 LPIPS 網路換來的假進步。
#
# --stack_asym 使每格同時產出 latent_opt（SD decoder）與 latent_opt_asym
# （asym decoder），故 lam x decoder 完全交叉，不需先跑一輪選 lam。
# asym_free / asym_leak 與 lam 無關，每格重算成本僅數秒，順便補上 E17
# 從未量過的 DISTS。
set -u
cd /work/nelson0314/WACV
source env.sh

for LAM in 0.1 1 10; do
  OUT="runs/e19_lam${LAM}"
  echo "=========== lam=${LAM} -> ${OUT} ==========="
  python scripts/e17_vae_floor.py \
      --out "${OUT}" \
      --steps 400 \
      --lr 0.02 \
      --lam "${LAM}" \
      --stack_asym
  echo "=========== lam=${LAM} done rc=$? ==========="
done
echo "ALL_DONE"
