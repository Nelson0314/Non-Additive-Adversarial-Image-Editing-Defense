#!/bin/bash
# E21 主網格：新保真約束（LPIPS ∩ 鈍化）下的三個設定的比較
#
# 這是 E20 的直接後果。E20 證明了 E15 的匹配只對齊 LPIPS，而 site S 在那個
# 匹配點上丟掉 15% 的高頻；新約束加上鈍化 hinge 之後，site S 的既有運作點
# 0/6 可行（runs/p4_constraint_check/）。因此 E15 的「1.15× 領先」失效，
# 必須在新可行域內重新測量。
#
# 三個設定
#   S_bic  site S + bicubic 重取樣  —— E20 §5.2 量出它在同一 LPIPS 下保留
#          99.9% 銳利度（bilinear 只有 85.0%）。這是讓非加性重回可行域的
#          假設，本實驗就是要驗證它。
#   S_bil  site S + bilinear（E15 的設定）—— 對照組。它應該被鈍化 hinge
#          節流，其 net 相對 E15 的下降幅度就是「E15 那部分效果是靠由模糊換來的」
#          的直接量度。
#   P      site P 加性基準 —— 不受新約束影響（E20 實測 6/6 可行），
#          其結果應與 E15 幾乎相同，可作為管線未被改壞的對照。
#
# 其餘協議與 E15 完全相同（步數、k_inv、n_edit、strength、beta_linf=0、
# 逐位置的 lr），使兩者可以逐格對照。唯一的差別是保真約束與重取樣模式。
set -e
cd /work/nelson0314/WACV
source env.sh
mkdir -p runs/logs

COMMON="--steps 25 --k_inv 10 --n_edit 10 --n_eot 1 --strength 0.5 --beta_linf 0"

for tau in 0.02 0.05 0.10; do
  EXTRA=""
  # 泛化評測只在主匹配點跑，與 E15 的作法一致
  if [ "$tau" = "0.05" ]; then EXTRA="--eval_strengths 0.3,0.5,0.7"; fi

  python scripts/run_defense.py --sites S --ranks 32 --lr 0.1 \
    --warp_resample bicubic \
    --tau_lpips $tau $COMMON $EXTRA \
    --out runs/e21_Sbic_tau$tau > runs/logs/e21_Sbic_tau$tau.log 2>&1
  echo "done S_bicubic tau=$tau"

  python scripts/run_defense.py --sites S --ranks 32 --lr 0.1 \
    --warp_resample bilinear \
    --tau_lpips $tau $COMMON $EXTRA \
    --out runs/e21_Sbil_tau$tau > runs/logs/e21_Sbil_tau$tau.log 2>&1
  echo "done S_bilinear tau=$tau"

  python scripts/run_defense.py --sites P --ranks 16 --lr 0.008 \
    --tau_lpips $tau $COMMON $EXTRA \
    --out runs/e21_P_tau$tau > runs/logs/e21_P_tau$tau.log 2>&1
  echo "done P tau=$tau"
done
echo "E21 ALL DONE"
