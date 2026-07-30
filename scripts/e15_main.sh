#!/bin/bash
# E15 主網格：非加性（site S，空間變形）對加性（site P，像素低秩）
#
# 這是本階段唯一的必要問題：在匹配人眼可辨失真下，非加性能否在「抵抗文字
# 編輯」上勝過加性。
#
# 協議
# - 匹配軸為 LPIPS(x_def, x_base)，掃描 τ ∈ {0.02, 0.05, 0.10}，比較兩條
#   曲線而非單點，結論不受匹配點選擇左右。
# - beta_linf=0：E13 實測 site S 完全由 L∞ 節流（LPIPS 罰則為 0、L∞ 罰則
#   41.2）。空間變形把邊緣移兩像素即可讓 L∞ 接近 1 而人眼幾乎無感，以它
#   節流非加性並不對等。L∞ 仍照常記錄，只是不進梯度。
# - 學習率逐位置以同一判準探測（E14）：τ 內 tail5_shift 最大者。
#   site S → 0.1，site P → 0.008。不強行取同一個 lr：兩個位置的 φ 單位
#   不同（位移像素 vs 像素值），同一數值不代表同一步長；判準相同才是對等。
# - 全程跑評測（不加 --no_eval）：訓練期的 shift 是對訓練噪聲種子的過擬合
#   結果，site P 實測過擬合 3.30–3.56 倍，site S 的倍率未知。唯一可比的是
#   未見種子下的評測值。
set -e
cd /work/nelson0314/WACV
source env.sh
mkdir -p runs/logs

COMMON="--steps 25 --k_inv 10 --n_edit 10 --n_eot 1 --strength 0.5 --beta_linf 0"

for tau in 0.02 0.05 0.10; do
  EXTRA=""
  # 跨 prompt / 跨強度的泛化評測只在主匹配點跑一次，避免成本三倍
  if [ "$tau" = "0.05" ]; then EXTRA="--eval_strengths 0.3,0.5,0.7"; fi

  python scripts/run_defense.py --sites S --ranks 32 --lr 0.1 \
    --tau_lpips $tau $COMMON $EXTRA \
    --out runs/e15_S_tau$tau > runs/logs/e15_S_tau$tau.log 2>&1
  echo "done S tau=$tau"

  python scripts/run_defense.py --sites P --ranks 16 --lr 0.008 \
    --tau_lpips $tau $COMMON $EXTRA \
    --out runs/e15_P_tau$tau > runs/logs/e15_P_tau$tau.log 2>&1
  echo "done P tau=$tau"
done
echo "E15 ALL DONE"
