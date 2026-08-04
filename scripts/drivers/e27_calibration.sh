#!/bin/bash
# E27 校準四輪在 Lightning AI H100 上實際執行的指令。
#
# 原本只存在於遠端的 /teamspace/studios/this_studio/WACV/，用途與本目錄其他
# 腳本相同：保留參數的出處。報告裡的每一輪對應下面哪一段，在此可以查到。
#
# 四輪的差別只在每次放寬一個候選有效約束，過程與結論見
# docs/RESULTS_E25-E31.md §2。
set -e
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python3
cd /teamspace/studios/this_studio/WACV
export HF_HOME=/teamspace/studios/this_studio/.cache/huggingface

BASE="--size 512 --steps 60 --k_inv 10 --n_edit 10 --limit 2 --no_eval \
  --guidance_scale 7.5 --beta_linf 0 --tau_lpips 0.05"

# 第一輪：預設的 color_max_dev=0.15。結果 LPIPS hinge 0/60 步啟動。
for LR in 0.03 0.1 0.3; do
  "$PY" scripts/run_defense.py --sites C --ranks 32 --lr "$LR" $BASE \
    --out "runs/e27_lrC_$LR"
done

# 第二輪：max_dev 放寬到 1.0。仍只有 1–8/60。
for LR in 0.03 0.1 0.3; do
  "$PY" scripts/run_defense.py --sites C --ranks 32 --lr "$LR" $BASE \
    --color_max_dev 1.0 --out "runs/e27b_lrC_$LR"
done

# 第三輪：再放寬 margin 到 1.0。仍未綁住，暴露出原始 lpips 項。
for LR in 0.1 0.3; do
  "$PY" scripts/run_defense.py --sites C --ranks 32 --lr "$LR" $BASE \
    --color_max_dev 2.0 --margin 1.0 --out "runs/e27c_C_lr$LR"
done
"$PY" scripts/run_defense.py --sites P --ranks 16 --lr 0.008 $BASE \
  --margin 1.0 --out "runs/e27c_P_lr0.008"

# 第四輪：alpha_lpips=0。τ 終於成為有效約束，並定出兩個條件的 lr。
for LR in 0.1 0.3; do
  "$PY" scripts/run_defense.py --sites C --ranks 32 --lr "$LR" $BASE \
    --color_max_dev 2.0 --margin 1.0 --alpha_lpips 0 --out "runs/e27d_C_lr$LR"
done
for LR in 0.008 0.03; do
  "$PY" scripts/run_defense.py --sites P --ranks 16 --lr "$LR" $BASE \
    --margin 1.0 --alpha_lpips 0 --out "runs/e27d_P_lr$LR"
done
