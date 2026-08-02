#!/bin/bash
# E31 R2：正對照搜尋的主網格。
# 設計見 docs/specs/2026-08-02-e31-positive-control.md §5。
#
# 只跑加性臂 site P。目的不是比較加性與非加性——E29 已證實兩臂在現行運作點
# 都是零，兩個零之間的比較沒有內容。目的是找到**任何一個**擋得下編輯的運作
# 點，為量測裝置建立本專案從未有過的正對照。
#
# 三個軸各自對應一個文獻依據：
#
#   目標函數 —— untargeted 是 E2–E29 一路用的，其最大化的量已被 E25 判定
#     與防禦成功不對應；targeted 是 PhotoGuard 的 diffusion attack 形式
#     （Salman et al., arXiv:2302.06588），推向灰色目標，成功的樣子是輸出
#     劣化；suppress 是 cross-attention 免疫那條線（arXiv:2509.10359、
#     arXiv:2512.14333）的著力點。後兩者從未在真實 SD 上產生過資料。
#
#   τ_lpips —— 0.10 保住與 E27–E29 的可比性；0.28 進文獻區間。現有資料無法
#     分辨「這個方法無效」與「任何方法在這個預算下都無效」，因為從未在文獻
#     的預算區間量過。
#
#   strength —— 0.5 是本專案現行的威脅模型；0.3 對齊 PhotoGuard 的 SDEdit
#     評測。strength 同時決定防禦訓練與攻擊，白盒下兩者必須一致，故它是
#     格子的軸而非只是評測參數。在 0.5 的全域編輯下輸出主要由 prompt 重新
#     生成，「語意不符」在原理上幾乎不可達成——這很可能是 E29 為負的最大
#     單一原因，必須量出來而不是推論。
#
# 成本：12 格 × 2 圖。全鏈格約 100 步 × 2.36 s + 33.7 s 評測 ≈ 4.5 分／圖，
# crossattn 格較低。合計約 1.5–2 小時（H100、TF32 開）。
#
# 前置：e31_calibration.sh 的綁定者判定全部是 LPIPS hinge。LR_028 由該輪
# 定出，執行前必須以環境變數傳入實際值——預設只是佔位。
#
# 用法：
#   LR_028=0.1 bash scripts/drivers/e31_grid.sh
#   PY=/path/to/python LR_028=0.1 bash scripts/drivers/e31_grid.sh
set -euo pipefail

PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
cd "$(dirname "$0")/../.."
mkdir -p runs/logs

# τ=0.10 的 lr 沿用 E29 定出的 0.03；τ=0.28 的由 R1 定出。
LR_010="${LR_010:-0.03}"
LR_028="${LR_028:-0.03}"

BASE="--sites P --ranks 16 --size 512 --steps 150 --k_inv 10 --n_edit 10 \
  --limit 2 --guidance_scale 7.5 --beta_linf 0 --margin 1.0 --alpha_lpips 0 \
  --stop_on_plateau"

{
  echo "=== E31 R2 主網格開始 $(date -Is) ==="
  echo "LR_010=$LR_010  LR_028=$LR_028"
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

  for TAU in 0.10 0.28; do
    if [ "$TAU" = "0.10" ]; then LR="$LR_010"; else LR="$LR_028"; fi
    for S in 0.5 0.3; do
      "$PY" scripts/run_defense.py $BASE --tau_lpips "$TAU" --lr "$LR" \
        --strength "$S" --defense_mode untargeted \
        --out "runs/e31_untargeted_tau${TAU}_s${S}"

      # targeted 的 L_def = LPIPS(y_def, y_target) 是最小化，本來就有界，
      # 故 margin 對這一格完全不起作用。報告中須寫明，否則 margin 欄位會
      # 被誤讀成「設得太小所以沒發展起來」。
      "$PY" scripts/run_defense.py $BASE --tau_lpips "$TAU" --lr "$LR" \
        --strength "$S" --defense_mode targeted \
        --target_image data/targets/gray.png \
        --out "runs/e31_targeted_tau${TAU}_s${S}"

      "$PY" scripts/run_defense.py $BASE --tau_lpips "$TAU" --lr "$LR" \
        --strength "$S" --defense_mode crossattn --attn_mode suppress \
        --attn_timesteps 4 --out "runs/e31_attn_tau${TAU}_s${S}"
    done
  done

  echo "=== 綁定者判定 ==="
  "$PY" scripts/e27_binding_check.py runs/e31_*_tau*_s*
  echo "=== E31 R2 主網格結束 $(date -Is) ==="
} 2>&1 | tee runs/logs/e31_grid.log
