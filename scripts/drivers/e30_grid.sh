#!/bin/bash
# 【2026-08-02 起這支不該直接執行】E29 在本腳本會用到的最寬鬆運作點
# （τ=0.10、上限 150 步、平台停止）上實測，兩臂都沒有阻止編輯達成 prompt，
# 證據見 `docs/RESULTS_E29_negative.md`。照本腳本跑完 36 格只會得到同一個
# 結論的複本。檔案保留是因為它記錄了那個計畫實際打算下什麼參數。
# 那兩個結構性問題已由 E31 處理：設計見
# `docs/specs/2026-08-02-e31-positive-control.md`，網格改由
# `scripts/drivers/e31_grid.sh` 承擔（site P 單臂、三個目標函數、
# 兩個失真預算、兩個 strength）。本檔不再有後續，保留為那個計畫的紀錄。
#
# E30：主網格。修好攻擊端（E26）、判準（E25）與三道保真約束（E20/E28）之後的
# 第一次對等比較。
#
# 網格：2 site（C 非加性／P 加性基準）× 3 τ（0.05 / 0.02 / 0.10）× 6 圖 = 36 格。
# τ 的順序取 0.05 先跑：那是與 E21/E23 對應的格子，機時不足時至少有關鍵比較。
# site S 不在網格內（使用者 2026-08-01 決定）。
#
# 成本（H100 實測 2.5 s/step、40 s/格評測、GPU 使用率 82–92% 故並行無益）：
#   平均 60 步 → 約 2 小時；上限 150 步用滿 → 約 4.2 小時。
#
# 學習率必須由 E29 校準的結果填入，本腳本不給預設值。E27 定出的 0.3 / 0.03
# 是在沒有色度約束的程式上量的，直接沿用等於假設色度約束不影響解。
#
# 用法：
#   LR_C=0.3 LR_P=0.03 bash scripts/drivers/e30_grid.sh
set -euo pipefail

PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
cd "$(dirname "$0")/../.."
mkdir -p runs/logs

: "${LR_C:?請以 LR_C=<E29 校準出的 site C 學習率> 傳入}"
: "${LR_P:?請以 LR_P=<E29 校準出的 site P 學習率> 傳入}"

# `--stop_on_plateau` 是協議的一部分，不是可選項。固定步數會讓不同格子被
# 不同的東西綁住（E21–E23 §5.4：τ=0.05 由 25 步改到 100 步，site S 對 site P
# 的比值由 1.14× 反轉為 0.85×）。
BASE="--size 512 --steps 150 --k_inv 10 --n_edit 10 \
  --guidance_scale 7.5 --beta_linf 0 --margin 1.0 --alpha_lpips 0 --stop_on_plateau"

{
  echo "=== E30 主網格開始 $(date -Is)  LR_C=$LR_C LR_P=$LR_P ==="
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

  for TAU in 0.05 0.02 0.10; do
    "$PY" scripts/run_defense.py --sites C --ranks 32 --lr "$LR_C" \
      --tau_lpips "$TAU" $BASE --color_max_dev 2.0 --out "runs/e30_C_tau$TAU"
    "$PY" scripts/run_defense.py --sites P --ranks 16 --lr "$LR_P" \
      --tau_lpips "$TAU" $BASE --out "runs/e30_P_tau$TAU"
  done

  echo "=== 彙整 ==="
  "$PY" scripts/e27_report.py --prefix e30
  echo "=== 綁定者診斷 ==="
  "$PY" scripts/e27_binding_check.py runs/e30_C_tau0.05 runs/e30_P_tau0.05 \
    runs/e30_C_tau0.02 runs/e30_P_tau0.02 runs/e30_C_tau0.10 runs/e30_P_tau0.10
  echo "=== E30 主網格結束 $(date -Is) ==="
} 2>&1 | tee runs/logs/e30_grid.log
