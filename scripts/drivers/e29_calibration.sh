#!/bin/bash
# 【已執行，2026-08-02】結果在 `runs/e29_*`，判定見
# `docs/RESULTS_E29_negative.md` §3：site C 的四個學習率（連同 `e29b` 的兩個，
# 共跨 3 倍範圍）全部由色度 hinge 綁住，本腳本設定的判準未通過。後續的
# `e30_grid.sh` 因此沒有執行。再調 site C 的學習率已列為死路
# （`docs/NEXT_SESSION.md` §6）。
#
# E29：加入色度約束之後的學習率重新校準（主網格 E30 的前置）。
#
# 為什麼要重跑一次校準。E27 第四輪（`e27d_*`）定出 site C lr=0.3、
# site P lr=0.03，但那一輪跑在**還沒有色度約束**的程式上。E28 之後
# `gamma_chroma=100`、`tau_chroma=0.8` 預設開啟，而 site C 在舊約束下的解
# 色度偏壓是 4.97，遠超過 0.8——它一定會被壓下來。壓下來之後 lr=0.3 還適不
# 適用是未知的，故必須重測。
#
# 通過的判準（`docs/NEXT_SESSION.md` §5）：`e27_binding_check.py` 對每一格的
# 判定都是「LPIPS hinge」。不是硬上界、不是防禦 margin、不是色度 hinge。
# 這一步不通過就不要開主網格——被別道約束綁住的格子，τ 對它不起作用，
# 「匹配失真的比較」不成立。
#
# 成本：8 格 × 60 步 × 2.5 s/step ≈ 20 分鐘（H100 實測，`--no_eval` 不評測）。
#
# 用法：
#   bash scripts/drivers/e29_calibration.sh
#   PY=/path/to/python bash scripts/drivers/e29_calibration.sh   # 覆寫直譯器
set -euo pipefail

# 背景腳本不是 login shell，`python3` 會取到系統直譯器（缺 numpy），
# 故一律用絕對路徑。預設值是 Lightning AI Studio 的 conda env。
PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
cd "$(dirname "$0")/../.."
mkdir -p runs/logs

BASE="--size 512 --steps 60 --k_inv 10 --n_edit 10 --limit 2 --no_eval \
  --guidance_scale 7.5 --beta_linf 0 --tau_lpips 0.05 --margin 1.0 --alpha_lpips 0"

{
  echo "=== E29 校準開始 $(date -Is) ==="
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

  # 非加性臂：site C（色度矩陣場）。lr 由本輪重新選定。
  for LR in 0.1 0.3; do
    "$PY" scripts/run_defense.py --sites C --ranks 32 --lr "$LR" $BASE \
      --color_max_dev 2.0 --out "runs/e29_C_lr$LR"
  done

  # 加性基準：site P。它沒有色度矩陣，色度約束理論上不該綁住它；
  # 仍然一起跑，因為「理論上不該」在本專案已經被推翻過三次。
  for LR in 0.03 0.1; do
    "$PY" scripts/run_defense.py --sites P --ranks 16 --lr "$LR" $BASE \
      --out "runs/e29_P_lr$LR"
  done

  echo "=== 綁定者診斷 ==="
  "$PY" scripts/e27_binding_check.py runs/e29_C_lr0.1 runs/e29_C_lr0.3 \
    runs/e29_P_lr0.03 runs/e29_P_lr0.1
  echo "=== E29 校準結束 $(date -Is) ==="
} 2>&1 | tee runs/logs/e29_calibration.log
