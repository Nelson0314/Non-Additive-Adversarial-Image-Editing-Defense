#!/bin/bash
# E31 R1：τ=0.28 的學習率校準（12 格主網格 e31_grid.sh 的前置）。
#
# 設計見 docs/specs/2026-08-02-e31-positive-control.md §7.3、§8。
#
# 為什麼要重校。E29 定出的 site P lr=0.03 是在 τ_lpips=0.05 下校的。E31 把
# 失真預算拉到文獻區間（0.28；DCT-Shield 自報 0.267，PhotoGuard／MIST／AdvDM／
# SDS／DiffusionGuard 為 0.284–0.362），預算變 5.6 倍後該 lr 適不適用是未知的。
#
# 通過的判準（規格 §8）：`e27_binding_check.py` 對每一格的判定都必須是
# 「LPIPS hinge」。不是鈍化、不是色度、不是硬上界、不是防禦 margin。
# 這一步不通過就不要開主網格——被別道約束綁住的格子，τ 對它不起作用，
# 「匹配失真的比較」不成立。本機的 p13 探針已先給出數量級預估，見
# runs/p13_budget_probe/probe.csv；此處是實測，兩者要對照。
#
# 成本：4 格 × 60 步 × 2.36 s/step ≈ 10 分鐘（H100、TF32 開、--no_eval）。
# crossattn 那一格的每步成本未量過，本腳本一併量——它以 attn_timesteps=4 次
# 單步前向取代 n_edit=10 步的鏈，預期較低，但那是推測不是量測。
#
# 兩道次要門檻逐預算而定（規格 §12）：沿用 E20／E28 的絕對值 0.04 與 0.8 會把
# 可達的 LPIPS 封在 0.15–0.20（p13 實測連 i.i.d. 白高斯雜訊在 0.20 的 acut 都
# 已達 0.0414），gate 會因與防禦無關的理由失敗。τ=0.28 對應的值由
# scripts/p14_budget_thresholds.py 定出，執行前必須傳入。
#
# 用法：
#   TA_028=... TC_028=... bash scripts/drivers/e31_calibration.sh
#   PY=/path/to/python TA_028=... TC_028=... bash scripts/drivers/e31_calibration.sh
set -euo pipefail

# 背景腳本不是 login shell，`python3` 會取到系統直譯器（缺 numpy），
# 故一律用絕對路徑。預設值是 Lightning AI Studio 的 conda env。
PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
cd "$(dirname "$0")/../.."
mkdir -p runs/logs

TA_028="${TA_028:-0.0}" ; TC_028="${TC_028:-0.0}"
for V in "$TA_028" "$TC_028"; do
  if [ "$V" = "0.0" ]; then
    echo "錯誤：次要門檻仍是佔位值 0.0。請由 runs/p14_budget_thresholds/" \
         "thresholds.csv 取 budget=0.28 那一列的 tau_acut / tau_chroma。" \
         "門檻設為 0 會讓兩道 hinge 從第一步就飽和，校準結果作廢。" >&2
    exit 1
  fi
done

BASE="--sites P --ranks 16 --size 512 --steps 60 --k_inv 10 --n_edit 10 \
  --limit 2 --no_eval --guidance_scale 7.5 --beta_linf 0 --margin 1.0 \
  --alpha_lpips 0 --tau_lpips 0.28 --strength 0.5 \
  --tau_acut $TA_028 --tau_chroma $TC_028"

{
  echo "=== E31 R1 校準開始 $(date -Is) ==="
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

  # 加性臂在新預算下的 lr。E29 的 0.03 是 τ=0.05 下校的，此處跨一個數量級。
  for LR in 0.03 0.1 0.3; do
    "$PY" scripts/run_defense.py --lr "$LR" $BASE --out "runs/e31_P_tau0.28_lr$LR"
  done

  # crossattn 路徑的每步成本與可行性。它走另一條迴圈（optimize_crossattn），
  # 成本模型與上面三格不同，必須單獨量一次而非由上面外推。
  "$PY" scripts/run_defense.py --lr 0.03 $BASE \
    --defense_mode crossattn --attn_mode suppress --attn_timesteps 4 \
    --out runs/e31_P_tau0.28_attn

  echo "=== 綁定者判定（gate）==="
  "$PY" scripts/e27_binding_check.py runs/e31_P_tau0.28_*
  echo "=== E31 R1 校準結束 $(date -Is) ==="
} 2>&1 | tee runs/logs/e31_calibration.log
