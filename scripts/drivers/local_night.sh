#!/bin/bash
# 本機 GPU 的夜間序列（2026-08-03）。四個工作串起來跑，避免互相搶 4 GB 顯存。
#
# 為什麼要串成一支：本機是 RTX 2050 4 GB，`e31_local_probe.py` 實測無梯度
# 512² SDEdit 的峰值就已達 4873 MB（靠 Windows 共享記憶體外溢）。兩個 GPU
# 工作並行必然互搶，而 CPU 工作與 GPU 工作並行時，CPU 那側的 LPIPS 會把
# GPU 工作的 Python 執行緒餓住——實測 `p14` 在 CPU 上跑時，`e31_make_edits`
# 的單張耗時由 222 s 拉長到 30 分鐘以上。
#
# 順序的理由：
#   1. e31_train_probe  — 決定本機能不能跑縮小版訓練，後面的規劃靠它
#   2. p14_budget_thresholds — GPU 上跑，比 CPU 快一個量級
#   3. p11_degrade_ladder    — 來源已由 e31_make_edits 補齊，重跑一次
#
# 用法：bash scripts/drivers/local_night.sh
set -euo pipefail

PY="${PY:-C:/Users/nelso/miniconda3/envs/wacv/python.exe}"
cd "$(dirname "$0")/../.."
mkdir -p runs/logs
export PYTHONIOENCODING=utf-8

echo "=== 本機夜間序列開始 $(date +%H:%M:%S) ==="

echo "--- [1/3] 訓練可行性探針 $(date +%H:%M:%S) ---"
"$PY" scripts/e31_train_probe.py > runs/logs/e31_train_probe.log 2>&1 || \
  echo "  訓練探針結束（含 OOM 也算有效結果，見 log）"
grep -a "probe\]" runs/logs/e31_train_probe.log || true

echo "--- [2/3] 逐預算門檻 $(date +%H:%M:%S) ---"
"$PY" scripts/p14_budget_thresholds.py > runs/logs/p14.log 2>&1
tail -12 runs/logs/p14.log

echo "--- [3/3] 劣化階梯（來源已補齊）$(date +%H:%M:%S) ---"
"$PY" scripts/p11_degrade_ladder.py > runs/logs/p11.log 2>&1
tail -4 runs/logs/p11.log

echo "=== 本機夜間序列結束 $(date +%H:%M:%S) ==="
