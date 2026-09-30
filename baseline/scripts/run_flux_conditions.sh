#!/usr/bin/env bash
# 依序對指定條件執行 FLUX 全表，每個條件載入一次模型。可重複執行：
# run_flux_edits 會略過 CSV 中已完成且協定一致的格。
#   bash scripts/run_flux_conditions.sh <條件>...
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
[ $# -ge 1 ] || { echo "需要至少一個條件" >&2; exit 2; }
for arm in "$@"; do
  echo "=== $arm ==="
  "$PY" -m immunization_baseline.cli.run_flux_edits --arm "$arm"
done
echo "DONE: $*"
