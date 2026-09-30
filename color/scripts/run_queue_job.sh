#!/usr/bin/env bash
# 佇列 runner：`run_queue_job.sh <工作> <卡號>`，於 color 專案根執行一件工作。
set -uo pipefail
job="$1"; gpu="$2"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$COLOR_ROOT" || exit 1
kind=${job%%:*}; arm=$(echo "$job" | cut -d: -f2)
img=$(echo "$job" | cut -d: -f3); steps=$(echo "$job" | cut -d: -f4)

merge_shards() {
  local arm="$1" src="artifacts/defense_shards/$1" dst="artifacts/defenses/$1"
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  cp -f "$src"/*/*.png "$dst"/ || return $?
  "$PY" - "$src" "$dst/results.csv" <<'PYEOF'
import csv, sys
from pathlib import Path
from immunization_core.io import write_sorted_csv
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/results.csv")):
    with f.open(encoding="utf-8", newline="") as stream:
        rows += list(csv.DictReader(stream))
write_sorted_csv(Path(sys.argv[2]), rows)
print(f"merged {len(rows)} rows", file=sys.stderr)
PYEOF
  [ "$?" -eq 0 ] || return 1
  local n; n=$(ls -1 "$dst"/*__"$arm"__def.png 2>/dev/null | wc -l)
  [ "$n" -eq 8 ] || { echo "[FATAL] merge $arm 只有 $n 張防禦圖" >&2; return 1; }
  "$PY" -m immunization_color.stages write "$arm" defense
}

case "$kind" in
  pilot) DEF_OUT="artifacts/defense_pilots/$arm/$img" \
           bash scripts/generate_condition.sh "$arm" --images "$img" --steps "$steps" ;;
  def)   DEF_OUT="artifacts/defense_shards/$arm/$img" \
           bash scripts/generate_condition.sh "$arm" --images "$img" ;;
  chain) merge_shards "$arm" && bash scripts/evaluate_condition.sh "$gpu" "$arm" ;;
  readout) bash scripts/measure_condition_results.sh "$gpu" ;;
  fid)   "$PY" -m immunization_color.cli.measure_defense_fidelity --arms $FID_ARMS --output-csv results/fidelity.csv ;;
  *) echo "[FATAL] 未知的工作：$job" >&2; exit 2 ;;
esac
