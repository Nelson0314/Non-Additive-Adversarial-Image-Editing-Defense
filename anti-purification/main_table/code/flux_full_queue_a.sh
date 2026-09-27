#!/bin/bash
# Runs the FLUX full-table arms assigned to this queue, sequentially, one
# model load per arm. Safe to re-run: edit_flux_preview.py skips cells
# already in the arm's CSV.
set -e
cd "$HOME/image-immunization"
source "$HOME/env.sh" >/dev/null 2>&1
ARMS="$@"
for arm in $ARMS; do
  echo "=== $arm ==="
  "$PY" main_table/code/edit_flux_preview.py --arm "$arm"
done
echo "QUEUE DONE: $ARMS"
