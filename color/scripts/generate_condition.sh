#!/usr/bin/env bash
# 條件名 → 產防禦圖的指令；條件與參數的正本為 configs/conditions.yaml。
#
# 用法：bash scripts/generate_condition.sh <條件名> [額外參數…]
#        產出寫到 artifacts/defenses/<條件>/；DEF_OUT 可覆寫（分片執行用，
#        results.csv 每寫一列就整份重寫，不可共用目錄）。額外參數附在指令最後。
# 於 color 專案根執行；PY 由 scripts/env.sh 設定。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$COLOR_ROOT" || exit 1
ARM="$1"
OUT="${DEF_OUT:-artifacts/defenses/$ARM}"
EXTRA_TEXT=$("$PY" -m immunization_color.conditions "$ARM") || exit $?
EXTRA=()
[ -z "$EXTRA_TEXT" ] || mapfile -t EXTRA <<< "$EXTRA_TEXT"
exec "$PY" -m immunization_color.cli.generate_color_defenses --arm "$ARM" --out "$OUT" \
  --data data/portraits "${EXTRA[@]}" "${@:2}"
