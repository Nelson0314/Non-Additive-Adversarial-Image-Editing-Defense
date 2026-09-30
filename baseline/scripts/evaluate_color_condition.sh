#!/usr/bin/env bash
# `color` 條件的主表評測：兩條單卡鏈，完成狀態寫在 runtime/color_condition/。
#
#   bash scripts/evaluate_color_condition.sh main <卡> <color 防禦圖目錄>
#       匯入 → ip2p／inpaint 編輯 → 淨化 → 淨化後重新編輯 → UltraEdit
#   bash scripts/evaluate_color_condition.sh flux <卡>
#       等待匯入完成 → FLUX 全表
#
# <color 防禦圖目錄> 為 color 專案產出的 `<圖>__color__def.png`（color/artifacts/defenses/color），經
# import_defense_artifacts 整理成主表版面，並以同一段程式重算保真欄。
# 取卡經 vendor/scripts/gpu_lease.sh，計入全局授權卡數。
set -uo pipefail
mode=$1; card=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source "$GPU_TOOLS/gpu_lease.sh"
gpu_policy_init || exit $?
lease_acquire "$card" "color_condition_$mode" "$(gpu_global_cap)" || { echo "CARD $card NOT FREE OR CAP REACHED"; exit 3; }
trap 'lease_release "$card"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES=$card TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CLI=immunization_baseline.cli
A=artifacts
S=runtime/color_condition
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }
mkdir -p "$S"
step() { echo "[STEP] $(date -Is) $*"; }
edit_stage() {
  local defended=$1 out=$2 scenario=$3 suffix=$4 rc
  "$PY" -m $CLI.check_edit_completion --data-root data/portraits --defenses-dir "$defended" \
      --output-dir "$out" --scenario "$scenario" --suffix "$suffix"
  rc=$?
  [ "$rc" -eq 0 ] && return 0
  [ "$rc" -eq 1 ] || return "$rc"
  "$PY" -m $CLI.run_edits --data-root data/portraits --defenses-dir "$defended" \
      --output-dir "$out" --scenarios "$scenario" --suffix "$suffix" --require-new-arm || return $?
  "$PY" -m $CLI.check_edit_completion --data-root data/portraits --defenses-dir "$defended" \
      --output-dir "$out" --scenario "$scenario" --suffix "$suffix"
}

if [ "$mode" = main ]; then
  color_defenses=${3:?需要 color 防禦圖目錄}
  if [ ! -f "$S/import.done" ]; then
    step "匯入防禦圖"
    mkdir -p $A/color_import
    for f in "$color_defenses"/*__color__def.png; do
      [ -f "$f" ] || { echo "[FATAL] $color_defenses 中沒有 __color__def.png" >&2; exit 1; }
      n=$(basename "$f" __color__def.png)
      ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$A/color_import/${n}__color__defended.png"
    done
    "$PY" -m $CLI.import_defense_artifacts --source-dir $A/color_import --variant color \
        --norm delta_e00_cap --budget 32 --data-root data/portraits \
        --source-settings "$color_defenses/results.csv" \
        --output-dir $A/defenses/color || exit 1
    touch "$S/import.done"
  fi
  for SC in ip2p inpaint; do
    step "防禦後編輯 $SC"
    edit_stage $A/defenses/color $A/defended_edits/color "$SC" _color || exit $?
  done
  if [ ! -f $A/purified/color/purified.csv ]; then
    step "淨化"
    "$PY" -m $CLI.apply_purifiers --defenses-dir $A/defenses/color --output-dir $A/purified/color || exit 1
  fi
  for PUR in $PURIFIERS; do
    for SC in ip2p inpaint; do
      step "淨化後編輯 $PUR $SC"
      edit_stage "$A/purified/color/$PUR" "$A/purified_edits/color/$PUR" \
          "$SC" "_color_$PUR" || exit $?
    done
  done
  touch "$S/main_edits.done"
  step "UltraEdit"
  "$PY" -m $CLI.run_ultraedit_edits --arms color --variant add --guidance 2.5 \
      --image-guidance 1.5 || exit 1
  touch "$S/ultraedit.done"
elif [ "$mode" = flux ]; then
  until [ -f "$S/import.done" ]; do sleep 60; done
  step "FLUX"
  "$PY" -m $CLI.run_flux_edits --arm color || exit 1
  touch "$S/flux.done"
else
  echo "unknown mode $mode"; exit 2
fi
step "CHAIN DONE $mode"
