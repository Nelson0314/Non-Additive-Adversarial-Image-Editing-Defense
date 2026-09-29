#!/bin/bash
# `color`（lab 的現行顏色方法）取代主表的顏色列：兩條單卡鏈，狀態檔在 runs/state_color/。
#
#   bash main_table/code/color_row_chain.sh main <卡>   # basic-1：匯入 → ip2p/inpaint 編輯 → 淨化 → 淨化後重編 → UltraEdit
#   bash main_table/code/color_row_chain.sh flux <卡>   # basic-2（FLUX 權重只在這台）：等匯入完成 → FLUX 全表
#
# 防禦圖取 lab 已產出的 `lab/runs/defence/color/<圖>__color__def.png`（lab/code/color_defence.py
# 預設參數），經 immunise_as_condition.py 整理成主表版面並以同一段程式重算保真欄。
# 其餘每一步與原本顏色列的鏈（runs/edit_defended/_launch/colour_chain.sh）相同，只把腳本
# 路徑換成 main_table/code/。
set -uo pipefail
mode=$1; card=$2
source "$HOME/env.sh" >/dev/null 2>&1
cd "$HOME/image-immunization" || exit 1
bash scripts/free_cards.sh --assert "$card" || { echo "CARD $card NOT FREE"; exit 3; }
lease="$HOME/lab_leases/$(hostname)-$card"
echo "$(hostname) $$ color_row_$mode" > "$lease"
trap 'rm -f "$lease"' EXIT
export CUDA_VISIBLE_DEVICES=$card TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8
C=main_table/code
S=runs/state_color
mkdir -p "$S"
step() { echo "[STEP] $(date -Is) $*"; }
edit_stage() {
  local defended=$1 out=$2 scenario=$3 suffix=$4 rc
  "$PY" "$C/check_edit_complete.py" --data data/portraits --defended "$defended" \
      --out "$out" --scenario "$scenario" --suffix "$suffix"
  rc=$?
  [ "$rc" -eq 0 ] && return 0
  [ "$rc" -eq 1 ] || return "$rc"
  "$PY" "$C/edit_preflight.py" --data data/portraits --defended "$defended" \
      --out "$out" --scenarios "$scenario" --suffix "$suffix" --require-new-arm || return $?
  "$PY" "$C/check_edit_complete.py" --data data/portraits --defended "$defended" \
      --out "$out" --scenario "$scenario" --suffix "$suffix"
}

if [ "$mode" = main ]; then
  if [ ! -f "$S/import.done" ]; then
    step "匯入防禦圖"
    mkdir -p runs/color_import
    for f in lab/runs/defence/color/*__color__def.png; do
      n=$(basename "$f" __color__def.png)
      ln -sf "$PWD/$f" "runs/color_import/${n}__color__defended.png"
    done
    "$PY" $C/immunise_as_condition.py --run runs/color_import --variant color \
        --norm delta_e00_cap --budget 32 --data data/portraits \
        --out runs/defence_portraits/color || exit 1
    touch "$S/import.done"
  fi
  for SC in ip2p inpaint; do
    step "防禦後編輯 $SC"
    edit_stage runs/defence_portraits/color runs/edit_defended/color "$SC" _color || exit $?
  done
  if [ ! -f runs/purified/color/purified.csv ]; then
    step "淨化"
    "$PY" $C/purify_run.py --defended runs/defence_portraits/color --out runs/purified/color || exit 1
  fi
  for PUR in jpeg50 crop_resize0.1 blur1 rotate15 jpeg30 jpeg80 blur2; do
    for SC in ip2p inpaint; do
      step "淨化後編輯 $PUR $SC"
      edit_stage "runs/purified/color/$PUR" "runs/edit_purified/color/$PUR" \
          "$SC" "_color_$PUR" || exit $?
    done
  done
  touch "$S/main_edits.done"
  step "UltraEdit"
  "$PY" $C/edit_ultraedit_full.py --arms color --variant add --guidance 2.5 \
      --image-guidance 1.5 || exit 1
  touch "$S/ultraedit.done"
elif [ "$mode" = flux ]; then
  until [ -f "$S/import.done" ]; do sleep 60; done
  step "FLUX"
  "$PY" $C/edit_flux_preview.py --arm color || exit 1
  touch "$S/flux.done"
else
  echo "unknown mode $mode"; exit 2
fi
step "CHAIN DONE $mode"
