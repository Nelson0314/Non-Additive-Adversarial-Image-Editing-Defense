#!/usr/bin/env bash
# Pexels 人像資料組的等失真臂（ip2p）：八個條件以主表等失真臂的 eps 倍率求解
# （`results/aligned/defense_<條件>.csv` 的 `eps_scale`，於 data/portraits 八張上對到防禦圖 LPIPS 0.3344），
# 其餘協定（指令、種子、步數、七道淨化、量測與補充指標）與 scripts/evaluate_pexels_portraits.sh 相同。
#
# 未防禦分母沿用原生評測的產物：本臂的 undefended_edits/ 與 purified_edits/undefended/ 為指向原生產物的
# 符號連結，本腳本不寫入其中。因此只處理原生評測已完成未防禦步驟的影像組；一輪掃描下來沒有可處理的影像組時
# 以結束碼 4 結束（不佔卡等待），由呼叫端稍後重新取卡執行。
# 多個 worker 以 runtime/pexels_portraits/aligned/locks/<影像組> 目錄鎖分工，影像組依參數順序處理。
#
# 每個影像組的位置（<根> 為 image_sets.json 的 root 欄，空字串表示第一批 97 張）：
#   產物       artifacts/pexels_portraits/aligned/<根>/
#   狀態       runtime/pexels_portraits/aligned/<根>/*.done
#   讀數       runtime/pexels_portraits/aligned/<根>/results/
#
# 用法（遠端；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定，例如 vendor/scripts/run_with_gpu_lease.sh）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_aligned.sh <影像組> [<影像組>...]
set -uo pipefail
[ $# -ge 1 ] || { echo "需要至少一個影像組" >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/pexels_portraits
CONDITIONS="mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp"
LOCKS=runtime/pexels_portraits/aligned/locks
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }
mkdir -p "$LOCKS"

# 條件的 eps 倍率；主表等失真臂的表中須恰有一個值。
eps_scale() {
  "$PY" - "$1" <<'EOF'
import csv, sys
path = f"results/aligned/defense_{sys.argv[1]}.csv"
values = {row["eps_scale"] for row in csv.DictReader(open(path, encoding="utf-8"))}
if len(values) != 1:
    raise SystemExit(f"{path} 的 eps_scale 不唯一：{sorted(values)}")
print(values.pop())
EOF
}

set_root() {
  "$PY" - "$DATA" "$1" <<'EOF'
import json, sys
from pathlib import Path
sets = {s["name"]: s for s in json.loads((Path(sys.argv[1]) / "image_sets.json").read_text(encoding="utf-8"))["sets"]}
if sys.argv[2] not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {sys.argv[2]}")
print(sets[sys.argv[2]]["root"])
EOF
}

# 原生評測的未防禦步驟（編輯與七道淨化後編輯）皆已完成。
native_ready() {
  local NS=$1 P
  [ -f "$NS/undefended_edits.done" ] || return 1
  for P in $PURIFIERS; do [ -f "$NS/undefended_edits_$P.done" ] || return 1; done
}

link_undefended() {  # <原生產物根> <本臂產物根>
  local NA=$BASELINE_ROOT/$1 A=$2
  mkdir -p "$A/purified_edits"
  [ -L "$A/undefended_edits" ] || ln -s "$NA/undefended_edits" "$A/undefended_edits" || return 1
  [ -L "$A/purified_edits/undefended" ] || ln -s "$NA/purified_edits/undefended" "$A/purified_edits/undefended" || return 1
}

process_set() {  # <影像組> <根>
  local SET=$1 root=$2 C P scale todo first
  local D=runtime/pexels_portraits/data/$SET
  local NA=artifacts/pexels_portraits${root:+/$root}
  local A=artifacts/pexels_portraits/aligned${root:+/$root}
  local S=runtime/pexels_portraits/aligned${root:+/$root}
  local R=$S/results
  mkdir -p "$S" "$R"
  link_undefended "$NA" "$A" || return 1
  chain_step "等失真影像組 $SET（產物 $A）"
  for C in $CONDITIONS; do
    if [ ! -f "$S/defenses_$C.done" ]; then
      scale=$(eps_scale "$C") || return 1
      todo=$(chain_missing_defenses "$D" "$A/defenses/$C") || return 1
      if [ -n "$todo" ]; then
        first=${todo%% *}
        chain_step "防禦生成 $C eps_scale=$scale（$(echo $todo | wc -w) 張）"
        # shellcheck disable=SC2086
        "$PY" -m $CHAIN_CLI.generate_defenses --data-root "$D" --output-dir "$A/defenses/$C" \
            --conditions "$C" --eps-scale "$scale" --images $todo --tag "from_$first" || return 1
      fi
      [ -z "$(chain_missing_defenses "$D" "$A/defenses/$C")" ] || { echo "[FATAL] $C 防禦圖仍不齊" >&2; return 1; }
      touch "$S/defenses_$C.done"
    fi
    if [ ! -f "$S/edits_$C.done" ]; then
      chain_step "防禦後編輯 $C"
      chain_edit_arm "$D" "$A/defenses/$C" "$A/defended_edits/$C" "_$C" || return 1
      touch "$S/edits_$C.done"
    fi
    if [ ! -f "$S/purified_$C.done" ]; then
      chain_step "淨化 $C"
      "$PY" -m $CHAIN_CLI.apply_purifiers --defenses-dir "$A/defenses/$C" --output-dir "$A/purified/$C" || return 1
      touch "$S/purified_$C.done"
    fi
    for P in $PURIFIERS; do
      [ -f "$S/edits_${C}_$P.done" ] && continue
      chain_step "淨化後編輯 $C $P"
      chain_edit_arm "$D" "$A/purified/$C/$P" "$A/purified_edits/$C/$P" "_${C}_$P" || return 1
      touch "$S/edits_${C}_$P.done"
    done
    if [ ! -f "$S/measured_$C.done" ]; then
      chain_step "量測 $C"
      "$PY" -m $CHAIN_CLI.measure_edit_displacement --defended-edits-root "$A/defended_edits" \
          --undefended-edits-root "$A/undefended_edits" --data-root "$D" \
          --output-csv "$R/displacement_$C.csv" --conditions "$C" || return 1
      "$PY" -m $CHAIN_CLI.measure_purified_displacement --purified-edits-root "$A/purified_edits" \
          --displacement-csv "$R/displacement_$C.csv" --data-root "$D" \
          --output-csv "$R/retention_$C.csv" --conditions "$C" || return 1
      chain_merge_defense_csv "$D" "$A/defenses/$C" "$R/defense_$C.csv" || return 1
      touch "$S/measured_$C.done"
    fi
  done
  if [ ! -f "$S/additional_metrics.done" ]; then
    chain_step "補充指標 fidelity、aesthetic"
    "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage fidelity \
        --defenses-root "$A/defenses" --output-dir "$R" || return 1
    "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage aesthetic \
        --defenses-root "$A/defenses" --output-dir "$R" || return 1
    touch "$S/additional_metrics.done"
  fi
  touch "$S/chain.done"
  chain_step "CHAIN DONE $SET"
}

while true; do
  remaining=0 progressed=0
  for SET in "$@"; do
    root=$(set_root "$SET") || exit 1
    S=runtime/pexels_portraits/aligned${root:+/$root}
    [ -f "$S/chain.done" ] && continue
    remaining=1
    native_ready "runtime/pexels_portraits${root:+/$root}" || continue
    mkdir "$LOCKS/$SET" 2>/dev/null || continue
    echo "$(hostname) card $CUDA_VISIBLE_DEVICES pid $$ $(date -Is)" > "$LOCKS/$SET/owner"
    if process_set "$SET" "$root"; then
      rm -r "$LOCKS/$SET"
      progressed=1
    else
      echo "[FATAL] 影像組 $SET 失敗；鎖 $LOCKS/$SET 保留" >&2
      exit 1
    fi
  done
  [ $remaining -eq 0 ] && break
  [ $progressed -eq 0 ] && { chain_step "沒有可處理的影像組"; exit 4; }
done
chain_step "ALL SETS DONE $*"
