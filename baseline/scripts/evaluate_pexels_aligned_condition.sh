#!/usr/bin/env bash
# Pexels 人像資料組的等失真臂（ip2p），只處理指定的一個條件；協定與 scripts/evaluate_pexels_aligned.sh 相同
# （eps 倍率取 `results/aligned/defense_<條件>.csv` 的 `eps_scale`，指令、種子、步數、七道淨化、量測同原生評測）。
# 用於 evaluate_pexels_aligned.sh 的八個條件以外的條件（例如 photoguard_encoder：eps_scale 0.2614，即 2/255，1000 步），
# 與該腳本在同一影像組上並行時互不干擾：
#   完成標記  runtime/pexels_portraits/aligned/<根>/ 下的逐條件標記（defenses_、edits_、purified_、measured_<條件>.done）
#             與 chain_<條件>.done；不寫 chain.done、additional_metrics.done
#   鎖        runtime/pexels_portraits/aligned/locks/<條件>/<影像組>
#   防禦圖失真與美學  只量本條件，輸出 runtime/pexels_portraits/aligned/<根>/results/<條件>/（fidelity.csv、aesthetic.csv），
#             以 runtime/pexels_portraits/aligned/<根>/defenses_root_<條件>/<條件>（指向防禦圖目錄的符號連結）為 --defenses-root
# 未防禦分母沿用原生評測的產物（undefended_edits/ 與 purified_edits/undefended/ 的符號連結，與 evaluate_pexels_aligned.sh 相同）。
# 只處理原生評測已完成未防禦步驟的影像組；一輪掃描下來沒有可處理的影像組時以結束碼 4 結束。
#
# 用法（遠端；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_aligned_condition.sh <條件> <影像組> [<影像組>...]
set -uo pipefail
[ $# -ge 2 ] || { echo "用法：$0 <條件> <影像組> [<影像組>...]" >&2; exit 2; }
C=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/pexels_portraits
LOCKS=runtime/pexels_portraits/aligned/locks/$C
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }
mkdir -p "$LOCKS"

scale=$("$PY" - "$C" <<'EOF'
import csv, sys
path = f"results/aligned/defense_{sys.argv[1]}.csv"
values = {row["eps_scale"] for row in csv.DictReader(open(path, encoding="utf-8"))}
if len(values) != 1:
    raise SystemExit(f"{path} 的 eps_scale 不唯一：{sorted(values)}")
print(values.pop())
EOF
) || exit 1

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

native_ready() {
  local NS=$1 P
  [ -f "$NS/undefended_edits.done" ] || return 1
  for P in $PURIFIERS; do [ -f "$NS/undefended_edits_$P.done" ] || return 1; done
}

link_undefended() {  # <原生產物根> <本臂產物根>；與 evaluate_pexels_aligned.sh 並行時對方可能先建立
  local NA=$BASELINE_ROOT/$1 A=$2
  mkdir -p "$A/purified_edits"
  [ -L "$A/undefended_edits" ] || ln -s "$NA/undefended_edits" "$A/undefended_edits" 2>/dev/null || [ -L "$A/undefended_edits" ] || return 1
  [ -L "$A/purified_edits/undefended" ] || ln -s "$NA/purified_edits/undefended" "$A/purified_edits/undefended" 2>/dev/null \
    || [ -L "$A/purified_edits/undefended" ] || return 1
}

process_set() {  # <影像組> <根>
  local SET=$1 root=$2 P todo first
  local D=runtime/pexels_portraits/data/$SET
  local NA=artifacts/pexels_portraits${root:+/$root}
  local A=artifacts/pexels_portraits/aligned${root:+/$root}
  local S=runtime/pexels_portraits/aligned${root:+/$root}
  local R=$S/results
  mkdir -p "$S" "$R"
  link_undefended "$NA" "$A" || return 1
  chain_step "等失真影像組 $SET 條件 $C eps_scale=$scale（產物 $A）"
  if [ ! -f "$S/defenses_$C.done" ]; then
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
  if [ ! -f "$S/additional_metrics_$C.done" ]; then
    chain_step "補充指標 fidelity、aesthetic（$C）"
    mkdir -p "$S/defenses_root_$C" "$R/$C"
    [ -L "$S/defenses_root_$C/$C" ] || ln -s "$BASELINE_ROOT/$A/defenses/$C" "$S/defenses_root_$C/$C" || return 1
    "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage fidelity \
        --defenses-root "$S/defenses_root_$C" --output-dir "$R/$C" || return 1
    "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage aesthetic \
        --defenses-root "$S/defenses_root_$C" --output-dir "$R/$C" || return 1
    touch "$S/additional_metrics_$C.done"
  fi
  touch "$S/chain_$C.done"
  chain_step "CHAIN DONE $SET $C"
}

while true; do
  remaining=0 progressed=0
  for SET in "$@"; do
    root=$(set_root "$SET") || exit 1
    S=runtime/pexels_portraits/aligned${root:+/$root}
    [ -f "$S/chain_$C.done" ] && continue
    remaining=1
    native_ready "runtime/pexels_portraits${root:+/$root}" || continue
    mkdir "$LOCKS/$SET" 2>/dev/null || continue
    echo "$(hostname) card $CUDA_VISIBLE_DEVICES pid $$ $(date -Is)" > "$LOCKS/$SET/owner"
    if process_set "$SET" "$root"; then
      rm -r "$LOCKS/$SET"
      progressed=1
    else
      echo "[FATAL] 影像組 $SET 條件 $C 失敗；鎖 $LOCKS/$SET 保留" >&2
      exit 1
    fi
  done
  [ $remaining -eq 0 ] && break
  [ $progressed -eq 0 ] && { chain_step "沒有可處理的影像組"; exit 4; }
done
chain_step "ALL SETS DONE $C $*"
