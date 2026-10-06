#!/usr/bin/env bash
# ip2p 評測鏈的函式庫（由入口腳本 source）：對一個資料根依序執行未防禦分母與各條件的
# 防禦生成 → ip2p 編輯 → 七道淨化 → 淨化後編輯 → 量測。呼叫前須已 source scripts/env.sh、
# 位於 baseline 專案根，並設定 CUDA_VISIBLE_DEVICES。
#
#   run_evaluation_chain <資料根> <產物根> <狀態目錄> <讀數目錄> <條件...>
#
# 產物根的子樹與主表 artifacts/ 相同。每一步完成後在狀態目錄寫入 <步驟>.done，重新呼叫時跳過；
# 未標記的編輯步驟整個 arm 重跑。防禦生成以 results_*.csv 已有的影像列判定完成，只補缺的影像。
# 讀數目錄得到 defense_<條件>.csv、displacement_<條件>.csv、retention_<條件>.csv，
# 以及 fidelity.csv、aesthetic.csv（防禦圖的補充失真與無參考美學指標，CPU）。
# 任何一步失敗即回傳非零，不跳過。

CHAIN_CLI=immunization_baseline.cli

chain_step() { echo "[STEP] $(date -Is) $*"; }

chain_missing_defenses() {  # <資料根> <條件的防禦圖目錄>
  "$PY" - "$1" "$2" <<'EOF'
import csv, sys
from pathlib import Path
data, out = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.stem for c in ("man", "woman") for p in (data / c).glob("*.png"))
done = set()
for f in out.glob("results_*.csv"):
    with f.open(encoding="utf-8", newline="") as s:
        done |= {r["image"] for r in csv.DictReader(s)}
print(" ".join(n for n in names if n not in done or not (out / f"{n}__{out.name}__def.png").is_file()))
EOF
}

chain_merge_defense_csv() {  # <資料根> <條件的防禦圖目錄> <輸出 CSV>
  "$PY" - "$1" "$2" "$3" <<'EOF'
import csv, sys
from pathlib import Path
from immunization_core.io import write_csv
data, out, target = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
names = sorted(p.stem for c in ("man", "woman") for p in (data / c).glob("*.png"))
rows = {}
for f in sorted(out.glob("results_*.csv")):
    with f.open(encoding="utf-8", newline="") as s:
        for r in csv.DictReader(s):
            rows[r["image"]] = r
missing = [n for n in names if n not in rows]
if missing:
    raise SystemExit(f"{out} 缺防禦列：{missing}")
write_csv(target, [rows[n] for n in names])
EOF
}

chain_edit_arm() {  # <資料根> <防禦圖目錄或空字串> <輸出目錄> <suffix>
  local data=$1 defended=$2 out=$3 suffix=$4
  if [ -n "$defended" ]; then
    "$PY" -m $CHAIN_CLI.run_edits --data-root "$data" --defenses-dir "$defended" --output-dir "$out" \
        --scenarios ip2p --suffix "$suffix" || return $?
    "$PY" -m $CHAIN_CLI.evaluate_edit_completion --data-root "$data" --defenses-dir "$defended" \
        --output-dir "$out" --scenario ip2p --suffix "$suffix"
  else
    "$PY" -m $CHAIN_CLI.run_edits --data-root "$data" --output-dir "$out" --scenarios ip2p --suffix "$suffix"
  fi
}

run_evaluation_chain() {
  local D=$1 A=$2 S=$3 R=$4; shift 4
  local conditions="$*" purifiers C PUR todo first
  purifiers=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
    || { echo "[FATAL] 無法讀取淨化協定" >&2; return 1; }
  mkdir -p "$S" "$R"

  # 1. 未防禦分母
  if [ ! -f "$S/undefended_edits.done" ]; then
    chain_step "未防禦編輯 $D"
    chain_edit_arm "$D" "" "$A/undefended_edits" _si18 || return 1
    touch "$S/undefended_edits.done"
  fi
  if [ ! -f "$S/undefended_purified.done" ]; then
    chain_step "未防禦淨化"
    "$PY" -m $CHAIN_CLI.apply_purifiers --data-root "$D" --output-dir "$A/purified/undefended" || return 1
    touch "$S/undefended_purified.done"
  fi
  for PUR in $purifiers; do
    [ -f "$S/undefended_edits_$PUR.done" ] && continue
    chain_step "未防禦淨化後編輯 $PUR"
    chain_edit_arm "$D" "$A/purified/undefended/$PUR" "$A/purified_edits/undefended/$PUR" "_undefended_$PUR" || return 1
    touch "$S/undefended_edits_$PUR.done"
  done

  # 2. 各條件
  for C in $conditions; do
    if [ ! -f "$S/defenses_$C.done" ]; then
      todo=$(chain_missing_defenses "$D" "$A/defenses/$C") || return 1
      if [ -n "$todo" ]; then
        first=${todo%% *}
        chain_step "防禦生成 $C（$(echo $todo | wc -w) 張）"
        # shellcheck disable=SC2086
        "$PY" -m $CHAIN_CLI.generate_defenses --data-root "$D" --output-dir "$A/defenses/$C" \
            --conditions "$C" --images $todo --tag "from_$first" || return 1
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
    for PUR in $purifiers; do
      [ -f "$S/edits_${C}_$PUR.done" ] && continue
      chain_step "淨化後編輯 $C $PUR"
      chain_edit_arm "$D" "$A/purified/$C/$PUR" "$A/purified_edits/$C/$PUR" "_${C}_$PUR" || return 1
      touch "$S/edits_${C}_$PUR.done"
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

  # 3. 防禦圖的補充指標（讀取 defenses/ 下的全部條件；CPU）
  if [ ! -f "$S/additional_metrics.done" ]; then
    chain_step "補充指標 fidelity、aesthetic"
    "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage fidelity \
        --defenses-root "$A/defenses" --output-dir "$R" || return 1
    "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage aesthetic \
        --defenses-root "$A/defenses" --output-dir "$R" || return 1
    touch "$S/additional_metrics.done"
  fi
  chain_step "CHAIN DONE $D"
}
