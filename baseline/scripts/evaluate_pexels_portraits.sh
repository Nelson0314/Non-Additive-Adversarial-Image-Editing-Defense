#!/usr/bin/env bash
# Pexels 人像資料組（data/pexels_portraits）上十一個外部條件的原生預算評測，只跑 ip2p 場景。
#
# 取得指定卡的租約後，整條鏈在同一張卡上依序進行：
#   1. 未防禦分母：ip2p 編輯（arm ip2p_si18）、七道淨化、淨化後編輯。
#   2. 每個條件依求解成本由低到高：防禦生成 → ip2p 編輯 → 七道淨化 → 淨化後編輯
#      → 編輯結果 LPIPS（displacement_<條件>.csv）與保留率（retention_<條件>.csv）
#      → 合併防禦端讀數（defense_<條件>.csv）。
# 指令、種子、步數、淨化協定與主表相同（data/pexels_portraits/prompts.yaml 逐字沿用主表）。
# color 條件不在此鏈中。
#
# 續跑：每一步完成後在 runtime/pexels_portraits/ 寫入完成標記，重啟時跳過已標記的步驟；
# 未標記的編輯步驟整個 arm 重跑。防禦生成以 results_*.csv 已有的影像列判定完成，只補缺的影像。
# 量測 CSV 寫在 runtime/pexels_portraits/results/，入版控時複製到 results/pexels_portraits/。
#
# 用法（遠端；卡號由呼叫端以 measure_free_gpus.sh 與 nvidia-smi 的 compute app 清單確認後指定，
# 全局上限依 GPU_CAP 或租約目錄的既有紀錄）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_portraits.sh <卡號>
set -uo pipefail
card=${1:?需要卡號}
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source "$GPU_TOOLS/gpu_lease.sh"
gpu_policy_init || exit $?
lease_acquire "$card" pexels_portraits "$(gpu_global_cap)" || { echo "CARD $card NOT FREE OR CAP REACHED"; exit 3; }
trap 'lease_release "$card"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES=$card TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CLI=immunization_baseline.cli
D=data/pexels_portraits
A=artifacts/pexels_portraits
S=runtime/pexels_portraits
R=$S/results
CONDITIONS="diffvax mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp photoguard_c photoguard_linf"
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }
mkdir -p "$S" "$R"

step() { echo "[STEP] $(date -Is) $*"; }
done_mark() { [ -f "$S/$1.done" ]; }
mark() { touch "$S/$1.done"; }

# 影像名清單（依類別目錄），供防禦生成判定缺哪些影像。
missing_defenses() {
  "$PY" - "$D" "$A/defenses/$1" <<'EOF'
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

merge_defense_csv() {
  "$PY" - "$D" "$A/defenses/$1" "$R/defense_$1.csv" <<'EOF'
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

edit_arm() {  # <防禦圖目錄或空字串> <輸出目錄> <suffix>
  local defended=$1 out=$2 suffix=$3
  if [ -n "$defended" ]; then
    "$PY" -m $CLI.run_edits --data-root $D --defenses-dir "$defended" --output-dir "$out" \
        --scenarios ip2p --suffix "$suffix" || return $?
    "$PY" -m $CLI.evaluate_edit_completion --data-root $D --defenses-dir "$defended" \
        --output-dir "$out" --scenario ip2p --suffix "$suffix"
  else
    "$PY" -m $CLI.run_edits --data-root $D --output-dir "$out" --scenarios ip2p --suffix "$suffix"
  fi
}

# 1. 未防禦分母
if ! done_mark undefended_edits; then
  step "未防禦編輯"
  edit_arm "" $A/undefended_edits _si18 || exit 1
  mark undefended_edits
fi
if ! done_mark undefended_purified; then
  step "未防禦淨化"
  "$PY" -m $CLI.apply_purifiers --data-root $D --output-dir $A/purified/undefended || exit 1
  mark undefended_purified
fi
for PUR in $PURIFIERS; do
  done_mark "undefended_edits_$PUR" && continue
  step "未防禦淨化後編輯 $PUR"
  edit_arm "$A/purified/undefended/$PUR" "$A/purified_edits/undefended/$PUR" "_undefended_$PUR" || exit 1
  mark "undefended_edits_$PUR"
done

# 2. 各條件
for C in $CONDITIONS; do
  if ! done_mark "defenses_$C"; then
    todo=$(missing_defenses "$C") || exit 1
    if [ -n "$todo" ]; then
      first=${todo%% *}
      step "防禦生成 $C（$(echo $todo | wc -w) 張）"
      # shellcheck disable=SC2086
      "$PY" -m $CLI.generate_defenses --data-root $D --output-dir "$A/defenses/$C" \
          --conditions "$C" --images $todo --tag "from_$first" || exit 1
    fi
    [ -z "$(missing_defenses "$C")" ] || { echo "[FATAL] $C 防禦圖仍不齊" >&2; exit 1; }
    mark "defenses_$C"
  fi
  if ! done_mark "edits_$C"; then
    step "防禦後編輯 $C"
    edit_arm "$A/defenses/$C" "$A/defended_edits/$C" "_$C" || exit 1
    mark "edits_$C"
  fi
  if ! done_mark "purified_$C"; then
    step "淨化 $C"
    "$PY" -m $CLI.apply_purifiers --defenses-dir "$A/defenses/$C" --output-dir "$A/purified/$C" || exit 1
    mark "purified_$C"
  fi
  for PUR in $PURIFIERS; do
    done_mark "edits_${C}_$PUR" && continue
    step "淨化後編輯 $C $PUR"
    edit_arm "$A/purified/$C/$PUR" "$A/purified_edits/$C/$PUR" "_${C}_$PUR" || exit 1
    mark "edits_${C}_$PUR"
  done
  if ! done_mark "measured_$C"; then
    step "量測 $C"
    "$PY" -m $CLI.measure_edit_displacement --defended-edits-root $A/defended_edits \
        --undefended-edits-root $A/undefended_edits --data-root $D \
        --output-csv "$R/displacement_$C.csv" --conditions "$C" || exit 1
    "$PY" -m $CLI.measure_purified_displacement --purified-edits-root $A/purified_edits \
        --displacement-csv "$R/displacement_$C.csv" --data-root $D \
        --output-csv "$R/retention_$C.csv" --conditions "$C" || exit 1
    merge_defense_csv "$C" || exit 1
    mark "measured_$C"
  fi
done
step "CHAIN DONE"
