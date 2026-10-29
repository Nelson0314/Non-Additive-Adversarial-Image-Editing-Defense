#!/usr/bin/env bash
# Pexels 人像的後續量測（CPU 部分），對一個影像組執行 `measure_additional_metrics`：
#   edit_aesthetic       未淨化編輯結果的七項無參考美學指標（含未防禦編輯），aligned（等失真九個條件）、color3d 兩組各一次
#   fidelity、aesthetic  color3d_pexels 防禦圖對原圖的失真與防禦圖美學（交付 PNG）；
#                        等失真臂的同名表已由 evaluate_pexels_aligned.sh、evaluate_pexels_aligned_condition.sh 逐影像組產生
# 前置條件：等失真臂該影像組的 chain.done、chain_photoguard_encoder.done，以及 scripts/evaluate_pexels_followup.sh 的 color3d 分組
# （它建立 color3d 的本組位移表與防禦圖連結）。不成立時以結束碼 4 結束。
#
# 輸出：runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/（edit_aesthetic.csv；color3d 另有
# fidelity.csv、aesthetic.csv）；edit_aesthetic 的輸入為該組各條件位移表的合併（edit_aesthetic_plan.csv）。
#
# 用法（遠端 baseline 專案根，不需 GPU）：ENV_FILE=~/env.sh bash scripts/measure_pexels_followup_cpu.sh <影像組>
set -uo pipefail
[ $# -eq 1 ] || { echo "用法：$0 <影像組>" >&2; exit 2; }
SET=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

ALIGNED_CONDITIONS="mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp photoguard_encoder"
root=$("$PY" - "$SET" <<'EOF'
import json, sys
sets = {s["name"]: s for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]}
if sys.argv[1] not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {sys.argv[1]}")
print(sets[sys.argv[1]]["root"])
EOF
) || exit 1
AS=runtime/pexels_portraits/aligned${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
F=runtime/pexels_portraits/followup${root:+/$root}
mkdir -p "$F/done"
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }
[ -f "$AS/chain.done" ] || not_ready "等失真 $SET"
[ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET 的 photoguard_encoder"
[ -f "$F/done/color3d.done" ] || not_ready "$SET 的 color3d 分組（evaluate_pexels_followup.sh）"

edit_aesthetic() {  # <分組> <位移表...>
  local group=$1 R=$F/results/$1; shift
  [ -f "$F/done/edit_aesthetic_$group.done" ] && return 0
  mkdir -p "$R"
  "$PY" - "$R/edit_aesthetic_plan.csv" "$@" <<'EOF' || return 1
import csv, sys
rows = [r for t in sys.argv[2:] for r in csv.DictReader(open(t, encoding="utf-8")) if r["scenario"] == "ip2p"]
fields = list(rows[0])
with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[PLAN] {sys.argv[1]}（{len(rows)} 列）")
EOF
  chain_step "編輯結果美學 $group"
  "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage edit_aesthetic \
      --displacement-csv "$R/edit_aesthetic_plan.csv" --output-dir "$R" || return 1
  touch "$F/done/edit_aesthetic_$group.done"
}

chain_step "後續量測 CPU $SET"
aligned=()
for c in $ALIGNED_CONDITIONS; do aligned+=("$AS/results/displacement_$c.csv"); done
edit_aesthetic aligned "${aligned[@]}" || exit 1
edit_aesthetic color3d "$F/results/color3d/displacement_color3d_pexels.csv" || exit 1
if [ ! -f "$F/done/color3d_fidelity.done" ]; then
  chain_step "color3d_pexels 防禦圖失真與美學"
  "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage fidelity \
      --defenses-root "$V/color3d/defenses" --output-dir "$F/results/color3d" || exit 1
  "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage aesthetic \
      --defenses-root "$V/color3d/defenses" --output-dir "$F/results/color3d" || exit 1
  touch "$F/done/color3d_fidelity.done"
fi
touch "$F/done/cpu.done"
chain_step "FOLLOWUP CPU DONE $SET"
