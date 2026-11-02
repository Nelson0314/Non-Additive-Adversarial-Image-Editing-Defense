#!/usr/bin/env bash
# Pexels 人像一個影像組的後續量測中不含 VAE-Trans 的部分：scripts/evaluate_pexels_followup.sh 與
# scripts/measure_pexels_followup_cpu.sh 的同名步驟，產物、讀數與完成標記的位置相同，已完成的步驟不重做；
# 日後補 VAE-Trans 時，原腳本依這些標記只做 VAE-Trans 的步驟。
#
# 工作（第二個參數）：
#   color3d_prep        color3d_pexels 的本組防禦圖連結與位移子表（CPU；標記 done/color3d_prep.done）
#   semantic <分組>      aligned（等失真九個條件）或 color3d 的語意與身分，未淨化與七道（GPU；標記 done/semantic_<分組>.done）
#   cpu                 編輯結果美學（aligned、color3d）與 color3d_pexels 的防禦圖失真與美學（CPU；標記 done/cpu.done）
# 前置條件不成立時以結束碼 4 結束。
#
# 用法（遠端 baseline 專案根；semantic 的 CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_followup_semantic.sh <影像組> <color3d_prep|semantic <分組>|cpu>
set -uo pipefail
[ $# -ge 2 ] || { echo "用法：$0 <影像組> <color3d_prep|semantic <分組>|cpu>" >&2; exit 2; }
SET=$1 JOB=$2 GROUP=${3:-}
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALIGNED_CONDITIONS="mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp photoguard_encoder"
CT=../color/trials/lut3d_defense_and_naturalness
CP=$CT/artifacts_pexels
C3=color3d_pexels
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }
root=$("$PY" - "$SET" <<'EOF'
import json, sys
sets = {s["name"]: s for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]}
if sys.argv[1] not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {sys.argv[1]}")
print(sets[sys.argv[1]]["root"])
EOF
) || exit 1
D=runtime/pexels_portraits/data/$SET
AS=runtime/pexels_portraits/aligned${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
F=runtime/pexels_portraits/followup${root:+/$root}
mkdir -p "$F/done" "$F/results/aligned" "$F/results/color3d"
IMAGES=$(ls "$D"/man/*.png "$D"/woman/*.png 2>/dev/null | xargs -n1 basename | sed 's/\.png$//' | sort)
[ -n "$IMAGES" ] || { echo "[FATAL] $D 沒有影像" >&2; exit 1; }
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }
aligned_ready() {
  [ -f "$AS/chain.done" ] || not_ready "等失真 $SET"
  [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET 的 photoguard_encoder"
}

case $JOB in
  color3d_prep)
    export CUDA_VISIBLE_DEVICES=""
    if [ ! -f "$F/done/color3d_prep.done" ]; then
      chain_step "color3d_pexels 防禦圖連結與位移子表 $SET"
      mkdir -p "$V/color3d/defenses/$C3" || exit 1
      for im in $IMAGES; do
        for f in "${im}__${C3}__def.png" "${im}__orig.png"; do
          [ -f "$CP/defenses/$C3/$f" ] || { echo "[FATAL] 缺 $CP/defenses/$C3/$f" >&2; exit 1; }
          ln -sfn "$BASELINE_ROOT/$CP/defenses/$C3/$f" "$V/color3d/defenses/$C3/$f" || exit 1
        done
      done
      # shellcheck disable=SC2086
      "$PY" - "$CT/results/color3d_pexels/displacement.csv" "$F/results/color3d/displacement_$C3.csv" $IMAGES <<'EOF' || exit 1
import csv, sys
src, out, images = sys.argv[1], sys.argv[2], set(sys.argv[3:])
rows = [r for r in csv.DictReader(open(src, encoding="utf-8")) if r["image"] in images and r["scenario"] == "ip2p"]
if len(rows) != 4 * len(images) or {r["image"] for r in rows} != images:
    raise SystemExit(f"{src} 中本影像組的列不齊：{len(rows)} 列，應為 {4 * len(images)}")
for r in rows:
    for k in ("defended_png", "undefended_png"):
        r[k] = "../color/" + r[k]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[SUBSET] {out}（{len(rows)} 列）")
EOF
      touch "$F/done/color3d_prep.done"
    fi
    ;;
  semantic)
    : "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
    "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
    tables=()
    case $GROUP in
      aligned) aligned_ready; for c in $ALIGNED_CONDITIONS; do tables+=("$AS/results/displacement_$c.csv"); done ;;
      color3d) [ -f "$F/done/color3d_prep.done" ] || [ -f "$F/done/color3d.done" ] || not_ready "$SET 的 color3d_prep"
               tables=("$F/results/color3d/displacement_$C3.csv") ;;
      *) echo "[FATAL] 未知的分組 $GROUP" >&2; exit 2 ;;
    esac
    if [ ! -f "$F/done/semantic_$GROUP.done" ]; then
      chain_step "語意與身分 $SET $GROUP（未淨化與七道，GPU $CUDA_VISIBLE_DEVICES）"
      # 參數與 evaluate_pexels_followup.sh 相同，同一輸出檔中斷後由 measure_semantic_identity 接續
      # shellcheck disable=SC2086
      "$PY" -m $CHAIN_CLI.measure_semantic_identity --displacement-csv "${tables[@]}" --data-root "$D" \
          --output-csv "$F/results/$GROUP/semantic_identity.csv" --purifiers $PURIFIERS --device cuda || exit 1
      touch "$F/done/semantic_$GROUP.done"
    fi
    ;;
  cpu)
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
    aligned_ready
    [ -f "$F/done/color3d_prep.done" ] || [ -f "$F/done/color3d.done" ] || not_ready "$SET 的 color3d_prep"
    edit_aesthetic() {  # <分組> <位移表...>
      local group=$1 R=$F/results/$1; shift
      [ -f "$F/done/edit_aesthetic_$group.done" ] && return 0
      "$PY" - "$R/edit_aesthetic_plan.csv" "$@" <<'EOF' || return 1
import csv, sys
rows = [r for t in sys.argv[2:] for r in csv.DictReader(open(t, encoding="utf-8")) if r["scenario"] == "ip2p"]
with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[PLAN] {sys.argv[1]}（{len(rows)} 列）")
EOF
      chain_step "編輯結果美學 $group"
      "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage edit_aesthetic \
          --displacement-csv "$R/edit_aesthetic_plan.csv" --output-dir "$R" || return 1
      touch "$F/done/edit_aesthetic_$group.done"
    }
    aligned=(); for c in $ALIGNED_CONDITIONS; do aligned+=("$AS/results/displacement_$c.csv"); done
    edit_aesthetic aligned "${aligned[@]}" || exit 1
    edit_aesthetic color3d "$F/results/color3d/displacement_$C3.csv" || exit 1
    if [ ! -f "$F/done/color3d_fidelity.done" ]; then
      chain_step "color3d_pexels 防禦圖失真與美學"
      "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage fidelity \
          --defenses-root "$V/color3d/defenses" --output-dir "$F/results/color3d" || exit 1
      "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage aesthetic \
          --defenses-root "$V/color3d/defenses" --output-dir "$F/results/color3d" || exit 1
      touch "$F/done/color3d_fidelity.done"
    fi
    touch "$F/done/cpu.done"
    ;;
  *) echo "[FATAL] 未知的工作 $JOB" >&2; exit 2 ;;
esac
chain_step "FOLLOWUP SEMANTIC DONE $SET $JOB $GROUP"
