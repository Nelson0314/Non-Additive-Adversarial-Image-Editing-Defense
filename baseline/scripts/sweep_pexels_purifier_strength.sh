#!/usr/bin/env bash
# Pexels 人像的淨化強度掃描（ip2p），對一個影像組的一個條件執行。
# 條件：undefended（分母）、color3d_pexels、等失真臂的 dct_shield、mist、dia_pt。
#
# 13 個設定與 data/portraits 的掃描（results/sweeps/ip2p/purifier_strength/）相同：jpeg95、90、70、50、30、10，blur0.5、1、2、3，
# resample_roundtrip384、256、128。blur1、blur2、jpeg30、jpeg50 與七道協定相同（portraits 上兩者送進編輯器的輸入 md5 相同），
# 重用七道的淨化後編輯（目錄符號連結，只讀）；其餘 9 個設定以 `apply_purifier_sweep` 淨化後編輯。
# 量測：`measure_purified_displacement`（核心的 retention 流程）。未淨化的位移取自各條件既有的位移表。
#
# 位置（<根> 為 image_sets.json 的 root 欄，空字串表示第一批 97 張）：
#   產物  artifacts/sweeps/ip2p/purifier_strength/pexels_portraits/<根>/{purified,purified_edits}/<條件>/<設定>/
#   狀態  runtime/pexels_portraits/purifier_strength/<根>/done/
#   讀數  runtime/pexels_portraits/purifier_strength/<根>/results/retention_<條件>.csv
# 前置條件：undefended 需原生評測的七道未防禦編輯；其餘條件需 undefended 完成；dct_shield、mist、dia_pt 需等失真臂
# 該條件的 measured_<條件>.done；color3d_pexels 需 scripts/evaluate_pexels_followup.sh 的 color3d 分組。不成立時以結束碼 4 結束。
#
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/sweep_pexels_purifier_strength.sh <影像組> <條件>
set -uo pipefail
[ $# -eq 2 ] || { echo "用法：$0 <影像組> <條件>" >&2; exit 2; }
SET=$1 C=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NEW_SPECS="jpeg:95 jpeg:90 jpeg:70 jpeg:10 blur:0.5 blur:3 resample_roundtrip:384 resample_roundtrip:256 resample_roundtrip:128"
REUSED="blur1 blur2 jpeg30 jpeg50"
CP=../color/trials/lut3d_defense_and_naturalness/artifacts_pexels

root=$("$PY" - "$SET" <<'EOF'
import json, sys
sets = {s["name"]: s for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]}
if sys.argv[1] not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {sys.argv[1]}")
print(sets[sys.argv[1]]["root"])
EOF
) || exit 1
D=runtime/pexels_portraits/data/$SET
NS=runtime/pexels_portraits${root:+/$root}
NA=artifacts/pexels_portraits${root:+/$root}
AS=runtime/pexels_portraits/aligned${root:+/$root}
AA=artifacts/pexels_portraits/aligned${root:+/$root}
FS=runtime/pexels_portraits/followup${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
A=artifacts/sweeps/ip2p/purifier_strength/pexels_portraits${root:+/$root}
S=runtime/pexels_portraits/purifier_strength${root:+/$root}
mkdir -p "$S/done" "$S/results"
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }

case $C in
  undefended)
    for P in $REUSED; do [ -f "$NS/undefended_edits_$P.done" ] || not_ready "原生 $SET 的未防禦 $P"; done
    source_args=(--data-root "$D")
    reuse_root=$BASELINE_ROOT/$NA/purified_edits/undefended
    ;;
  color3d_pexels)
    [ -f "$S/done/undefended.done" ] || not_ready "$SET 的未防禦分母"
    [ -f "$FS/done/color3d.done" ] || not_ready "$SET 的 followup color3d"
    source_args=(--defenses-dir "$V/color3d/defenses/$C" --condition "$C")
    reuse_root=$BASELINE_ROOT/$CP/purified_edits/$C
    displacement=$FS/results/color3d/displacement_$C.csv
    ;;
  dct_shield|mist|dia_pt)
    [ -f "$S/done/undefended.done" ] || not_ready "$SET 的未防禦分母"
    [ -f "$AS/measured_$C.done" ] || not_ready "等失真 $SET 的 $C"
    source_args=(--defenses-dir "$AA/defenses/$C" --condition "$C")
    reuse_root=$BASELINE_ROOT/$AA/purified_edits/$C
    displacement=$AS/results/displacement_$C.csv
    ;;
  *) echo "[FATAL] 未知的條件 $C" >&2; exit 2 ;;
esac

chain_step "淨化強度掃描 $SET $C（GPU $CUDA_VISIBLE_DEVICES）"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
mkdir -p "$A/purified_edits/$C"
for P in $REUSED; do
  [ -d "$reuse_root/$P" ] || { echo "[FATAL] 缺重用的編輯 $reuse_root/$P" >&2; exit 1; }
  [ -L "$A/purified_edits/$C/$P" ] || ln -s "$reuse_root/$P" "$A/purified_edits/$C/$P" || exit 1
done
if [ ! -f "$S/done/purified_$C.done" ]; then
  chain_step "淨化 $C（9 個設定）"
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.apply_purifier_sweep "${source_args[@]}" --output-dir "$A/purified/$C" --purifiers $NEW_SPECS || exit 1
  touch "$S/done/purified_$C.done"
fi
for d in "$A/purified/$C"/*/; do
  P=$(basename "$d")
  [ -f "$S/done/edits_${C}_$P.done" ] && continue
  chain_step "淨化後編輯 $C $P"
  chain_edit_arm "$D" "$A/purified/$C/$P" "$A/purified_edits/$C/$P" "_${C}_$P" || exit 1
  touch "$S/done/edits_${C}_$P.done"
done
if [ "$C" != undefended ] && [ ! -f "$S/done/retention_$C.done" ]; then
  chain_step "量測 $C"
  [ -L "$A/purified_edits/undefended" ] || [ -d "$A/purified_edits/undefended" ] || { echo "[FATAL] 缺分母 $A/purified_edits/undefended" >&2; exit 1; }
  "$PY" -m $CHAIN_CLI.measure_purified_displacement --purified-edits-root "$A/purified_edits" \
      --displacement-csv "$displacement" --data-root "$D" --output-csv "$S/results/retention_$C.csv" --conditions "$C" || exit 1
  touch "$S/done/retention_$C.done"
fi
[ "$C" = undefended ] && touch "$S/done/undefended.done"
chain_step "SWEEP DONE $SET $C"
