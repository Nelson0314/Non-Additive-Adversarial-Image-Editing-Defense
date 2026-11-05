#!/usr/bin/env bash
# BlurGuard 在 Pexels 100 張子集上的 ip2p 評測鏈（淨化前、七道），一個影像組一次（佇列工作 blurguard_<影像組>，GPU 1 張）。
# 子集：runtime/pexels_portraits/blurguard/images.txt（generate_blurguard_subset.py）；求解設定：runtime/pexels_portraits/blurguard/setting.env
# （BG_ARGS，傳給 generate_blurguard_defenses）。資料根只含本組被選中的影像（符號連結）；產物寫入原生評測的產物根
# artifacts/pexels_portraits/<根>/{defenses,defended_edits,purified,purified_edits}/blurguard/，未防禦分母沿用原生評測。
# 狀態 runtime/pexels_portraits/blurguard/state/<影像組>/；讀數 runtime/pexels_portraits/blurguard/results/<影像組>/
# （defense_、displacement_、retention_blurguard.csv，fidelity.csv、aesthetic.csv）。
# 用法（遠端 baseline 專案根）：ENV_FILE=~/env.sh bash runtime/pexels_portraits/blurguard/run_blurguard_set.sh <影像組>
set -uo pipefail
SET=$1
source scripts/env.sh || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=${BLURGUARD_ROOT:-runtime/pexels_portraits/blurguard}   # 執行根；setting.env 與 scale_blurguard.py 固定取 runtime/pexels_portraits/blurguard/
G=runtime/pexels_portraits/blurguard
[ -f $G/setting.env ] || { echo "[NOT READY] 缺 $G/setting.env"; exit 4; }
source $G/setting.env
root=$("$PY" -c "import json,sys; print({s['name']: s['root'] for s in json.load(open('data/pexels_portraits/image_sets.json'))['sets']}['$SET'])") || exit 1
NS=runtime/pexels_portraits${root:+/$root}
NA=artifacts/pexels_portraits${root:+/$root}
DS=runtime/pexels_portraits/data/$SET
D=$B/data/$SET S=$B/state/$SET R=$B/results/$SET
mkdir -p "$S" "$R"
picked=$(for f in "$DS"/man/*.png "$DS"/woman/*.png; do n=$(basename "$f" .png); grep -qx "$n" $B/images.txt && echo "$(basename "$(dirname "$f")")/$n"; done)
[ -n "$picked" ] || { echo "[SKIP] $SET 沒有選中的影像"; touch "$S/chain.done"; exit 0; }
for p in $picked; do mkdir -p "$D/${p%/*}"; ln -sfn "$BASELINE_ROOT/$DS/$p.png" "$D/$p.png"; done
for e in "$DS"/*; do case $(basename "$e") in man|woman) ;; *) ln -sfn "$BASELINE_ROOT/$e" "$D/$(basename "$e")" ;; esac; done  # prompts.yaml、masks/ 等
echo "[STEP] $(date -Iseconds) $SET：$(echo $picked | wc -w) 張（GPU $CUDA_VISIBLE_DEVICES；$BG_ARGS）"
purifiers=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) || exit 1
[ -f "$NS/undefended_edits.done" ] || { echo "[FATAL] 原生 $SET 缺未防禦編輯" >&2; exit 1; }
touch "$S/undefended_edits.done" "$S/undefended_purified.done" "$S/additional_metrics.done"
for P in $purifiers; do
  [ -f "$NS/undefended_edits_$P.done" ] || { echo "[FATAL] 原生 $SET 缺未防禦 $P" >&2; exit 1; }
  touch "$S/undefended_edits_$P.done"
done
names=$(for p in $picked; do echo -n "${p#*/} "; done)
# 求解輸出到 $B/raw/<影像組>/，再以 scale_blurguard.py 依 setting.env 的 BG_SCALE 事後縮放後寫入產物根（results_*.csv 標 modified_from_paper）
RAW=$B/raw/$SET
missing=$(for n in $names; do [ -f "$RAW/${n}__blurguard__def.png" ] || echo -n "$n "; done)
if [ -n "$missing" ]; then
  # shellcheck disable=SC2086
  "$PY" -m immunization_baseline.cli.generate_blurguard_defenses --data-root "$D" --output-dir "$RAW" \
      --images $missing --tag "$(basename "$B")_$SET" $BG_ARGS || exit 1
fi
if [ ! -f "$S/scaled.done" ]; then
  : "${BG_SCALE:?setting.env 需要 BG_SCALE}"
  CUDA_VISIBLE_DEVICES= "$PY" $G/scale_blurguard.py apply "$RAW" "$NA/defenses/blurguard" "$BG_SCALE" || exit 1
  touch "$S/scaled.done"
fi
run_evaluation_chain "$D" "$NA" "$S" "$R" blurguard || exit 1
# 防禦圖補充指標只讀本批影像（產物根的 defenses/blurguard/ 可能含另一批）
mkdir -p "$R/defenses_root/blurguard"
for n in $names; do for g in "${n}__blurguard__def.png" "${n}__orig.png"; do
  ln -sfn "$BASELINE_ROOT/$NA/defenses/blurguard/$g" "$R/defenses_root/blurguard/$g"; done; done
if [ ! -f "$S/blurguard_fidelity.done" ]; then
  CUDA_VISIBLE_DEVICES= "$PY" -m immunization_baseline.cli.measure_additional_metrics --stage fidelity --defenses-root "$R/defenses_root" --output-dir "$R" || exit 1
  CUDA_VISIBLE_DEVICES= "$PY" -m immunization_baseline.cli.measure_additional_metrics --stage aesthetic --defenses-root "$R/defenses_root" --output-dir "$R" || exit 1
  touch "$S/blurguard_fidelity.done"
fi
touch "$S/chain.done"
echo "[STEP] $(date -Iseconds) BLURGUARD SET DONE $SET"
