#!/usr/bin/env bash
# Pexels 人像的跨編輯器 baseline 列：photoguard_encoder、dct_shield、dct_shield_y、mist_textural 以替代編輯器的 VAE 求解後，
# 由該編輯器編輯（未淨化、七道、VAE-Trans）。協定同 results/cross_model/README.md：UltraEdit（SD3）guidance 2.5、image guidance 1.5、
# variant add、512²、50 步，306 張，只在 basic-1；FLUX.1-Kontext guidance 3.5、28 步、NF4、1024²，只跑 color trial
# pexels_subsets/flux.txt 的 50 張，只在 basic-2。
#
# 工作（第二個參數起）：
#   sd3 undefended <影像組>     UltraEdit 未防禦分母。產物與完成標記沿用 color trial（artifacts_pexels/cross_model/sd3/、
#                               runtime/cross_model_pexels/sd3/jobs/），已有 done 標記的步驟不重跑，呼叫 color 的
#                               tools/run_pexels_ultraedit_job.sh 補缺，完成後寫回 done 標記
#   sd3 <條件> <影像組>         求解、淨化、編輯
#   sd3 readouts                合併逐組表，量位移、保留率、防禦圖失真、語意與身分
#   flux <條件> <塊>            flux.txt 每 10 張一塊（0–4）的求解、淨化、編輯；未防禦分母由 color 完成，直接共用
#   flux readouts               同 sd3 readouts
#
# 位置：產物 artifacts/pexels_portraits/cross_model/<sd3|flux>/（sd3 的防禦圖與淨化後輸入在 sets/<影像組>/，flux 的在 sets/chunk_<塊>/）；
#       狀態與讀數 runtime/pexels_portraits/cross_model/<sd3|flux>/（done/、results/）。
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_cross_model.sh <sd3|flux> <工作…>
set -uo pipefail
[ $# -ge 2 ] || { echo "用法：$0 <sd3|flux> <工作…>" >&2; exit 2; }
M=$1; J=$2; ARG=${3:-}
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
CONDS="photoguard_encoder dct_shield dct_shield_y mist_textural"
T=$BASELINE_ROOT/../color/trials/lut3d_defense_and_naturalness
ENC=$T/artifacts/purification/vae_trans/model/encoder.pt
UE="--guidance 2.5 --image-guidance 1.5 --variant add"
PD=data/pexels_portraits
CA=$T/artifacts_pexels/cross_model/$M
CR=$T/results/cross_model_pexels/$M
CJ=$T/runtime/cross_model_pexels/$M/jobs
X=artifacts/pexels_portraits/cross_model/$M
S=runtime/pexels_portraits/cross_model/$M
R=$S/results
case $M in
  sd3)  VAE=BleachNick/SD3_UltraEdit_freeform; RES=512 ;;
  flux) VAE=black-forest-labs/FLUX.1-Kontext-dev; RES=1024 ;;
  *) echo "[FATAL] 未知的編輯器 $M" >&2; exit 2 ;;
esac
mkdir -p "$S/done" "$R/vae_trans" "$R/additional_metrics"
SETS=$("$PY" -c 'import json; print(" ".join(s["name"] for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]))') || exit 1

# 求解缺少的防禦圖、七道與 VAE-Trans 的淨化後輸入。<資料根> <本批根> <條件> <標記前綴> <影像…>
solve_and_purify() {
  local data=$1 base=$2 c=$3 tag=$4; shift 4
  local d=$base/defenses/$c p=$base/purified_inputs n=$# todo="" im op f name
  if [ ! -f "$S/done/defenses_${c}_$tag.done" ]; then
    for im in "$@"; do [ -f "$d/${im}__${c}__def.png" ] || todo="$todo $im"; done
    if [ -n "$todo" ]; then
      chain_step "求解 $c（$(echo $todo | wc -w) 張）"
      # shellcheck disable=SC2086
      "$PY" -m $CHAIN_CLI.generate_defenses --data-root "$data" --output-dir "$d" --conditions "$c" \
          --vae-model "$VAE" --vae-resolution "$RES" --images $todo --tag "from_$(echo $todo | cut -d" " -f1)" || return 1
    fi
    [ "$(ls "$d" | grep -c "__${c}__def\.png$")" -eq "$n" ] || { echo "[FATAL] $d 不是 $n 張防禦圖" >&2; return 1; }
    touch "$S/done/defenses_${c}_$tag.done"
  fi
  if [ ! -f "$S/done/purified_${c}_$tag.done" ]; then
    chain_step "淨化 $c"
    "$PY" -m $CHAIN_CLI.apply_purifiers --defenses-dir "$d" --output-dir "$base/purified/$c" --condition "$c" || return 1
    for op in $OPS; do
      mkdir -p "$p/$c/$op/ip2p_${c}_$op"
      for f in "$base/purified/$c/$op"/*__def.png; do
        name=$(basename "$f"); cp "$f" "$p/$c/$op/ip2p_${c}_$op/${name%__def.png}__orig.png" || return 1
      done
    done
    "$PY" -m $CHAIN_CLI.apply_vae_trans --defenses-dir "$d" --condition "$c" --encoder "$ENC" \
        --output-root "$p" --fidelity-csv "$R/vae_trans/fidelity_${c}_$tag.csv" || return 1
    touch "$S/done/purified_${c}_$tag.done"
  fi
}

concat_csv() {  # <輸出> <輸入…>；同欄位，取第一個的表頭
  local out=$1; shift
  mkdir -p "$(dirname "$out")"
  { head -n 1 "$1"; for f in "$@"; do tail -n +2 "$f"; done; } > "$out.incoming" && mv "$out.incoming" "$out"
}

semantic() {  # <場景> <編輯表…>
  local scenario=$1; shift
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.measure_semantic_identity --device cuda --scenario "$scenario" --data-root "$PD" \
      --purifiers $OPS vae_trans --output-csv "$R/semantic_identity.csv" --edits-csv "$@"
}

fidelity() {  # <防禦圖根> <輸出目錄>
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 "$PY" -m $CHAIN_CLI.measure_additional_metrics --stage fidelity \
      --defenses-root "$1" --output-dir "$2"
}

chain_step "跨編輯器 $M $J $ARG（GPU $CUDA_VISIBLE_DEVICES）"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1

if [ "$M" = sd3 ] && [ "$J" = undefended ]; then
  mkdir -p "$CJ"
  for step in prep_undefended edit_undefended; do
    [ -e "$CJ/${step}_$ARG.done" ] && { chain_step "$step $ARG：color 已完成"; continue; }
    [ ! -e "$CJ/${step}_$ARG.claim" ] || { echo "[FATAL] color 的工作 ${step}_$ARG 有認領標記" >&2; exit 1; }
    bash "$T/tools/run_pexels_ultraedit_job.sh" "${step}_$ARG" || exit 1
    touch "$CJ/${step}_$ARG.done"
  done

elif [ "$M" = sd3 ] && [ "$J" = readouts ]; then
  mkdir -p "$X/edits/purified_edits"
  [ -L "$X/edits/purified_edits/undefended" ] || ln -s "$CA/edits/purified_edits/undefended" "$X/edits/purified_edits/undefended" || exit 1
  roots=""
  for s in $SETS; do roots="$roots runtime/pexels_portraits/data/$s"; done
  tables="$R/edits_undefended.csv"
  files=(); for s in $SETS; do files+=("$CR/chunks/$s/edits_undefended.csv"); done
  concat_csv "$R/edits_undefended.csv" "${files[@]}" || exit 1
  for c in $CONDS; do
    # shellcheck disable=SC2086
    "$PY" "$T/tools/run_ultraedit_pexels_edits.py" --write-preflight "$X/edits" "$c" $roots || exit 1
    files=(); for s in $SETS; do files+=("$R/chunks/$s/edits_$c.csv"); done
    concat_csv "$R/edits_$c.csv" "${files[@]}" || exit 1
    tables="$tables $R/edits_$c.csv"
  done
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.measure_edit_displacement --defended-edits-root "$X/edits/defended_edits" \
      --undefended-edits-root "$CA/edits/undefended_edits" --ip2p-arm ultraedit_undefended --data-root "$PD" \
      --conditions $CONDS --output-csv "$R/displacement.csv" || exit 1
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.measure_purified_displacement --purified-edits-root "$X/edits/purified_edits" \
      --displacement-csv "$R/displacement.csv" --data-root "$PD" --conditions $CONDS --output-csv "$R/retention.csv" || exit 1
  for s in $SETS; do fidelity "$X/sets/$s/defenses" "$R/additional_metrics/$s" || exit 1; done
  # shellcheck disable=SC2086
  semantic ultraedit $tables || exit 1

elif [ "$M" = sd3 ]; then
  c=$J s=$ARG base=$X/sets/$s D=runtime/pexels_portraits/data/$s
  # shellcheck disable=SC2046
  solve_and_purify "$D" "$base" "$c" "$s" $(ls "$D"/man/*.png "$D"/woman/*.png 2>/dev/null | xargs -n1 basename | sed 's/\.png$//' | sort) || exit 1
  if [ ! -f "$S/done/edits_${c}_$s.done" ]; then
    chain_step "編輯 $c $s"
    mkdir -p "$R/chunks/$s"
    # shellcheck disable=SC2086
    "$PY" -m $CHAIN_CLI.run_ultraedit_edits --data-root "$D" --arms "$c" --purifiers none $OPS vae_trans $UE \
        --defenses-root "$base/defenses" --purified-edits-root "$base/purified_inputs" --output-dir "$X/edits" \
        --output-csv-dir "$R/chunks/$s" || exit 1
    touch "$S/done/edits_${c}_$s.done"
  fi

elif [ "$J" = readouts ]; then
  mkdir -p "$X/edits" "$X/purified_edits"
  [ -L "$X/edits/undefended" ] || ln -s "$CA/edits/undefended" "$X/edits/undefended" || exit 1
  [ -L "$X/purified_edits/undefended" ] || ln -s "$CA/purified_edits/undefended" "$X/purified_edits/undefended" || exit 1
  tables=""
  for c in undefended $CONDS; do
    for op in none $OPS vae_trans; do
      if [ "$c" = undefended ]; then files=("$CR/chunks/$op/edits_${c}__"*.csv); else files=("$R/chunks/$op/edits_${c}__"*.csv); fi
      if [ "$op" = none ]; then out=$R/edits_$c.csv; else out=$R/purified_edits/$op/edits_$c.csv; fi
      concat_csv "$out" "${files[@]}" || exit 1
      [ "$(( $(wc -l < "$out") - 1 ))" -eq 200 ] || { echo "[FATAL] $out 不是 200 格" >&2; exit 1; }
      tables="$tables $out"
    done
  done
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.measure_flux_displacement --data-root "$PD" --edits-root "$X/edits" --edits-csv-dir "$R" \
      --conditions $CONDS --output-csv "$R/displacement.csv" || exit 1
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.measure_flux_retention --data-root "$PD" --purified-edits-root "$X/purified_edits" \
      --displacement-csv "$R/displacement.csv" --conditions $CONDS --output-csv "$R/retention.csv" || exit 1
  for k in 0 1 2 3 4; do fidelity "$X/sets/chunk_$k/defenses" "$R/additional_metrics/chunk_$k" || exit 1; done
  # shellcheck disable=SC2086
  semantic flux $tables || exit 1

else
  c=$J k=$ARG base=$X/sets/chunk_$k
  mapfile -t imgs < <(sed 's/\r$//' "$T/pexels_subsets/flux.txt" | sed -n "$(( k * 10 + 1 )),$(( (k + 1) * 10 ))p")
  [ "${#imgs[@]}" -eq 10 ] || { echo "[FATAL] 塊 $k 不是 10 張" >&2; exit 1; }
  solve_and_purify "$PD" "$base" "$c" "chunk_$k" "${imgs[@]}" || exit 1
  for op in none $OPS vae_trans; do
    [ -f "$S/done/edits_${c}_${op}_$k.done" ] && continue
    chain_step "編輯 $c $op 塊 $k"
    mkdir -p "$R/chunks/$op"
    if [ "$op" = none ]; then
      "$PY" -m $CHAIN_CLI.run_flux_edits --arm "$c" --data-root "$PD" --images "${imgs[@]}" --defenses-dir "$base/defenses/$c" \
          --output-dir "$X/edits/$c" --output-csv "$R/chunks/none/edits_${c}__$k.csv" || exit 1
    else
      "$PY" -m $CHAIN_CLI.run_flux_edits --arm "$c" --data-root "$PD" --images "${imgs[@]}" --purifier "$op" \
          --purified-edits-root "$base/purified_inputs" --output-dir "$X/purified_edits/$c/$op" \
          --output-csv "$R/chunks/$op/edits_${c}__$k.csv" || exit 1
    fi
    touch "$S/done/edits_${c}_${op}_$k.done"
  done
fi
chain_step "CROSS MODEL DONE $M $J $ARG"
