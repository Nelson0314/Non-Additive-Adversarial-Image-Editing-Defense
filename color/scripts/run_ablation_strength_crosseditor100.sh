#!/usr/bin/env bash
# 消融降強度擴充到 crosseditor100 全部 100 張：B2 0.85 倍、B3 0.55 倍（各補 70 張，30 張樣本沿用）與 B3 0.40 倍（100 張），
# 目標 t ＝ r · Full 同一張的防禦圖 LPIPS，容差 ±0.015（generate_ablation_strength_defenses.py）。
# 協定：ip2p text 7.5、image 1.8、50 步、主種子 20260812；4 句 × none／jpeg50／blur2／vae_trans。
# 單一 driver（遠端 basic-2，color 專案根，setsid 執行）依序跑下列各段；每段寫 [STEP] <段> rc=<rc>，失敗即寫 [FATAL] 並停止：
#   prep       工作清單 jobs/ablation_strength_crosseditor100.txt（條件 × 100 張，第三欄 jpeg50,blur2）與 B3 0.40 倍的 generator；
#              30 張樣本的既有 claim 使佇列跳過它們（不重解、不覆寫）；新影像已有 shard 即失敗
#   queue      run_lut_queue.sh main（LANES 條線、LANE_PREFIX color_abl_x，每條線經租約工具取一張完全空的卡，沒有空卡每 5 分鐘重試）
#   check      失敗的工作必須是 failed_no_feasible_checkpoint（求解無可行檢查點），其他原因即失敗
#   readout    各條件以 run_lut_queue.sh readout 重做佇列讀數（含 30 張樣本；併入 defenses／purified_edits、防禦圖 LPIPS），經租約取卡
#   vae        各條件一張卡：沒有既有 VAE-Trans 編輯的影像做 VAE-Trans 淨化與 ip2p 編輯，寫入 artifacts_pexels/vae_trans_crosseditor100/<條件>/
#              （既有 artifacts_pexels/vae_trans/<條件>/ 不動）；同時在 CPU 量防禦圖失真與美學（tools/run_ablation_followup.sh cpu）
#   sheets     generate_ablation_strength_crosseditor100_sheets.py（400 張 × 5 列）
#   summary    measure_ablation_strength_crosseditor100_summary.py
# 用法：bash scripts/run_ablation_strength_crosseditor100.sh all | <段>
set -uo pipefail
COLOR=$HOME/image-immunization/color
cd "$COLOR" || exit 1
T=trials/lut3d_defense_and_naturalness
A=$COLOR/$T/artifacts_pexels  # 絕對路徑：bl() 先 cd 到 baseline 再執行，相對路徑會解析到 baseline 底下
B=$HOME/image-immunization/baseline
IMAGES=$COLOR/results/ablation_strength_crosseditor100/images.txt
JOBS_FILE=$T/jobs/ablation_strength_crosseditor100.txt
LOGS=$T/logs/ablation_strength_crosseditor100
CONDS="color_baseline_affine_lpips0p40_pexels color_baseline_huesat_lpips0p85_pexels color_baseline_affine_lpips0p55_pexels"
NEW_COND=color_baseline_affine_lpips0p40_pexels
GEN=../../scripts/generate_ablation_strength_defenses.py
ENC=$COLOR/$T/artifacts/purification/vae_trans/model/encoder.pt
export ENV_FILE=${ENV_FILE:-$HOME/env.sh} LEASE_LIMIT=${LEASE_LIMIT:-6}
mkdir -p "$LOGS"

bl() { ( cd "$B" && unset PYTHONPATH && source scripts/env.sh && "$PY" -m "$@" ); }
leased() {  # leased <租約名> <指令...>：取一張完全空的卡執行；沒有空卡每 5 分鐘重試
  HOSTS=$(hostname) INTERVAL=300 bash "$T/tools/run_with_lease_retry.sh" "$1" "$LOGS/lease_$1.log" "${@:2}"
}

prep() {
  [ "$(wc -l < "$IMAGES")" -eq 100 ] || { echo "[FATAL] $IMAGES 不是 100 張" >&2; return 1; }
  [ -e "$JOBS_FILE" ] && { echo "[FATAL] $JOBS_FILE 已存在，不覆寫" >&2; return 1; }
  [ -e "$T/conds/$NEW_COND.generator" ] || [ -e "$A/defense_shards/$NEW_COND" ] && { echo "[FATAL] $NEW_COND 已存在" >&2; return 1; }
  local c img
  for c in $CONDS; do
    [ -e "$A/vae_trans_crosseditor100/$c" ] && { echo "[FATAL] vae_trans_crosseditor100/$c 已存在" >&2; return 1; }
    while read -r img; do
      if [ -e "$T/runtime/claims/${c}__$img" ]; then
        [ -e "$T/runtime/done/${c}__$img" ] || { echo "[FATAL] $c $img 有 claim 但未完成" >&2; return 1; }
      elif [ -e "$A/defense_shards/$c/$img" ]; then
        echo "[FATAL] $c $img 沒有 claim 卻已有 shard" >&2; return 1
      fi
    done < "$IMAGES"
  done
  printf '%s\n' "$GEN" > "$T/conds/$NEW_COND.generator"
  for c in $CONDS; do while read -r img; do echo "$c $img jpeg50,blur2"; done < "$IMAGES"; done > "$JOBS_FILE.tmp"
  mv "$JOBS_FILE.tmp" "$JOBS_FILE"
  echo "[PREP] $(wc -l < "$JOBS_FILE") 行；待解 $(awk '{print $1"__"$2}' "$JOBS_FILE" | while read -r k; do [ -e "$T/runtime/claims/$k" ] || echo "$k"; done | wc -l) 項"
}

queue() {
  LANES=${LANES:-6} LANE_PREFIX=color_abl_x JOBS=$JOBS_FILE bash "$T/run_lut_queue.sh" main
}

check() {
  local c img n_bad=0 n_fail=0
  for c in $CONDS; do
    while read -r img; do
      [ -e "$T/runtime/done/${c}__$img" ] && continue
      [ -e "$T/runtime/failed/${c}__$img" ] || { echo "[FATAL] $c $img 既未完成也未失敗"; n_bad=$((n_bad + 1)); continue; }
      if grep -q "failed_no_feasible_checkpoint" "$A/defense_shards/$c/$img/results.csv" 2>/dev/null \
          && [ ! -e "$A/defense_shards/$c/$img/${img}__${c}__def.png" ]; then
        echo "[UNREACHABLE] $c $img"; n_fail=$((n_fail + 1))
      else
        echo "[FATAL] $c $img 失敗原因不是 failed_no_feasible_checkpoint"; n_bad=$((n_bad + 1))
      fi
    done < "$IMAGES"
  done
  echo "[CHECK] 無可行 $n_fail 項，其他失敗 $n_bad 項"
  [ "$n_bad" -eq 0 ]
}

readout() {
  local c
  for c in $CONDS; do
    JOBS=$JOBS_FILE leased "color_abl_x_readout" bash "$T/run_lut_queue.sh" readout "$c" || return 1
  done
}

vae_one() {  # vae_one <條件>：在已取卡的行程內執行
  local c=$1 img n=0
  local D=$A/defenses/$c OLD=$A/vae_trans/$c/edits/$c/vae_trans/ip2p_${c}_vae_trans V=$A/vae_trans_crosseditor100/$c
  local S=$V/staging_defenses IN=$V/inputs/$c/vae_trans/ip2p_${c}_vae_trans NEW=""
  [ -e "$V" ] && { echo "[FATAL] $V 已存在" >&2; return 1; }
  mkdir -p "$S"
  for f in "$D"/*__"${c}"__def.png; do
    img=$(basename "$f" "__${c}__def.png")
    [ -e "$OLD/${img}__p0.png" ] && continue
    cp "$f" "$D/${img}__orig.png" "$S/" || return 1
    NEW="$NEW $img"; n=$((n + 1))
  done
  echo "[VAE] $c 新增 $n 張"
  [ "$n" -gt 0 ] || return 0
  bl immunization_baseline.cli.apply_vae_trans --defenses-dir "$S" --condition "$c" --encoder "$ENC" \
      --output-root "$V/inputs" --fidelity-csv "$V/vae_trans_fidelity.csv" || return 1
  for f in "$IN"/*__orig.png; do ln -f "$f" "${f%__orig.png}__vae_trans__def.png"; done
  bl immunization_baseline.cli.run_edits --data-root "$B/data/pexels_portraits" --defenses-dir "$IN" \
      --output-dir "$V/edits/$c/vae_trans" --scenarios ip2p --suffix "_${c}_vae_trans" --images $NEW || return 1
  local got; got=$(ls "$V/edits/$c/vae_trans/ip2p_${c}_vae_trans" | grep -c "__p[0-3]\.png$")
  [ "$got" -eq $((4 * n)) ] || { echo "[FATAL] $c VAE-Trans 編輯 $got 張，應為 $((4 * n))" >&2; return 1; }
  echo "[VAE-DONE] $(date -Is) $c n=$n"
}

vae() {
  local c pids="" rc=0 p
  for c in $CONDS; do
    ( CUDA_VISIBLE_DEVICES= bash "$T/tools/run_ablation_followup.sh" cpu "$c" > "$LOGS/cpu_$c.log" 2>&1 ) &
    pids="$pids $!"
    leased "color_abl_x_vae_$c" bash scripts/run_ablation_strength_crosseditor100.sh vae_one "$c" &
    pids="$pids $!"; sleep 20
  done
  for p in $pids; do wait "$p" || rc=1; done
  return $rc
}

sheets() { source scripts/env.sh && CUDA_VISIBLE_DEVICES= "$PY" scripts/generate_ablation_strength_crosseditor100_sheets.py; }
summary() { source scripts/env.sh && CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_ablation_strength_crosseditor100_summary.py; }

run_step() {
  echo "[STEP-START] $(date -Is) $1"
  "$1"; local rc=$?
  echo "[STEP] $(date -Is) $1 rc=$rc"
  [ "$rc" -eq 0 ] || { echo "[FATAL] $(date -Is) $1 失敗，停止"; exit "$rc"; }
}

case "${1:-}" in
  all) for s in prep queue check readout vae sheets summary; do run_step "$s"; done; echo "[CROSSEDITOR100-ALL] $(date -Is) rc=0" ;;
  prep|queue|check|readout|vae|sheets|summary) run_step "$1" ;;
  vae_one) shift; vae_one "$@" ;;
  *) echo "用法：$0 all|prep|queue|check|readout|vae|sheets|summary" >&2; exit 2 ;;
esac
