#!/usr/bin/env bash
# 消融降強度實驗：B2、B3 各 3 個防禦圖 LPIPS 目標（Full 同一張的 0.85、0.70、0.55 倍）× 30 張 × ip2p 4 句 × 4 淨化。
# 求解、編輯、jpeg50／blur2 淨化與佇列讀數用 trial run_lut_queue.sh；VAE-Trans 與防禦圖美學用 trial tools/run_ablation_followup.sh。
# 子指令（遠端 basic-1，color 專案根）：
#   prep           寫 conds/<條件>.generator 與 jobs/ablation_strength.txt（條件依序 × 樣本影像，第三欄 jpeg50,blur2）
#   queue          LANES 條線（預設 3；basic-1 與 basic-2 各自在本機開，LANE_PREFIX 分別為 color_abl_s、color_abl_t）跑工作清單，每條線經租約工具取一張完全空的卡（LEASE_LIMIT 預設 6，全局租約上限）
#   vae <條件...>  依序做各條件的 VAE-Trans 淨化與編輯（需 GPU，經租約工具呼叫），同時在 CPU 背景量防禦圖失真與美學
#   followup       queue 結束後，每個條件一條線（各取一張卡）做 vae；條件的佇列讀數未完成時先等待
#   all            prep → queue → followup → 拼圖與彙整（CPU）
set -uo pipefail
COLOR=$HOME/image-immunization/color
cd "$COLOR" || exit 1
T=trials/lut3d_defense_and_naturalness
B=$HOME/image-immunization/baseline
SAMPLE=$COLOR/results/ablation_strength/images.txt
JOBS_FILE=$T/jobs/ablation_strength.txt
LOGS=$T/logs/ablation_strength
CONDS="color_baseline_huesat_lpips0p85_pexels color_baseline_huesat_lpips0p70_pexels color_baseline_huesat_lpips0p55_pexels
       color_baseline_affine_lpips0p85_pexels color_baseline_affine_lpips0p70_pexels color_baseline_affine_lpips0p55_pexels"
GEN=../../scripts/generate_ablation_strength_defenses.py
export ENV_FILE=${ENV_FILE:-$HOME/env.sh}
mkdir -p "$LOGS"

prep() {
  [ -s "$SAMPLE" ] || { echo "[FATAL] 缺 $SAMPLE" >&2; return 1; }
  [ -e "$JOBS_FILE" ] && { echo "[FATAL] $JOBS_FILE 已存在，不覆寫" >&2; return 1; }
  for c in $CONDS; do
    [ -e "$T/conds/$c.generator" ] || [ -e "$T/conds/$c.args" ] && { echo "[FATAL] conds/$c 已存在" >&2; return 1; }
    [ -e "$T/artifacts_pexels/defense_shards/$c" ] && { echo "[FATAL] defense_shards/$c 已存在" >&2; return 1; }
  done
  for c in $CONDS; do printf '%s\n' "$GEN" > "$T/conds/$c.generator"; done
  for c in $CONDS; do while read -r img; do echo "$c $img jpeg50,blur2"; done < "$SAMPLE"; done > "$JOBS_FILE.tmp"
  mv "$JOBS_FILE.tmp" "$JOBS_FILE"
  echo "[PREP] $(wc -l < "$JOBS_FILE") 項"
}

queue() {  # LANE_PREFIX 依主機區分（log 檔名）；兩台主機的線共用 NFS 上的 claims，不會重複認領
  LANES=${LANES:-3} LANE_PREFIX=${LANE_PREFIX:-color_abl_s} JOBS=$JOBS_FILE LEASE_LIMIT=${LEASE_LIMIT:-6} bash "$T/run_lut_queue.sh" main
}

ready() {  # ready <條件>：樣本 30 張皆有 done／failed 標記，且有交付影像時佇列讀數已完成
  local c="$1" img n_done=0
  while read -r img; do
    [ -e "$T/runtime/done/${c}__$img" ] && { n_done=$((n_done + 1)); continue; }
    [ -e "$T/runtime/failed/${c}__$img" ] || return 1
  done < "$SAMPLE"
  [ "$n_done" -eq 0 ] || grep -q "READOUT-DONE.* $c（" "$T"/logs/lane_color_abl_[st]_*.log 2>/dev/null
}

vae() {
  local c
  for c in "$@"; do
    until ready "$c"; do sleep 120; done
    # 條件的全部影像皆無可行檢查點時佇列不做讀數、沒有防禦圖可淨化；記錄後處理下一個條件
    ls "$T/artifacts_pexels/defenses/$c/"*__def.png >/dev/null 2>&1 || { echo "[NO-DEFENSES] $(date -Is) $c"; continue; }
    ( CUDA_VISIBLE_DEVICES= bash "$T/tools/run_ablation_followup.sh" cpu "$c" > "$LOGS/followup_cpu_$c.log" 2>&1 ) &
    bash "$T/tools/run_ablation_followup.sh" vae "$c" || { echo "[FATAL] $(date -Is) $c VAE-Trans 失敗"; wait; return 1; }
  done
  wait
}

followup() {
  local c pids=""
  for c in $CONDS; do
    LEASE_LIMIT=${LEASE_LIMIT:-6} HOSTS=$(hostname) bash "$T/tools/run_with_lease_retry.sh" "color_abl_vae_$c" "$LOGS/vae_$c.log" \
        bash scripts/run_ablation_strength.sh vae "$c" &
    pids="$pids $!"; sleep 20
  done
  local rc=0 p
  for p in $pids; do wait "$p" || rc=1; done
  return $rc
}

finish() {
  source scripts/env.sh || return 1
  CUDA_VISIBLE_DEVICES= "$PY" scripts/generate_ablation_strength_sheets.py || return 1
  CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_ablation_strength_summary.py || return 1
}

case "${1:-}" in
  prep) prep ;;
  queue) queue ;;
  vae) shift; vae "$@" ;;
  followup) followup ;;
  finish) finish ;;
  all) prep && queue && followup && finish; echo "[STRENGTH-ALL] $(date -Is) rc=$?" ;;
  *) echo "用法：$0 prep|queue|vae <條件...>|followup|finish|all" >&2; exit 2 ;;
esac
